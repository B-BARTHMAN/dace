import dace
import dace.transformation as xf
from dace.sdfg import nodes, graph as gr
from dace.transformation.blockedfp.cast_inout_util import cast_inout_util
from typing import List

from dace.transformation.blockedfp import libraries as bfplib

N = dace.symbol("N")
M = dace.symbol("M")
L = dace.symbol("L")
S = dace.symbol("S")

class BlockedFPGEMMTransform(xf.SingleStateTransformation):

    A_access = xf.PatternNode(dace.nodes.AccessNode)
    B_access = xf.PatternNode(dace.nodes.AccessNode)
    C_access = xf.PatternNode(dace.nodes.AccessNode)
    Library = xf.PatternNode(dace.nodes.LibraryNode)
    
    def __init__(self, blocking_factor: int = 16, use_int8: bool = False):
        self._blocking_factor = blocking_factor
        self._use_int8 = use_int8  # <-- new parameter

    @classmethod
    def expressions(cls):
        state = gr.OrderedMultiDiConnectorGraph()
        state.add_nedge(cls.A_access, cls.Library, dace.Memlet())
        state.add_nedge(cls.B_access, cls.Library, dace.Memlet())
        state.add_nedge(cls.Library, cls.C_access, dace.Memlet())
        return [state]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return self.Library.label == "_MatMult_"

    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG):
        for edge in graph.out_edges(self.A_access):
            if edge.dst != self.Library:
                continue
            shape = sdfg.arrays[edge.data.data].shape
            symbol_N = str(dace.symbolic.SymExpr(f"{shape[0]}/{self._blocking_factor}"))
            symbol_M = str(dace.symbolic.SymExpr(f"{shape[1]}/{self._blocking_factor}"))
        
        for edge in graph.out_edges(self.B_access):
            if edge.dst != self.Library:
                continue
            shape = sdfg.arrays[edge.data.data].shape
            symbol_L = str(dace.symbolic.SymExpr(f"{shape[1]}/{self._blocking_factor}"))
        
        # Pick library node class
        libnode_class = bfplib.BFPGemmInt8Node if self._use_int8 else bfplib.BFPGemmNode
        libnode = libnode_class(
            name=f"_BFP_MatMult_gemm{self.A_access.data}",
            symbol_mapping={"N": symbol_N, "M": symbol_M, "L": symbol_L, "S": "S"}
        )
        graph.add_node(libnode)

        # Pick castin/castout classes
        castin_class = bfplib.BFPCastinNodeInt8 if self._use_int8 else bfplib.BFPCastinNode
        castout_class = bfplib.BFPCastoutNodeInt8 if self._use_int8 else bfplib.BFPCastoutNode

        # Define input/output names depending on int8 flag
        if self._use_int8:
            a_in = ("a_bias", "a_scale", "a_ints")
            b_in = ("b_bias", "b_scale", "b_ints")
            c_out = ("c_bias", "c_scale", "c_ints")
        else:
            a_in = ("a_bias", "a_scale", "a_fp")
            b_in = ("b_bias", "b_scale", "b_fp")
            c_out = ("c_bias", "c_scale", "c_fp")

        # -------------------
        # Connect A access
        for edge in graph.out_edges(self.A_access):
            if edge.dst != self.Library:
                continue

            castin = castin_class(
                name=f"BFPCastin_{edge.data.data}",
                symbol_mapping={"N": symbol_N, "M": symbol_M, "S": "S"}
            )
            graph.add_node(castin)

            info = cast_inout_util(sdfg, edge, self._blocking_factor, self._use_int8)

            out1 = graph.add_access(info.fp)
            out2 = graph.add_access(info.bias)
            out3 = graph.add_access(info.scale)

            # Original array -> castin
            graph.add_edge(edge.src, edge.src_conn, castin, "array", edge.data)

            # Map outputs in correct order
            conn_name = "ints" if self._use_int8 else "fp"
            graph.add_edge(castin, conn_name, out1, None, dace.Memlet(data=info.fp, subset=info.fp_subset))
            graph.add_edge(out1, None, libnode, a_in[2], dace.Memlet(data=info.fp, subset=info.fp_subset))

            graph.add_edge(castin, "bias", out2, None, dace.Memlet(data=info.bias, subset=info.block_subset))
            graph.add_edge(out2, None, libnode, a_in[0], dace.Memlet(data=info.bias, subset=info.block_subset))

            graph.add_edge(castin, "scale", out3, None, dace.Memlet(data=info.scale, subset=info.block_subset))
            graph.add_edge(out3, None, libnode, a_in[1], dace.Memlet(data=info.scale, subset=info.block_subset))

        # -------------------
        # Connect B access
        for edge in graph.out_edges(self.B_access):
            if edge.dst != self.Library:
                continue

            castin = castin_class(
                name=f"BFPCastin_{edge.data.data}",
                symbol_mapping={"N": symbol_M, "M": symbol_L, "S": "S"}
            )
            graph.add_node(castin)

            info = cast_inout_util(sdfg, edge, self._blocking_factor, self._use_int8)

            out1 = graph.add_access(info.fp)
            out2 = graph.add_access(info.bias)
            out3 = graph.add_access(info.scale)

            graph.add_edge(edge.src, edge.src_conn, castin, "array", edge.data)
            
            conn_name = "ints" if self._use_int8 else "fp"
            graph.add_edge(castin, conn_name, out1, None, dace.Memlet(data=info.fp, subset=info.fp_subset))
            graph.add_edge(out1, None, libnode, b_in[2], dace.Memlet(data=info.fp, subset=info.fp_subset))

            graph.add_edge(castin, "bias", out2, None, dace.Memlet(data=info.bias, subset=info.block_subset))
            graph.add_edge(out2, None, libnode, b_in[0], dace.Memlet(data=info.bias, subset=info.block_subset))

            graph.add_edge(castin, "scale", out3, None, dace.Memlet(data=info.scale, subset=info.block_subset))
            graph.add_edge(out3, None, libnode, b_in[1], dace.Memlet(data=info.scale, subset=info.block_subset))

        # -------------------
        # Connect C access
        for edge in graph.out_edges(self.Library):
            if edge.dst != self.C_access:
                continue

            castout = castout_class(
                name=f"BFPCastout_{edge.data.data}",
                symbol_mapping={"N": symbol_N, "M": symbol_L, "S": "S"}
            )
            graph.add_node(castout)

            info = cast_inout_util(sdfg, edge, self._blocking_factor, self._use_int8)

            out1 = graph.add_access(info.fp)
            out2 = graph.add_access(info.bias)
            out3 = graph.add_access(info.scale)
            
            conn_name = "ints" if self._use_int8 else "fp"
            graph.add_edge(libnode, c_out[2], out1, None, dace.Memlet(data=info.fp, subset=info.fp_subset))
            graph.add_edge(out1, None, castout, conn_name, dace.Memlet(data=info.fp, subset=info.fp_subset))

            graph.add_edge(libnode, c_out[0], out2, None, dace.Memlet(data=info.bias, subset=info.block_subset))
            graph.add_edge(out2, None, castout, "bias", dace.Memlet(data=info.bias, subset=info.block_subset))

            graph.add_edge(libnode, c_out[1], out3, None, dace.Memlet(data=info.scale, subset=info.block_subset))
            graph.add_edge(out3, None, castout, "scale", dace.Memlet(data=info.scale, subset=info.block_subset))

            # castout → original array
            graph.add_edge(castout, "array", edge.dst, edge.dst_conn, edge.data)

        # Delete old library node
        graph.remove_node(self.Library)
