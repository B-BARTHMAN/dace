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
    
    def __init__(self, blocking_factor: int = 16):
        self._blocking_factor = blocking_factor

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
            
            # Get symbols for shape
            shape = sdfg.arrays[edge.data.data].shape
            symbol_N = str(dace.symbolic.SymExpr(f"{shape[0]}/{self._blocking_factor}"))
            symbol_M = str(dace.symbolic.SymExpr(f"{shape[1]}/{self._blocking_factor}"))
        
        for edge in graph.out_edges(self.B_access):
            if edge.dst != self.Library:
                continue
            
            # Get symbols for shape
            shape = sdfg.arrays[edge.data.data].shape
            symbol_L = str(dace.symbolic.SymExpr(f"{shape[1]}/{self._blocking_factor}"))
        
        # Add new library node
        libnode = bfplib.BFPGemmNode(
            name = "_BFP_MatMult_gemm",
            symbol_mapping={
                "N": symbol_N,
                "M": symbol_M,
                "L": symbol_L,
                "S": "S"
                }
            )
        graph.add_node(libnode)
        
        # Connect A access
        for edge in graph.out_edges(self.A_access):
            if edge.dst != self.Library:
                continue
            
            castin = bfplib.BFPCastinNode(
                name = f"BFPCastin_{edge.data.data}",
                symbol_mapping={
                    "N": symbol_N,
                    "M": symbol_M,
                    "S": "S"
                }
            )
            graph.add_node(castin)      

            info = cast_inout_util(sdfg, edge, self._blocking_factor)       

            fp = graph.add_access(info.fp)
            bias = graph.add_access(info.bias)
            scale = graph.add_access(info.scale)        

            # Original array -> castin
            graph.add_edge(edge.src, edge.src_conn, castin, "array", edge.data)     

            # fp path
            graph.add_edge(
                castin, "fp", fp, None,
                dace.Memlet(data=info.fp, subset=info.fp_subset)
            )
            graph.add_edge(
                fp, None, libnode, "a_fp",
                dace.Memlet(data=info.fp, subset=info.fp_subset)
            )       

            # bias path
            graph.add_edge(
                castin, "bias", bias, None,
                dace.Memlet(data=info.bias, subset=info.block_subset)
            )
            graph.add_edge(
                bias, None, libnode, "a_bias",
                dace.Memlet(data=info.bias, subset=info.block_subset)
            )       

            # scale path
            graph.add_edge(
                castin, "scale", scale, None,
                dace.Memlet(data=info.scale, subset=info.block_subset)
            )
            graph.add_edge(
                scale, None, libnode, "a_scale",
                dace.Memlet(data=info.scale, subset=info.block_subset)
            )
        
        # Connect B access
        for edge in graph.out_edges(self.B_access):
            if edge.dst != self.Library:
                continue
            
            castin = bfplib.BFPCastinNode(
                name = f"BFPCastin_{edge.data.data}",
                symbol_mapping={
                    "N": symbol_M,
                    "M": symbol_L,
                    "S": "S"
                }
            )
            graph.add_node(castin)      

            info = cast_inout_util(sdfg, edge, self._blocking_factor)       

            fp = graph.add_access(info.fp)
            bias = graph.add_access(info.bias)
            scale = graph.add_access(info.scale)        

            # Original array -> castin
            graph.add_edge(edge.src, edge.src_conn, castin, "array", edge.data)     

            # fp path
            graph.add_edge(
                castin, "fp", fp, None,
                dace.Memlet(data=info.fp, subset=info.fp_subset)
            )
            graph.add_edge(
                fp, None, libnode, "b_fp",
                dace.Memlet(data=info.fp, subset=info.fp_subset)
            )       

            # bias path
            graph.add_edge(
                castin, "bias", bias, None,
                dace.Memlet(data=info.bias, subset=info.block_subset)
            )
            graph.add_edge(
                bias, None, libnode, "b_bias",
                dace.Memlet(data=info.bias, subset=info.block_subset)
            )       

            # scale path
            graph.add_edge(
                castin, "scale", scale, None,
                dace.Memlet(data=info.scale, subset=info.block_subset)
            )
            graph.add_edge(
                scale, None, libnode, "b_scale",
                dace.Memlet(data=info.scale, subset=info.block_subset)
            )
        
        # Connect C access
        for edge in graph.out_edges(self.Library):
            if edge.dst != self.C_access:
                continue
            
            castout = bfplib.BFPCastoutNode(
                name = f"BFPCastout_{edge.data.data}",
                symbol_mapping={
                    "N": symbol_N,
                    "M": symbol_L,
                    "S": "S"
                }
            )
            graph.add_node(castout)

            info = cast_inout_util(sdfg, edge, self._blocking_factor)

            fp = graph.add_access(info.fp)
            bias = graph.add_access(info.bias)
            scale = graph.add_access(info.scale)

            # fp path
            graph.add_edge(
                libnode, "c_fp", fp, None,
                dace.Memlet(data=info.fp, subset=info.fp_subset)
            )
            graph.add_edge(
                fp, None, castout, "fp",
                dace.Memlet(data=info.fp, subset=info.fp_subset)
            )

            # bias path
            graph.add_edge(
                libnode, "c_bias", bias, None,
                dace.Memlet(data=info.bias, subset=info.block_subset)
            )
            graph.add_edge(
                bias, None, castout, "bias",
                dace.Memlet(data=info.bias, subset=info.block_subset)
            )

            # scale path
            graph.add_edge(
                libnode, "c_scale", scale, None,
                dace.Memlet(data=info.scale, subset=info.block_subset)
            )
            graph.add_edge(
                scale, None, castout, "scale",
                dace.Memlet(data=info.scale, subset=info.block_subset)
            )

            # castout → original array
            graph.add_edge(castout, "array", edge.dst, edge.dst_conn, edge.data)

        # Delete all old nodes
        graph.remove_node(self.Library)