import dace
from dace.library import expansion, node
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from typing import Dict

N = dace.symbol("N")
M = dace.symbol("M")
L = dace.symbol("L")
S = dace.symbol("S")

@dace.program
def bfpgemm(
    a_scale: dace.float64[N, M], a_bias: dace.float64[N, M], a_fp: dace.float32[N * S, M * S],
    b_scale: dace.float64[M, L], b_bias: dace.float64[M, L], b_fp: dace.float32[M * S, L * S],
    c_scale: dace.float64[N, L], c_bias: dace.float64[N, L], c_fp: dace.float32[N * S, L * S],
    ):
    
    full_res_A = dace.define_local((N * S, M * S), dtype=dace.float64)
    full_res_B = dace.define_local((M * S, L * S), dtype=dace.float64)
    
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        full_res_A[S * block_i + i, S * block_j + j] = a_scale[block_i, block_j] * a_fp[S * block_i + i, S * block_j + j] + a_bias[block_i, block_j]
    for block_i, block_j, i, j in dace.map[0:M, 0:L, 0:S, 0:S]:
        full_res_B[S * block_i + i, S * block_j + j] = b_scale[block_i, block_j] * b_fp[S * block_i + i, S * block_j + j] + b_bias[block_i, block_j]
    
    full_res = dace.define_local((N * S, L * S), dtype=dace.float64)
    full_res = full_res_A @ full_res_B
    
    for block_i, block_j in dace.map[0:N, 0:L]:
        c_scale[block_i, block_j] = 1
        c_bias[block_i, block_j] = 0
    
    for block_i, block_j, i, j in dace.map[0:N, 0:L, 0:S, 0:S]:
        c_bias[block_i, block_j] += full_res[S * block_i + i, S * block_j + j] / (S * S)
    for block_i, block_j in dace.map[0:N, 0:L]:
        for i, j in dace.map[0:S, 0:S]:
            c_scale[block_i, block_j] = max(abs(full_res[S * block_i + i, S * block_j + j] - c_bias[block_i, block_j]), c_scale[block_i, block_j])
    for block_i, block_j, i, j in dace.map[0:N, 0:L, 0:S, 0:S]:
        c_fp[S * block_i + i, S * block_j + j] = (full_res[S * block_i + i, S * block_j + j] - c_bias[block_i, block_j]) / c_scale[block_i, block_j]

@expansion
class ExpandBFPGemmNode(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node: "BFPGemmNode", parent_state, parent_sdfg, *args, **kwargs):
        
        a_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_scale")]
        a_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_bias")]
        a_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_fp")]

        b_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_scale")]
        b_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_bias")]
        b_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_fp")]

        c_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "c_scale")]
        c_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "c_bias")]
        c_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "c_fp")]
        
        assert all([a_scale, a_bias, a_fp, b_scale, b_bias, b_fp, c_scale, c_bias, c_fp])
        
        sdfg = bfpgemm.to_sdfg(
            a_scale, a_bias, a_fp,
            b_scale, b_bias, b_fp,
            c_scale, c_bias, c_fp,
        )
        sdfg.simplify()
        nested = dace.nodes.NestedSDFG(
            label= node.label,
            sdfg= sdfg,
            inputs={"a_scale", "a_bias", "a_fp", "b_scale", "b_bias", "b_fp"},
            outputs={"c_scale", "c_bias", "c_fp"},
            symbol_mapping=node.symbol_mapping
        )
        return nested

@node
class BFPGemmNode(dace.nodes.LibraryNode):
    implementations = {
        "pure": ExpandBFPGemmNode
    }
    default_implementation = "pure"
    
    symbol_mapping = Property(
        dtype=dict,
        default={}
    )
    
    def __init__(self, name: str, symbol_mapping: Dict[str, str] = None):
        self.symbol_mapping = symbol_mapping
        super().__init__(name, inputs={"a_scale", "a_bias", "a_fp", "b_scale", "b_bias", "b_fp"}, outputs={"c_scale", "c_bias", "c_fp"})