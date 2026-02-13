import dace
from dace.library import expansion, node
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from typing import Dict
import numpy as np

N = dace.symbol("N")
M = dace.symbol("M")
L = dace.symbol("L")
S = dace.symbol("S")

@dace.program
def bfpgemm_int8(
    a_bias: dace.float64[N, M], a_scale: dace.int8[N, M], a_ints: dace.int8[N * S, M * S],
    b_bias: dace.float64[M, L], b_scale: dace.int8[M, L], b_ints: dace.int8[M * S, L * S],
    c_bias: dace.float64[N, L], c_scale: dace.int8[N, L], c_ints: dace.int8[N * S, L * S],
    ):
    
    full_res_A = dace.define_local((N * S, M * S), dtype=dace.float64)
    full_res_B = dace.define_local((M * S, L * S), dtype=dace.float64)
    
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        full_res_A[S * block_i + i, S * block_j + j] = a_bias[block_i, block_j] + (2.0 ** a_scale[block_i, block_j]) * dace.float64(a_ints[S * block_i + i, S * block_j + j])
    for block_i, block_j, i, j in dace.map[0:M, 0:L, 0:S, 0:S]:
        full_res_B[S * block_i + i, S * block_j + j] = b_bias[block_i, block_j] + (2.0 ** b_scale[block_i, block_j]) * dace.float64(b_ints[S * block_i + i, S * block_j + j])
        
    full_res = dace.define_local((N * S, L * S), dtype=dace.float64)
    full_res = full_res_A @ full_res_B
    
    # Calculate bias
    for block_i, block_j in dace.map[0:N, 0:L]:
        c_bias[block_i] = 0
    for block_i, block_j, i, j in dace.map[0:N, 0:L, 0:S, 0:S]:
        c_bias[block_i, block_j] += full_res[S * block_i + i, S * block_j + j] / (S * S)
    
    # Calculate offset
    offset = dace.define_local((N * S, L * S), dtype=dace.float64)
    for block_i, block_j, i, j in dace.map[0:N, 0:L, 0:S, 0:S]:
        offset[S * block_i + i, S * block_j + j] = (
            full_res[S * block_i + i, S * block_j + j]
            - c_bias[block_i, block_j]
        )
    
    # Calculate scale (temp)
    temp = dace.define_local((N, L), dtype=dace.float64)
    for block_i, block_j in dace.map[0:N, 0:L]:
        temp[block_i, block_j] = 0.0001
    for block_i, block_j in dace.map[0:N, 0:L]:
        for i, j in dace.map[0:S, 0:S] @ dace.ScheduleType.Sequential:
            temp[block_i, block_j] = max(
                abs(offset[S * block_i + i, S * block_j + j]),
                temp[block_i, block_j]
            )
    for block_i, block_j in dace.map[0:N, 0:L]:
        temp[block_i, block_j] /= 127.0
    for block_i, block_j in dace.map[0:N, 0:L]:
        c_scale[block_i, block_j] = min(
            max(
                dace.int8(np.ceil(np.log2(temp[block_i, block_j]))),
                -128
            ),
            127
        )
    for block_i, block_j in dace.map[0:N, 0:L]:
        temp[block_i, block_j] = 2.0 ** c_scale[block_i, block_j]
    
    # Calculate ints
    for block_i, block_j, i, j in dace.map[0:N, 0:L, 0:S, 0:S]:
        c_ints[S * block_i + i, S * block_j + j] = dace.int8(
            min(
                max(
                    offset[S * block_i + i, S * block_j + j]
                    / temp[block_i, block_j],
                    -128
                ),
                127
            )
        )

@expansion
class ExpandBFPGemmInt8Node(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node: "BFPGemmInt8Node", parent_state, parent_sdfg, *args, **kwargs):
        # Input arrays
        a_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_bias")]
        a_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_scale")]
        a_ints  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_ints")]

        b_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_bias")]
        b_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_scale")]
        b_ints  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "b_ints")]

        # Output arrays
        c_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "c_bias")]
        c_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "c_scale")]
        c_ints  = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "c_ints")]

        assert all([a_bias, a_scale, a_ints, b_bias, b_scale, b_ints, c_bias, c_scale, c_ints])

        # Generate SDFG from bfpgemm_int8 program
        sdfg = bfpgemm_int8.to_sdfg(a_bias, a_scale, a_ints, b_bias, b_scale, b_ints, c_bias, c_scale, c_ints)
        sdfg.simplify()

        nested = dace.nodes.NestedSDFG(
            label=node.label,
            sdfg=sdfg,
            inputs={"a_bias", "a_scale", "a_ints", "b_bias", "b_scale", "b_ints"},
            outputs={"c_bias", "c_scale", "c_ints"},
            symbol_mapping=node.symbol_mapping
        )
        return nested

@node
class BFPGemmInt8Node(dace.nodes.LibraryNode):
    implementations = {
        "pure": ExpandBFPGemmInt8Node
    }
    default_implementation = "pure"

    symbol_mapping = Property(
        dtype=dict,
        default={}
    )

    def __init__(self, name: str, symbol_mapping: Dict[str, str] = None):
        self.symbol_mapping = symbol_mapping or {}
        super().__init__(
            name,
            inputs={"a_bias", "a_scale", "a_ints", "b_bias", "b_scale", "b_ints"},
            outputs={"c_bias", "c_scale", "c_ints"}
        )