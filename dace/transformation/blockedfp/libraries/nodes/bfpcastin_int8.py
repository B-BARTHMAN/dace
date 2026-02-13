import dace
from dace.library import expansion, node
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from typing import Dict
import numpy as np

N = dace.symbol("N")
M = dace.symbol("M")
S = dace.symbol("S")

@dace.program
def bfpcastin_int8_1d(
    array: dace.float64[N * S],
    bias: dace.float64[N], scale: dace.int8[N], ints: dace.int8[N * S]
    ):
    
    # Calculate bias
    for block_i in dace.map[0:N]:
        bias[block_i] = 0
    for block_i, i in dace.map[0:N, 0:S]:
        bias[block_i] += array[S * block_i + i] / S
    
    # Calculate offset
    offset = dace.define_local((N*S,), dtype=dace.float64)
    for block_i, i in dace.map[0:N, 0:S]:
        offset[S*block_i + i] = array[S * block_i + i] - bias[block_i]
    
    # Calculate Scale
    temp = dace.define_local((N), dtype=dace.float64)
    for block_i in dace.map[0:N]:
        temp[block_i] = 0.0001
    for block_i in dace.map[0:N]:
        for i in dace.map[0:S] @ dace.ScheduleType.Sequential:
            temp[block_i] = max(abs(offset[S*block_i + i]), temp[block_i])
    for block_i in dace.map[0:N]:
        temp[block_i] /= 127.0
    for block_i in dace.map[0:N]:
        scale[block_i] = min(max(dace.int8(np.ceil(np.log2(temp[block_i]))), -128), 127)
    for block_i in dace.map[0:N]:
        temp[block_i] = 2.0 ** scale[block_i]
    
    # Calculate ints
    for block_i, i in dace.map[0:N, 0:S]:
        ints[S * block_i + i] = dace.int8(min(max(offset[S*block_i + i] / temp[block_i], -128), 127))

@dace.program
def bfpcastin_int8_2d(
    array: dace.float64[N * S, M * S],
    bias: dace.float64[N, M],
    scale: dace.int8[N, M],
    ints: dace.int8[N * S, M * S]
    ):
    
    # Calculate bias
    for block_i, block_j in dace.map[0:N, 0:M]:
        bias[block_i, block_j] = 0
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        bias[block_i, block_j] += (
            array[S * block_i + i, S * block_j + j] / (S * S)
        )
    
    # Calculate offset
    offset = dace.define_local((N * S, M * S), dtype=dace.float64)
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        offset[S * block_i + i, S * block_j + j] = (
            array[S * block_i + i, S * block_j + j]
            - bias[block_i, block_j]
        )
    
    # Calculate scale (temp)
    temp = dace.define_local((N, M), dtype=dace.float64)
    for block_i, block_j in dace.map[0:N, 0:M]:
        temp[block_i, block_j] = 0.0001
    for block_i, block_j in dace.map[0:N, 0:M]:
        for i, j in dace.map[0:S, 0:S] @ dace.ScheduleType.Sequential:
            temp[block_i, block_j] = max(
                abs(offset[S * block_i + i, S * block_j + j]),
                temp[block_i, block_j]
            )
    for block_i, block_j in dace.map[0:N, 0:M]:
        temp[block_i, block_j] /= 127.0
    for block_i, block_j in dace.map[0:N, 0:M]:
        scale[block_i, block_j] = min(
            max(
                dace.int8(np.ceil(np.log2(temp[block_i, block_j]))),
                -128
            ),
            127
        )
    for block_i, block_j in dace.map[0:N, 0:M]:
        temp[block_i, block_j] = 2.0 ** scale[block_i, block_j]
    
    # Calculate ints
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        ints[S * block_i + i, S * block_j + j] = dace.int8(
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
class ExpandBFPCastinNodeInt8(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node: "BFPCastinNodeInt8", parent_state, parent_sdfg, *args, **kwargs):
        array = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "array")]
        bias = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "bias")]
        scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "scale")]
        ints = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "ints")]
        
        assert all([array, bias, scale, ints])
        
        if len(array.shape) == 1:
            sdfg = bfpcastin_int8_1d.to_sdfg(
                array,
                bias, scale, ints
            )
        elif len(array.shape) == 2:
            sdfg = bfpcastin_int8_2d.to_sdfg(
                array,
                bias, scale, ints
            )
        else:
            raise ValueError(
                f"BFP Castin can not handle dimensions '{len(array.shape)}'"
            )
        
        sdfg.simplify()
        nested = dace.nodes.NestedSDFG(
            label=node.label,
            sdfg=sdfg,
            inputs={"array"},
            outputs={"bias", "scale", "ints"},
            symbol_mapping=node.symbol_mapping
        )
        return nested

@node
class BFPCastinNodeInt8(dace.nodes.LibraryNode):
    implementations = {
        "pure": ExpandBFPCastinNodeInt8
    }
    default_implementation = "pure"
    
    symbol_mapping = Property(
        dtype=dict,
        default={}
    )
    
    def __init__(self, name, symbol_mapping: Dict[str, str] = None):
        self.symbol_mapping = symbol_mapping
        super().__init__(name, inputs={"array"}, outputs={"bias", "scale", "ints"})