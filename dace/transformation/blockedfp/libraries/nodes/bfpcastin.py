import dace
from dace.library import expansion, node
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from typing import Dict

N = dace.symbol("N")
M = dace.symbol("M")
S = dace.symbol("S")

@dace.program
def bfpcastin_1d(
    array: dace.float64[N * S],
    scale: dace.float64[N], bias: dace.float64[N], fp: dace.float32[N * S]
    ):
    
    for block_i in dace.map[0:N]:
        scale[block_i] = 1
        bias[block_i] = 0
    for block_i, i in dace.map[0:N, 0:S]:
        bias[block_i] += array[S * block_i + i] / S
    for block_i, i in dace.map[0:N, 0:S]:
        scale[block_i] = max(abs(array[S * block_i + i] - bias[block_i]), scale[block_i])
    for block_i, i in dace.map[0:N, 0:S]:
        fp[S * block_i + i] = (array[S * block_i + i] - bias[block_i]) / scale[block_i]

@dace.program
def bfpcastin_2d(
    array: dace.float64[N * S, M * S],
    scale: dace.float64[N, M], bias: dace.float64[N, M], fp: dace.float32[N * S, M * S]
    ):
    
    for block_i, block_j in dace.map[0:N, 0:M]:
        scale[block_i, block_j] = 1
        bias[block_i, block_j] = 0
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        bias[block_i, block_j] += array[S * block_i + i, S * block_j + j] / (S * S)
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        scale[block_i, block_j] = max(abs(array[S * block_i + i, S * block_j + j] - bias[block_i, block_j]), scale[block_i, block_j])
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        fp[S * block_i + i, S * block_j + j] = (array[S * block_i + i, S * block_j + j] - bias[block_i, block_j]) / scale[block_i, block_j]
    
@expansion
class ExpandBFPCastinNode(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node: "BFPCastinNode", parent_state, parent_sdfg, *args, **kwargs):
        array = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "array")]
        scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "scale")]
        bias = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "bias")]
        fp = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "fp")]
        
        assert all([array, scale, bias, fp])
        
        if len(array.shape) == 1:
            sdfg = bfpcastin_1d.to_sdfg(
                array,
                scale, bias, fp
            )
        elif len(array.shape) == 2:
            sdfg = bfpcastin_2d.to_sdfg(
                array,
                scale, bias, fp
            )
        else:
            raise ValueError(f"BFP Castin can not handle dimensions '{len(array.shape)}'")
        
        sdfg.simplify()
        nested = dace.nodes.NestedSDFG(
            label= node.label,
            sdfg= sdfg,
            inputs={"array"}, 
            outputs={"scale", "bias", "fp"},
            symbol_mapping=node.symbol_mapping
        )
        return nested

@node
class BFPCastinNode(dace.nodes.LibraryNode):
    implementations = {
        "pure": ExpandBFPCastinNode
    }
    default_implementation = "pure"
    
    symbol_mapping = Property(
        dtype=dict,
        default={}
    )
    
    def __init__(self, name, symbol_mapping: Dict[str, str] = None):
        self.symbol_mapping = symbol_mapping
        super().__init__(name, inputs={"array"}, outputs={"scale", "bias", "fp"})

bfpcastin_1d.to_sdfg().compile()