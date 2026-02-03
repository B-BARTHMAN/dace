import dace
from dace.library import expansion, node
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from typing import Dict

N = dace.symbol("N")
M = dace.symbol("M")
S = dace.symbol("S")

@dace.program
def bfpcastout_1d(
    scale: dace.float64[N], bias: dace.float64[N], fp: dace.float32[N * S],
    array: dace.float64[N * S]
    ):
    
    for block_i, i in dace.map[0:N, 0:S]:
        array[S * block_i + i] = scale[block_i] * fp[S * block_i + i] + bias[block_i]

@dace.program
def bfpcastout_2d(
    array: dace.float64[N * S, M * S],
    scale: dace.float64[N, M], bias: dace.float64[N, M], fp: dace.float32[N * S, M * S]
    ):
    
    for block_i, block_j, i, j in dace.map[0:N, 0:M, 0:S, 0:S]:
        array[S * block_i + i, S * block_j + j] = scale[block_i, block_j] * fp[S * block_i + i, S * block_j + j] + bias[block_i, block_j]
    
@expansion
class ExpandBFPCastoutNode(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node, parent_state, parent_sdfg, *args, **kwargs):
        scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "scale")]
        bias = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "bias")]
        fp = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "fp")]
        array = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "array")]
        
        assert all([scale, bias, fp, array])
        
        if len(array.shape) == 1:
            sdfg = bfpcastout_1d.to_sdfg(
                scale, bias, fp,
                array
            )
        elif len(array.shape) == 2:
            sdfg = bfpcastout_2d.to_sdfg(
                scale, bias, fp,
                array
            )
        else:
            raise ValueError(f"BFP Castout can not handle dimensions '{len(array.shape)}'")
        
        sdfg.simplify()
        nested = dace.nodes.NestedSDFG(
            label= node.label,
            sdfg= sdfg,
            inputs={"scale", "bias", "fp"}, 
            outputs={"array"},
            symbol_mapping=node.symbol_mapping
        )
        return nested

@node
class BFPCastoutNode(dace.nodes.LibraryNode):
    implementations = {
        "pure": ExpandBFPCastoutNode
    }
    default_implementation = "pure"
    
    symbol_mapping = Property(
        dtype=dict,
        default={}
    )
    
    def __init__(self, name, symbol_mapping: Dict[str, str] = None):
        self.symbol_mapping = symbol_mapping
        super().__init__(name, inputs={"scale", "bias", "fp"}, outputs={"array"})