import dace
from dace.library import expansion, node
from dace.properties import Property
from dace.transformation.transformation import ExpandTransformation
from typing import Dict

N = dace.symbol("N")
M = dace.symbol("M")
S = dace.symbol("S")

@dace.program
def bfpcastout_int8_1d(
    bias: dace.float64[N], scale: dace.int16[N], ints: dace.int16[N * S],
    array: dace.float64[N * S]
    ):
    
    scale_fp = dace.define_local((N), dtype=dace.float64)
    
    for block_i in dace.map[0:N]:
        scale_fp[block_i] = 2.0 ** scale[block_i]
    
    for block_i, i in dace.map[0:N, 0:S]:
        array[S * block_i + i] = bias[block_i] + scale_fp[block_i] * ints[S * block_i + i]

@expansion
class ExpandBFPCastoutNodeInt8(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node: "BFPCastoutNodeInt8", parent_state, parent_sdfg, *args, **kwargs):
        bias = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "bias")]
        scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "scale")]
        ints = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "ints")]
        array = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "array")]
        
        assert all([array, bias, scale, ints])
        
        if len(array.shape) == 1:
            sdfg = bfpcastout_int8_1d.to_sdfg(
                bias, scale, ints,
                array
            )
        elif len(array.shape) == 2:
            raise ValueError("NOT YET")
            # sdfg = bfpcastin_2d.to_sdfg(
            #     array,
            #     scale, bias, fp
            # )
        else:
            raise ValueError(f"BFP Castin can not handle dimensions '{len(array.shape)}'")
        
        sdfg.simplify()
        nested = dace.nodes.NestedSDFG(
            label= node.label,
            sdfg= sdfg,
            inputs={"bias", "scale", "ints"},
            outputs={"array"}, 
            symbol_mapping=node.symbol_mapping
        )
        return nested

@node
class BFPCastoutNodeInt8(dace.nodes.LibraryNode):
    implementations = {
        "pure": ExpandBFPCastoutNodeInt8
    }
    default_implementation = "pure"
    
    symbol_mapping = Property(
        dtype=dict,
        default={}
    )
    
    def __init__(self, name, symbol_mapping: Dict[str, str] = None):
        self.symbol_mapping = symbol_mapping
        super().__init__(name, inputs={"bias", "scale", "ints"}, outputs={"array"})
