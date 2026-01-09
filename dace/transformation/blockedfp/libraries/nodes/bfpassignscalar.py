import dace
from dace.library import expansion, node
from dace.transformation.transformation import ExpandTransformation

@dace.program
def bfpassignscalar(a_scale: dace.float64[1], a_bias: dace.float64[1], a_fp: dace.float32[16], out_scale: dace.float64[1], out_bias: dace.float64[1], out_fp: dace.float32[16]):
    for i in dace.map[0:16]:
        out_fp[i] = a_fp[i]
    out_scale[0] = a_scale[0]
    out_bias[0] = a_bias[0]
    
@expansion
class ExpandBFPAssignScalarNode(ExpandTransformation):
    environments = []
 
    @staticmethod
    def expansion(node: "BFPAssignScalarNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFGState) -> dace.nodes.Tasklet:
     
        a_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_scale")]
        a_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_bias")]
        a_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.in_edges(node) if e.dst_conn == "a_fp")]
        out_scale = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "out_scale")]
        out_bias  = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "out_bias")]
        out_fp    = parent_sdfg.arrays[next(e.data.data for e in parent_state.out_edges(node) if e.src_conn == "out_fp")]
     
        assert all([a_scale, a_bias, a_fp, out_scale, out_bias, out_fp])
     
        sdfg = bfpassignscalar.to_sdfg(
            a_scale, a_bias, a_fp, out_scale, out_bias, out_fp
        )
        sdfg.simplify()
        return sdfg

# @dace.library.expansion
# class ExpandBFPAssignScalarNode(ExpandTransformation):
#     environments = []
    
#     @staticmethod
#     def expansion(node: "BFPAssignScalarNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFGState) -> dace.nodes.Tasklet:
#         code = "out_scale = a_scale; out_bias = a_bias; out_fp = a_fp;"
#         return dace.nodes.Tasklet(
#             node.name,
#             node.in_connectors,
#             node.out_connectors,
#             code
#         )
        

@node
class BFPAssignScalarNode(dace.sdfg.nodes.LibraryNode):
    
    implementations = {
        "pure": ExpandBFPAssignScalarNode,
    }
    default_implementation = 'pure'
    
    def __init__(self, name):
        super().__init__(name, inputs={"a_scale", "a_bias", "a_fp"}, outputs={"out_scale", "out_bias", "out_fp"})