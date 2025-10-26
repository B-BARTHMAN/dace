import dace
from dace.transformation.transformation import ExpandTransformation

@dace.library.expansion
class ExpandBFPAddNode(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node: "BFPAddNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFGState) -> dace.nodes.Tasklet:
        code = "_out = _scale * _fp + _bias + _other"
        return dace.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code
        )
        

@dace.library.node
class BFPAddNode(dace.sdfg.nodes.LibraryNode):
    
    implementations = {
        "pure": ExpandBFPAddNode,
    }
    default_implementation = 'pure'
    
    def __init__(self, name):
        super().__init__(name, inputs={"_scale", "_bias", "_fp", "_other"}, outputs={"_out"})