import dace
from dace.transformation.transformation import ExpandTransformation

@dace.library.expansion
class ExpandBFPAssignScalarNode(ExpandTransformation):
    environments = []
    
    @staticmethod
    def expansion(node: "BFPAssignScalarNode", parent_state: dace.SDFGState, parent_sdfg: dace.SDFGState) -> dace.nodes.Tasklet:
        code = "out_fp = a_fp"
        return dace.nodes.Tasklet(
            node.name,
            node.in_connectors,
            node.out_connectors,
            code
        )
        

@dace.library.node
class BFPAssignScalarNode(dace.sdfg.nodes.LibraryNode):
    
    implementations = {
        "pure": ExpandBFPAssignScalarNode,
    }
    default_implementation = 'pure'
    
    def __init__(self, name):
        super().__init__(name, inputs={"a_scale", "a_bias", "a_fp"}, outputs={"out_scale", "out_bias", "out_fp"})