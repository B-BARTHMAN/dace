import dace
from dace.transformation.blockedfp.passes.add_new_arrays_pass import AddNewArraysPass
from typing import List

from tests.transformations.blockedfp.programs import programs

def add_new_arrays_condition(sdfg: dace.SDFG, arrays: List[str]):    
    # Try to validate and compile the program
    sdfg.validate()
    sdfg.compile()
    
    for state in sdfg.states():
        # Check if any Memlets contain one of the old arrays
        for edge in state.edges():
            if edge.data.data in arrays:
                raise ValueError(f"Edge '{edge}' contains data that it shouldn't.")
        # Check if any AccessNode accesses one of the old arrays
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                if node.data in arrays:
                    raise ValueError(f"Node {node} accesses data that it shouldn't")
        
        
def test_add_new_arrays():
    for program in programs:
        sdfg = program.to_sdfg()
        arrays = program.argnames
        
        # Run Pass
        AddNewArraysPass(arrays).apply_pass(sdfg, {})
        
        # Run conditions
        add_new_arrays_condition(sdfg, arrays)

test_add_new_arrays()