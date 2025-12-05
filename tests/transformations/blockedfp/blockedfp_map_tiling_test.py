import dace
from typing import List

# Testing Framework
from tests.transformations.blockedfp.programs import programs

# What we're testing
from dace.transformation.blockedfp.passes.add_new_arrays_pass import AddNewArraysPass
from dace.transformation.blockedfp.passes.map_tiling_pass import MapTilingPass

def map_tiling_condition(sdfg: dace.SDFG, arrays: List[str]):    
    # Try to validate and compile the program
    sdfg.validate()
    sdfg.compile()
    
    for state in sdfg.states():
        # Check all access nodes of the arrays, they have to write to a map that writes to another map
        for node in state.nodes():
            if not (isinstance(node, dace.nodes.AccessNode) and node.data in arrays):
                continue
            for edge in state.out_edges(node):
                if edge.data.data != node.data:
                    continue
                if not isinstance(edge.dst, dace.nodes.MapEntry):
                    raise ValueError(f"Edge '{edge}' doesn't write to a map but '{edge.dst}'")
                for edge_inner in state.out_edges(edge.dst):
                    if edge_inner.data != node.data:
                        continue
                    if not isinstance(edge_inner.dst, dace.nodes.MapEntry):
                        raise ValueError(f"Inner Edge '{edge_inner}' doesn't write to a map but '{edge_inner.dst}'")


def test_add_new_arrays():
    for program in programs:
        sdfg = program.to_sdfg()
        arrays = program.argnames
        
        # Run AddNewArrays Pass
        AddNewArraysPass(arrays).apply_pass(sdfg, {})
        # Run MapTiling Pass
        MapTilingPass(arrays).apply_pass(sdfg, {})
        
        # Run conditions
        map_tiling_condition(sdfg, arrays)

test_add_new_arrays()