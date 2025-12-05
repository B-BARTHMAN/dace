import dace
from typing import List

# Testing Framework
from tests.transformations.blockedfp.programs import programs

# What we're testing
from dace.transformation.blockedfp.passes.add_new_arrays_pass import AddNewArraysPass
from dace.transformation.blockedfp.passes.map_tiling_pass import MapTilingPass
from dace.transformation.blockedfp.passes.split_dimensions_pass import SplitDimensionsPass
from dace.transformation.blockedfp.passes.add_auxiliary_arrays_pass import AddAuxiliaryArraysPass
from dace.transformation.blockedfp.passes.split_tasklets_pass import SplitTaskletsPass

def add_auxiliary_arrays_condition(sdfg: dace.SDFG, arrays: List[str]):    
    # Try to validate and compile the program
    sdfg.validate()
    sdfg.compile()
    
    # What do I test here?


def test_add_auxiliary_arrays():
    for program in programs:
        sdfg = program.to_sdfg()
        arrays = program.argnames
        
        # Run AddNewArrays Pass
        AddNewArraysPass(arrays).apply_pass(sdfg, {})
        # Run MapTiling Pass
        MapTilingPass(arrays).apply_pass(sdfg, {})
        # Run SplitDimensions Pass
        SplitDimensionsPass(arrays).apply_pass(sdfg, {})
        # Run AddAuxiliaryArrays Pass
        AddAuxiliaryArraysPass(arrays).apply_pass(sdfg, {})
        # Run SplitTasklets Pass
        #SplitTaskletsPass().apply_pass(sdfg, {})
        
        sdfg.view()
        
        # Run conditions
        add_auxiliary_arrays_condition(sdfg, arrays)

test_add_auxiliary_arrays()