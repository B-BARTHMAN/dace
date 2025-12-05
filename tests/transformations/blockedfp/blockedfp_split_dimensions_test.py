import dace
from typing import List

# Testing Framework
from tests.transformations.blockedfp.programs import programs

# What we're testing
from dace.transformation.blockedfp.passes.add_new_arrays_pass import AddNewArraysPass
from dace.transformation.blockedfp.passes.map_tiling_pass import MapTilingPass
from dace.transformation.blockedfp.passes.split_dimensions_pass import SplitDimensionsPass

def split_dimensions_condition(sdfg: dace.SDFG, arrays: List[str], blocking_factor: int = 16):    
    # Try to validate and compile the program
    sdfg.validate()
    sdfg.compile()
    
    for array in arrays:
        array_before = sdfg.arrays[array]
        array_after  = sdfg.arrays[f"{array}_fp"]
        
        dim_before = len(array_before.shape)
        dim_after  = len(array_after.shape)
        
        assert dim_after == 2 * dim_before
        
        for dim in array_after.shape[dim_before:]:
            assert dim == blocking_factor


def test_split_dimenions():
    for program in programs:
        sdfg = program.to_sdfg()
        arrays = program.argnames
        
        # Run AddNewArrays Pass
        AddNewArraysPass(arrays).apply_pass(sdfg, {})
        # Run MapTiling Pass
        MapTilingPass(arrays).apply_pass(sdfg, {})
        # Run SplitDimensions Pass
        SplitDimensionsPass(arrays).apply_pass(sdfg, {})
        
        sdfg.view()
        
        # Run conditions
        split_dimensions_condition(sdfg, arrays)

test_split_dimenions()