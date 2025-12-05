import dace
from typing import List

# Testing Framework
from tests.transformations.blockedfp.programs import programs

# What we're testing
from dace.transformation.blockedfp.passes.add_new_arrays_pass import AddNewArraysPass
from dace.transformation.blockedfp.passes.map_tiling_pass import MapTilingPass
from dace.transformation.blockedfp.passes.split_dimensions_pass import SplitDimensionsPass
from dace.transformation.blockedfp.passes.add_auxiliary_arrays_pass import AddAuxiliaryArraysPass

def add_auxiliary_arrays_condition(sdfg: dace.SDFG, arrays: List[str]):    
    # Try to validate and compile the program
    sdfg.validate()
    sdfg.compile()
    
    for array in arrays:
        fp_shape = sdfg.arrays[f"{array}_fp"]
        required_auxiliary_shape = fp_shape[:(len(fp_shape)//2)]
        assert sdfg.arrays[f"{array}_bias"] == required_auxiliary_shape
        assert sdfg.arrays[f"{array}_scale"] == required_auxiliary_shape


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
        
        # Run conditions
        add_auxiliary_arrays_condition(sdfg, arrays)

test_add_auxiliary_arrays()