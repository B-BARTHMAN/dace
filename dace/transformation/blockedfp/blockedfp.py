import dace
from typing import List
import dace.transformation.pass_pipeline as ppl

from dace.transformation.blockedfp.passes.add_new_arrays_pass import AddNewArraysPass
from dace.transformation.blockedfp.passes.map_tiling_pass import MapTilingPass
from dace.transformation.blockedfp.passes.split_dimensions_pass import SplitDimensionsPass
from dace.transformation.blockedfp.passes.add_auxiliary_arrays_pass import AddAuxiliaryArraysPass
from dace.transformation.blockedfp.passes.split_tasklets_pass import SplitTaskletsPass
from dace.transformation.blockedfp.passes.replace_tasklets_pass import ReplaceTaskletsPass
from dace.transformation.blockedfp.passes.cast_in_out_pass import CastInOutPass

class BlockedFP(ppl.Pass):
    """
    A pass that transforms a given array into a blocked floating point format
    """
    
    def __init__(self, names: List[str], blocking_factor: int = 16):
        
        self.__names = names
        self.__blocking_factor = blocking_factor
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _) -> None:
        # Run AddNewArrays Pass
        AddNewArraysPass(self.__names, self.__blocking_factor).apply_pass(sdfg, {})
        # Run MapTiling Pass
        MapTilingPass(self.__names, self.__blocking_factor).apply_pass(sdfg, {})
        # Run SplitDimensions Pass
        SplitDimensionsPass(self.__names, self.__blocking_factor).apply_pass(sdfg, {})
        # Run AddAuxiliaryArrays Pass
        AddAuxiliaryArraysPass(self.__names, self.__blocking_factor).apply_pass(sdfg, {})
        # Run SplitTasklets Pass
        SplitTaskletsPass().apply_pass(sdfg, {})
        # Run ReplaceTasklets Pass
        ReplaceTaskletsPass(self.__names, self.__blocking_factor).apply_pass(sdfg, {})
        # Run CastInOutPass
        CastInOutPass(self.__names).apply_pass(sdfg, {})
        sdfg.simplify()