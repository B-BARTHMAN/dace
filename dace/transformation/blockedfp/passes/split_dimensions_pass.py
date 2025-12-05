import dace
import dace.transformation.pass_pipeline as ppl

from dace.transformation.layout.split_dimension import SplitDimensions

from typing import List

class SplitDimensionsPass(ppl.Pass):
    
    def __init__(self, names: List[str], blocking_factor: int = 16):
        
        self.__names = names
        self.__blocking_factor = blocking_factor
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _) -> None:
        
        # Tile the map
        split_map = {
            f"{name}_fp": (
                [True] * len(sdfg.arrays[name].shape),
                [self.__blocking_factor] * len(sdfg.arrays[name].shape)
            )
            for name in self.__names
        }
        SplitDimensions(split_map=split_map).apply_pass(sdfg, {})