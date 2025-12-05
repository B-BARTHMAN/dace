import dace
import dace.transformation.pass_pipeline as ppl
from typing import List

class AddAuxiliaryArraysPass(ppl.Pass):
    
    def __init__(self, names: List[str], blocking_factor: int = 16):
        
        self.__names = names
        self.__blocking_factor = blocking_factor
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _) -> None:
        
        # Add new arrays
        for name in self.__names:
            array: dace.data.Array = sdfg.arrays[f"{name}_fp"]
            shape = array.shape[:(len(array.shape)//2)]
            sdfg.add_array(f"{name}_scale", shape, dace.float64, transient=True)
            sdfg.add_array(f"{name}_bias", shape, dace.float64, transient=True)