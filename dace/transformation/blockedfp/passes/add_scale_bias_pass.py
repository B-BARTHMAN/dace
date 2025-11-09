import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.passes.split_tasklets import SplitTasklets
from dataclasses import dataclass
from typing import Dict, Any, Set, Tuple, List
from dace.data import Array


@dataclass(unsafe_hash=True)
class AddScaleBias(ppl.Pass):
    
    def __init__(self, names: List[str], block_size: Tuple[int]):
        self._names = names
        self._block_size = block_size
    
    # this pass has to run after the blocking step
    def depends_on(self) -> Set[ppl.Pass]:
        return {SplitTasklets}
    
    def modifies(self) -> ppl.Modifies:
        ppl.Modifies.AccessNodes
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]) -> None:
        
        for name in self._names:
            array : Array = sdfg.arrays[name]
            split_shape = array.shape # actually shape is already split, so i need to change this somehow
            shape = split_shape[:-len(self._block_size)]
            
            # add scale, bias array
            sdfg.add_array(f"{name}_scale", shape, dace.float64)
            sdfg.add_array(f"{name}_bias", shape, dace.float64)
    