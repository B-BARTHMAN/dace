import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.layout.split_dimension import SplitDimensions
from dataclasses import dataclass
from typing import Dict, Any, Set, Tuple
from dace.data import Array


@dataclass(unsafe_hash=True)
class AddScaleBias(ppl.Pass):
    
    def __init__(self, name: str, block_size: Tuple[int]):
        self._name = name
        self._block_size = block_size
    
    # this pass has to run after the blocking step
    def depends_on(self) -> Set[ppl.Pass]:
        return {SplitDimensions}
    
    def modifies(self) -> ppl.Modifies:
        ppl.Modifies.AccessNodes
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]) -> None:
        
        array : Array = sdfg.arrays[self._name]
        split_shape = array.shape # actually shape is already split, so i need to change this somehow
        shape = split_shape[:-len(self._block_size)]
        
        # add scale, bias array
        sdfg.add_array(f"{self._name}_scale", shape, dace.float64)
        sdfg.add_array(f"{self._name}_bias", shape, dace.float64)
    