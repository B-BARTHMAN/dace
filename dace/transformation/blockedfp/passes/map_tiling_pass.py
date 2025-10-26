import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.dataflow import MapTiling
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(unsafe_hash=True)
class MapTilingPass(ppl.Pass):
    
    def __init__(self, name: str):
        self._name = name
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]) -> None:
        
        for state in sdfg.states():
            for map_entry in [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]:
                
                # primitive check if this map reads from our desired array
                skip = True
                for edge in state.in_edges(map_entry):
                    if edge.data.data == f"{self._name}":
                        skip = False
                        break
                    map_entry
                
                if skip:
                    continue
                  
                MapTiling.apply_to(
                    sdfg=sdfg,
                    options={
                        "tile_sizes": [16],
                        "divides_evenly": True,
                        "tile_trivial": True
                    },
                    map_entry=map_entry
                )