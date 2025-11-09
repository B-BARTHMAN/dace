import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.dataflow import MapTiling
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass(unsafe_hash=True)
class MapTilingPass(ppl.Pass):
    
    def __init__(self, names: List[str]):
        self._names = names
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]) -> None:
        
        for state in sdfg.states():
            for map_entry in [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]:
                
                # Get the corresponding exit node
                map_exit = state.exit_node(map_entry)
                
                # primitive check if this map reads from or writes to one of our desired arrays
                tile = False
                for edge in state.in_edges(map_entry):
                    if edge.data.data in self._names:
                        tile = True
                        break
                for edge in state.out_edges(map_exit):
                    if edge.data.data in self._names:
                        tile = True
                        break
                
                if not tile:
                    continue
                  
                MapTiling.apply_to(
                    sdfg=sdfg,
                    options={
                        "tile_sizes": [16],
                        "divides_evenly": True,
                        "tile_trivial": False,
                        "skew": False
                    },
                    map_entry=map_entry
                )