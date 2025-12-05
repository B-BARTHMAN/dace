import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.dataflow import MapTiling
from typing import List

class MapTilingPass(ppl.Pass):
    
    def __init__(self, names: List[str], blocking_factor: int = 16):
        self.__names = [f"{name}_fp" for name in names]
        self.__blocking_factor = blocking_factor
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _) -> None:
        
        for state in sdfg.states():
            for map_entry in [n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)]:
                
                # Get the corresponding exit node
                map_exit = state.exit_node(map_entry)
                
                # primitive check if this map reads from or writes to one of our desired arrays
                tile = any(edge.data.data in self.__names for edge in state.in_edges(map_entry)) \
                or any(edge.data.data in self.__names for edge in state.out_edges(map_exit))
                
                if not tile:
                    continue
                
                MapTiling.apply_to(
                    sdfg=sdfg,
                    options={
                        "tile_sizes": [self.__blocking_factor],
                        "divides_evenly": True,
                        "tile_trivial": False,
                        "skew": False
                    },
                    map_entry=map_entry
                )