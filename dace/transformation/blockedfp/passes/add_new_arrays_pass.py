import dace
import dace.transformation.pass_pipeline as ppl
from typing import List

class AddNewArraysPass(ppl.Pass):
    
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
            array: dace.data.Array = sdfg.arrays[name]
            sdfg.add_array(f"{name}_fp", array.shape, dace.float32, transient=True)

        # Make sdfg use new arrays
        for state in sdfg.states():
            
            # Transform all AccessNodes
            for node in state.nodes():
                if isinstance(node, dace.nodes.AccessNode) and node.data in self.__names:
                    node.data = f"{node.data}_fp"
            
            # Transform all memlets
            for edge in state.edges():
                if edge.data.data in self.__names:
                    
                    # rename connectors
                    if isinstance(edge.src, dace.nodes.MapEntry) or isinstance(edge.src, dace.nodes.MapExit):
                        edge.src.add_out_connector(f"OUT_{edge.data.data}_fp")
                        edge.src.remove_out_connector(f"OUT_{edge.data.data}")
                        edge.src_conn = f"OUT_{edge.data.data}_fp"
                        
                    if isinstance(edge.dst, dace.nodes.MapEntry) or isinstance(edge.dst, dace.nodes.MapExit):
                        edge.dst.add_in_connector(f"IN_{edge.data.data}_fp")
                        edge.dst.remove_in_connector(f"IN_{edge.data.data}")
                        edge.dst_conn = f"IN_{edge.data.data}_fp"
                        
                    edge.data.data = f"{edge.data.data}_fp"