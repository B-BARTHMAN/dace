import dace
import dace.transformation.pass_pipeline as ppl
from dace.transformation.blockedfp.passes.add_scale_bias_pass import AddScaleBias
import dace.transformation as xf
from dataclasses import dataclass
from dace.sdfg.state import StateSubgraphView
from dace.sdfg.utils import node_path_graph
from typing import Dict, Any, Set, List

@dataclass(unsafe_hash=True)
class ExtendTaskletsPass(ppl.Pass):
    
    def __init__(self, name: str):
        self._name = name
    
    # this pass has to run after the add scale/bias step
    def depends_on(self) -> Set[ppl.Pass]:
        return {AddScaleBias}
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _: Dict[str, Any]):
        sdfg.apply_transformations(ExtendTaskletsTransform, options={"name": self._name}, validate=False)


class ExtendTaskletsTransform(xf.SingleStateTransformation):
    
    access = xf.PatternNode(dace.nodes.AccessNode)
    outer_map = xf.PatternNode(dace.nodes.MapEntry)
    inner_map = xf.PatternNode(dace.nodes.MapEntry)
    tasklet = xf.PatternNode(dace.nodes.Tasklet)
    
    def __init__(self, name: str):
        self._name = name
    
    @classmethod
    def expressions(cls) -> List[StateSubgraphView]:
        return [node_path_graph(cls.access, cls.outer_map, cls.inner_map, cls.tasklet)]
    
    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive = False) -> bool:
        
        if self.access.data != self._name:
            return False
        
        return True
    
    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG) -> None:
        
        # Step 1: Add new access nodes
        scale_acccess = graph.add_access(f"{self._name}_scale")
        bias_acccess = graph.add_access(f"{self._name}_bias")
        
        # Step 2: Add connectors to outer map
        self.outer_map.add_in_connector(f"IN_{self._name}_scale")
        self.outer_map.add_out_connector(f"OUT_{self._name}_scale")
        self.outer_map.add_in_connector(f"IN_{self._name}_bias")
        self.outer_map.add_out_connector(f"OUT_{self._name}_bias")
        
        # Step 3: Connect access to outer map
        outer_subset = self._get_connector_memlet_range(graph, self.outer_map, f"IN_{self._name}")
        graph.add_edge(scale_acccess, None, self.outer_map, f"IN_{self._name}_scale", dace.Memlet(data=f"{self._name}_scale", subset=outer_subset))
        graph.add_edge(bias_acccess, None, self.outer_map, f"IN_{self._name}_bias", dace.Memlet(data=f"{self._name}_bias", subset=outer_subset))
        
        # Step 4: Add connectors to inner map
        self.inner_map.add_in_connector(f"IN_{self._name}_scale")
        self.inner_map.add_out_connector(f"OUT_{self._name}_scale")
        self.inner_map.add_in_connector(f"IN_{self._name}_bias")
        self.inner_map.add_out_connector(f"OUT_{self._name}_bias")
        
        # Step 5: Connect outer map to inner map
        inner_subset = self._get_connector_memlet_range(graph, self.inner_map, f"IN_{self._name}")
        graph.add_edge(self.outer_map, f"OUT_{self._name}_scale", self.inner_map, f"IN_{self._name}_scale", dace.Memlet(data=f"{self._name}_scale", subset=inner_subset))
        graph.add_edge(self.outer_map, f"OUT_{self._name}_bias", self.inner_map, f"IN_{self._name}_bias", dace.Memlet(data=f"{self._name}_bias", subset=inner_subset))
        
        # Step 6: Add more connectors to tasklet?
        self.tasklet.add_in_connector(f"scale")
        self.tasklet.add_in_connector(f"bias")
        
        # Step 7: Connect inner map to tasklet
        connector = self._get_tasklet_connector(graph, self.tasklet, self._name)
        tasklet_subset = self._get_connector_memlet_range(graph, self.tasklet, connector)
        graph.add_edge(self.inner_map, f"OUT_{self._name}_scale", self.tasklet, "scale", dace.Memlet(data=f"{self._name}_scale", subset=tasklet_subset))
        graph.add_edge(self.inner_map, f"OUT_{self._name}_bias", self.tasklet, "bias", dace.Memlet(data=f"{self._name}_bias", subset=tasklet_subset))
        
    
    # this part is a bit ugly using the subsets, to extract the used range of the array
    def _get_connector_memlet_range(self, state: dace.SDFGState, node: dace.nodes.MapEntry, connector: str) -> dace.subsets.Range:
        # find all edges
        edges = list(state.in_edges_by_connector(node, connector))
        if len(edges) != 1:
            raise ValueError("This graph has a weird shape")
        edge = edges[0]
        return dace.subsets.Range([edge.data.src_subset.ranges[0]])
    
    def _get_tasklet_connector(self, state: dace.SDFGState, tasklet: dace.nodes.Tasklet, name: str) -> str:
        for connector in tasklet.in_connectors:
            for edge in state.edges_by_connector(tasklet, connector):
                if edge.data.data == name:
                    return connector