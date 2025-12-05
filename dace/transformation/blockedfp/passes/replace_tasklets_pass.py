import dace
import dace.transformation.pass_pipeline as ppl
import dace.transformation as xf
import dace.sdfg.tasklet_utils as tutil
import dace.sdfg.propagation

from dace.transformation.blockedfp.passes.extend_map_pass import ExtendMapPass
import dace.transformation.blockedfp.libraries as bfplib

from typing import List


# This pass is pretty huge and does multiple steps:
#   1. It extends the maps with the connections from the auxiliary arrays
#   2. It replaces every tasklet with a library node
#   3. It connects up all the library nodes
#   4. It expands all the library nodes
class ReplaceTaskletsPass(ppl.Pass):
    
    def __init__(self, names: List[str], blocking_factor: int = 16):
        
        self.__names = names
        self.__blocking_factor = blocking_factor
    
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Everything
    
    def should_reapply(self, _) -> bool:
        return False
    
    def apply_pass(self, sdfg: dace.SDFG, _) -> None:
        
        ExtendMapPass(self.__names).apply_pass(sdfg, {})
        sdfg.apply_transformations(ReplaceTaskletsTransform, options={"names": self.__names, "blocking_factor": self.__blocking_factor}, validate=False)
        sdfg.expand_library_nodes()

class ReplaceTaskletsTransform(xf.SingleStateTransformation):
    
    map_entry = xf.PatternNode(dace.nodes.MapEntry)
    tasklet = xf.PatternNode(dace.nodes.Tasklet)
    
    def __init__(self, names: List[str], blocking_factor: int = 16):
        self.__names = [f"{name}_fp" for name in names]
        self.__blocking_factor = blocking_factor
    
    @classmethod
    def expressions(cls) -> List[dace.sdfg.state.StateSubgraphView]:
        return [dace.sdfg.utils.node_path_graph(cls.map_entry, cls.tasklet)]
    
    def can_be_applied(self, graph: dace.SDFGState, expr_index: int, sdfg: dace.SDFG, permissive = False) -> bool:
        
        # Check if map reads from one of our arrays
        for edge in graph.in_edges(self.map_entry):
            if edge.data.data in self.__names:
                return True
        # Check if map writes to one of our arrays
        for edge in graph.out_edges(graph.exit_node(self.map_entry)):
            if edge.data.data in self.__names:
                return True
        
        return False
    
    def apply(self, graph: dace.SDFGState, sdfg: dace.SDFG) -> None:
        map_exit: dace.nodes.MapExit = graph.exit_node(self.map_entry)
        
        # Extend the intermediate access nodes
        for node in graph.all_nodes_between(self.map_entry, map_exit):
            if isinstance(node, dace.nodes.AccessNode):
                self.__extend_access_node(node, graph, sdfg)
        
        # Now iterate through all tasklets, change them to a sick new library node, and connect things up
        for node in graph.all_nodes_between(self.map_entry, map_exit):
            if isinstance(node, dace.nodes.Tasklet):
                self.__extend_tasklet(node, graph, sdfg)
                
        # Now iterate through all tasklets again and kill them
        for node in graph.all_nodes_between(self.map_entry, map_exit):
            if isinstance(node, dace.nodes.Tasklet):
                graph.remove_node(node)

        # directly connect outer map to first libnode now
        for in_edge in graph.in_edges(self.map_entry):
            for out_edge in graph.out_edges(self.map_entry):
                # Ugly check if the connectors are supposed to be the same
                if in_edge.dst_conn[3:] != out_edge.src_conn[4:]:
                    continue
                
                # directly connect outer map over inner map
                graph.add_edge(in_edge.src, in_edge.src_conn, out_edge.dst, out_edge.dst_conn, in_edge.data)
          
        for in_edge in graph.in_edges(map_exit):
            for out_edge in graph.out_edges(map_exit):
                # Ugly check if the connectors are supposed to be the same
                if in_edge.dst_conn[3:] != out_edge.src_conn[4:]:
                    continue
                # directly connect outer map over inner map
                expr = str(out_edge.data)
                expr = expr.replace("31/16", "1")
                expr = expr.replace("\n", "").replace(" ", "")

                #graph.add_edge(in_edge.src, in_edge.src_conn, out_edge.dst, out_edge.dst_conn, out_edge.data)
                graph.add_edge(in_edge.src, in_edge.src_conn, out_edge.dst, out_edge.dst_conn, dace.Memlet(expr))

        # Remove inner map
        graph.remove_nodes_from([self.map_entry, map_exit])
        dace.sdfg.propagation.propagate_memlets_state(sdfg, graph)
    
    def __extend_access_node(self, node: dace.nodes.AccessNode, state: dace.SDFGState, sdfg: dace.SDFG) -> None:
        # Get the name of the array
        name = node.data
        
        # Change fp type and shape
        data_desc = sdfg.arrays[name]
        sdfg.remove_data(name, validate=False)
        sdfg.add_array(name, (16,), dtype=dace.float32, storage=data_desc.storage, location=data_desc.location, lifetime=data_desc.lifetime, transient=data_desc.transient)
        #sdfg.arrays[name].dtype = dace.dtypes.float32
        # very ugly crude approcimation for a block
        #sdfg.arrays[name].shape = sdfg.arrays[self.__names[0]].shape[(len(sdfg.arrays[self.__names[0]].shape) // 2):]
        
        # Add new arrays
        sdfg.add_array(f"{name}_bias", shape=(1,), dtype=dace.dtypes.float64, transient=True)
        sdfg.add_array(f"{name}_scale", shape=(1,), dtype=dace.dtypes.float64, transient=True)
        
        # Add Access nodes
        state.add_access(f"{name}_bias")
        state.add_access(f"{name}_scale")
        
        # Add name to array
        self.__names.append(name)
    
    def __extend_tasklet(self, tasklet: dace.nodes.Tasklet, state: dace.SDFGState, sdfg: dace.SDFG) -> None:
     
        # Get classification of this tasklet
        classification = tutil.classify_tasklet(state, tasklet)
     
        # Find corresponding library node and add it
        if classification["type"] is tutil.TaskletType.ARRAY_ARRAY or classification["type"] is tutil.TaskletType.ARRAY_SCALAR or classification["type"] is tutil.TaskletType.SCALAR_ARRAY or classification["type"] is tutil.TaskletType.SCALAR_SCALAR:
            match classification["op"]:
                case '+':
                    libnode = bfplib.BFPAddNode(tasklet.name)
                case '*':
                    libnode = bfplib.BFPMultNode(tasklet.name)
                case op:
                    raise ValueError(f"Unsopported op encountered with array op '{op}'")
        elif classification["type"] is tutil.TaskletType.ARRAY_SYMBOL or classification["type"] is tutil.TaskletType.SCALAR_SYMBOL:
            match classification["op"]:
                case '+':
                    libnode = bfplib.BFPAddSymNode(tasklet.name, classification["constant1"] or classification["constant2"])
                case '*':
                    libnode = bfplib.BFPMultSymNode(tasklet.name, classification["constant1"] or classification["constant2"])
                case op:
                    raise ValueError(f"Unsupported op encountered with symbol op '{op}'")
        elif classification["type"] is tutil.TaskletType.ARRAY_SCALAR_ASSIGNMENT or classification["type"] is tutil.TaskletType.ARRAY_ARRAY_ASSIGNMENT:
            libnode = bfplib.BFPAssignScalarNode(tasklet.name)
        else:
            raise ValueError(f"WHAT THE FUCK AM I SEEING, {classification['type']}")

        # Add new libnode
        state.add_node(libnode)

        # Connect libnode
        self.__connect_libnodes(tasklet, libnode, state, sdfg)
    
    def __connect_libnodes(self, tasklet: dace.nodes.Tasklet, libnode: dace.nodes.LibraryNode, state: dace.SDFGState, sdfg: dace.SDFG) -> None:
        
        # Get classification of this tasklet
        classification = tutil.classify_tasklet(state, tasklet)
        
        prefix = "a"
        for edge in state.in_edges(tasklet):
            
            # is this edge coming out of a mapentry
            from_map = isinstance(edge.src, dace.nodes.MapEntry)
            
            # calculate src
            fp_src = edge.src
            bias_src = edge.src if from_map else self.__get_access_node(f"{edge.src.data}_bias", state)
            scale_src = edge.src if from_map else self.__get_access_node(f"{edge.src.data}_scale", state)

            # calculate src connectors
            fp_src_con = edge.src_conn if from_map else None
            bias_src_con = f"{edge.src_conn[:-3]}_bias" if from_map else None
            scale_src_con = f"{edge.src_conn[:-3]}_scale" if from_map else None
            
            # calculate memlets
            fp_memlet = edge.data if from_map else sdfg.make_array_memlet(edge.src.data)
            bias_memlet = dace.Memlet() if from_map else sdfg.make_array_memlet(f"{edge.src.data}_bias")
            scale_memlet = dace.Memlet() if from_map else sdfg.make_array_memlet(f"{edge.src.data}_scale")
            
            state.add_edge(fp_src, fp_src_con, libnode, f"{prefix}_fp", fp_memlet)
            state.add_edge(bias_src, bias_src_con, libnode, f"{prefix}_bias", bias_memlet)
            state.add_edge(scale_src, scale_src_con, libnode, f"{prefix}_scale", scale_memlet)
            
            # increment prefix
            prefix = chr(ord(prefix)+1)
        
        for edge in state.out_edges(tasklet):
       
            # is this edge coming out of a mapentry
            to_map = isinstance(edge.dst, dace.nodes.MapExit)
            
            # calculate src
            fp_dst = edge.dst
            bias_dst = edge.dst if to_map else self.__get_access_node(f"{edge.dst.data}_bias", state)
            scale_dst = edge.dst if to_map else self.__get_access_node(f"{edge.dst.data}_scale", state)
            
            # calculate src connectors
            fp_dst_con = edge.dst_conn if to_map else None
            bias_dst_con = f"{edge.dst_conn[:-3]}_bias" if to_map else None
            scale_dst_con = f"{edge.dst_conn[:-3]}_scale" if to_map else None
            
            # calculate memlets
            fp_memlet = edge.data if to_map else sdfg.make_array_memlet(edge.dst.data)
            bias_memlet = dace.Memlet() if to_map else sdfg.make_array_memlet(f"{edge.dst.data}_bias")
            scale_memlet = dace.Memlet() if to_map else sdfg.make_array_memlet(f"{edge.dst.data}_scale")
    
            state.add_edge(libnode, "out_fp", fp_dst, fp_dst_con, fp_memlet)
            state.add_edge(libnode, "out_bias", bias_dst, bias_dst_con, bias_memlet)
            state.add_edge(libnode, "out_scale", scale_dst, scale_dst_con, scale_memlet)
    
    def __get_access_node(self, name: str, state: dace.SDFGState) -> dace.nodes.AccessNode:
        for node in state.nodes():
            # Check if the node is an AccessNode
            if not isinstance(node, dace.nodes.AccessNode):
                continue
            if node.data == name:
                return node
        raise ValueError(f"Access node for '{name}' was not found")