import dace
from typing import List
from dace.transformation import Pipeline, Pass
from dace.transformation.blockedfp.passes.map_tiling_pass import MapTilingPass
from dace.transformation.layout.split_dimension import SplitDimensions
from dace.transformation.blockedfp.passes.add_scale_bias_pass import AddScaleBias
from dace.transformation.blockedfp.passes.change_fp_type_pass import ChangeFPType
from dace.transformation.blockedfp.passes.extend_tasklets_pass import ExtendTaskletsPass
from dace.transformation.blockedfp.passes.replace_tasklets_pass import ReplaceTaskletsPass

class BlockedFP(Pipeline):
    """
    A pipeline that transforms a given array into a blocked floating point format
    """
    
    def __init__(self, name: str, block_size: List[int]):
        
        self._name = name
        
        if not block_size:
            raise ValueError("block_size must contain at least one dimension!")
        
        self._block_size = block_size
        
        split_map = {
            name: ([True for _ in block_size], block_size)
        }
        
        passes: list[Pass] = [
            MapTilingPass(name=name),
            SplitDimensions(split_map=split_map),
            AddScaleBias(name=name, block_size=block_size),
            ChangeFPType(name=name),
            ExtendTaskletsPass(name=name),
            ReplaceTaskletsPass(name=name),
        ]
        
        super().__init__(passes=passes)