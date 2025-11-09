import dace
from typing import List
from dace.transformation import Pipeline, Pass
from dace.transformation.blockedfp.passes.map_tiling_pass import MapTilingPass
from dace.transformation.layout.split_dimension import SplitDimensions
from dace.transformation.passes.split_tasklets import SplitTasklets
from dace.transformation.blockedfp.passes.add_scale_bias_pass import AddScaleBias
from dace.transformation.blockedfp.passes.change_fp_type_pass import ChangeFPType
from dace.transformation.blockedfp.passes.extend_map_pass import ExtendMapPass
from dace.transformation.blockedfp.passes.extend_tasklets_pass import ExtendTaskletsPass
#from dace.transformation.blockedfp.passes.replace_tasklets_pass import ReplaceTaskletsPass

from typing import List

class BlockedFP(Pipeline):
    """
    A pipeline that transforms a given array into a blocked floating point format
    """
    
    def __init__(self, names: List[str], block_size: List[int]):
        
        self._names = names
        
        if not block_size:
            raise ValueError("block_size must contain at least one dimension!")
        
        self._block_size = block_size
        
        split_map = {
            name: ([True for _ in block_size], block_size) for name in names
        }
        
        passes: list[Pass] = [
            MapTilingPass(names=names),
            SplitDimensions(split_map=split_map),
            SplitTasklets(),
            AddScaleBias(names=names, block_size=block_size),
            ChangeFPType(names=names),
            ExtendMapPass(names=names),
            ExtendTaskletsPass(names=names),
            #ReplaceTaskletsPass(name=name),
        ]
        
        super().__init__(passes=passes)