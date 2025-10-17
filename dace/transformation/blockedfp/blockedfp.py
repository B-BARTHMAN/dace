import dace
from typing import List
from dace.transformation import Pipeline, Pass
from dace.transformation.layout.split_dimension import SplitDimensions
from dace.transformation.blockedfp.passes.add_scale_bias_pass import AddScaleBias

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
            SplitDimensions(split_map=split_map),
            AddScaleBias(name=name, block_size=block_size),
        ]
        
        super().__init__(passes=passes)