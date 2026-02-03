import dace
from dace.sdfg.graph import MultiConnectorEdge
from dataclasses import dataclass

@dataclass
class CastInOutInfo:
    fp: str
    bias: str
    scale: str
    fp_subset: dace.subsets.Subset
    block_subset: dace.subsets.Subset


def cast_inout_util(
    sdfg: dace.SDFG,
    edge: MultiConnectorEdge[dace.Memlet],
    blocking_factor: int = 16
) -> CastInOutInfo:

    name = edge.data.data
    array = sdfg.arrays[name]

    fp_name = f"{name}_fp"
    bias_name = f"{name}_bias"
    scale_name = f"{name}_scale"

    if fp_name not in sdfg.arrays:
        sdfg.add_array(
            name=fp_name,
            shape=array.shape,
            dtype=dace.float32,
            storage=array.storage,
            location=array.location,
            transient=True
        )

        block_shape = tuple(
            dace.symbolic.SymExpr(f"({d}) / {blocking_factor}")
            for d in array.shape
        )

        sdfg.add_array(
            name=bias_name,
            shape=block_shape,
            dtype=dace.float64,
            storage=array.storage,
            location=array.location,
            transient=True
        )

        sdfg.add_array(
            name=scale_name,
            shape=block_shape,
            dtype=dace.float64,
            storage=array.storage,
            location=array.location,
            transient=True
        )

    fp_subset = edge.data.subset

    block_subset = dace.subsets.Range([
        (
            dace.symbolic.SymExpr(f"{r[0]} / {blocking_factor}"),
            dace.symbolic.SymExpr(f"({r[1] + 1} / {blocking_factor}) - 1"),
            1
        )
        for r in edge.data.subset.ranges
    ])

    return CastInOutInfo(
        fp=fp_name,
        bias=bias_name,
        scale=scale_name,
        fp_subset=fp_subset,
        block_subset=block_subset
    )
