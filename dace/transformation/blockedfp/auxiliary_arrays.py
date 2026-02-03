import dace
from typing import List, Tuple, Union

def compute_blocked_shape(
    shape: Tuple[Union[int, dace.symbolic.symbol]],
    blocking_factor: int = 16
) -> Tuple[Union[int, dace.symbolic.symbol, dace.symbolic.SymExpr]]:
    """Compute blocked shape via floor-division per dimension."""
    return tuple(
        x // blocking_factor if isinstance(x, int)
        else dace.symbolic.SymExpr(f"({x} / {blocking_factor})")#else dace.symbolic.SymExpr(f"int_floor({x}, {blocking_factor})")
        for x in shape
    )

def add_auxiliary_arrays(
    sdfg: dace.SDFG,
    array_names: List[str],
    blocking_factor: int = 16
) -> None:
    """
    Add transient full-precision, scale, and bias arrays for given SDFG arrays.
    """
    # Ensure all arrays exist
    for name in array_names:
        if name not in sdfg.arrays:
            raise ValueError(f"Array '{name}' was not found in the sdfg.")

    for name in array_names:
        array = sdfg.arrays[name]

        # Full-precision copy
        sdfg.add_array(
            f"{name}_fp", array.shape, dace.dtypes.float32,
            array.storage, array.location, transient=True
        )

        # Blocked scale and bias arrays
        block_shape = compute_blocked_shape(array.shape, blocking_factor)
        sdfg.add_array(
            f"{name}_scale", block_shape, array.dtype,
            array.storage, array.location, transient=True
        )
        sdfg.add_array(
            f"{name}_bias", block_shape, array.dtype,
            array.storage, array.location, transient=True
        )