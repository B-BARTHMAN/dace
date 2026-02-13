import dace as dc
import numpy as np
import pytest
from dace.transformation.blockedfp.blockedfp import BlockedFP

# Symbols
NI = dc.symbol("NI")
NJ = dc.symbol("NJ")
NK = dc.symbol("NK")
NM = dc.symbol("NM")
NL = dc.symbol("NL")
S  = dc.symbol("S")


@dc.program
def k3mm_kernel(
    A: dc.float64[S * NI, S * NK],
    B: dc.float64[S * NK, S * NJ],
    C: dc.float64[S * NJ, S * NM],
    D: dc.float64[S * NM, S * NL],
) -> dc.float64[S * NI, S * NL]:
    return A @ B @ C @ D


# Compile once per block size
@pytest.fixture(scope="module", params=[2, 4, 8, 16])
def compiled_k3mm(request):
    s = request.param

    sdfg = k3mm_kernel.to_sdfg()
    BlockedFP(s).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    sdfg.validate()

    return sdfg.compile(), s


# Parametrized test (reduced)
@pytest.mark.parametrize("ni", [1, 2, 64])
@pytest.mark.parametrize("nj", [1, 2, 64])
@pytest.mark.parametrize("nk", [1, 2, 64])
@pytest.mark.parametrize("nm", [1, 2, 64])
@pytest.mark.parametrize("nl", [1, 2, 64])
@pytest.mark.parametrize("scale", [1, 10, 100])
def test_k3mm(compiled_k3mm, ni, nj, nk, nm, nl, scale):
    compiled_fn, s = compiled_k3mm

    ni_size = ni * s
    nj_size = nj * s
    nk_size = nk * s
    nm_size = nm * s
    nl_size = nl * s

    A = np.random.uniform(-scale, scale, (ni_size, nk_size))
    B = np.random.uniform(-scale, scale, (nk_size, nj_size))
    C = np.random.uniform(-scale, scale, (nj_size, nm_size))
    D = np.random.uniform(-scale, scale, (nm_size, nl_size))

    expected = A @ B @ C @ D

    result = compiled_fn(
        A=A,
        B=B,
        C=C,
        D=D,
        NI=ni,
        NJ=nj,
        NK=nk,
        NM=nm,
        NL=nl,
        S=s,
    )

    error = np.linalg.norm(result - expected) / (ni_size * nl_size)

    assert error < 1e-2, (
        f"Failed with NI={ni}, NJ={nj}, NK={nk}, NM={nm}, NL={nl}, "
        f"scale={scale}, blocksize={s}, error={error}"
    )
