import dace as dc
import numpy as np
import pytest
from dace.transformation.blockedfp.blockedfp import BlockedFP

# Symbols
NI = dc.symbol("NI")
NJ = dc.symbol("NJ")
NK = dc.symbol("NK")
NL = dc.symbol("NL")
S  = dc.symbol("S")


@dc.program
def k2mm_kernel(alpha: dc.float64,
                beta: dc.float64,
                A: dc.float64[S * NI, S * NK],
                B: dc.float64[S * NK, S * NJ],
                C: dc.float64[S * NJ, S * NL],
                D: dc.float64[S * NI, S * NL]) -> None:
    D[:] = alpha * (A @ B @ C) + beta * D


# Compile once per block size (EXACTLY like gemm test)
@pytest.fixture(scope="module", params=[2, 4, 8, 16])
def compiled_k2mm(request):
    s = request.param

    sdfg = k2mm_kernel.to_sdfg()
    BlockedFP(s).apply_pass(sdfg, {})
    sdfg.expand_library_nodes()
    sdfg.validate()

    return sdfg.compile(), s


# Parametrized test
@pytest.mark.parametrize("ni", [1, 2, 8, 64])
@pytest.mark.parametrize("nj", [1, 2, 8, 64])
@pytest.mark.parametrize("nk", [1, 2, 8, 64])
@pytest.mark.parametrize("nl", [1, 2, 8, 64])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("scale", [1, 10, 100])
def test_k2mm(compiled_k2mm, ni, nj, nk, nl, alpha, beta, scale):
    compiled_fn, s = compiled_k2mm

    ni_size = ni * s
    nj_size = nj * s
    nk_size = nk * s
    nl_size = nl * s

    # Random inputs
    A = np.random.uniform(-scale, scale, (ni_size, nk_size))
    B = np.random.uniform(-scale, scale, (nk_size, nj_size))
    C = np.random.uniform(-scale, scale, (nj_size, nl_size))
    D = np.random.uniform(-scale, scale, (ni_size, nl_size))

    D_orig = D.copy()

    # Reference
    expected = alpha * (A @ B @ C) + beta * D_orig

    # Run kernel
    compiled_fn(
        alpha=alpha,
        beta=beta,
        A=A,
        B=B,
        C=C,
        D=D,
        NI=ni,
        NJ=nj,
        NK=nk,
        NL=nl,
        S=s,
    )

    error = np.linalg.norm(D - expected) / (ni_size * nl_size)

    assert error < 1e-2, (
        f"Failed with NI={ni}, NJ={nj}, NK={nk}, NL={nl}, "
        f"alpha={alpha}, beta={beta}, scale={scale}, "
        f"blocksize={s}, error={error}"
    )