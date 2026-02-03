import dace
import numpy as np
import matplotlib.pyplot as plt
from dace.transformation.blockedfp.blockedfp import BlockedFP

N = dace.symbol("N")
M = dace.symbol("M")
L = dace.symbol("L")

@dace.program
def gemm(A: dace.float64[16 * N, 16 * M], 
         B: dace.float64[16 * M, 16 * L], 
         C: dace.float64[16 * N, 16 * L]) -> None:
    C[:] = A @ B  # modify in-place

# Compile SDFG once
sdfg = gemm.to_sdfg()
BlockedFP().apply_pass(sdfg, {})
sdfg.expand_library_nodes()
sdfg.validate()
compiled = sdfg.compile()

# Parameters
ks = [1, 2, 4, 8, 16]        # matrix size multipliers
scales = [1, 2, 4, 8, 16]    # value range for A and B

# Collect errors
errors = {}

for k in ks:
    n = 16 * k
    errs = []
    for scale in scales:
        A = np.random.uniform(-scale, scale, (n, n))
        B = np.random.uniform(-scale, scale, (n, n))
        C = np.zeros((n, n))

        expected = A @ B

        compiled(A=A, B=B, C=C, N=n // 16, M=n // 16, L=n // 16)

        error = np.linalg.norm(C - expected) / (n * n)
        errs.append(error)
        print(f"k={k}, scale={scale}, error={error:.3e}")

    errors[k] = errs

# Plot
plt.figure(figsize=(8,6))
for k, errs in errors.items():
    plt.plot(scales, errs, marker='o', label=f"k={k} (size {16*k})")

#plt.xscale('log', base=2)
#plt.yscale('log')
plt.xlabel("Scale of random values")
plt.ylabel("Error (normalized)")
plt.title("Blocked GEMM Error vs Value Scale")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig("gemm_error.png")
