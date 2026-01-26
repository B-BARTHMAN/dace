import dace
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-GUI backend
import matplotlib.pyplot as plt
from dace.transformation.blockedfp.blockedfp import blocked_fp

# --------------------------
# DaCe program
# --------------------------

Nsym = dace.symbol("N")

@dace.program
def vec_add(A: dace.float64[16*Nsym],
            B: dace.float64[16*Nsym],
            C: dace.float64[16*Nsym]):
    for i in dace.map[0:16*Nsym]:
        C[i] = A[i] + B[i]

# --------------------------
# Parameter sweep
# --------------------------

Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
bounds = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

# Compile once
sdfg = vec_add.to_sdfg()
blocked_fp(sdfg, ["A", "B", "C"])
sdfg.validate()
compiled = sdfg.compile()

# Collect results: dict of N -> list of errors
results = {N: [] for N in Ns}

for N in Ns:
    size = 16 * N
    for bound in bounds:
        A = np.random.uniform(-bound, bound, size)
        B = np.random.uniform(-bound, bound, size)
        C = np.zeros_like(A)

        expected = A + B
        compiled(A=A, B=B, C=C, N=N)

        err = np.linalg.norm(C - expected)
        results[N].append(err)

# --------------------------
# Plot
# --------------------------

plt.figure(figsize=(10, 6))

for N in Ns:
    plt.plot(bounds, results[N], marker='o', label=f"N={N}")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Bound")
plt.ylabel(r"$\|C - (A + B)\|_2$")
plt.title("Vector Add Error vs Bound for Different N")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(title="Array size factor N")
plt.tight_layout()
plt.savefig("vec_add_error.png", dpi=150)
print("Plot saved to vec_add_error.png")
