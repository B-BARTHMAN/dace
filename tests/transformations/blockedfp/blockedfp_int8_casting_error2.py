import os
import dace
import numpy as np
import matplotlib.pyplot as plt

from dace.transformation.blockedfp.libraries import (
    BFPCastinNodeInt8,
    BFPCastoutNodeInt8,
)


def build_sdfg():
    N = dace.symbol("N")
    S = dace.symbol("S")

    sdfg = dace.SDFG("CASTING_TEST")
    sdfg.add_symbol("N", stype=int)
    sdfg.add_symbol("S", stype=int)

    sdfg.add_array(
        name="array",
        shape=(N * S,),
        dtype=dace.float64
    )

    state = sdfg.add_state("casting", is_start_state=True)

    bias = state.add_transient("bias", shape=(N,), dtype=dace.float64)
    scale = state.add_transient("scale", shape=(N,), dtype=dace.int8)
    ints = state.add_transient("ints", shape=(N * S,), dtype=dace.int8)

    access_in = state.add_access("array")
    access_out = state.add_access("array")

    castin = BFPCastinNodeInt8("castin", symbol_mapping={"N": "N", "S": "S"})
    castout = BFPCastoutNodeInt8("castout", symbol_mapping={"N": "N", "S": "S"})
    state.add_nodes_from([castin, castout])

    state.add_edge(access_in, None, castin, "array",
                   sdfg.make_array_memlet("array"))
    state.add_edge(castin, "bias", bias, None,
                   sdfg.make_array_memlet("bias"))
    state.add_edge(castin, "scale", scale, None,
                   sdfg.make_array_memlet("scale"))
    state.add_edge(castin, "ints", ints, None,
                   sdfg.make_array_memlet("ints"))
    state.add_edge(bias, None, castout, "bias",
                   sdfg.make_array_memlet("bias"))
    state.add_edge(scale, None, castout, "scale",
                   sdfg.make_array_memlet("scale"))
    state.add_edge(ints, None, castout, "ints",
                   sdfg.make_array_memlet("ints"))
    state.add_edge(castout, "array", access_out, None,
                   sdfg.make_array_memlet("array"))

    sdfg.expand_library_nodes()
    sdfg.validate()

    return sdfg.compile()


def main():
    bounds = [1, 2, 4, 8, 16, 32, 64, 128, 256]   # k (bounds)
    n_values = [1, 2, 4, 8, 16, 32]              # number of blocks
    s_values = [2, 4, 8, 16, 32]                 # block size
    num_runs = 100                               # averaging runs

    output_dir = "casting_error_plots_vs_n"
    os.makedirs(output_dir, exist_ok=True)

    sdfg = build_sdfg()

    # errors[k][s] -> list over n
    errors = {
        k: {s: [] for s in s_values}
        for k in bounds
    }

    # Run experiments
    for k in bounds:
        for s in s_values:
            for n in n_values:
                run_errors = []

                for _ in range(num_runs):
                    A = np.random.uniform(-k, k, size=(n * s,))
                    expected = A.copy()

                    sdfg(array=A, N=n, S=s)

                    err = np.linalg.norm(A - expected) / (n * s)
                    run_errors.append(err)

                avg_error = np.mean(run_errors)
                errors[k][s].append(avg_error)

    # Plot and save
    for k in bounds:
        plt.figure(figsize=(7, 5))

        for s in s_values:
            plt.plot(n_values, errors[k][s], marker="o", label=f"s={s}")

        plt.xlabel("Number of blocks (n)")
        plt.ylabel("L2 norm error")
        plt.title(
            f"Error vs number of blocks (bounds k={k})\n"
            f"Averaged over {num_runs} runs per datapoint"
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(
            output_dir, f"error_vs_n_k{k}_avg{num_runs}.png"
        )
        plt.savefig(filename, dpi=200)
        plt.close()

        print(f"Saved: {filename}")


main()