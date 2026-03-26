"""Find the minimum-CNOT circuit for Dicke state D(4,2).

Sweeps CNOT count k = 2..8. For each k, samples random CNOT topologies
and optimizes Ry rotation angles via L-BFGS-B to maximize state fidelity.
Stops when fidelity >= 0.9999 (exact preparation found).

Requirements: numpy, scipy
Usage: python dicke42_search.py
"""

import numpy as np
from scipy.optimize import minimize


def dicke_state(n: int, k: int) -> np.ndarray:
    """Return the Dicke state D(n,k) as a real vector."""
    d = 2**n
    state = np.zeros(d)
    count = sum(1 for i in range(d) if bin(i).count("1") == k)
    amp = 1.0 / np.sqrt(count)
    for i in range(d):
        if bin(i).count("1") == k:
            state[i] = amp
    return state


def precompute_cnot_matrices(n: int) -> dict:
    """Precompute CNOT matrices for all directed qubit pairs."""
    d = 2**n
    mats = {}
    for c in range(n):
        for t in range(n):
            if c == t:
                continue
            m = np.zeros((d, d))
            for b in range(d):
                ctrl_bit = (b >> (n - 1 - c)) & 1
                new_b = b ^ ((1 << (n - 1 - t)) if ctrl_bit else 0)
                m[new_b, b] = 1.0
            mats[(c, t)] = m
    return mats


def make_ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


def simulate_circuit(n, topology, params, cx_mats):
    """Simulate Ry+CNOT circuit on |0...0>, return output state vector."""
    d = 2**n
    state = np.zeros(d)
    state[0] = 1.0
    idx = 0
    for layer in range(len(topology) + 1):
        if layer > 0:
            state = cx_mats[topology[layer - 1]] @ state
        ry_layer = np.array([1.0])
        for q in range(n):
            ry_layer = np.kron(ry_layer, make_ry(params[idx]))
            idx += 1
        state = ry_layer @ state
    return state


def search_dicke(n, k_hamming, max_cnot=8, n_topologies=500, n_restarts=6, seed=42):
    """Find minimum CNOT count to prepare D(n, k_hamming) from |0...0>."""
    np.random.seed(seed)
    target = dicke_state(n, k_hamming)
    cx_mats = precompute_cnot_matrices(n)
    cx_options = list(cx_mats.keys())
    n_options = len(cx_options)

    print(f"Searching D({n},{k_hamming}): {n} qubits, {n_options} CNOT options")

    for num_cx in range(2, max_cnot + 1):
        best_fid = 0.0
        best_topo = None
        best_params = None
        n_params = n * (num_cx + 1)
        n_topo = min(n_topologies, n_options**num_cx)

        for trial in range(n_topo):
            topo = tuple(
                cx_options[np.random.randint(n_options)] for _ in range(num_cx)
            )
            for _ in range(n_restarts):
                p0 = np.random.uniform(-np.pi, np.pi, n_params)

                def neg_fidelity(p, topo=topo):
                    s = simulate_circuit(n, topo, p, cx_mats)
                    return -(np.dot(target, s) ** 2)

                res = minimize(
                    neg_fidelity,
                    p0,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-14},
                )
                fid = -res.fun
                if fid > best_fid:
                    best_fid = fid
                    best_topo = topo
                    best_params = res.x.copy()

        print(f"  k={num_cx}: fidelity={best_fid:.10f}  topology={best_topo}")

        if best_fid >= 0.9999:
            print(f"\nExact preparation found at k={num_cx} CNOT gates.")
            print(f"Topology: {best_topo}")
            print(f"Parameters: {best_params.round(16).tolist()}")
            return num_cx, best_topo, best_params

    print("No exact preparation found.")
    return None, None, None


if __name__ == "__main__":
    search_dicke(n=4, k_hamming=2)
