#!/usr/bin/env python3
"""Fast Dicke state minimum-CNOT search using Ry+CNOT circuits."""
import numpy as np
from scipy.optimize import minimize
from functools import reduce
import json, time

def dicke(n, k):
    d = 2**n; s = np.zeros(d)
    cnt = sum(1 for i in range(d) if bin(i).count('1') == k)
    amp = 1.0/np.sqrt(cnt)
    for i in range(d):
        if bin(i).count('1') == k: s[i] = amp
    return s

def precompute_cx(n):
    d = 2**n; mats = {}
    for c in range(n):
        for t in range(n):
            if c == t: continue
            m = np.zeros((d, d))
            for b in range(d):
                cb = (b >> (n-1-c)) & 1
                nb = b ^ ((1 << (n-1-t)) if cb else 0)
                m[nb, b] = 1.0
            mats[(c,t)] = m
    return mats

def make_ry(theta):
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([[c, -s], [s, c]])

def circuit_eval(n, topo, params, cx_mats):
    d = 2**n; state = np.zeros(d); state[0] = 1.0; idx = 0
    for layer in range(len(topo)+1):
        if layer > 0:
            state = cx_mats[topo[layer-1]] @ state
        mats = [make_ry(params[idx+q]) for q in range(n)]
        ry = reduce(np.kron, mats)
        state = ry @ state
        idx += n
    return state

OUT = "/home/churchill/repos/zx-webs/data_agent8"

# Targets: Dicke states where optimal CNOT is unknown
targets = [
    ("D(4,2)", 4, 2),
    ("D(5,2)", 5, 2),
    ("D(4,3)", 4, 3),
    ("D(5,3)", 5, 3),
]

all_results = []

for state_name, n, k_hw in targets:
    target = dicke(n, k_hw)
    cx_mats = precompute_cx(n)
    cx_opts = list(cx_mats.keys())
    n_cx_opts = len(cx_opts)

    # Timing
    topo = tuple(cx_opts[:3])
    p = np.random.randn(n*4)
    t0 = time.time()
    for _ in range(500): circuit_eval(n, topo, p, cx_mats)
    ms = (time.time()-t0)/500*1000
    print(f"\n{'='*60}", flush=True)
    print(f"{state_name}: {n}q, eval={ms:.2f}ms, {n_cx_opts} CX options", flush=True)
    print(f"{'='*60}", flush=True)

    found_k = None
    for num_cx in range(2, min(3*n, 15)):
        t0 = time.time()
        best_fid = 0; best_topo = None; best_p = None
        np_ = n * (num_cx + 1)
        n_topo = min(500, n_cx_opts**num_cx)
        n_restart = 6

        for trial in range(n_topo):
            np.random.seed(42 + num_cx*10000 + trial)
            tp = tuple(cx_opts[np.random.randint(n_cx_opts)] for _ in range(num_cx))
            for r in range(n_restart):
                p0 = np.random.uniform(-np.pi, np.pi, np_)
                def neg_fid(p, tp=tp):
                    s = circuit_eval(n, tp, p, cx_mats)
                    return -(np.dot(target, s)**2)
                try:
                    res = minimize(neg_fid, p0, method='L-BFGS-B',
                                 options={'maxiter':100, 'ftol':1e-14})
                    f = -res.fun
                    if f > best_fid:
                        best_fid = f; best_topo = tp; best_p = res.x.copy()
                except: pass

        el = time.time() - t0
        marker = '  <<< EXACT >>>' if best_fid >= 0.9999 else ''
        print(f"  k={num_cx}: fid={best_fid:.10f} ({el:.1f}s){marker}", flush=True)

        if best_fid >= 0.9999:
            found_k = num_cx
            # Refine
            for r in range(300):
                p0 = np.random.uniform(-np.pi, np.pi, np_)
                def neg_fid(p, tp=best_topo):
                    s = circuit_eval(n, tp, p, cx_mats)
                    return -(np.dot(target, s)**2)
                res = minimize(neg_fid, p0, method='L-BFGS-B',
                             options={'maxiter':500, 'ftol':1e-15})
                if -res.fun > best_fid: best_fid = -res.fun; best_p = res.x.copy()

            out_state = circuit_eval(n, best_topo, best_p, cx_mats)
            print(f"  Refined: fid={best_fid:.15f}", flush=True)
            print(f"  Topology: {best_topo}", flush=True)
            print(f"  Params: {np.round(best_p, 8).tolist()}", flush=True)

            all_results.append({
                "state": state_name, "n_qubits": n, "hamming_weight": k_hw,
                "min_cnot": num_cx, "fidelity": best_fid,
                "topology": [list(t) for t in best_topo],
                "params": best_p.tolist(),
            })
            break

    if found_k is None:
        print(f"  NOT FOUND (searched up to k={min(3*n, 14)})", flush=True)
        all_results.append({
            "state": state_name, "n_qubits": n, "hamming_weight": k_hw,
            "min_cnot": None, "note": "not found",
        })

    # Save incrementally
    with open(f"{OUT}/dicke_all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

# Summary
print(f"\n{'='*60}", flush=True)
print("SUMMARY", flush=True)
print(f"{'='*60}", flush=True)
for r in all_results:
    print(f"  {r['state']}: min CNOT = {r.get('min_cnot', 'N/A')}, "
          f"fid = {r.get('fidelity', 'N/A')}", flush=True)
