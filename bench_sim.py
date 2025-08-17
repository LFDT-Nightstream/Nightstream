import os
import numpy as np
import time


def ring_mul(a, b, n):
    prod = np.polymul(a, b)
    if len(prod) > n:
        for i in range(n, len(prod)):
            # Use modular arithmetic to avoid overflow
            prod[i - n] = (prod[i - n] - prod[i]) % (2**61)
        prod = prod[:n]
    return prod


def ajtai_commit(n, k, d, sigma=3.2):
    # Use smaller integers to avoid overflow
    A = np.random.randint(0, 2**31, (k, d, n))
    w = np.random.randint(0, 2**31, (d, n))
    e = np.random.normal(0, sigma, (k, n)).astype(int) % (2**31)
    c = np.zeros((k, n))
    for i in range(k):
        for j in range(d):
            c[i] += np.polymul(A[i, j], w[j])[:n]
        c[i] += e[i]
    return c


def multilinear_sumcheck(evals, claim):
    current = claim
    l = len(evals).bit_length() - 1
    for _ in range(l):
        half = len(evals) // 2
        s0 = sum(evals[:half])
        s1 = sum(evals[half:])
        uni = np.array([s0, s1 - s0])
        r = np.random.rand()
        evals[:half] = (1 - r) * evals[:half] + r * evals[half:]
        evals = evals[:half]
        current = np.polyval(uni[::-1], r)
    return current


# simulate_fold function removed - no longer needed


def benchmark(func, args, samples=100):
    times = []
    for _ in range(samples):
        start = time.perf_counter_ns()
        func(*args)
        times.append(time.perf_counter_ns() - start)
    return np.mean(times) / 1e6  # ms


if __name__ == "__main__":
    samples = 100
    if os.environ.get("RUN_LONG_TESTS"):
        samples = 1000
    with open('sim_bench.txt', 'w') as f:
        n = 64
        # Use smaller integers to avoid overflow in numpy operations
        a = np.random.randint(0, 2**31, n)
        b = np.random.randint(0, 2**31, n)
        time_ms = benchmark(ring_mul, (a, b, n), samples)
        f.write(f"ring_mul_n64: {time_ms:.2f} ms\n")

        time_ms = benchmark(ajtai_commit, (4, 2, 4), samples)
        f.write(f"commit_small: {time_ms:.2f} ms\n")

        evals_n = 1024 if not os.environ.get("RUN_LONG_TESTS") else 16384
        evals = np.random.rand(evals_n)
        claim = evals.sum()
        time_ms = benchmark(multilinear_sumcheck, (evals.copy(), claim), samples)
        f.write(f"multilinear_sumcheck_N{evals_n}: {time_ms:.2f} ms\n")

        # Removed full_fold_toy benchmark - no corresponding working Rust implementation

    print("Python simulations saved to sim_bench.txt")
