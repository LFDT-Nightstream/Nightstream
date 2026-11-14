# neo-params

The parameter set and its checks.

- `NeoParams::nightstream_p64_k14()`: the only profile, "Nightstream P64, k_rho = 14". It uses
  q = 2^64 − 59, η = 128, d = 64, b = 2, B = 2^14, κ = 18, T = 128 and s = 2. The constants are in the
  module `nightstream_p64_k14`. See `decisions/field-2p64-59-ring-x64.md`.
- The RLC guard `(k + 1)·T·(b − 1) < B`, and the PiCCS security census for a CCS shape.
- `poseidon2`: the sponge shape (width, rate, capacity, digest length).
