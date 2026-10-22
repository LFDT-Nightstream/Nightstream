# neo-params

Typed parameter sets and their checks.

- `NeoParams::nightstream_goldilocks_k16()`: the Nightstream Goldilocks production profile, with
  k_rho = 16. It uses b = 2, B = 2^16, κ = 22, T = 216, η = 81 and s = 2.
- `NeoParams::goldilocks_paper_b2()`: the SuperNeo v1.2 Section 8.2 values (k_rho = 14, B = 2^14). They are
  reference values only. Do not use them for a production proof.
- The RLC guard `(k + 1)·T·(b − 1) < B`, and the PiCCS security census for a CCS shape.
- The `goldilocks_auto_*` selectors start from the Section 8.2 values. Tests use them.
