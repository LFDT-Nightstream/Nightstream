# Security Parameters

## Status

Open question.

This note records the current security-parameter issue in `neo-fold-clean`.
It is not a final protocol claim. The goal is to keep the calculation and
decision points easy to audit.

## Paper Facts

SuperNeo runs sumcheck over an extension field:

```text
K = F_{q^s}
```

The basic sumcheck soundness error is bounded by roughly:

```text
error <= ell * d_sc / |K|
```

But SuperNeo Appendix D.4 charges two terms for Π_CCS:

```text
epsilon_SC = max(u, 2b + 1, 2) * log(m) / |K|
epsilon_SZ = (2K_fresh + k_rho) * max(log(m), k_rho * t * d) / |K|
```

So the full Π_CCS effective security bits are roughly:

```text
factor     = epsilon_SC_numerator + epsilon_SZ_numerator
lambda_eff ~= log2(|K|) - log2(factor) - safety_margin
```

For Goldilocks:

```text
q ~= 2^64
|K| = q^s
log2(|K|) ~= 64 * s
```

## Appendix B.2 Goldilocks Facts

```text
q     = 2^64 - 2^32 + 1
d     = 54
b     = 2
k_rho = 14
B     = 2^14
T     = 216
s     = 2
|C|   ~= 2^125
|K|   ~= 2^128
```

The important split: Appendix B.2 gives `|C| ~= 2^125`, but Π_CCS
soundness also pays for concrete shape factors over `K`. With `s = 2`,
`|K| ~= 2^128`, so only a few bits are available if we want strict
125-bit effective Π_CCS soundness.

## How ell Is Calculated

For the current optimized R1CS/CCS path, we size the sumcheck domain by
the coefficient-expanded CCS shape:

```text
shape_size  = max(ccs.n, ccs.m)
padded_size = next_power_of_two(shape_size)
ell         = ceil(log2(d * padded_size))
```

where:

```text
ccs.n = number of CCS rows / constraints
ccs.m = number of CCS variables
d     = ring degree, 54 for Goldilocks Appendix B.2
```

We use `max(ccs.n, ccs.m)` because FE checks are row-driven, while
NC/witness checks can be width-driven.

## How d_sc Is Calculated

For the current R1CS-to-CCS path, the helper uses SuperNeo D.4:

```text
d_sc = max(u, 2b + 1, 2)
```

where:

```text
u = CCS polynomial degree
b = norm bound
```

For the standard R1CS-to-CCS embedding:

```text
u    = 2
b    = 2
d_sc = max(2, 5, 2)
d_sc = 5
```

The checked polynomial is not just the R1CS degree. The norm-check term
contributes the `2b + 1` part.

The helper also charges the D.4 Schwartz-Zippel term. Because
`FoldSchedule` can choose larger batches, the conservative config charges
the maximum fresh count allowed by the Appendix B.2 RLC guard:

```text
(K_fresh + k_rho) * T * (b - 1) < B
(K_fresh + 14) * 216 * 1 < 16384
K_fresh <= 61
```

## Example: Current Fibonacci Bits Test

Current perf shape:

```text
ccs.n         = 60
ccs.m         = 54
d             = 54
b             = 2
u             = 2
s             = 2
safety_margin = 2
```

Calculate `ell`:

```text
shape_size  = max(60, 54) = 60
padded_size = next_power_of_two(60) = 64
d * padded_size = 54 * 64 = 3456
ell = ceil(log2(3456))
ell = 12
```

Calculate `d_sc`:

```text
d_sc = max(u, 2b + 1, 2)
d_sc = max(2, 5, 2)
d_sc = 5
```

Calculate the full D.4 factor:

```text
epsilon_SC_numerator = 12 * 5 = 60
k_rho * t * d        = 14 * 3 * 54 = 2268
epsilon_SZ_numerator = (2 * 61 + 14) * max(12, 2268)
                     = 136 * 2268
                     = 308448
factor               = 60 + 308448
                     = 308508
```

Calculate effective lambda:

```text
log2(|K|) ~= 128
log2(308508) ~= 18.23

lambda_eff ~= 128 - 18.23 - 2
lambda_eff ~= 107
```

So under `s = 2`, this shape meets the current executable full-D.4 floor:

```text
MIN_EFFECTIVE_LAMBDA = 100
```

The helper still chooses `lambda = 107` for this concrete shape because it
searches for the strongest lambda that satisfies the `s = 2` policy above
the configured floor. It does not meet 120-bit or strict 125-bit effective
Π_CCS soundness under this conservative D.4 accounting.

## What Changes With s = 3

With `s = 3`:

```text
log2(|K|) ~= 192
lambda_eff ~= 192 - 18.23 - 2
lambda_eff ~= 171
```

So `s = 3` would comfortably cover strict 125-bit effective Π_CCS
soundness for this example.

But `s = 3` is not a config-only change. It requires extension-field
support for `F_{q^3}` across sumcheck, NIFS, transcript challenge sampling,
CE claims, serialization/digests, and eventually the decider/F' verifier
gadgets.

## Current Implementation Policy

For now, `neo-fold-clean` keeps:

```text
s = 2
Appendix B.2 Goldilocks core parameters
```

For executable R1CS/CCS paths, it uses a shape-effective lambda and rejects
shapes below:

```text
MIN_EFFECTIVE_LAMBDA = 100
EXTENSION_SAFETY_MARGIN_BITS = 2
```

This is an explicit floor. The implementation should not silently drift to
lower effective security.

## Open Questions

- Is the 100-bit full-D.4 floor acceptable for the current product target?
- Do we require a 120-bit effective floor, which requires `s = 3` under this
  conservative accounting?
- Do we require strict effective lambda >= 125 for all production circuits?
- Should we implement `s = 3`?
- Is the current `ell` / D.4 factor policy conservative relative to the paper
  proof, and can it be tightened without inventing security?
- Should public docs distinguish "Appendix B.2 core parameters" from
  "strict effective 125-bit soundness"?

## Resolution Criteria

This question is closed when we choose one of:

- Keep `s = 2`, document the shape-effective lambda model, and enforce an
  agreed minimum effective lambda.
- Implement `s = 3` and make strict >=125 effective sumcheck soundness the
  production target.
- Justify a tighter soundness calculation that keeps `s = 2` at the desired
  threshold for the intended circuit shapes.
