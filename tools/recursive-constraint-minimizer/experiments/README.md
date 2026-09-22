# Phi81 quotient candidate

`phi81_quotient.py` checks the proposed product relation over the exact
Goldilocks field, with `Phi81(X) = X^54 + X^27 + 1`:

```text
A(t) B(t) = H(t) + Phi81(t) Q(t), for t = 0, ..., 107
H = output - prior
```

Each polynomial has 54 coefficient slots. Thus the residual has degree at
most 107. All 108 nodes are distinct in Goldilocks. A valid product has a
quotient of degree at most 52; the final quotient coefficient must be zero.
Keeping 54 quotient slots requires the full 108-node check.

Run from the repository root with cvc5 on `PATH`:

```sh
timeout --signal=KILL 300 python3 tools/recursive-constraint-minimizer/experiments/phi81_quotient.py
```

The 300-second cap comes from the root `AGENTS.md`. No additional dependencies
or solver limits are used. The script writes `phi81_quotient.json` beside it.

The recorded run used cvc5 1.3.4 with its finite-field `gb` solver:

| Check | Result | Solver time |
|---|---|---:|
| A degree-at-most-107 polynomial vanishes at 108 nodes but has a nonzero coefficient | `unsat` | 98.453 s |
| A monic degree-107 polynomial vanishes at the first 107 nodes | `sat` | 2.086 s |

The script separately constructs the Lagrange coefficient matrix and checks
that its product with the 108-node Vandermonde matrix is the identity. This
check uses exact modular integer arithmetic.

The SAT model is replayed as a false ring product. For
`P(X) = product (X - t), t = 0, ..., 106`, take `A = B = 0`,
`H = -remainder(P, Phi81)`, and `Q = -quotient(P, Phi81)`. The script checks
every residual coefficient and every retained evaluation. Here
`H[0] = 9190115465959932585`, so the ring product is false. The omitted node
107 has residual `8463668624014718480`. The full model and replay coefficients
are in the result JSON.

These checks concern the polynomial contract. They do not prove the Lean
implementation, its matrix placement, package identity, Rust conformance, or
security composition. Acceptance still requires the Lean soundness and
constructive completeness proofs through the selected consumer.

## Matrix cost to measure

The candidate reduces a full base-field ring product from 1,836 rows and
1,782 group witnesses to 108 rows and 54 quotient witnesses. Its evaluation
rows have longer linear forms. For the current 41-coordinate field slots,
the following counts model the nonzero sparse coefficients, including row
selectors:

| Product | Current grouped plan | Quotient evaluation plan |
|---|---:|---:|
| First source, with zero prior | 477,846 | 947,864 |
| Later source | 480,060 | 1,184,803 |

The old Phi81 reduction contains 3,996 nonzero scalar products per ring.
Each uses two field forms. The 1,782 group witnesses occur in both their
product row and the final sum. A later source also has 54 output and 54 prior
forms. This gives `(2*3996 + 2*1782 + 108)*41 + 1836` entries.

The 108 evaluation nodes give `1 + 107*54 = 5779` nonzero weights per
polynomial. Phi81 is nonzero at all selected nodes. For a later source,
five polynomials give `5*5779*41 + 108` entries. The first source omits the
prior polynomial.

These are counts derived from the candidate's forms, not measurements of an
emitted package or runtime. They assume distinct retained fields in each
form and omit stored zero coefficients. Measure the selected streamer,
emitter, and matrix products before claiming a speed or memory improvement.
PiRLC's extension-valued evaluation families use two base-field components;
the challenge is a base-field ring value, so the same product contract applies
to each component without extension-field cross terms.

## Poseidon reduction feasibility

`poseidon_reduction.py` checks two restricted candidates against checkpoint
`8b7c07d8`. It changes no production relation, profile, or package.

```sh
timeout --signal=KILL 300 python3 tools/recursive-constraint-minimizer/experiments/poseidon_reduction.py
```

All seven controls passed with cvc5 1.3.4. Exact queries, timings, integer
replay, and coefficient data are in `poseidon_reduction.json`.

| Candidate or control | Result |
|---|---|
| Encode the missing value with 40 signed trits and an unrestricted integer modulus quotient | `unsat` |
| Encode it with the explicit 41-trit witness | `sat` |
| Produce it with the seventh-power S-box | `sat` |
| Nonzero bivariate relation of total degree at most 8 for `Y = X^49` | `unsat` |
| Degree-49 relation `Y - X^49` | `sat` |
| Incorrect replacement `Y = X^7` | `sat` counterexample |
| Remove the first link from `Z = X^7; Y = Z^7` | `sat` counterexample |

Forty signed trits have `3^40 = 12157665459056928801` possible encodings,
fewer than the Goldilocks modulus. With the existing radix-three weights,
their integer sum lies between `-6078832729528464400` and
`6078832729528464400`. The value `6078832729528464401` has no such encoding,
even modulo the field. It is the seventh power of `3194645001229403778`.
The 41-trit witness is forty `-1` digits followed by `1`. Dropping its last
digit changes the field value. The query uses exact integers and an
unrestricted modulus quotient, with no bitvectors or overflow assumption.
`tests/ConstraintReductionResearch.lean` now proves the cardinality bound,
the field recomposition bound, and this exact S-box counterexample. It does
not assert reachability of that S-box input in the complete protocol.

The degree-eight class has 45 monomials. Substitution maps `X^i Y^j` to
`X^(i+49j)`. These exponents are distinct and at most 392. The script checks
the resulting coefficient-map injection directly; it does not sample inputs.
The polynomial root bound needs 393 distinct nodes, which fit in Goldilocks.
The same Lean module now proves universal field vanishing forces all 45
coefficients to zero, using the field root bound and exponent injection.
This excludes only this scalar bivariate class with no extra witnesses. It
does not prove a lower bound for full Poseidon rounds, affine mixing, other
encodings, lookups, or equations over additional state coordinates.

The exact high-degree replacement has efficient witness maps: discard `Z`
in one direction and compute `Z = X^7` in the other. Its degree is outside
the fixed profile. Other replacements still need soundness, completeness,
and efficient witness reconstruction proofs.

Primary-source findings for further candidate selection:

- [CLAP, section VI](https://arxiv.org/html/2405.12115v2#S6) describes affine
  substitution, duplicate-expression removal, and removal of repeated range
  checks. These support searches for equal input expressions and existing
  range proofs. They establish no additional saving in this package.
- [Plonky3's Poseidon2 AIR](https://github.com/Plonky3/Plonky3/blob/main/poseidon2-air/src/air.rs)
  uses degree seven with no internal S-box register, or an extra cube
  register with degree-three constraints. Adding that register cannot reduce
  committed width here. Its tests also forge one retained value and recompute
  later rounds, which is a useful check when proposing a row removal.
- [cvc5's field documentation](https://cvc5.github.io/docs/latest/theories/finite_field.html)
  supports exact prime-field equations. The coefficient query uses that
  theory; the encoding query needs the separate integer-to-field argument.
- [FF_CVC5_Lean](https://github.com/NethermindEth/FF_CVC5_Lean) demonstrates
  Lean reconstruction of cvc5 field reasoning. This is a proof-integration
  reference, not an installed dependency or an authority for these results.
