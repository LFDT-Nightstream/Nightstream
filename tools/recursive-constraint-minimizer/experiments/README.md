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
