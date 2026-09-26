# Phi81 quotient values as the witness basis

The exact arithmetic experiment confirms the predicted **111,598,761 fewer
matrix nonzeros** when the 54 quotient slots hold `Q(0), ..., Q(53)` instead
of coefficients. It keeps all 108 checks at `0, ..., 107`. This candidate
does not reduce committed coordinates or rows.

The [script](phi81_quotient_basis.py) and [recorded result](phi81_quotient_basis.json)
change no production code, package, fixture, or Rust file. Run with:

```sh
timeout --signal=KILL 300 python3 -B tools/recursive-constraint-minimizer/experiments/phi81_quotient_basis.py
```

The 300-second cap is the root `AGENTS.md` non-Lean test cap. The recorded
run took 4.090 seconds.

## Exact maps

Work in Goldilocks, `p = 18446744069414584321`. For `0 <= j < 54`, define

```text
L_j(X) = product_(0 <= m < 54, m != j) (X-m)/(j-m)
v_j    = Q(j) = sum_(0 <= i < 54) j^i q_i
q_i    = sum_(0 <= j < 54) coefficient_i(L_j) v_j
Q(t)   = sum_(0 <= j < 54) L_j(t) v_j
```

All denominators are nonzero in the selected field. The script constructs
the 54 by 54 Vandermonde matrix `V` and the Lagrange coefficient matrix `I`.
It checks every entry of `V*I = I*V = identity`. It also checks both matrix
equalities connecting these maps to evaluation at all 108 nodes. These
coefficient checks cover every field vector; they do not sample witnesses.

This gives exact forward and inverse witness maps with quadratic field work
in the fixed dimension 54. Both representations use 54 general-field values,
which can use the existing 41-trit encoding. A production proof must connect
these field maps to that encoding and the selected extraction path.

The old padding fact is about coefficient `q_53`, not value `Q(53)`. For
`A = X^53` and `B = X`, division by `X^54 + X^27 + 1` gives `Q = 1` and
`H = -1-X^27`. Thus `q_53 = 0` while `Q(53) = 1`. The script replays all 108
equations for this product. Pinning the last new value to zero would lose
completeness.

## Code and coefficient replay

The experiment reads the current source and the existing formula library:

- `Phi81ProductPlan.evaluateForm` uses the powers `t^i`; `outputForm` scales
  the quotient by `Phi81(t)`.
- `SharedFormulas.phi81Interface` assigns quotient inputs `109..162`.
  `ProductSumRow.Forms.meaningfulForm` puts the result in port 4.
- Every quotient coefficient in all 108 saved formula rows was compared
  with `Phi81(t)*t^i`, including stored zero terms.
- The four source families have dimensions `(17,22,1)`, `(17,5,1)`,
  `(17,1,2)`, `(17,14,2)`. They give 969 base-field ring products. The script
  checks their full lane/cell indexing is a disjoint map onto 52,326 slots.
- `ProductRetainedBlock` selects field slots; `BalancedTernary.width` is 41.
  `RetainedSlot.recomposeForms` uses powers of 3. All 41 scaled weights of
  every nonzero quotient term remain nonzero in Goldilocks.
- `PiRLCRetainedGeometry` places the quotient block before the selector and
  product-output blocks. Quotient terms do not share columns with the prior
  or output terms in the same result form.
- Rust's `matrix_program/phi81.rs` loads those slots and calls the saved Lean
  template. `template.rs` substitutes forms; `form.rs` removes zero scaling
  and reconstructs field slots with radix-three runs.

These reads verify the local cost calculation. They do not run the full
matrix streamer or emit a candidate package.

## Separate cost axes

| Quantity | Coefficient basis | Value basis |
|---|---:|---:|
| Quotient field slots per ring | 54 | 54 |
| Product rows per ring | 108 | 108 |
| Normalized quotient weights per ring | 5,779 | 2,970 |
| Expanded quotient matrix nonzeros per ring | 236,939 | 121,770 |
| Complete committed coordinates | 184,359,564 | 184,359,564 |
| Complete logical rows | 4,703,127 | 4,703,127 |
| Complete matrix nonzeros | 3,001,571,645 | 2,889,972,884 predicted |

The first 54 new evaluation forms are singletons. The other 54 each have
54 nonzero weights. Every selected `Phi81(t)` is nonzero. Therefore the
exact saving is `(5779-2970)*41*969 = 111598761` entries, about 3.72% of
checkpoint matrix nonzeros. The predicted total remains 554,150,409 entries
above the original baseline.

The current serialized template contains 5,832 quotient terms per ring,
including zeros. The table counts normalized nonzeros. A dense implementation
of Lagrange forms could still serialize zero terms; this experiment makes no
package-size claim. The complete checkpoint counts come from the saved
[conformance record](checkpoint-metrics.json), not a new complete execution.
Proving time, witness-generation time, and memory have not been measured for
this candidate.

## Controls and tool choice

| Check | Result | Time |
|---|---|---:|
| Exact matrices, current formula replay, and counts | Passed | 0.163 s |
| Nonzero degree-at-most-53 polynomial vanishes at all 54 basis nodes | cvc5 `unsat` | 3.773 s |
| Monic degree-53 polynomial vanishes at the first 53 basis nodes | cvc5 `sat` | 0.144 s |

The SAT model is exactly `product_(m=0..52)(X-m)`. Its value at the missing
node 53 is `7335203899209668766`, so the control detects the lost coordinate.
This control concerns all degree-at-most-53 polynomials. It is not a
counterexample to a separate construction restricted to degree at most 52.
The JSON records both complete SMT queries and solver output. The unchanged
108-node residual-degree obligation retains the existing
[quotient experiment](phi81_quotient.json); the controls here test the new
basis map.

SageMath, SymPy, `galois`, and the cvc5 Python bindings are absent locally.
The cvc5 CLI is version 1.3.4 with CoCoALib. Python integer arithmetic and
modular inverses complete the exact matrix checks in less than 0.2 seconds,
so this experiment needs no large installation.

The primary documentation checked was SageMath's
[finite-field construction](https://doc.sagemath.org/html/en/reference/finite_rings/sage/rings/finite_rings/finite_field_constructor.html)
and [Lagrange interpolation](https://doc.sagemath.org/html/en/reference/polynomial_rings/sage/rings/polynomial/polynomial_ring.html#sage.rings.polynomial.polynomial_ring.PolynomialRing_field.lagrange_polynomial),
plus [cvc5's exact prime-field syntax](https://cvc5.github.io/docs/cvc5-1.3.4/theories/finite_field.html).
The search queries were `site.doc.sagemath.org Lagrange polynomial finite field
interpolation polynomial_ring constructor` and `site.cvc5.github.io finite
field theory finite field arithmetic Python API`; the source paths and local
inspection queries are recorded in the JSON.

This is an exact-arithmetic candidate check, not Lean proof authority.
Selection still needs Lean equivalence, constructive encoding and extraction
maps, matrix placement, and selected consumer conformance. It makes no
progress toward the requested additional 50% coordinate reduction.
