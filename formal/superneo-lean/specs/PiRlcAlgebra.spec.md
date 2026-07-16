# Fixed Pi_RLC algebra contract

This component owns the concrete algebraic obligations enforced after the
fifteen transcript-derived rho polynomials have been sampled. It does not own
the sampler, Pi_CCS output reconstruction, or Pi_DEC validation.

| Protocol phase | Constraint family | Fixed shape | Mathematical obligation | Lean file |
|---|---|---:|---|---|
| Semantics | Ring action | degree 54 | multiplication in `F[X] / (X^54 + X^27 + 1)` | `Semantics/RingAction.lean` |
| Semantics | Combination | 15 inputs | exact parent sum of ring actions | `Semantics/Combination.lean` |
| Semantics | Projection-binding shape | 15 outputs; each 54-coefficient `y_ring`/`y_zcol` projection is paired with one 64-entry carrier and ten zero tail entries | exact family arities, active-prefix equality, and padding | `Semantics/ProjectionBindingShape.lean` |
| Claims | Commitment | 18 lanes | `c = sum_i rho_i * c_i` | `Claims/Commitment.lean` |
| Claims | Advice | 3 coordinates by 18 lanes | present-case `ops`, `is`, and `fs` combinations | `Claims/Adv.lean` |
| Claims | Public X | 5 active columns in a caller-sized full carrier | active combination plus zero inactive input/parent columns | `Claims/X.lean` |
| Claims | `y_ring` | 3 rows by 2 limbs | padded limbwise ring combination | `Claims/YRing.lean` |
| Claims | `y_zcol` | 1 vector by 2 limbs | padded limbwise ring combination | `Claims/YZcol.lean` |
| Claims | Padding | lanes 54 through 63 | canonical zero outside the active action | `Claims/Padding.lean` |
| Authority | Common values | `s_col`, fold digest | inputs share one value and the parent is bound once | `Authority/Consistency.lean` |
| Diagnostic | Child/next-running `y_zcol` no-read model | one paper `CE.Statement` plus two legacy sidecars | projection and no-read extensionality only; no protocol erasure permission | `Authority/ChildYZcolElision.lean` |
| Refinement | Exact materialization | 15 products | exact product columns substitute into the parent sum | `Refinement/ExactMaterialization.lean` |
| Refinement | Scalar product sums | finite products/groups | generic substitution and carry telescoping | `Refinement/ProductSum.lean` |
| Refinement | Projection-binding serialization | plain fixed profile | exact version-one field framing and 3,616-field SIS preimage | `Refinement/ProjectionBindingSerialization.lean` |

The paper-level refinement compares two encodings of one exact 15-input ring
combination:

```text
exists products,
  products_i = ringAction(rho_i, input_i) and
  parent = sum_i products_i

iff

parent = sum_i ringAction(rho_i, input_i)
```

The child/next-running no-read model uses the paper's `CE.Statement` directly.
A legacy carrier may repeat one `y_zcol` sidecar in its child and next-running
encodings, but projection simply discards both values. Predicates are invariant
under sidecar mutation only when they were defined to factor through that
projection. This is diagnostic, not a semantic argument for erasure: the
optimized implementation's raw projection is linear across Pi_RLC/Pi_DEC and
must be closed by the delayed authority transition.

This exact equation is not the deterministic semantics of the production
one-point projection. Accepted production rows imply coefficient equality or
a named bad-root event; the probability reduction additionally requires the
identity and quotient advice to be bound before transcript-derived beta.

## Refinement Layers

The model layer exposes these theorem obligations:

| Obligation | Lean result |
|---|---|
| Mixed SSA | `MixedSsaExecution` admits both product and linear fresh-column definitions with explicit topological references |
| Fresh geometry | `decodeMixedSsaFrom_length` appends exactly one reconstructed value per instruction |
| Executable decoder | `mixedSsaExecution_iff_eq_decoder` proves source satisfaction exactly when the output equals `decodeMixedSsaFrom` |
| Unique reconstruction | `mixedSsaExecution_unique` proves all fresh columns are uniquely determined by authoritative inputs |
| Retained boundary | `RetainedMatrixFullColumnRank` states injectivity of the exact retained-coefficient matrix; `retainedValues_unique_of_fullColumnRank` derives uniqueness |
| Arity bound | `BoundedProductGroup` requires every nonempty group to contain at most `maxProductTerms = 18` products |
| Carry rows | `boundedCarryEncoding_iff_direct` proves the bounded carry chain equivalent to the unsplit ordered product sum |

A production `ProductSumBatchTrace` bridge is valid only if it establishes:

1. Every parsed Rust product or linear row instantiates the corresponding
   `SsaInstruction`, with the same external/prior column partition.
2. The symbolic identities used by the lowering denote the decoded mixed SSA
   program, including constants and retained linear terms.
3. The concrete retained-coefficient matrix satisfies
   `RetainedMatrixFullColumnRank`; defining the condition is not a proof for
   the production matrix.
4. No removed temporary escapes the validated row/column interval or appears
   as an external identity input.
5. Rust `chunks(18)` produces the ordered nonempty `BoundedProductGroup`s used
   by the carry theorem.
6. The Rust decoder and Lean `decodeMixedSsaFrom` reconstruct the same columns
   in the same topological order.

The production exact-or-bad-root reduction and concrete transcript binding of
the identities, quotient advice, and beta are separate required conformance
layers.

## Fixed projection-binding profile

The plain fixed profile has fifteen Pi_CCS outputs, eighteen commitment lanes,
no advice material, five active X lanes, six `y_ring` limb lanes, and two
`y_zcol` limb lanes. Each serialized combined polynomial is exactly the first
54 entries of its paired carrier, and each division quotient has 53
coefficients. Every paired `y_ring` and `y_zcol` carrier has 64 entries, with
the ten entries at lanes 54 through 63 fixed to zero and excluded from the
active serialization.

With the exact version-one label framing, this profile's projection-binding
SIS preimage has 3,616 field elements.

The 6,889-field figure below is a counterfactual diagnostic only: it adds all
three advice coordinates while retaining the plain profile's five X lanes.
Nebula extends the public input, so its active X-family arity must be derived
from its own `m_in`; a Nebula profile must not reuse the 6,889-field figure.

| Serialized family | Plain fields |
|---|---:|
| Domain | 8 |
| Combined commitment | 978 |
| Commitment quotients | 1,062 |
| X combined/quotient pairs | 600 |
| `y_ring` limb pairs | 726 |
| `y_zcol` limb pairs | 242 |
| Plain total | 3,616 |
| Counterfactual same-X advice leaves, added | 33 |
| Counterfactual same-X advice quotients, added | 3,240 |
| Counterfactual same-X total | 6,889 |

A fixed-F-prime implementation must provide a generated conformance artifact
showing that its native verifier, circuit emitter, and concrete serializer
instantiate this profile. Full-history dimensions are not evidence for these
family counts.

The X refinement surface includes `InactiveXZero` and its coefficientwise
equivalence. A production bridge is valid only if it proves that Rust's
row-major `D * m_in` matrices, `superneo_public_x_cols(m_in)`, and emitted
`padding.x` rows instantiate that model; an ownership-map entry is not a
conformance proof.
