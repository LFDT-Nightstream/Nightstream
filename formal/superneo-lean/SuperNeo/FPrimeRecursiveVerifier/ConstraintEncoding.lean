import SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.BooleanPairRows
import SuperNeo.FPrimeRecursiveVerifier.ConstraintEncoding.ResidualPairFamilies

/-!
Owns: the public import root and ownership map for protocol-neutral constraint
encodings reused across recursive-verifier phases.

Does not own: protocol coordinate selection, generated artifacts, Rust
lowering, matrix materialization, or row-removal decisions.

Emits constraints: no. Child modules specify reusable schedules and prove
their mathematical meaning.

Authority boundary: each protocol consumer must independently prove that its
authoritative coordinate sequence and generated row/matrix artifact
instantiate the selected child encoding.

| Child | Mathematical obligation | Candidate consumers | Concrete bridge owner |
|---|---|---|---|
| `BooleanPairRows` | Pair adjacent Boolean residuals with nonresidue seven and retain an ordinary odd tail | common Boolean membership; Pi_RLC acceptance tree; Pi_RLC Mod-5 quotient | each generated global or leaf-local row/matrix artifact |
| `ResidualPairFamilies` | Specialize the same exact schedule to arbitrary, one-product R1CS, and centered-unit residuals | common one-product rows and centered-field membership | stage/family-local generated row and selector artifact |

Specs: `specs/ConstraintEncoding.spec.md` and
`specs/ResidualPairFamilies.spec.md`.
-/
