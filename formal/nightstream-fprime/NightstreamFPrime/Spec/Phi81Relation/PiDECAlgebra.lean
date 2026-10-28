import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Algebra
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix.UniformSignedDigits
import NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.PaperVerifier

/-!
Curated construction surface for the concrete typed Phi81 `PiDEC` algebra and
paper verifier.

Protocol: SuperNeo `Pi_DEC`.

Owns: the dependency boundary and theorem owner of each concrete algebra field,
the exact paper public-input splitter, and the Phi81 matrix-evaluation arity.

Does not own: Ajtai binding or MSIS security, child CE membership, PiCCS or
PiRLC acceptance, NIFS composition, Rust/R1CS refinement, constraint
accounting, security reductions, or permission to remove rows.
-/
