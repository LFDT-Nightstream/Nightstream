import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Algebra
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Challenge
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Norm
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.PublicInput

/-!
Curated construction surface for the concrete typed Phi81 `PiRLC.Algebra`.

Protocol: SuperNeo `Pi_RLC`.

Owns: the theorem and dependency boundary of every concrete algebra field. Each
child computes its verifier operation independently and proves the
corresponding homomorphism or bound. `Algebra.concrete` assembles those closed
fields without adding an oracle or caller-supplied law.

Does not own: Fiat--Shamir transcript derivation, Rust/R1CS refinement,
constraint accounting, security reductions, or permission to remove rows.
-/
