import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteBoolean
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ResidualSemantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.JointComposition
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.VerifierSemantics
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity

/-!
Paper-anchored algebraic skeleton for SuperNeo `Pi_CCS` (Section 7.3 and
Appendix D.4).

Owns: the paper-level one-joint algebra and finite Boolean domains;
source-derived CCS, norm, and carried residuals; the explicit `Q` and
`T_abs - sum_x Q` identity; the finite SumCheck semantic reduction; the typed
abstract transcript and output interface; and the Phi81 layouts, coefficient
kernel, and necessity countermodels listed below.

Does not own: production data and layout refinement, the concrete modulus
Euclid proof, approval of the candidate target correction, equivalence with
the production two-SumCheck FE/NC protocol, executable Poseidon2 transcript
security, concrete output projection, Rust/R1CS refinement, or constraint
counts.

Emits constraints: no.

Authority boundary: this module is derived from the paper, not from the
existing circuit. It cannot authorize a production constraint removal until
the production data/layout refinement, SplitNc refinement, and exact
Rust/R1CS refinement boundaries are separately closed.

Open obligations are recorded in the child contract headers and
`specs/fpr-nifs-bridge.md`. This facade intentionally exports no editable
status datatype: changing a diagnostic list is not a proof.

| Child group | Stable mathematical ownership | Excluded boundary |
|---|---|---|
| `FiniteBoolean` | domains, equality weights, interpolation, reproduction, and finite sums | protocol policy |
| `ResidualSemantics` | independent CCS, strict norm, and carried-evaluation tables | transcript execution |
| `JointComposition` | target convention, signed `Q` identity, coefficients, and semantic closure | approval of the paper correction |
| `VerifierSemantics` | SumCheck truth path, output point, abstract Fiat--Shamir, and protocol checker | concrete Poseidon2 security |
| `Phi81` | Goldilocks/Phi81 placement, coefficient kernel, and source connectivity | Rust/R1CS refinement |
| `Necessity` | countermodels for nonlinear terminals, carrier retention, coefficient binding, domain separation, and unchecked SumCheck coefficients above the paper degree | production bug claims |
-/
