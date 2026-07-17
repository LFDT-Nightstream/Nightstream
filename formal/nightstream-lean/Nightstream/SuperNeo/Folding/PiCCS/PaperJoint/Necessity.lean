import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.NonlinearTerminal
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.DomainSeparation

/-!
Necessity countermodels for the paper-joint `Pi_CCS` model.

Owns: only explicit witnesses showing that selected semantic obligations
cannot be omitted from this model.

Does not own: completeness of the eventual production check set, probability
bounds, Rust bug claims, R1CS row ownership, or permission to remove rows.

Emits constraints: no.

| Necessity family | Invalid weakening witnessed |
|---|---|
| nonlinear terminal | interpolating residual values is not the same as evaluating the nonlinear paper polynomial off-cube |
| coefficient connectivity | unbound coefficient matrices can change acceptance while visible sources stay fixed |
| padded carrier | projecting away the completed suffix can hide a changed coefficient image |
| domain separation | one square Boolean domain cannot represent both row and complete Phi81 carrier indices |
-/
