import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConstraintPolynomialLift.Evaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TableResiduals

/-!
Independent residual semantics for the paper-joint `Pi_CCS` model.

Owns: only source-derived CCS, strict-norm, and carried-evaluation residuals,
including their Boolean-table and lifted-polynomial meanings.

Does not own: signed joint composition, SumCheck acceptance, transcript
authority, concrete Phi81 packing, or Rust/R1CS refinement.

Emits constraints: no.

Authority boundary: every residual is derived from mathematical matrices,
assignments, or claims. No verifier message or existing circuit defines truth.

| Residual family | Mathematical obligation |
|---|---|
| CCS | independent matrix images satisfy the paper constraint polynomial |
| norm | every authoritative coordinate satisfies the strict centered bound |
| carried evaluation | claimed evaluations equal equality-weighted matrix-image evaluations |
| table/lift bridge | Boolean residual tables and their polynomial lifts agree on the cube |
-/
