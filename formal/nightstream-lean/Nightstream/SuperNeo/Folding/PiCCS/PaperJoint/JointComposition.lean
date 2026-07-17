import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Coefficients
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetConvention
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientObject
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement

/-!
Signed joint-polynomial composition for paper `Pi_CCS`.

Owns: only construction of the single paper joint object from independent
CCS, norm, and carried-evaluation sources; its target convention; coefficient
serialization; and the exact signed identity used by SumCheck.

Does not own: acceptance of a transcript, approval of the candidate paper
target correction, concrete Poseidon2, Phi81 packing, or implementation
refinement.

Emits constraints: no.

| Composition family | Mathematical obligation |
|---|---|
| target and coefficients | the chosen exponent convention and constant-first serialization are explicit |
| signed identity | `T_abs - sum_x Q` equals the signed residual combination |
| source connectivity | one authoritative source family constructs every joint input |
| protocol polynomial | the nonlinear off-cube polynomial agrees with residual truth on Boolean points |
-/
