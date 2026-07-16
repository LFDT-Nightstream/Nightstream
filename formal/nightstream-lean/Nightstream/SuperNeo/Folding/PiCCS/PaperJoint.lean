import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Coefficients
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanHypercubeSum
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetConvention
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientObject
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81Evaluation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.NonlinearTerminal
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedProtocolVerifier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.DomainSeparation
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TableResiduals
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Sampling
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.OutputPoint

/-!
Paper-anchored algebraic skeleton for SuperNeo `Pi_CCS` (Section 7.3 and
Appendix D.4).

Owns: a candidate one-joint coefficient-block skeleton, a canonical finite
Boolean-table coefficient transform and table-level zero residualization,
one shared typed Boolean-domain order, independently derived paper-level CCS
matrix/polynomial residual tables, independently derived strict-norm residual
tables for all paper-model sources, and the exact base-field residual for the
production strict `b = 2` norm, plus an explicit equality-weighted hypercube
sum theorem for the canonical MLE and independently derived carried
claimed-minus-matrix-evaluation residuals,
an explicit pointwise `F`/`NC`/`Eval`/`Q` construction and exact signed
`T_abs - sum_x Q` identity over extension-carrier tables,
an exact signed constant-first gamma coefficient serialization whose Horner
evaluation equals that identity, one explicit residual-table MLE used only as
an algebraic audit path, a distinct off-cube protocol polynomial that evaluates
underlying images before applying nonlinear formulas, verifier-owned finite
SumCheck initial and output-message terminal binding with a
deterministic table-truth/signed-mixing-root/round-collision dichotomy, exact
alpha/gamma specialization of the independent unsampled coefficient object,
construction of the sole joint object from independent CCS, norm, and carried
mathematical inputs with exact semantic-truth equivalence, one authoritative
`K+k` assignment family whose CCS, norm, and carried views are derived and
proven connected,
one authoritative field-matrix family whose carried coefficient matrices are
derived through a named bilinear kernel, together with the conditional
constant-term connection required by the paper embedding,
the exact partial logical-column to 54-lane block layout, plus a distinct
complete carrier domain where fresh CCS data is zero-extended but folded CE
data may use every completed coordinate,
the concrete Phi81 coefficient kernel derived from the closed-form bar
transform and actual cyclotomic multiplication, with a kernel-checked finite
basis proof of the paper constant-term law,
an explicit countermodel proving that the still-independent carried
coefficient matrices are a security-relevant connectivity hole,
an arithmetic impossibility proof showing that the square paper row/column
bijection cannot index a complete 54-lane Phi81 carrier,
an abstract typed Fiat--Shamir schedule in which alpha, gamma, and every
SumCheck challenge are verifier-derived and absent from the certificate, an
actual protocol verifier whose terminal uses the nonlinear paper polynomial
and whose outgoing transcript absorbs the typed output message,
and binding output points to one joint SumCheck challenge vector.

Does not own: external Boolean-leaf/bit-order alignment, concrete
relation refinement for the CCS tables, extension-field placement of CCS
residuals, production assignment/order refinement for the norm tables,
production matrix-cache/layout, fresh-to-carrier completion, mixed-CE carrier
preservation, and base-to-extension refinement for carried
evaluations, proof that Rust's runtime Gram inversion and matrix cache realize
the closed-form Phi81 transform, a kernel proof of the
concrete modulus Euclid property,
extension-field placement of the norm residual, approval of the corrected
target convention, production construction/refinement of the joint inputs and
protocol image tables, the semantic degree bound, SumCheck probability bounds,
projection of typed output messages from concrete CE instances,
root-counting probability, concrete transcript encoding and Fiat--Shamir security,
output-projection sufficiency, the production two-SumCheck FE/NC protocol,
Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: this module is derived from the paper, not from the
existing circuit. It cannot authorize a production constraint removal until
the production data/layout refinement, SplitNc refinement, and exact
Rust/R1CS refinement boundaries are separately closed.

| Protocol | Phase | Constraint family | Mathematical result |
|---|---|---|---|
| `Pi_CCS` | table transform | CCS / norm / carried evaluation | coefficient-zero iff leaf-zero and canonical-polynomial evaluation equals independent recursive MLE |
| `Pi_CCS` | hypercube expansion | `eq(x,r)` / canonical MLE | recursive evaluation equals the explicit sum of `eq(x,r) * table[x]` |
| `Pi_CCS` | CCS residual construction | matrices / sparse `f` / fresh sources | explicit row residuals; table zero iff every independently derived CCS row is zero |
| `Pi_CCS` | strict norm semantics | base-field norm residual | `(z+1)z(z-1)=0` iff centered `|z|<2`, conditional only on no zero divisors |
| `Pi_CCS` | norm residual construction | all `K+k` typed source assignments | canonical cubic tables; table zero iff semantic `normBounded 2` |
| `Pi_CCS` | carried residual construction | running / matrix / coefficient | claimed minus explicit equality-weighted matrix-image MLE; zero iff every evaluation claim holds |
| `Pi_CCS` | carried target audit | local versus shifted exponent layouts | exact target-shift identity and exponent-zero support mismatch witness |
| `Pi_CCS` | pre-SumCheck algebra | signed block composition | explicit pointwise `Q`; exact `T_abs - sum_x Q = -CCS - norm + carried` identity |
| `Pi_CCS` | gamma serialization | signed CCS / norm / carried blocks | exact constant-first positions and Horner equality with `T_abs - sum_x Q` |
| `Pi_CCS` | signed coefficient object | finite alpha polynomials / carried scalars | coefficient truth iff explicit CCS/norm/carried table obligations; exact specialization into Horner coefficients |
| `Pi_CCS` | residual-table audit path | canonical MLE | agrees with residual leaves but is not treated as the nonlinear off-cube paper polynomial |
| `Pi_CCS` | off-cube protocol polynomial | image MLEs / nonlinear CCS and norm / carried evaluations | actual `Q(r')` agrees with the signed residual object exactly on Boolean vertices |
| `Pi_CCS` | off-cube terminal necessity | nonlinear construction order | a concrete finite-field countermodel separates residual-table MLE from the paper polynomial away from the Boolean cube |
| `Pi_CCS` | output terminal | typed prover evaluations at verifier point | terminal is message-derived; mismatch with actual `Q(r')` is a named event |
| `Pi_CCS` | SumCheck initial | verifier target / semantic hypercube sum | executable acceptance yields table truth, signed mixing root, or named round collision without a caller-supplied expected callback or honesty premise |
| `Pi_CCS` | joint semantic closure | CCS / norm / carried source data | the sole constructed joint coefficient truth is equivalent to the three independently defined semantic families |
| paper `Pi_CCS` | source connectivity | one `K+k` assignment family / square Boolean-column bijection | CCS, norm, and carried views provably read the same `z_i` values and norm covers every paper-model column |
| `Pi_CCS` | coefficient source connectivity | sole field matrix / block layout / bilinear kernel | every carried coefficient matrix is derived from `M`; the constant coefficient agrees with the CCS image under the explicit paper kernel law |
| coefficient embedding | logical column layout | flat column / 54-lane block / padding hole | exact quotient-remainder round trips at the original CCS width |
| coefficient embedding | complete carrier layout | original width / completed width / carried suffix | fresh assignment and matrix suffixes are zero-derived, while every completed coordinate remains available to folded CE |
| coefficient embedding | concrete Phi81 kernel | closed-form bar basis / cyclotomic product | the constant coefficient is the Kronecker inner product, checked by the Lean kernel over all 54-by-54 basis pairs |
| `Pi_CCS` | coefficient connectivity necessity | CCS matrices / carried coefficient matrices | identical non-coefficient sources can flip semantic truth when the coefficient view is unbound |
| coefficient embedding | carrier-retention necessity | original-width projection / completed tail / nonconstant image | two carried assignments with the same original projection produce different derived coefficient images |
| production shape | row/column separation necessity | Boolean row cube / complete Phi81 carrier | no square `ColumnLayout` can index a carrier whose width is a multiple of 54 |
| `Pi_CCS` | Fiat--Shamir authority | context / alpha / gamma / SumCheck rounds | certificate carries only finite messages; all challenges are deterministically verifier-derived in paper order |
| `Pi_CCS` | transcript-bound protocol verifier | nonlinear terminal / output message / outgoing state | actual polynomial is checked at the derived point and output values are absorbed before transcript handoff |
| `Pi_CCS` | SumCheck output | challenge point `r'` | `BoundOutputs.outputPoint_eq_roundChallenges` |
| production refinement | FE/NC split | not modeled here | explicitly open |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Boundaries intentionally left open by this paper-level slice. -/
inductive OpenBoundary where
  | booleanLeafOrderAlignment
  | ccsConcreteRelationRefinement
  | ccsResidualExtensionPlacement
  | normConcreteAssignmentRefinement
  | goldilocksModulusEuclid
  | normResidualExtensionPlacement
  | carrierCompletionRefinement
  | rowColumnDomainSeparation
  | carriedCoefficientMatrixRefinement
  | carriedBaseExtensionLift
  | targetConventionReview
  | protocolImageConstructionRefinement
  | semanticRoundDegreeBound
  | outputMessageCeProjection
  | outputProjectionSufficiency
  | mixingRootProbability
  | concreteTranscriptEncoding
  | fiatShamirSecurity
  | splitNcRefinement
  | rustR1csRefinement
deriving Repr, DecidableEq

/-- Explicit census preventing the finite deterministic slice from being
mistaken for end-to-end protocol or implementation assurance. -/
def openBoundaries : List OpenBoundary :=
  [.booleanLeafOrderAlignment, .ccsConcreteRelationRefinement,
    .ccsResidualExtensionPlacement,
    .normConcreteAssignmentRefinement, .goldilocksModulusEuclid,
    .normResidualExtensionPlacement, .carrierCompletionRefinement,
    .carriedCoefficientMatrixRefinement, .carriedBaseExtensionLift,
    .targetConventionReview,
    .protocolImageConstructionRefinement,
    .semanticRoundDegreeBound,
    .outputMessageCeProjection, .outputProjectionSufficiency,
    .mixingRootProbability, .concreteTranscriptEncoding,
    .fiatShamirSecurity, .splitNcRefinement, .rustR1csRefinement]

/-- Diagnostic status only. It is not an assurance predicate and cannot be
discharged by editing a list. -/
inductive CoverageStatus where
  | incomplete (remaining : List OpenBoundary)
deriving Repr, DecidableEq

/-- Current diagnostic status of this isolated slice. -/
def coverageStatus : CoverageStatus :=
  .incomplete openBoundaries

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
