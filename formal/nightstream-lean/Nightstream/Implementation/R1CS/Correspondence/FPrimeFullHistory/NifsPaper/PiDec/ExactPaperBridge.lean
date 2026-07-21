import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDec.PublicInputBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiDecEvaluationBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PointBridge
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra

/-!
Conditional exact-paper refinement of the decoded strict `PiDEC` carrier.

Assurance tier: model-level representation refinement. The exported theorem
starts from `PiDecStrictCompiler.Accepted`; it does not derive that predicate
from generated R1CS rows and is not Rust-conformant.

Owns: typed decoding of the strict parent and child message fields; exact
commitment and evaluation recomposition from strict acceptance; and promotion
to `PiDEC.PaperVerifier.OutputAccepted` once the two facts absent from the
strict carrier are supplied explicitly.

Does not own: R1CS-row soundness, a semantic source or private opening,
canonical child-X enforcement, active-relation matrix-count alignment,
transcript authority, Ajtai binding, costs, or row removal.

Emits constraints: no.

Authority boundary: the current strict verifier checks only weighted
recomposition of child X values. `PaperPremises.canonicalChildPublicInput`
therefore remains a premise: it says each decoded child X is the exact public
digit computed from the parent by the paper verifier. The current decoded
evaluation carrier has three rows, while a semantic profile owns its matrix
count; `PaperPremises.parentEvaluationSize` and
`childEvaluationSize` expose that alignment instead of silently truncating or
padding.

| Stage path | Fact | Source | Permits row removal? |
|---|---|---|---|
| `nifs.pi_dec.paper.decode.commitment` | strict flat commitment fields decode as 18 typed Phi81 rings | computed | no |
| `nifs.pi_dec.paper.decode.public_input` | all 270 X fields use the proved lane-major transpose | existing typed bridge | no |
| `nifs.pi_dec.paper.decode.evaluations` | active limbs decode to typed Phi81 evaluations | existing typed bridge | no |
| `nifs.pi_dec.paper.commitment` | typed parent commitment is the radix recomposition | strict acceptance + derived transport | no |
| `nifs.pi_dec.paper.evaluations` | typed parent evaluations are the radix recomposition | strict acceptance + exact arity | no |
| `nifs.pi_dec.paper.public_split` | each physical child X equals the verifier-computed digit | explicit missing premise | no |
| `nifs.pi_dec.paper.acceptance` | decoded output satisfies exact operational paper acceptance | derived, conditional | no |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

namespace PublicInputBridge

export Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.PublicInputBridge
  (decodedPublicInput)

end PublicInputBridge

namespace EvaluationBridge

export Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDecEvaluationBridge
  (combineEvaluations_eq_of_size)

end EvaluationBridge

namespace PointBridge

export Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PointBridge
  (pointOfLength)

end PointBridge

/-- Interpret the exact 18-by-54 strict commitment payload as the independent
typed Phi81 commitment carrier. The production layout facts below ensure that
the total `getD` operation is never used out of range in the refinement
theorem. -/
def decodedCommitment
    (assignment : Nat -> Nat) (claim : ClaimLayout) :
    PiRLCAlgebra.Commitment.Value productionProfile.commitmentWidth :=
  fun row coefficient =>
    residue (assignment (claim.commitment.dataCols.getD
      (row.val * ringDegree + coefficient.val) 0))

/-- Dimension-checked interpretation of the verifier-owned parent point. -/
def decodedParentPoint
    (dimensions : Dimensions) (assignment : Nat -> Nat)
    (pointDimension :
      layout.parent.rCols.length = dimensions.shape.rowVariables) :
    Phi81Relation.Point dimensions.shape :=
  PointBridge.pointOfLength dimensions.shape assignment
    { r := layout.parent.rCols } pointDimension

/-- Typed parent reconstructed from the strict carrier. -/
def decodedParent
    (dimensions : Dimensions)
    (system : Phi81Relation.Structure dimensions.shape)
    (assignment : Nat -> Nat)
    (pointDimension :
      layout.parent.rCols.length = dimensions.shape.rowVariables) :
    CE.Instance
      (Phi81Relation.Structure dimensions.shape)
      (Phi81Relation.PublicInput dimensions.shape)
      (Phi81Relation.Point dimensions.shape)
      Phi81Relation.Evaluation
      (PiRLCAlgebra.Commitment.Value productionProfile.commitmentWidth) where
  constraintSystem := system
  commitment := decodedCommitment assignment layout.parent
  publicInput := PublicInputBridge.decodedPublicInput dimensions assignment
    layout.parent
  point := decodedParentPoint dimensions assignment pointDimension
  evaluations := decodedEvaluations assignment layout.parent
  stage := .combined

/-- Full typed child family reconstructed from strict child commitment,
public-input, and evaluation fields. Structure, point, and fresh stage are
verifier-computed exactly as in the paper operational verifier. -/
def decodedOutput
    (dimensions : Dimensions)
    (system : Phi81Relation.Structure dimensions.shape)
    (assignment : Nat -> Nat)
    (pointDimension :
      layout.parent.rCols.length = dimensions.shape.rowVariables) :
    Fin productionGlobalParams.k ->
      CE.Instance
        (Phi81Relation.Structure dimensions.shape)
        (Phi81Relation.PublicInput dimensions.shape)
        (Phi81Relation.Point dimensions.shape)
        Phi81Relation.Evaluation
        (PiRLCAlgebra.Commitment.Value productionProfile.commitmentWidth) :=
  fun child => {
    constraintSystem := system
    commitment := decodedCommitment assignment (childLayout child)
    publicInput := PublicInputBridge.decodedPublicInput dimensions assignment
      (childLayout child)
    point := decodedParentPoint dimensions assignment pointDimension
    evaluations := decodedEvaluations assignment (childLayout child)
    stage := .fresh
  }

/-- Facts not implied by strict weighted recomposition but required by the
exact Section-7.5 operational verifier. The arity fields are representation
facts, not arithmetic acceptance checks. -/
structure PaperPremises
    (dimensions : Dimensions)
    (key : PiRLCAlgebra.Commitment.Key dimensions.shape
      productionProfile.commitmentWidth)
    (assignment : Nat -> Nat) : Prop where
  canonicalChildPublicInput : forall child,
    PublicInputBridge.decodedPublicInput dimensions assignment
        (childLayout child) =
      (PiDECAlgebra.PaperVerifier.publicInputSplit key).split
        (PublicInputBridge.decodedPublicInput dimensions assignment
          layout.parent) child
  parentEvaluationSize :
    (decodedEvaluations assignment layout.parent).size =
      dimensions.shape.matrixCount
  childEvaluationSize : forall child,
    (decodedEvaluations assignment (childLayout child)).size =
      dimensions.shape.matrixCount

/-- The committed legacy strict carrier contains exactly three active
evaluation rows. This is an artifact shape fact, not authority for a semantic
profile's matrix count. -/
theorem decodedParentEvaluationSize (assignment : Nat -> Nat) :
    (decodedEvaluations assignment layout.parent).size = 3 := by
  simp [decodedEvaluations, layout]

/-- Consequently, the conditional exact-arity premise can hold only for a
three-matrix semantic shape. In particular it cannot silently justify a
larger active relation. -/
theorem PaperPremises.matrixCount_eq_three
    {dimensions : Dimensions}
    {key : PiRLCAlgebra.Commitment.Key dimensions.shape
      productionProfile.commitmentWidth}
    {assignment : Nat -> Nat}
    (premises : PaperPremises dimensions key assignment) :
    dimensions.shape.matrixCount = 3 := by
  have size := premises.parentEvaluationSize
  rw [decodedParentEvaluationSize assignment] at size
  exact size.symm

private theorem residue_add (left right : Nat) :
    residue (left + right) = residue left + residue right := by
  apply Fin.ext
  change (left + right) % goldilocksP =
    (left % goldilocksP + right % goldilocksP) % goldilocksP
  exact Nat.add_mod left right goldilocksP

private theorem residue_mul (left right : Nat) :
    residue (left * right) = residue left * residue right := by
  apply Fin.ext
  change (left * right) % goldilocksP =
    (left % goldilocksP * (right % goldilocksP)) % goldilocksP
  exact Nat.mul_mod left right goldilocksP

private theorem residue_mod (value : Nat) :
    residue (value % goldilocksP) = residue value := by
  apply Fin.ext
  simp [residue]

private theorem residue_rawLcEval (assignment : Nat -> Nat) :
    forall terms : List (Nat × Nat),
      residue (rawLcEval assignment terms) =
        terms.foldr (fun term suffix =>
          residue term.2 * residue (assignment term.1) + suffix) 0 := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [rawLcEval, List.foldr_cons]
      rw [residue_add, residue_mul, inductionHypothesis]

private theorem residue_lcEval
    (assignment : Nat -> Nat) (terms : List (Nat × Nat)) :
    residue (lcEval assignment terms) =
      terms.foldr (fun term suffix =>
        residue term.2 * residue (assignment term.1) + suffix) 0 := by
  rw [lcEval_eq_raw_mod, residue_mod]
  exact residue_rawLcEval assignment terms

private theorem recomposesField
    (assignment : Nat -> Nat) (parent : Nat)
    (column : ClaimLayout -> Nat)
    (recomposes : PiDecStrictCompiler.Recomposes assignment parent
      (layout.children.map column)
      (PiDecStrictCompiler.radixPowers layout.radix
        layout.children.length)) :
    residue (assignment parent) =
      combineScalar fun child =>
        residue (assignment (column (childLayout child))) := by
  unfold PiDecStrictCompiler.Recomposes at recomposes
  rw [recomposes, residue_lcEval]
  have range14 : List.range 14 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] := by
    decide
  simp [combineScalar, radixWeights, productionGlobalParams, childLayout,
    layout, PiDecStrictCompiler.radixPowers, range14]

private theorem commitmentLane
    (assignment : Nat -> Nat)
    (lane : Nat)
    (laneLt : lane < layout.parent.commitment.dataCols.length)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    residue (assignment
      (layout.parent.commitment.dataCols.getD lane 0)) =
      combineScalar fun child =>
        residue (assignment
          ((childLayout child).commitment.dataCols.getD lane 0)) := by
  have recomposes := accepted.commitment lane laneLt
  have normalized : PiDecStrictCompiler.Recomposes assignment
      (layout.parent.commitment.dataCols.getD lane 0)
      (layout.children.map fun claim =>
        claim.commitment.dataCols.getD lane 0)
      (PiDecStrictCompiler.radixPowers layout.radix
        layout.children.length) := by
    simpa [List.map_map] using recomposes
  exact recomposesField assignment _
    (fun claim => claim.commitment.dataCols.getD lane 0) normalized

private theorem combineCommitments_apply
    {count verifierRows : Nat}
    (weights : Fin count -> F)
    (items : Fin count -> PiRLCAlgebra.Commitment.Value verifierRows)
    (row : Fin verifierRows) (coefficient : Fin ringDegree) :
    (PiDECAlgebra.Commitment.combineCommitments weights items row) coefficient =
      PiDec.Weights.combineScalars weights
        (fun child => items child row coefficient) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [PiDECAlgebra.Commitment.combineCommitments,
        PiDECAlgebra.Commitment.commitmentScale,
        PiRLCAlgebra.Commitment.commitmentAdd,
        EvaluationHomomorphism.CarrierAction.ringFScale,
        ringFAdd,
        PiDec.Weights.combineScalars]
      rw [inductionHypothesis
        (fun child => weights child.succ)
        (fun child => items child.succ)]

set_option maxRecDepth 524288 in
private theorem parentCommitmentLength :
    layout.parent.commitment.dataCols.length =
      productionProfile.commitmentWidth * ringDegree := by
  decide

/-- Strict commitment recomposition transports to the independent typed
18-ring commitment carrier. -/
theorem strictAccepted_typedCommitmentEquation
    (assignment : Nat -> Nat)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedCommitment assignment layout.parent =
      PiDECAlgebra.Commitment.recomposeCommitment fun child =>
        decodedCommitment assignment (childLayout child) := by
  funext row coefficient
  let lane := row.val * ringDegree + coefficient.val
  have laneLt : lane < layout.parent.commitment.dataCols.length := by
    rw [parentCommitmentLength]
    have rowLt := row.isLt
    have coefficientLt := coefficient.isLt
    simp only [lane, productionProfile, ringDegree] at rowLt coefficientLt |-
    omega
  calc
    decodedCommitment assignment layout.parent row coefficient =
        residue (assignment
          (layout.parent.commitment.dataCols.getD lane 0)) := by
      rfl
    _ = combineScalar fun child =>
          residue (assignment
            ((childLayout child).commitment.dataCols.getD lane 0)) :=
      commitmentLane assignment lane laneLt accepted
    _ = PiDec.Weights.combineScalars
          EvaluationHomomorphism.PiDEC.radixWeight
          (fun child => decodedCommitment assignment
            (childLayout child) row coefficient) := by
      rw [PiDec.Weights.combineScalar_eq]
      rfl
    _ = (PiDECAlgebra.Commitment.recomposeCommitment
          (fun child => decodedCommitment assignment
            (childLayout child)) row) coefficient := by
      symm
      exact combineCommitments_apply
        EvaluationHomomorphism.PiDEC.radixWeight
        (fun child => decodedCommitment assignment (childLayout child))
        row coefficient

private theorem yLane
    (assignment : Nat -> Nat) (row lane : Nat)
    (rowLt : row < layout.parent.yRingCols.length)
    (laneLt : lane < (layout.parent.yRingCols.getD row []).length)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    residue (assignment
      ((layout.parent.yRingCols.getD row []).getD lane 0)) =
      combineScalar fun child =>
        residue (assignment
          (((childLayout child).yRingCols.getD row []).getD lane 0)) := by
  exact recomposesField assignment _
    (fun claim => (claim.yRingCols.getD row []).getD lane 0)
    (accepted.y row lane rowLt laneLt)

private theorem childLayout_mem
    (child : Fin productionGlobalParams.k) :
    childLayout child ∈ layout.children := by
  unfold childLayout
  exact List.get_mem layout.children (Fin.cast production_child_count child)

private theorem childYRowsLength
    (child : Fin productionGlobalParams.k) :
    (childLayout child).yRingCols.length =
      layout.parent.yRingCols.length :=
  (production_public_shape.yShapes
    (childLayout child) (childLayout_mem child)).1

private theorem activeEvaluationLength
    (row : Nat) (rowLt : row < layout.parent.yRingCols.length) :
    2 * ringDegree <=
      (layout.parent.yRingCols.getD row []).length := by
  have member : layout.parent.yRingCols.getD row [] ∈
      layout.parent.yRingCols := by
    rw [← List.getElem_eq_getD
      (l := layout.parent.yRingCols) (i := row) []]
    exact List.getElem_mem rowLt
  have active := production_public_shape.activeEvaluationRows
    layout.parent (List.mem_cons_self) _ member
  simpa [layout, ringDegree] using active

private theorem decodedEvaluationEquation
    (assignment : Nat -> Nat) (row : Nat)
    (rowLt : row < layout.parent.yRingCols.length)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedEvaluation assignment
        (layout.parent.yRingCols.getD row []) =
      combineEvaluation fun child =>
        decodedEvaluation assignment
          ((childLayout child).yRingCols.getD row []) := by
  funext coefficient
  apply k_eq_of_coeffs
  · change residue (assignment
        ((layout.parent.yRingCols.getD row []).getD
          (2 * coefficient.val) 0)) =
      combineScalar fun child => residue (assignment
        (((childLayout child).yRingCols.getD row []).getD
          (2 * coefficient.val) 0))
    apply yLane assignment row (2 * coefficient.val) rowLt _ accepted
    have active := activeEvaluationLength row rowLt
    omega
  · change residue (assignment
        ((layout.parent.yRingCols.getD row []).getD
          (2 * coefficient.val + 1) 0)) =
      combineScalar fun child => residue (assignment
        (((childLayout child).yRingCols.getD row []).getD
          (2 * coefficient.val + 1) 0))
    apply yLane assignment row (2 * coefficient.val + 1) rowLt _ accepted
    have active := activeEvaluationLength row rowLt
    omega

/-- Strict evaluation recomposition transports to the active, unpadded
decoded arrays. This theorem does not assert that their row count matches a
semantic relation shape. -/
theorem strictAccepted_decodedEvaluationsEquation
    (assignment : Nat -> Nat)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedEvaluations assignment layout.parent =
      combineEvaluations fun child =>
        decodedEvaluations assignment (childLayout child) := by
  unfold decodedEvaluations combineEvaluations
  apply congrArg List.toArray
  apply List.ext_get
  · simp [firstIndex, childLayout, productionGlobalParams, layout]
  · intro row parentLt combinedLt
    simp only [List.get_eq_getElem, List.getElem_map, List.getElem_range]
    have rowLt : row < layout.parent.yRingCols.length := by
      simpa using parentLt
    calc
      decodedEvaluation assignment layout.parent.yRingCols[row] =
          decodedEvaluation assignment
            (layout.parent.yRingCols.getD row []) := by
        rw [← List.getElem_eq_getD
          (l := layout.parent.yRingCols) (i := row) []]
      _ = combineEvaluation fun child =>
          decodedEvaluation assignment
            ((childLayout child).yRingCols.getD row []) :=
        decodedEvaluationEquation assignment row rowLt accepted
      _ = combineEvaluation fun child =>
          ((childLayout child).yRingCols.map
            (decodedEvaluation assignment)).toArray.getD
              row ringKZero := by
        apply congrArg combineEvaluation
        funext child
        have childLt : row < (childLayout child).yRingCols.length := by
          rw [childYRowsLength child]
          exact rowLt
        simp [List.getD, childLt]

/-- Exact paper acceptance for the typed decoded carrier, conditional only on
strict semantic acceptance, point-dimension alignment, canonical child-X
identity, and exact semantic evaluation arity.

This is not a full row-to-paper theorem: generated rows to
`PiDecStrictCompiler.Accepted`, canonical child-X enforcement, and active
matrix-count alignment remain separate refinement edges. -/
theorem strictAccepted_refines_outputAccepted
    (dimensions : Dimensions)
    (key : PiRLCAlgebra.Commitment.Key dimensions.shape
      productionProfile.commitmentWidth)
    (system : Phi81Relation.Structure dimensions.shape)
    (assignment : Nat -> Nat)
    (pointDimension :
      layout.parent.rCols.length = dimensions.shape.rowVariables)
    (accepted : PiDecStrictCompiler.Accepted layout assignment)
    (premises : PaperPremises dimensions key assignment) :
    PiDEC.PaperVerifier.OutputAccepted
      (PiDECAlgebra.Algebra.concrete key)
      (PiDECAlgebra.PaperVerifier.publicInputSplit key)
      (PiDECAlgebra.PaperVerifier.evaluationArity key)
      (decodedParent dimensions system assignment pointDimension)
      (decodedOutput dimensions system assignment pointDimension) := by
  have commitmentEquation :=
    strictAccepted_typedCommitmentEquation assignment accepted
  have rawEvaluationEquation :=
    strictAccepted_decodedEvaluationsEquation assignment accepted
  have evaluationEquation :
      (decodedParent dimensions system assignment pointDimension).evaluations =
        (PiDECAlgebra.Algebra.concrete key).recomposeEvaluations
          (fun child =>
            (decodedOutput dimensions system assignment pointDimension child).evaluations) := by
    calc
      (decodedParent dimensions system assignment pointDimension).evaluations =
          combineEvaluations fun child =>
            decodedEvaluations assignment (childLayout child) :=
        rawEvaluationEquation
      _ = EvaluationHomomorphism.PiDEC.recomposeEvaluations
          (shape := dimensions.shape)
          (fun child => decodedEvaluations assignment (childLayout child)) := by
        exact EvaluationBridge.combineEvaluations_eq_of_size _
          premises.childEvaluationSize
      _ = (PiDECAlgebra.Algebra.concrete key).recomposeEvaluations
          (fun child =>
            (decodedOutput dimensions system assignment pointDimension child).evaluations) := by
        rfl
  refine {
    outputComputed := ?_
    checks := {
      parentCombined := rfl
      parentEvaluationSize := ?_
      messageEvaluationSize := ?_
      commitmentEquation := ?_
      evaluationEquation := ?_
    }
  }
  · funext child
    simp only [PiDEC.PaperVerifier.children,
      PiDEC.PaperVerifier.attemptForOutput,
      PiDEC.PaperVerifier.messagesOf, decodedParent, decodedOutput]
    rw [premises.canonicalChildPublicInput child]
  · exact premises.parentEvaluationSize
  · exact premises.childEvaluationSize
  · exact commitmentEquation
  · exact evaluationEquation

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec.ExactPaperBridge
