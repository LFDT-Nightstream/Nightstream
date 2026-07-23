import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecStrictProductionCompiler
import Nightstream.Implementation.R1CS.Correspondence.Gadgets.PiDecTypedCarrier

/-!
Typed paper bridge for the reduced production strict-`PiDEC` compiler.

Assurance tier: model-level representation refinement.

Owns: transport of the compiler's exact public split through the checked
270-coordinate transpose, transport of strict `sameR` into the typed point,
assembly of `PiDecTypedCarrier.Accepted`, and the resulting exact operational
paper acceptance theorem.

Commitment and semantic-evaluation transport are proved generically from the
strict equations and the typed profile. `TransportPremises` is only an
internal packaging boundary; `transportPremises_of_accepted` derives it.
Nothing in this file is artifact-checked or Rust-conformant.

Does not own: generated rows, Rust/audit identity, commitment binding, private
openings, transcript authority, costs, or row-removal authorization.
-/

namespace Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix
open Nightstream.SuperNeo.Folding
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiDecStrictCompiler

theorem childLayout_eq_profile
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (child : ChildIndex) :
    PiDecStrictProductionCompiler.childLayout layout child =
      profile.childLayout child := by
  unfold PiDecStrictProductionCompiler.childLayout
    PiDecTypedCarrier.Profile.childLayout
  rfl

private theorem parent_xRows_exact
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base) :
    layout.base.parent.xRows = ringDegree := by
  have length := valid.base.activeXLengths layout.base.parent (by simp)
  rw [profile.parentPublicLength, profile.activePublicColumns] at length
  simp only [alignedPublicWidth, publicRingColumns, ringDegree] at length ⊢
  omega

/-- The active typed carrier contains exactly `54 × 5 = 270` logical
public-X coordinates. This is a profile theorem, not a measured artifact
count. -/
theorem active_logicalXCount_270
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base) :
    PiDecStrictProductionCompiler.logicalXCount layout = 270 := by
  unfold PiDecStrictProductionCompiler.logicalXCount
  rw [parent_xRows_exact valid profile, profile.activePublicColumns]
  decide

/-- The chosen isolated lowering costs definitionally seventeen rows for each
of the 270 active coordinates: 270 recompositions plus 4,320 canonicality
rows. -/
theorem active_uniformXRows_count_4590
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base) :
    (PiDecStrictProductionCompiler.uniformXRows layout).length = 4590 := by
  rw [PiDecStrictProductionCompiler.uniformXRows_count,
    active_logicalXCount_270 valid profile]
  decide

private theorem public_coordinate_bounds
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (column : Fin shape.publicWidth) :
    column.val % ringDegree < layout.base.parent.xRows ∧
      column.val / ringDegree < activeColumns layout.base := by
  have columnLt : column.val < alignedPublicWidth := by
    rw [← profile.publicWidth]
    exact column.isLt
  rw [parent_xRows_exact valid profile, profile.activePublicColumns]
  simp only [alignedPublicWidth, ringDegree, publicRingColumns] at columnLt ⊢
  constructor
  · exact Nat.mod_lt _ (by decide)
  · omega

private theorem xColumn_eq_publicSlot
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (claim : ClaimLayout)
    (column : Fin shape.publicWidth)
    (coordinateLt : column.val / ringDegree < activeColumns layout.base) :
    xColumn layout.base claim (column.val % ringDegree)
        (column.val / ringDegree) =
      claim.xActiveCols.getD (PiDecTypedCarrier.publicSlot column) 0 := by
  unfold xColumn
  rw [if_pos coordinateLt, profile.activePublicColumns]
  unfold PiDecTypedCarrier.publicSlot
  rfl

/-- The uniform-X endpoint is exactly the typed paper verifier's public-input
split after the independently checked lane-major transpose. Only the
X-recomposition and canonicality leaves implement this endpoint; no other
strict-`PiDEC` row family contributes. -/
theorem canonicalPublicInput_of_uniformX
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    {assignment : Nat → Nat}
    (accepted : PiDecStrictProductionCompiler.UniformXAccepted layout
      assignment)
    (child : ChildIndex) :
    PiDecTypedCarrier.decodePublicInput profile assignment
        (profile.childLayout child) =
      PiDECAlgebra.PublicInput.splitPublicInput
        (PiDecTypedCarrier.decodePublicInput profile assignment
          layout.base.parent) child := by
  funext column
  have bounds := public_coordinate_bounds valid profile column
  have exactDigits := congrFun
    (accepted.childXExact
      (column.val % ringDegree) (column.val / ringDegree)
      bounds.1 bounds.2) child
  have childColumn := xColumn_eq_publicSlot profile
    (profile.childLayout child) column bounds.2
  have parentColumn := xColumn_eq_publicSlot profile
    layout.base.parent column bounds.2
  change PiDECAlgebra.Radix.fieldOfNat
      (assignment (xColumn layout.base
        (PiDecStrictProductionCompiler.childLayout layout child)
        (column.val % ringDegree) (column.val / ringDegree))) =
    splitScalar
      (PiDECAlgebra.Radix.fieldOfNat
        (assignment (xColumn layout.base layout.base.parent
          (column.val % ringDegree) (column.val / ringDegree)))) child
    at exactDigits
  rw [childLayout_eq_profile profile child] at exactDigits
  rw [childColumn, parentColumn] at exactDigits
  simpa [PiDecTypedCarrier.decodePublicInput,
    PiDecTypedCarrier.decodeField,
    PiDECAlgebra.PublicInput.splitPublicInput] using exactDigits

/-- The complete strict endpoint exposes the same public-input theorem by
projecting to its exact canonical-X obligation. -/
theorem canonicalPublicInput
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    {assignment : Nat → Nat}
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment)
    (child : ChildIndex) :
    PiDecTypedCarrier.decodePublicInput profile assignment
        (profile.childLayout child) =
      PiDECAlgebra.PublicInput.splitPublicInput
        (PiDecTypedCarrier.decodePublicInput profile assignment
          layout.base.parent) child :=
  canonicalPublicInput_of_uniformX valid profile accepted.uniformX child

private theorem k_eq_of_coefficients (left right : K)
    (c0 : left.c0 = right.c0) (c1 : left.c1 = right.c1) : left = right := by
  cases left
  cases right
  cases c0
  cases c1
  rfl

private theorem decodedPairs_eq_of_equalPairs
    (assignment : Nat → Nat) :
    ∀ (parent child : List (Nat × Nat)),
      parent.length = child.length →
      EqualPairs assignment parent child →
      child.map (PiDecTypedCarrier.decodeK assignment) =
        parent.map (PiDecTypedCarrier.decodeK assignment) := by
  intro parent
  induction parent with
  | nil =>
      intro child lengthEq equalPairs
      cases child with
      | nil => rfl
      | cons head tail => simp at lengthEq
  | cons parentHead parentTail inductionHypothesis =>
      intro child lengthEq equalPairs
      cases child with
      | nil => simp at lengthEq
      | cons childHead childTail =>
          have headEq := equalPairs (parentHead, childHead) (by simp)
          have tailEqualPairs :
              EqualPairs assignment parentTail childTail := by
            intro pair pairMember
            exact equalPairs pair (by simp [pairMember])
          have tailLength : parentTail.length = childTail.length := by
            simpa using lengthEq
          simp only [List.map_cons, List.cons.injEq]
          constructor
          · apply k_eq_of_coefficients
            · exact congrArg PiDECAlgebra.Radix.fieldOfNat headEq.1.symm
            · exact congrArg PiDECAlgebra.Radix.fieldOfNat headEq.2.symm
          · exact inductionHypothesis childTail tailLength tailEqualPairs

private theorem typedPoint_eq_of_coordinates
    {shape : Phi81Relation.Shape}
    (left right : Phi81Relation.Point shape)
    (coordinates : left.coordinates = right.coordinates) : left = right := by
  cases left
  cases right
  cases coordinates
  rfl

/-- Strict `sameR` transports to equality of the dimension-checked typed
points. -/
theorem commonPoint
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    {assignment : Nat → Nat}
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment)
    (child : ChildIndex) :
    PiDecTypedCarrier.decodePoint assignment (profile.childLayout child)
        (profile.childPointLength (profile.childLayout child)
          (profile.childLayout_mem child)) =
      PiDecTypedCarrier.decodePoint assignment layout.base.parent
        profile.parentPointLength := by
  apply typedPoint_eq_of_coordinates
  exact decodedPairs_eq_of_equalPairs assignment
    layout.base.parent.rCols (profile.childLayout child).rCols
    (profile.parentPointLength.trans
      (profile.childPointLength (profile.childLayout child)
        (profile.childLayout_mem child)).symm)
    (accepted.legacy.sameR (profile.childLayout child)
      (profile.childLayout_mem child))

/-! ## Active source-R1CS arithmetic -/

/-- For the active 270-public-coordinate, 13-matrix profile with physical
128-limb evaluation rows, the two reductions remove exactly 3,500 source
R1CS rows: `270 * 12 = 3,240` public-digit rows and
`13 * (128 - 108) = 260` padded recomposition rows. This is deliberately not
a selective-CCS or final production constraint count. -/
theorem active_source_rows_saved_3500
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (matrixCount : shape.matrixCount = 13)
    (yShape : PiDecStrictReducedY.UniformParentYWidth layout.base 128) :
    (PiDecStrictCompiler.rows layout.base).length =
      (PiDecStrictProductionCompiler.rows layout).length + 3500 := by
  have logicalCount := active_logicalXCount_270 valid profile
  have evaluationCount : layout.base.parent.yRingCols.length = 13 := by
    rw [profile.parentEvaluationCount, matrixCount]
  have semanticWidth : PiDecStrictReducedY.semanticYWidth layout.base =
      108 := by
    unfold PiDecStrictReducedY.semanticYWidth
    rw [profile.ringDimension, profile.extensionLimbs]
    decide
  have source := PiDecStrictProductionCompiler.combined_source_saving
    valid yShape
  simpa [logicalCount, evaluationCount, semanticWidth,
    PiDecStrictCanonicalX.rowsSavedPerCoordinate] using source

/-! ## Generic commitment and evaluation transport -/

/-- Local scalar fold definitionally matching the production radix fold. -/
private def scalarFold : {count : Nat} →
    (Fin count → F) → (Fin count → F) → F
  | 0, _, _ => 0
  | _ + 1, weights, values =>
      weights 0 * values 0 +
        scalarFold
          (fun index => weights index.succ)
          (fun index => values index.succ)

private theorem recomposeScalar_eq_scalarFold (values : ChildIndex → F) :
    recomposeScalar values =
      scalarFold EvaluationHomomorphism.PiDEC.radixWeight values := by
  rfl

private theorem profileChildLayouts
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base) :
    List.ofFn profile.childLayout = layout.base.children := by
  apply List.ext_get
  · simpa using profile.childCount.symm
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_ofFn]
    unfold PiDecTypedCarrier.Profile.childLayout
    rfl

private theorem recomposesField
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (assignment : Nat → Nat) (parent : Nat)
    (column : ClaimLayout → Nat)
    (recomposes : Recomposes assignment parent
      (layout.base.children.map column)
      (radixPowers layout.base.radix layout.base.children.length)) :
    PiDecTypedCarrier.decodeField assignment parent =
      recomposeScalar fun child =>
        PiDecTypedCarrier.decodeField assignment
          (column (profile.childLayout child)) := by
  let coordinateLayout' : PiDecStrictCanonicalX.Layout := {
    parentColumn := parent
    signColumn := 0
    signOutputColumn := 0
    digitColumns := fun child => column (profile.childLayout child)
  }
  have normalized : Recomposes assignment coordinateLayout'.parentColumn
      (PiDecStrictCanonicalX.childColumns coordinateLayout')
      PiDecStrictCanonicalX.powers := by
    change Recomposes assignment parent
      (List.ofFn fun child : ChildIndex => column (profile.childLayout child))
      PiDecStrictCanonicalX.powers
    rw [show (List.ofFn fun child : ChildIndex =>
        column (profile.childLayout child)) =
        (List.ofFn profile.childLayout).map column by
      simpa only [Function.comp_apply] using
        (List.map_ofFn (f := profile.childLayout) (g := column)).symm]
    rw [profileChildLayouts profile]
    simpa [PiDecStrictCanonicalX.powers, profile.radixTwo,
      profile.childCount] using recomposes
  have decoded :=
    PiDecStrictCanonicalX.decodedRecomposition_of_recomposes normalized
  simpa [coordinateLayout', PiDecStrictCanonicalX.decodedParent,
    PiDecStrictCanonicalX.decodedDigits,
    PiDecTypedCarrier.decodeField] using decoded.symm

private theorem combineCommitments_apply
    {count verifierRows : Nat}
    (weights : Fin count → F)
    (items : Fin count → PiRLCAlgebra.Commitment.Value verifierRows)
    (row : Fin verifierRows) (coefficient : Fin ringDegree) :
    (PiDECAlgebra.Commitment.combineCommitments weights items row) coefficient =
      scalarFold weights (fun child => items child row coefficient) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [PiDECAlgebra.Commitment.combineCommitments,
        PiDECAlgebra.Commitment.commitmentScale,
        PiRLCAlgebra.Commitment.commitmentAdd,
        CarrierAction.ringFScale, ringFAdd, scalarFold]
      rw [inductionHypothesis
        (fun child => weights child.succ)
        (fun child => items child.succ)]

/-- Legacy strict commitment recomposition transports to the exact typed
commitment carrier for any checked profile width. -/
theorem commitmentEquation
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment) :
    PiDecTypedCarrier.decodeCommitment (verifierRows := verifierRows)
        assignment layout.base.parent =
      PiDECAlgebra.Commitment.recomposeCommitment
        (verifierRows := verifierRows) fun child =>
        PiDecTypedCarrier.decodeCommitment (verifierRows := verifierRows) assignment
          (profile.childLayout child) := by
  funext row coefficient
  let lane := row.val * ringDegree + coefficient.val
  have laneLt : lane < layout.base.parent.commitment.dataCols.length := by
    rw [profile.parentCommitmentLength]
    have rowLt := row.isLt
    have coefficientLt := coefficient.isLt
    simp only [lane, ringDegree] at rowLt coefficientLt ⊢
    omega
  have source := accepted.legacy.commitment lane laneLt
  have normalized : Recomposes assignment
      (layout.base.parent.commitment.dataCols.getD lane 0)
      (layout.base.children.map fun claim =>
        claim.commitment.dataCols.getD lane 0)
      (radixPowers layout.base.radix layout.base.children.length) := by
    simpa [List.map_map] using source
  calc
    PiDecTypedCarrier.decodeCommitment assignment layout.base.parent
          row coefficient =
        PiDecTypedCarrier.decodeField assignment
          (layout.base.parent.commitment.dataCols.getD lane 0) := by rfl
    _ = recomposeScalar fun child =>
          PiDecTypedCarrier.decodeField assignment
            ((profile.childLayout child).commitment.dataCols.getD lane 0) :=
      recomposesField profile assignment _
        (fun claim => claim.commitment.dataCols.getD lane 0) normalized
    _ = scalarFold EvaluationHomomorphism.PiDEC.radixWeight
          (fun child =>
            PiDecTypedCarrier.decodeCommitment assignment
              (profile.childLayout child) row coefficient) := by
      rw [recomposeScalar_eq_scalarFold]
      rfl
    _ = (PiDECAlgebra.Commitment.recomposeCommitment
          (fun child => PiDecTypedCarrier.decodeCommitment assignment
            (profile.childLayout child)) row) coefficient := by
      symm
      exact combineCommitments_apply
        EvaluationHomomorphism.PiDEC.radixWeight
        (fun child => PiDecTypedCarrier.decodeCommitment assignment
          (profile.childLayout child)) row coefficient

private theorem combineEvaluations_c0
    {count : Nat} (weights : Fin count → F)
    (items : Fin count → Phi81Relation.Evaluation)
    (coefficient : Fin ringDegree) :
    (BaseLinear.combineEvaluations weights items coefficient).c0 =
      scalarFold weights (fun child => (items child coefficient).c0) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations, BaseLinear.evaluationAdd,
        BaseLinear.evaluationScale, K.add, K.mul, K.embed, scalarFold]
      rw [inductionHypothesis
        (fun child => weights child.succ)
        (fun child => items child.succ)]
      rw [Fin.mul_zero, Fin.zero_mul, Fin.add_zero]

private theorem combineEvaluations_c1
    {count : Nat} (weights : Fin count → F)
    (items : Fin count → Phi81Relation.Evaluation)
    (coefficient : Fin ringDegree) :
    (BaseLinear.combineEvaluations weights items coefficient).c1 =
      scalarFold weights (fun child => (items child coefficient).c1) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [BaseLinear.combineEvaluations, BaseLinear.evaluationAdd,
        BaseLinear.evaluationScale, K.add, K.mul, K.embed, scalarFold]
      rw [inductionHypothesis
        (fun child => weights child.succ)
        (fun child => items child.succ)]
      rw [Fin.zero_mul, Fin.add_zero]

private theorem decodedEvaluationEquation
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment)
    (row : Nat) (rowLt : row < layout.base.parent.yRingCols.length) :
    PiDecTypedCarrier.decodeEvaluation assignment
        (layout.base.parent.yRingCols.getD row []) =
      BaseLinear.combineEvaluations
        EvaluationHomomorphism.PiDEC.radixWeight fun child =>
          PiDecTypedCarrier.decodeEvaluation assignment
            ((profile.childLayout child).yRingCols.getD row []) := by
  funext coefficient
  apply k_eq_of_coefficients
  · change PiDecTypedCarrier.decodeField assignment
        ((layout.base.parent.yRingCols.getD row []).getD
          (2 * coefficient.val) 0) = _
    have laneLt := profile.parentEvaluationWidth row rowLt
    have coefficientLt := coefficient.isLt
    have source := accepted.legacy.y row (2 * coefficient.val) rowLt (by
      simp only [ringDegree] at coefficientLt laneLt ⊢
      omega)
    calc
      PiDecTypedCarrier.decodeField assignment
          ((layout.base.parent.yRingCols.getD row []).getD
            (2 * coefficient.val) 0) =
          recomposeScalar fun child =>
            PiDecTypedCarrier.decodeField assignment
              (((profile.childLayout child).yRingCols.getD row []).getD
                (2 * coefficient.val) 0) :=
        recomposesField profile assignment _
          (fun claim => (claim.yRingCols.getD row []).getD
            (2 * coefficient.val) 0) source
      _ = scalarFold EvaluationHomomorphism.PiDEC.radixWeight
          (fun child =>
            (PiDecTypedCarrier.decodeEvaluation assignment
              ((profile.childLayout child).yRingCols.getD row [])
                coefficient).c0) := by
        rw [recomposeScalar_eq_scalarFold]
        rfl
      _ = (BaseLinear.combineEvaluations
          EvaluationHomomorphism.PiDEC.radixWeight
          (fun child => PiDecTypedCarrier.decodeEvaluation assignment
            ((profile.childLayout child).yRingCols.getD row []))
          coefficient).c0 :=
        (combineEvaluations_c0 _ _ coefficient).symm
  · change PiDecTypedCarrier.decodeField assignment
        ((layout.base.parent.yRingCols.getD row []).getD
          (2 * coefficient.val + 1) 0) = _
    have laneLt := profile.parentEvaluationWidth row rowLt
    have coefficientLt := coefficient.isLt
    have source := accepted.legacy.y row (2 * coefficient.val + 1) rowLt (by
      simp only [ringDegree] at coefficientLt laneLt ⊢
      omega)
    calc
      PiDecTypedCarrier.decodeField assignment
          ((layout.base.parent.yRingCols.getD row []).getD
            (2 * coefficient.val + 1) 0) =
          recomposeScalar fun child =>
            PiDecTypedCarrier.decodeField assignment
              (((profile.childLayout child).yRingCols.getD row []).getD
                (2 * coefficient.val + 1) 0) :=
        recomposesField profile assignment _
          (fun claim => (claim.yRingCols.getD row []).getD
            (2 * coefficient.val + 1) 0) source
      _ = scalarFold EvaluationHomomorphism.PiDEC.radixWeight
          (fun child =>
            (PiDecTypedCarrier.decodeEvaluation assignment
              ((profile.childLayout child).yRingCols.getD row [])
                coefficient).c1) := by
        rw [recomposeScalar_eq_scalarFold]
        rfl
      _ = (BaseLinear.combineEvaluations
          EvaluationHomomorphism.PiDEC.radixWeight
          (fun child => PiDecTypedCarrier.decodeEvaluation assignment
            ((profile.childLayout child).yRingCols.getD row []))
          coefficient).c1 :=
        (combineEvaluations_c1 _ _ coefficient).symm

/-- Legacy strict semantic-prefix y recomposition transports to the exact
typed evaluation arrays; physical padding never enters the decoder. -/
theorem evaluationEquation
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment) :
    PiDecTypedCarrier.decodeEvaluations assignment layout.base.parent =
      EvaluationHomomorphism.PiDEC.recomposeEvaluations
        (shape := shape) fun child =>
          PiDecTypedCarrier.decodeEvaluations assignment
            (profile.childLayout child) := by
  apply Array.ext
  · simpa [PiDecTypedCarrier.decodeEvaluations,
      EvaluationHomomorphism.PiDEC.recomposeEvaluations] using
        profile.parentEvaluationCount
  · intro row leftLt rightLt
    have rowLt : row < layout.base.parent.yRingCols.length := by
      simpa [PiDecTypedCarrier.decodeEvaluations] using leftLt
    let matrix : Fin shape.matrixCount := ⟨row, by
      rw [← profile.parentEvaluationCount]
      exact rowLt⟩
    simp only [EvaluationHomomorphism.PiDEC.recomposeEvaluations,
      Array.getElem_ofFn]
    calc
      (PiDecTypedCarrier.decodeEvaluations assignment
          layout.base.parent)[row] =
          PiDecTypedCarrier.decodeEvaluation assignment
            (layout.base.parent.yRingCols.getD row []) := by
        simp [PiDecTypedCarrier.decodeEvaluations, List.getD, rowLt]
      _ = BaseLinear.combineEvaluations
          EvaluationHomomorphism.PiDEC.radixWeight (fun child =>
            PiDecTypedCarrier.decodeEvaluation assignment
              ((profile.childLayout child).yRingCols.getD row [])) :=
        decodedEvaluationEquation profile assignment accepted row rowLt
      _ = BaseLinear.combineEvaluations
          EvaluationHomomorphism.PiDEC.radixWeight (fun child =>
            (PiDecTypedCarrier.decodeEvaluations assignment
              (profile.childLayout child)).getD row
                BaseLinear.evaluationZero) := by
        apply congrArg (BaseLinear.combineEvaluations
          EvaluationHomomorphism.PiDEC.radixWeight)
        funext child
        have childMember := profile.childLayout_mem child
        have childLt : row < (profile.childLayout child).yRingCols.length := by
          rw [profile.childEvaluationCount
            (profile.childLayout child) childMember]
          exact matrix.isLt
        simp [PiDecTypedCarrier.decodeEvaluations, List.getD, childLt]

/-- Packaging of the two generic representation transports. The exported
unconditional bridge constructs this structure from strict acceptance. -/
structure TransportPremises
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat → Nat) : Prop where
  commitmentEquation :
    (PiDecTypedCarrier.decodedParent profile system assignment).commitment =
      (PiDECAlgebra.Algebra.concrete key).recomposeCommitment
        (fun child =>
          (PiDecTypedCarrier.decodedOutput profile system assignment child).commitment)
  evaluationEquation :
    (PiDecTypedCarrier.decodedParent profile system assignment).evaluations =
      (PiDECAlgebra.Algebra.concrete key).recomposeEvaluations
        (fun child =>
          (PiDecTypedCarrier.decodedOutput profile system assignment child).evaluations)

/-- Both representation transports follow from the legacy equations already
contained in reduced compiler acceptance. -/
theorem transportPremises_of_accepted
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment) :
    TransportPremises profile key system assignment where
  commitmentEquation := by
    change PiDecTypedCarrier.decodeCommitment assignment layout.base.parent =
      PiDECAlgebra.Commitment.recomposeCommitment fun child =>
        PiDecTypedCarrier.decodeCommitment assignment
          (profile.childLayout child)
    exact commitmentEquation profile assignment accepted
  evaluationEquation := by
    change PiDecTypedCarrier.decodeEvaluations assignment layout.base.parent =
      EvaluationHomomorphism.PiDEC.recomposeEvaluations
        (shape := shape) fun child =>
          PiDecTypedCarrier.decodeEvaluations assignment
            (profile.childLayout child)
    exact evaluationEquation profile assignment accepted

/-- Assembly bridge once the two derived representation transports are
packaged. `accepted_refines_typed` below supplies them automatically. -/
theorem toTypedAccepted
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment)
    (transport : TransportPremises profile key system assignment) :
    PiDecTypedCarrier.Accepted profile key system assignment where
  commitmentEquation := transport.commitmentEquation
  canonicalPublicInput := by
    intro child
    simpa [PiDecTypedCarrier.decodedParent,
      PiDecTypedCarrier.decodedOutput,
      PiDECAlgebra.PaperVerifier.publicInputSplit] using
        canonicalPublicInput valid profile accepted child
  evaluationEquation := transport.evaluationEquation
  commonPoint := by
    intro child
    simpa [PiDecTypedCarrier.decodedParent,
      PiDecTypedCarrier.decodedOutput] using
        commonPoint profile accepted child

/-- Exact operational paper acceptance, conditional on the two explicitly
named representation transports. -/
theorem refines_paper
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment)
    (transport : TransportPremises profile key system assignment) :
    PiDEC.PaperVerifier.OutputAccepted
      (PiDECAlgebra.Algebra.concrete key)
      (PiDECAlgebra.PaperVerifier.publicInputSplit key)
      (PiDECAlgebra.PaperVerifier.evaluationArity key)
      (PiDecTypedCarrier.decodedParent profile system assignment)
      (PiDecTypedCarrier.decodedOutput profile system assignment) := by
  exact PiDecTypedCarrier.accepted_refines_paper profile key system assignment
    (toTypedAccepted valid profile key system assignment accepted transport)

/-- Unconditional model-level bridge from the reduced compiler endpoint to
the exact typed paper endpoint. -/
theorem accepted_refines_typed
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment) :
    PiDecTypedCarrier.Accepted profile key system assignment := by
  exact toTypedAccepted valid profile key system assignment accepted
    (transportPremises_of_accepted profile key system assignment accepted)

/-- Unconditional model-level exact-paper consequence of reduced compiler
acceptance under the typed production profile. -/
theorem accepted_refines_paper
    {shape : Phi81Relation.Shape}
    {verifierRows : Nat}
    {layout : PiDecStrictProductionCompiler.Layout}
    (valid : PiDecStrictProductionCompiler.ShapeValid layout)
    (profile : PiDecTypedCarrier.Profile shape verifierRows layout.base)
    (key : PiRLCAlgebra.Commitment.Key shape verifierRows)
    (system : Phi81Relation.Structure shape)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictProductionCompiler.Accepted layout assignment) :
    PiDEC.PaperVerifier.OutputAccepted
      (PiDECAlgebra.Algebra.concrete key)
      (PiDECAlgebra.PaperVerifier.publicInputSplit key)
      (PiDECAlgebra.PaperVerifier.evaluationArity key)
      (PiDecTypedCarrier.decodedParent profile system assignment)
      (PiDecTypedCarrier.decodedOutput profile system assignment) := by
  exact PiDecTypedCarrier.accepted_refines_paper profile key system assignment
    (accepted_refines_typed valid profile key system assignment accepted)

end Nightstream.Implementation.R1CS.PiDecStrictProductionCompiler.PaperBridge
