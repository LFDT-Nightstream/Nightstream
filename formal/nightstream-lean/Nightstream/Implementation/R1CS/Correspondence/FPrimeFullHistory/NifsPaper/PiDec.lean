import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicCarrier

/-!
Π_DEC phase refinement for the fixed F' NIFS profile.

Owns: construction of a generic `PiDEC.Attempt` over the shared packed public
carrier, and the conditional implication from strict decoded acceptance to
`PiDEC.Accepted` for an independently supplied algebra.

Does not own: private CE openings, knowledge reduction, Π_CCS/Π_RLC linkage,
the packed-to-Concrete carrier refinement, Ajtai/MSIS security, `y_zcol`
delayed authority, or row removal.

Because its `PublicInput` is the 270-coefficient packed carrier, this theorem
does not close `REL-CONCRETE-PRODUCTION` or identify that carrier with the
paper's aligned `L_in`. `PublicInputBoundary.lean` keeps that gap explicit.

Emits constraints: no.

Authority boundary: the strict semantic equations are an explicit premise;
the theorem does not infer them from an old measured circuit or from a digest.

| Π_DEC public obligation | Strict source | Guarantee | Permits row removal? |
|---|---|---|---|
| parent/child stages | construction | combined parent, fresh children | no |
| same structure/point | one caller-supplied system; strict `sameR` | paper CE point equality | no |
| commitment recomposition | `Accepted.commitment` | paper commitment equation | no |
| packed-X recomposition | `Accepted.x` | packed public-carrier equation | no |
| evaluation recomposition | active first 108 limbs of `Accepted.y` | three `RingK` equations | no |
| R1CS rows → strict acceptance | current `Exact.*_sound` uses `native_decide` | **open trusted refinement edge** | no |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.PiDecStrictCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

universe uStructure

def attempt {Structure : Type uStructure} (system : Structure)
    (assignment : Nat → Nat) :
    PiDEC.Attempt Structure PackedPublicInput Point Evaluation PackedCommitment
      Concrete.productionGlobalParams where
  parent := decodedInstance system assignment layout.parent .combined
  children := fun index =>
    decodedInstance system assignment (childLayout index) .fresh

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

private theorem residue_rawLcEval (assignment : Nat → Nat) :
    ∀ terms : List (Nat × Nat),
      residue (rawLcEval assignment terms) =
        terms.foldr (fun term suffix =>
          residue term.2 * residue (assignment term.1) + suffix) 0 := by
  intro terms
  induction terms with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [rawLcEval, List.foldr_cons]
      rw [residue_add, residue_mul, inductionHypothesis]

private theorem residue_lcEval (assignment : Nat → Nat)
    (terms : List (Nat × Nat)) :
    residue (lcEval assignment terms) =
      terms.foldr (fun term suffix =>
        residue term.2 * residue (assignment term.1) + suffix) 0 := by
  rw [lcEval_eq_raw_mod, residue_mod]
  exact residue_rawLcEval assignment terms

private theorem recomposesField
    (assignment : Nat → Nat) (parent : Nat)
    (column : ClaimLayout → Nat)
    (recomposes : PiDecStrictCompiler.Recomposes assignment parent
      (layout.children.map column)
      (PiDecStrictCompiler.radixPowers layout.radix layout.children.length)) :
    residue (assignment parent) =
      combineScalar fun index => residue (assignment (column (childLayout index))) := by
  unfold PiDecStrictCompiler.Recomposes at recomposes
  rw [recomposes, residue_lcEval]
  have range14 : List.range 14 =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] := by
    decide
  simp [combineScalar, radixWeights, Concrete.productionGlobalParams,
    childLayout, FPrimeFullHistoryPiDec.layout,
    PiDecStrictCompiler.radixPowers, range14]

private theorem commitmentLane
    (assignment : Nat → Nat) (lane : Nat)
    (laneLt : lane < layout.parent.commitment.dataCols.length)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    residue (assignment (layout.parent.commitment.dataCols.getD lane 0)) =
      combineScalar fun index =>
        residue (assignment
          ((childLayout index).commitment.dataCols.getD lane 0)) := by
  have recomposes := accepted.commitment lane
  specialize recomposes laneLt
  exact recomposesField assignment _
    (fun claim => claim.commitment.dataCols.getD lane 0) recomposes

private theorem childLayout_mem
    (index : Fin Concrete.productionGlobalParams.k) :
    childLayout index ∈ layout.children := by
  unfold childLayout
  exact List.get_mem layout.children (Fin.cast production_child_count index)

private theorem childCommitmentLength
    (index : Fin Concrete.productionGlobalParams.k) :
    (childLayout index).commitment.dataCols.length =
      layout.parent.commitment.dataCols.length :=
  production_public_shape.commitmentLengths
    (childLayout index) (childLayout_mem index)

set_option maxRecDepth 524288 in
private theorem commitmentEquation
    (assignment : Nat → Nat)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedPackedCommitment assignment layout.parent =
      combinePackedCommitment fun index =>
        decodedPackedCommitment assignment (childLayout index) := by
  apply PackedCommitment.eq_of_data_eq
  apply List.ext_get
  · simp [decodedPackedCommitment, values, combinePackedCommitment, combineList,
      firstIndex, childLayout, Concrete.productionGlobalParams,
      FPrimeFullHistoryPiDec.layout]
  · intro lane parentLt combinedLt
    simp only [decodedPackedCommitment, combinePackedCommitment, values, combineList,
      List.get_eq_getElem, List.getElem_map, List.getElem_range]
    have laneLt : lane < layout.parent.commitment.dataCols.length := by
      simpa [decodedPackedCommitment, values] using parentLt
    calc
      residue (assignment layout.parent.commitment.dataCols[lane]) =
          residue (assignment
            (layout.parent.commitment.dataCols.getD lane 0)) := by
        rw [← List.getElem_eq_getD (l := layout.parent.commitment.dataCols)
          (i := lane) 0]
      _ = combineScalar fun index =>
          residue (assignment
            ((childLayout index).commitment.dataCols.getD lane 0)) :=
        commitmentLane assignment lane laneLt accepted
      _ = combineScalar fun index =>
          (values assignment (childLayout index).commitment.dataCols).getD lane 0 := by
        apply congrArg combineScalar
        funext index
        have childLt : lane <
            (childLayout index).commitment.dataCols.length := by
          rw [childCommitmentLength index]
          exact laneLt
        simp [values, List.getD, childLt]

set_option maxRecDepth 524288 in
private theorem xLane
    (assignment : Nat → Nat) (lane : Nat)
    (laneLt : lane < layout.parent.xActiveCols.length)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    residue (assignment
      (layout.parent.xActiveCols.getD lane 0)) =
      combineScalar fun index => residue (assignment
        ((childLayout index).xActiveCols.getD lane 0)) := by
  have laneBound : lane < 270 := by
    simpa [FPrimeFullHistoryPiDec.layout] using laneLt
  have rowLt : lane / 5 < layout.parent.xRows := by
    simpa [FPrimeFullHistoryPiDec.layout] using
      (show lane / 5 < 54 by omega)
  have columnLt : lane % 5 < activeColumns layout := by
    simpa [activeColumns, FPrimeFullHistoryPiDec.layout] using
      Nat.mod_lt lane (by decide : 0 < 5)
  have recomposes := accepted.x (lane / 5) (lane % 5) rowLt columnLt
  have field := recomposesField assignment
    (xColumn layout layout.parent (lane / 5) (lane % 5))
    (fun claim => xColumn layout claim (lane / 5) (lane % 5)) recomposes
  have laneIndex : lane / 5 * 5 + lane % 5 = lane := by
    omega
  simp [xColumn, columnLt] at field
  rw [show activeColumns layout = 5 by decide] at field
  rw [laneIndex] at field
  simpa [FPrimeFullHistoryPiDec.layout, List.getD, laneBound] using field

private theorem childXLength
    (index : Fin Concrete.productionGlobalParams.k) :
    (childLayout index).xActiveCols.length = layout.parent.xActiveCols.length :=
  production_public_shape.xLengths
    (childLayout index) (childLayout_mem index)

set_option maxRecDepth 524288 in
private theorem publicInputEquation
    (assignment : Nat → Nat)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedPackedInput assignment layout.parent =
      combinePackedPublicInput fun index =>
        decodedPackedInput assignment (childLayout index) := by
  apply PackedPublicInput.eq_of_data_eq
  apply List.ext_get
  · simp [decodedPackedInput, values, combinePackedPublicInput, combineList,
      firstIndex, childLayout,
      Concrete.productionGlobalParams, FPrimeFullHistoryPiDec.layout]
  · intro lane parentLt combinedLt
    simp only [decodedPackedInput, combinePackedPublicInput, values, combineList,
      List.get_eq_getElem,
      List.getElem_map, List.getElem_range]
    have laneLt : lane < layout.parent.xActiveCols.length := by
      simpa [decodedPackedInput, values] using parentLt
    calc
      residue (assignment layout.parent.xActiveCols[lane]) =
          residue (assignment (layout.parent.xActiveCols.getD lane 0)) := by
        rw [← List.getElem_eq_getD (l := layout.parent.xActiveCols)
          (i := lane) 0]
      _ = combineScalar fun index => residue (assignment
          ((childLayout index).xActiveCols.getD lane 0)) :=
        xLane assignment lane laneLt accepted
      _ = combineScalar fun index =>
          (decodedPackedInput assignment (childLayout index)).data.getD lane 0 := by
        apply congrArg combineScalar
        funext index
        have childLt : lane < (childLayout index).xActiveCols.length := by
          rw [childXLength index]
          exact laneLt
        simp [decodedPackedInput, values, List.getD, childLt]

private theorem yLane
    (assignment : Nat → Nat) (row lane : Nat)
    (rowLt : row < layout.parent.yRingCols.length)
    (laneLt : lane < (layout.parent.yRingCols.getD row []).length)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    residue (assignment
      ((layout.parent.yRingCols.getD row []).getD lane 0)) =
      combineScalar fun index => residue (assignment
        (((childLayout index).yRingCols.getD row []).getD lane 0)) := by
  exact recomposesField assignment _
    (fun claim => (claim.yRingCols.getD row []).getD lane 0)
    (accepted.y row lane rowLt laneLt)

private theorem childYRowsLength
    (index : Fin Concrete.productionGlobalParams.k) :
    (childLayout index).yRingCols.length =
      layout.parent.yRingCols.length :=
  (production_public_shape.yShapes
    (childLayout index) (childLayout_mem index)).1

private theorem childYRowLength
    (index : Fin Concrete.productionGlobalParams.k) (row : Nat)
    (rowLt : row < layout.parent.yRingCols.length) :
    ((childLayout index).yRingCols.getD row []).length =
      (layout.parent.yRingCols.getD row []).length :=
  (production_public_shape.yShapes
    (childLayout index) (childLayout_mem index)).2 row rowLt

private theorem getD_mem_of_lt {Carrier : Type}
    (values : List Carrier) (default : Carrier) (index : Nat)
    (indexLt : index < values.length) : values.getD index default ∈ values := by
  rw [← List.getElem_eq_getD (l := values) (i := index) default]
  exact List.getElem_mem indexLt

private theorem activeEvaluationLength (row : Nat)
    (rowLt : row < layout.parent.yRingCols.length) :
    2 * Concrete.ringDegree ≤
      (layout.parent.yRingCols.getD row []).length := by
  have member := getD_mem_of_lt layout.parent.yRingCols [] row rowLt
  have active := production_public_shape.activeEvaluationRows
    layout.parent (List.mem_cons_self) _ member
  simpa [FPrimeFullHistoryPiDec.layout, Concrete.ringDegree] using active

private theorem decodedEvaluationEquation
    (assignment : Nat → Nat) (row : Nat)
    (rowLt : row < layout.parent.yRingCols.length)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedEvaluation assignment (layout.parent.yRingCols.getD row []) =
      combineEvaluation fun index =>
        decodedEvaluation assignment
          ((childLayout index).yRingCols.getD row []) := by
  funext coefficient
  apply k_eq_of_coeffs
  · change residue (assignment
        ((layout.parent.yRingCols.getD row []).getD
          (2 * coefficient.val) 0)) =
      combineScalar fun index => residue (assignment
        (((childLayout index).yRingCols.getD row []).getD
          (2 * coefficient.val) 0))
    apply yLane assignment row (2 * coefficient.val) rowLt _ accepted
    have active := activeEvaluationLength row rowLt
    omega
  · change residue (assignment
        ((layout.parent.yRingCols.getD row []).getD
          (2 * coefficient.val + 1) 0)) =
      combineScalar fun index => residue (assignment
        (((childLayout index).yRingCols.getD row []).getD
          (2 * coefficient.val + 1) 0))
    apply yLane assignment row (2 * coefficient.val + 1) rowLt _ accepted
    have active := activeEvaluationLength row rowLt
    omega

set_option maxRecDepth 524288 in
private theorem evaluationEquation
    (assignment : Nat → Nat)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedEvaluations assignment layout.parent =
      combineEvaluations fun index =>
        decodedEvaluations assignment (childLayout index) := by
  unfold decodedEvaluations combineEvaluations
  apply congrArg List.toArray
  apply List.ext_get
  · simp [firstIndex, childLayout, Concrete.productionGlobalParams,
      FPrimeFullHistoryPiDec.layout]
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
      _ = combineEvaluation fun index =>
          decodedEvaluation assignment
            ((childLayout index).yRingCols.getD row []) :=
        decodedEvaluationEquation assignment row rowLt accepted
      _ = combineEvaluation fun index =>
          ((childLayout index).yRingCols.map
            (decodedEvaluation assignment)).toArray.getD
              row Concrete.ringKZero := by
        apply congrArg combineEvaluation
        funext index
        have childLt : row < (childLayout index).yRingCols.length := by
          rw [childYRowsLength index]
          exact rowLt
        simp [List.getD, childLt]

private theorem extensionValues_eq_of_equalPairs
    (assignment : Nat → Nat) :
    ∀ (parent child : List (Nat × Nat)),
      parent.length = child.length →
      PiDecStrictCompiler.EqualPairs assignment parent child →
      extensionValues assignment child = extensionValues assignment parent := by
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
              PiDecStrictCompiler.EqualPairs assignment parentTail childTail := by
            intro pair pairMem
            exact equalPairs pair (by simp [pairMem])
          have tailLength : parentTail.length = childTail.length := by
            simpa using lengthEq
          simp only [extensionValues, List.map_cons, List.cons.injEq]
          constructor
          · apply k_eq_of_coeffs
            · exact congrArg residue headEq.1.symm
            · exact congrArg residue headEq.2.symm
          · exact inductionHypothesis childTail tailLength tailEqualPairs

private theorem samePoint
    (assignment : Nat → Nat)
    (index : Fin Concrete.productionGlobalParams.k)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    decodedPoint assignment (childLayout index) =
      decodedPoint assignment layout.parent := by
  exact extensionValues_eq_of_equalPairs assignment
    layout.parent.rCols (childLayout index).rCols
    (production_public_shape.rShapes
      (childLayout index) (childLayout_mem index)).symm
    (accepted.sameR (childLayout index) (childLayout_mem index))

universe uAssignment

/-- The only algebra-specific facts needed to interpret the decoded packed
carrier as the public equations of an independently supplied paper algebra.
Private relation predicates and witness validity are deliberately absent. -/
structure PublicRecompositionMatches
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {semantics : RelationSemantics
      Structure Assignment PackedPublicInput Point Evaluation PackedCommitment}
    (algebra : PiDEC.Algebra
      Structure Assignment PackedPublicInput Point Evaluation PackedCommitment
        semantics Concrete.productionGlobalParams) : Prop where
  commitment : ∀ items,
    algebra.recomposeCommitment items = combinePackedCommitment items
  publicInput : ∀ items,
    algebra.recomposePublicInput items = combinePackedPublicInput items
  evaluations : ∀ items,
    algebra.recomposeEvaluations items = combineEvaluations items

/-- Conditional implementation correspondence: strict decoded equations imply
the generic Π_DEC public acceptance predicate for any relation semantics and
algebra whose three public recomposition operations match the fixed packed
carrier. This is not a proof of strict acceptance, the production paper
`L_in`, `CE.Holds`, or knowledge. -/
theorem strictAccepted_refines_paperPublicAccepted
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {semantics : RelationSemantics
      Structure Assignment PackedPublicInput Point Evaluation PackedCommitment}
    (algebra : PiDEC.Algebra
      Structure Assignment PackedPublicInput Point Evaluation PackedCommitment
        semantics Concrete.productionGlobalParams)
    (operations : PublicRecompositionMatches algebra)
    (system : Structure)
    (assignment : Nat → Nat)
    (accepted : PiDecStrictCompiler.Accepted layout assignment) :
    PiDEC.Accepted algebra (attempt system assignment) where
  parentCombined := rfl
  childFresh := fun _ => rfl
  sameStructure := fun _ => rfl
  samePoint := fun index => samePoint assignment index accepted
  commitmentEquation :=
    (commitmentEquation assignment accepted).trans
      (operations.commitment _).symm
  publicInputEquation :=
    (publicInputEquation assignment accepted).trans
      (operations.publicInput _).symm
  evaluationEquation :=
    (evaluationEquation assignment accepted).trans
      (operations.evaluations _).symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiDec
