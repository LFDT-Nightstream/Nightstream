import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.AlgebraRows
import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.RingCombinationSound
import Nightstream.Implementation.NebulaV2.NIFS.Core.PaperAlgebra
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.RingKModule

/-!
Contract: typed soundness of the complete V2 PiRLC algebra rows.

This file decodes all 110 ring families from their physical columns and proves
that aggregate row satisfaction gives the exact componentwise product-bundle,
public-input, and evaluation combinations selected by the paper PiRLC
algebra. The shared challenge rings are decoded once from the common symbol
columns.

It does not own transcript sampling, placement in the complete NIFS call, or
the PiCCS and PiDEC checks.

Emits constraints: no; it proves the meaning of 4,817,340 rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows
open Nightstream.Implementation.NebulaV2.ProductPiRlcRingCombinationSound
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81StrongSet
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Norm
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.Protocol.NebulaV2.CommitmentBundle

/-- The one shared set of 15 challenge rings, decoded from selected symbols. -/
def decodeChallenges
    (layout : Layout) (assignment : Nat -> Nat)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5) :
    Source -> RingF :=
  fun source lane =>
    embedCoefficient
      ⟨assignment (layout.challengeSymbol source lane), range source lane⟩

/-- The 15 authority-bearing source bundles. -/
def decodeInputBundles
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Source -> ProductCommitmentAlgebra.BundleValue :=
  fun source component row lane =>
    wireField assignment canonical
      (layout.inputBundle source component row lane)

/-- The one combined authority-bearing output bundle. -/
def decodeOutputBundle
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductCommitmentAlgebra.BundleValue :=
  fun component row lane =>
    wireField assignment canonical
      (layout.outputBundle component row lane)

/-- The source public input represented as exactly ten whole Phi81 rings. -/
def decodeInputPublicRings
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Source -> PublicBlock -> RingF :=
  fun source block lane =>
    wireField assignment canonical
      (layout.inputPublic source block lane)

/-- The output public input represented as exactly ten whole Phi81 rings. -/
def decodeOutputPublicRings
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    PublicBlock -> RingF :=
  fun block lane =>
    wireField assignment canonical
      (layout.outputPublic block lane)

/-- Reconstruct the exact 540-field public carrier from ten whole rings. -/
def publicOfRings
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (rings : PublicBlock -> RingF) :
    ProductPaperAlgebra.PublicInput logicalWidth publicFits :=
  fun column =>
    rings
      ⟨column.val / ringDegree, by
        have columnLt := column.isLt
        change column.val < 540 at columnLt
        simp only [ringDegree]
        omega⟩
      ⟨column.val % ringDegree, Nat.mod_lt _ (by simp [ringDegree])⟩

/-- Reconstructing the public carrier and selecting a complete ring is an
exact inverse operation. -/
theorem publicBlock_publicOfRings
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (rings : PublicBlock -> RingF) (block : PublicBlock) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
        (publicOfRings (logicalWidth := logicalWidth)
          (publicFits := publicFits) rings) block =
      rings block := by
  funext lane
  unfold
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
    publicOfRings
  have blockLt := block.isLt
  have laneLt := lane.isLt
  change block.val < 10 at blockLt
  change lane.val < 54 at laneLt
  have quotient :
      (block.val * 54 + lane.val) / 54 = block.val := by
    rw [Nat.mul_comm]
    rw [Nat.mul_add_div (by decide : 0 < 54),
      Nat.div_eq_of_lt laneLt, Nat.add_zero]
  have remainder :
      (block.val * 54 + lane.val) % 54 = lane.val := by
    rw [Nat.mul_comm]
    rw [Nat.mul_add_mod, Nat.mod_eq_of_lt laneLt]
  have blockEqual :
      (⟨(block.val * 54 + lane.val) / 54, by omega⟩ : PublicBlock) =
        block := by
    apply Fin.ext
    exact quotient
  have laneEqual :
      (⟨(block.val * 54 + lane.val) % 54, by omega⟩ : Fin ringDegree) =
        lane := by
    apply Fin.ext
    simp only [ringDegree]
    exact remainder
  change
    rings ⟨(block.val * 54 + lane.val) / 54, by omega⟩
        ⟨(block.val * 54 + lane.val) % 54, by omega⟩ =
      rings block lane
  rw [blockEqual, laneEqual]

def decodeInputPublic
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Source -> ProductPaperAlgebra.PublicInput logicalWidth publicFits :=
  fun source => publicOfRings (decodeInputPublicRings layout assignment canonical source)

def decodeOutputPublic
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductPaperAlgebra.PublicInput logicalWidth publicFits :=
  publicOfRings (decodeOutputPublicRings layout assignment canonical)

/-- One base-field limb of one source evaluation family. -/
def decodeInputEvaluationLimb
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (source : Source) (matrix : MatrixIndex) (limb : ExtensionLimb) : RingF :=
  fun lane =>
    wireField assignment canonical
      (layout.inputEvaluation source matrix limb lane)

/-- One base-field limb of the output evaluation family. -/
def decodeOutputEvaluationLimb
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (matrix : MatrixIndex) (limb : ExtensionLimb) : RingF :=
  fun lane =>
    wireField assignment canonical
      (layout.outputEvaluation matrix limb lane)

/-- The 15 complete quadratic-extension evaluation families. -/
def decodeInputEvaluations
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Source -> ProductPaperAlgebra.Evaluation :=
  fun source matrix lane =>
    ⟨decodeInputEvaluationLimb layout assignment canonical source matrix 0 lane,
      decodeInputEvaluationLimb layout assignment canonical source matrix 1 lane⟩

/-- The one complete combined quadratic-extension evaluation family. -/
def decodeOutputEvaluation
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductPaperAlgebra.Evaluation :=
  fun matrix lane =>
    ⟨decodeOutputEvaluationLimb layout assignment canonical matrix 0 lane,
      decodeOutputEvaluationLimb layout assignment canonical matrix 1 lane⟩

/-- Every aggregate family is the exact typed sum of its 15 source rings. -/
theorem familyEquation_of_rows
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies (rows layout) assignment)
    (family : Family) :
    outputRing (familyLayout layout family) assignment canonical =
      ProjectionPhi81.productSum (decodeChallenges layout assignment range)
        (inputRing (familyLayout layout family) assignment canonical) := by
  have derived := rows_imply_ring_combination canonical one
    (fun source lane => range source lane)
    (family_satisfies satisfied family)
  have challengeEqual :
      challengeRing (familyLayout layout family) assignment
          (fun source lane => range source lane) =
        decodeChallenges layout assignment range := by
    funext source lane
    rfl
  rw [← challengeEqual]
  exact derived

/-! ## Independent paper-algebra equalities -/

/-- The paper commitment fold is the same canonical head-first product sum. -/
theorem combineCommitments_eq_productSum
    {count verifierRows : Nat} (challenges : Fin count -> RingF)
    (values : Fin count ->
      Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.Value
        verifierRows)
    (row : Fin verifierRows) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments
        challenges values row =
      ProjectionPhi81.productSum challenges (fun source => values source row) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.combineCommitments,
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAdd,
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.commitmentAct,
        ProjectionPhi81.productSum]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => values source.succ)]

/-- The paper public-input fold at one field coordinate is the same product
sum over the coordinate's complete public ring. -/
theorem combinePublicInputs_coordinate
    {shape : Phi81Relation.Shape} {count : Nat}
    (challenges : Fin count -> RingF)
    (inputs : Fin count -> Phi81Relation.PublicInput shape)
    (column : Fin shape.publicWidth) :
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        challenges inputs column =
      ProjectionPhi81.productSum challenges
        (fun source =>
          Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
            (inputs source)
            (Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
              shape column))
        (Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
          column) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs,
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicAdd,
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicAct,
        ProjectionPhi81.productSum, ringFAdd]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => inputs source.succ)]

/-- Low extension limbs of the paper evaluation fold use the identical base
ring product sum. -/
theorem combineEvaluation_component0
    {count : Nat} (challenges : Fin count -> RingF)
    (values : Fin count -> RingK) :
    RingKModule.component0
        (PiRLCFinite.combineEvaluation challenges values) =
      ProjectionPhi81.productSum challenges
        (fun source => RingKModule.component0 (values source)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [PiRLCFinite.combineEvaluation, ProjectionPhi81.productSum,
        RingKModule.component0_add]
      rw [RingKModule.action_component0]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => values source.succ)]

/-- High extension limbs of the paper evaluation fold use the identical base
ring product sum. -/
theorem combineEvaluation_component1
    {count : Nat} (challenges : Fin count -> RingF)
    (values : Fin count -> RingK) :
    RingKModule.component1
        (PiRLCFinite.combineEvaluation challenges values) =
      ProjectionPhi81.productSum challenges
        (fun source => RingKModule.component1 (values source)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      simp only [PiRLCFinite.combineEvaluation, ProjectionPhi81.productSum,
        RingKModule.component1_add]
      rw [RingKModule.action_component1]
      rw [inductionHypothesis
        (fun source => challenges source.succ)
        (fun source => values source.succ)]

/-! ## Complete typed equations from aggregate rows -/

/-- All 72 commitment-ring families imply the exact mandatory product-bundle
parent. -/
theorem bundleEquation_of_rows
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies (rows layout) assignment) :
    decodeOutputBundle layout assignment canonical =
      ProductCommitmentAlgebra.combineBundles
        (decodeChallenges layout assignment range)
        (decodeInputBundles layout assignment canonical) := by
  funext component row
  unfold ProductCommitmentAlgebra.combineBundles
  rw [combineCommitments_eq_productSum]
  have derived := familyEquation_of_rows canonical one range satisfied
    (.commitment component row)
  change
    decodeOutputBundle layout assignment canonical component row =
      ProjectionPhi81.productSum
        (decodeChallenges layout assignment range)
        (fun source =>
          decodeInputBundles layout assignment canonical source component row)
      at derived
  exact derived

/-- All ten public-ring families imply the exact paper public-input parent. -/
theorem publicEquation_of_rows
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies (rows layout) assignment) :
    decodeOutputPublic (logicalWidth := logicalWidth)
        (publicFits := publicFits) layout assignment canonical =
      Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
        (decodeChallenges layout assignment range)
        (decodeInputPublic (logicalWidth := logicalWidth)
          (publicFits := publicFits) layout assignment canonical) := by
  funext column
  rw [combinePublicInputs_coordinate]
  let block :=
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlockIndex
      (ProductPaperAlgebra.FullShape logicalWidth publicFits) column
  let lane :=
    Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicLaneIndex
      column
  have derivedRing := familyEquation_of_rows canonical one range satisfied
    (.publicInput block)
  have derived := congrFun derivedRing lane
  have inputBlocks :
      (fun source =>
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.publicBlock
          (decodeInputPublic (logicalWidth := logicalWidth)
            (publicFits := publicFits) layout assignment canonical source)
          block) =
        fun source =>
          decodeInputPublicRings layout assignment canonical source block := by
    funext source
    exact publicBlock_publicOfRings
      (decodeInputPublicRings layout assignment canonical source) block
  rw [inputBlocks]
  change
    decodeOutputPublicRings layout assignment canonical block lane =
      ProjectionPhi81.productSum
        (decodeChallenges layout assignment range)
        (fun source =>
          decodeInputPublicRings layout assignment canonical source block)
        lane
  change
    decodeOutputPublicRings layout assignment canonical block =
      ProjectionPhi81.productSum
        (decodeChallenges layout assignment range)
        (fun source =>
          decodeInputPublicRings layout assignment canonical source block)
      at derivedRing
  exact congrFun derivedRing lane

/-- All 28 evaluation-limb families imply the exact quadratic-extension
evaluation parent. -/
theorem evaluationEquation_of_rows
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies (rows layout) assignment) :
    decodeOutputEvaluation layout assignment canonical =
      ProductPaperAlgebra.combineEvaluationFamily
        (decodeChallenges layout assignment range)
        (decodeInputEvaluations layout assignment canonical) := by
  funext matrix
  apply RingKModule.ext_components
  · have rowsEquation := familyEquation_of_rows canonical one range satisfied
      (.evaluation matrix 0)
    have foldEquation := combineEvaluation_component0
      (decodeChallenges layout assignment range)
      (fun source => decodeInputEvaluations layout assignment canonical source matrix)
    change
      decodeOutputEvaluationLimb layout assignment canonical matrix 0 =
        ProjectionPhi81.productSum
          (decodeChallenges layout assignment range)
          (fun source =>
            decodeInputEvaluationLimb layout assignment canonical source matrix 0)
      at rowsEquation
    change
      RingKModule.component0
          (ProductPaperAlgebra.combineEvaluationFamily
            (decodeChallenges layout assignment range)
            (decodeInputEvaluations layout assignment canonical) matrix) =
        ProjectionPhi81.productSum
          (decodeChallenges layout assignment range)
          (fun source =>
            decodeInputEvaluationLimb layout assignment canonical source matrix 0)
      at foldEquation
    exact rowsEquation.trans foldEquation.symm
  · have rowsEquation := familyEquation_of_rows canonical one range satisfied
      (.evaluation matrix 1)
    have foldEquation := combineEvaluation_component1
      (decodeChallenges layout assignment range)
      (fun source => decodeInputEvaluations layout assignment canonical source matrix)
    change
      decodeOutputEvaluationLimb layout assignment canonical matrix 1 =
        ProjectionPhi81.productSum
          (decodeChallenges layout assignment range)
          (fun source =>
            decodeInputEvaluationLimb layout assignment canonical source matrix 1)
      at rowsEquation
    change
      RingKModule.component1
          (ProductPaperAlgebra.combineEvaluationFamily
            (decodeChallenges layout assignment range)
            (decodeInputEvaluations layout assignment canonical) matrix) =
        ProjectionPhi81.productSum
          (decodeChallenges layout assignment range)
          (fun source =>
            decodeInputEvaluationLimb layout assignment canonical source matrix 1)
      at foldEquation
    exact rowsEquation.trans foldEquation.symm

/-- Direct row-level form of all three exact PiRLC parent equations. -/
theorem typedEquations_of_rows
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies (rows layout) assignment) :
    (decodeOutputBundle layout assignment canonical =
        ProductCommitmentAlgebra.combineBundles
          (decodeChallenges layout assignment range)
          (decodeInputBundles layout assignment canonical)) /\
      (decodeOutputPublic (logicalWidth := logicalWidth)
          (publicFits := publicFits) layout assignment canonical =
        Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
          (decodeChallenges layout assignment range)
          (decodeInputPublic (logicalWidth := logicalWidth)
            (publicFits := publicFits) layout assignment canonical)) /\
      (decodeOutputEvaluation layout assignment canonical =
        ProductPaperAlgebra.combineEvaluationFamily
          (decodeChallenges layout assignment range)
          (decodeInputEvaluations layout assignment canonical)) := by
  exact
    ⟨bundleEquation_of_rows canonical one range satisfied,
      publicEquation_of_rows canonical one range satisfied,
      evaluationEquation_of_rows canonical one range satisfied⟩

end Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSound
