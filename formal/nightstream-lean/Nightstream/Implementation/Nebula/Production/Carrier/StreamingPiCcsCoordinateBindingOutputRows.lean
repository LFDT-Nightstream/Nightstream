import Mathlib.Algebra.BigOperators.Fin
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingRows

/-!
Contract: exact semantic refinement of the 108 compact seeded Phi81 output
rows for one production PiCCS variable-coordinate commitment phase.

Assurance tier: generated-row to concrete Module-SIS map bridge.

Owns the residue interpretation of the compact block's dense linear value,
its equality to the verifier-owned Phi81 matrix action, and the equality of
that action to the exact finite Ajtai commitment on the phase-masked witness.

Does not own source-opening rows, Rust `rand_chacha` conformance, commitment
shape rows, phase scheduling, public-state placement, Module-SIS hardness, or
recursive lifecycle integration.

Emits constraints: no. It proves the meaning of the 108 rows emitted by the
compact block.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows

open scoped BigOperators
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingRows
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource

local instance : CommRing F :=
  CommRing.ofMinimalAxioms
    ConcreteCarrier.baseLaws.add_assoc
    ConcreteCarrier.baseLaws.zero_add
    Lean.Grind.Fin.neg_add_cancel
    ConcreteCarrier.baseLaws.mul_assoc
    ConcreteCarrier.baseLaws.mul_comm
    ConcreteCarrier.baseLaws.one_mul
    ConcreteCarrier.baseLaws.left_distrib

private theorem foldRange_residue
    (count initial : Nat) (term : Nat → Nat) :
    SeededPhi81RingRefinement.residueNat
        ((List.range count).foldl
          (fun accumulated index => accumulated + term index) initial) =
      SeededPhi81RingRefinement.residueNat initial +
        sumRange ConcreteCarrier.baseOps count
          (fun index => SeededPhi81RingRefinement.residueNat (term index)) := by
  induction count generalizing initial with
  | zero =>
      exact (ConcreteCarrier.baseLaws.add_zero _).symm
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rw [SeededPhi81RingRefinement.residueNat_add,
        inductionHypothesis, sumRange]
      exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem nestedFold_residue
    (outerCount innerCount : Nat) (term : Nat → Nat → Nat) :
    SeededPhi81RingRefinement.residueNat
        ((List.range outerCount).foldl (fun outer outerIndex =>
          (List.range innerCount).foldl (fun inner innerIndex =>
            inner + term outerIndex innerIndex) outer) 0) =
      sumRange ConcreteCarrier.baseOps outerCount fun outerIndex =>
        sumRange ConcreteCarrier.baseOps innerCount fun innerIndex =>
          SeededPhi81RingRefinement.residueNat
            (term outerIndex innerIndex) := by
  induction outerCount with
  | zero => rfl
  | succ outerCount inductionHypothesis =>
      rw [List.range_succ, List.foldl_append]
      simp only [List.foldl_cons, List.foldl_nil]
      rw [foldRange_residue, inductionHypothesis, sumRange]
      rfl

theorem linearValue_residue
    (block : SeededPhi81.Block) (assignment : Nat → Nat)
    (output coordinate : Nat) :
    SeededPhi81RingRefinement.residueNat
        (block.linearValue assignment output coordinate) =
      sumRange ConcreteCarrier.baseOps block.messageCols fun messageCol =>
        sumRange ConcreteCarrier.baseOps SeededPhi81.dimension fun messageRow =>
          SeededPhi81RingRefinement.residueNat
            (block.termValue assignment output coordinate
              messageCol messageRow) := by
  unfold SeededPhi81.Block.linearValue
  rw [SeededPhi81RingRefinement.residueNat_mod]
  exact nestedFold_residue block.messageCols SeededPhi81.dimension _

theorem coordinateBlock_linearValue_eq_ring_products
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (exact : SourceColumnsExact layout assignment fields)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    SeededPhi81RingRefinement.residueNat
        ((coordinateBlock production layout).linearValue assignment
          output.val coordinate.val) =
      sumRange ConcreteCarrier.baseOps messageColumnCount fun messageCol =>
        if messageColLt : messageCol < messageColumnCount then
          ringFMul
            (production.setup.verifierKey output
              ⟨messageCol, messageColLt⟩)
            (coefficientMap
              (maskedWitness fields layout.selected
                ⟨messageCol, messageColLt⟩)).coefficients
            coordinate
        else 0 := by
  rw [linearValue_residue, coordinateBlock_messageCols]
  apply sumRange_congr
  intro messageCol messageColLt
  rw [dif_pos messageColLt]
  let column : Fin messageColumnCount := ⟨messageCol, messageColLt⟩
  rw [CarrierAction.ringFMul_apply_eq_rightLinear]
  apply sumRange_congr
  intro messageRow messageRowLt
  have messageRowLtRing : messageRow < ringDegree := by
    simpa [SeededPhi81.dimension, SeededPhi81Sampler.dimension, ringDegree]
      using messageRowLt
  rw [dif_pos messageRowLtRing]
  let row : Fin ringDegree := ⟨messageRow, messageRowLtRing⟩
  unfold SeededPhi81.Block.termValue
  rw [SeededPhi81RingRefinement.residueNat_mul]
  rw [coordinateBlock_coefficient_residue production layout output column
    row coordinate]
  rw [coordinateBlock_inputValue_exact exact column row]
  rfl

private theorem finSum_eq_sumRange :
    ∀ {count : Nat} (term : Fin count → F),
      (∑ index, term index) =
        sumRange ConcreteCarrier.baseOps count fun index =>
          if indexLt : index < count then term ⟨index, indexLt⟩ else 0
  | 0, term => by
      rw [Fin.sum_univ_zero]
      rfl
  | count + 1, term => by
      rw [Fin.sum_univ_castSucc, sumRange]
      rw [finSum_eq_sumRange (fun index : Fin count => term index.castSucc)]
      congr 1
      · apply sumRange_congr
        intro index indexLt
        rw [dif_pos indexLt,
          dif_pos (Nat.lt_trans indexLt (Nat.lt_succ_self count))]
        congr 1
      · rw [dif_pos (Nat.lt_succ_self count)]
        congr 1

private def ringCoordinate (coordinate : Fin ringDegree) :
    ExecutablePhi81.Ring →+ F where
  toFun value := value.coefficients coordinate
  map_zero' := rfl
  map_add' := by
    intro left right
    rfl

private theorem commit_coordinate_generic
    {sourceShape : Nightstream.Protocol.Nebula.AjtaiBinding.Shape}
    (matrix : Matrix ExecutablePhi81.Ring sourceShape)
    (map : CoefficientVector sourceShape →+ ExecutablePhi81.Ring)
    (witness : Witness sourceShape)
    (output : Fin sourceShape.rows) (coordinate : Fin ringDegree) :
    ((commit matrix map witness output).coefficients coordinate) =
      ∑ messageCol,
        ringFMul (map (witness messageCol)).coefficients
          (matrix output messageCol).coefficients coordinate := by
  unfold commit
  calc
    ((∑ messageCol,
        map (witness messageCol) * matrix output messageCol) :
        ExecutablePhi81.Ring).coefficients coordinate =
        ∑ messageCol,
          (map (witness messageCol) *
            matrix output messageCol).coefficients coordinate := by
      simpa only [ringCoordinate] using
        (map_sum (ringCoordinate coordinate)
          (fun messageCol : Fin sourceShape.columns =>
            map (witness messageCol) * matrix output messageCol) Finset.univ)
    _ = ∑ messageCol,
        ringFMul (map (witness messageCol)).coefficients
          (matrix output messageCol).coefficients coordinate := by
      rfl

private theorem concreteCommitment_coordinate
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (witness : Witness shape)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    ((commit (seededMatrix setup) coefficientMap witness output
        ).coefficients coordinate) =
      ∑ messageCol,
        ringFMul (setup.verifierKey output messageCol)
          (coefficientMap (witness messageCol)).coefficients coordinate := by
  rw [commit_coordinate_generic]
  apply Finset.sum_congr rfl
  intro messageCol _
  change ringFMul
      (coefficientMap (witness messageCol)).coefficients
      (setup.verifierKey output messageCol) coordinate = _
  exact congrFun (RingFLaws.ringFMul_comm _ _) coordinate

/-- The exact finite Ajtai commitment coordinate is the residue of the dense
compact-row value on the same masked witness. -/
theorem maskedCommitment_coordinate_eq_linearValue
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (exact : SourceColumnsExact layout assignment fields)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    ((commit (seededMatrix production.setup) coefficientMap
        (maskedWitness fields layout.selected) output).coefficients
        coordinate) =
      SeededPhi81RingRefinement.residueNat
        ((coordinateBlock production layout).linearValue assignment
          output.val coordinate.val) := by
  rw [concreteCommitment_coordinate]
  rw [finSum_eq_sumRange]
  exact (coordinateBlock_linearValue_eq_ring_products exact output coordinate).symm

/-- Canonical 108-field view of the direct phase-masked Module-SIS
commitment. -/
def maskedConcreteBinding
    (production : ProductionSetup) (fields : Fields)
    (selected : Fin fieldCount → Bool) : OutputFields :=
  flattenCommitment
    (commit (seededMatrix production.setup) coefficientMap
      (maskedWitness fields selected))

@[simp]
theorem maskedConcreteBinding_outputIndex
    (production : ProductionSetup) (fields : Fields)
    (selected : Fin fieldCount → Bool)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    (maskedConcreteBinding production fields selected
      (outputIndex output coordinate)).val =
      ((commit (seededMatrix production.setup) coefficientMap
        (maskedWitness fields selected) output).coefficients coordinate).val := by
  unfold maskedConcreteBinding
  exact flattenCommitment_outputIndex _ output coordinate

private theorem getD_ofFn
    {alpha : Type} {count : Nat} (function : Fin count → alpha)
    (index : Nat) (fallback : alpha) (bound : index < count) :
    (List.ofFn function).getD index fallback =
      function ⟨index, bound⟩ := by
  have listBound : index < (List.ofFn function).length := by
    simpa using bound
  rw [List.getD_eq_getElem _ _ listBound]
  exact List.getElem_ofFn listBound

private theorem coordinateBlock_outputColumn
    (production : ProductionSetup) (layout : Layout)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    (coordinateBlock production layout).outputColumns.getD
        (output.val * SeededPhi81.dimension + coordinate.val) 0 =
      layout.outputColumn (outputIndex output coordinate) := by
  rw [coordinateBlock_outputColumns]
  have bound :
      output.val * SeededPhi81.dimension + coordinate.val <
        shape.rows * shape.degree := by
    have outputLt := output.isLt
    have coordinateLt := coordinate.isLt
    change output.val < 2 at outputLt
    change coordinate.val < 54 at coordinateLt
    change output.val * 54 + coordinate.val < 108
    omega
  rw [getD_ofFn layout.outputColumn _ 0 bound]
  congr 1
  apply Fin.ext
  rw [outputIndex_val]
  rfl

/-- Accepted compact definitions and authoritative source columns determine
the exact 108-field direct commitment. No claimed commitment value is a
premise. -/
theorem compact_output_exact
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (sourceExact : SourceColumnsExact layout assignment fields)
    (holds : (coordinateBlock production layout).Holds assignment)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    assignment (layout.outputColumn (outputIndex output coordinate)) =
      (maskedConcreteBinding production fields layout.selected
        (outputIndex output coordinate)).val := by
  have outputValue :=
    (coordinateBlock production layout).output_eq_linearValue holds
      output coordinate
  rw [coordinateBlock_outputColumn production layout output coordinate]
    at outputValue
  rw [outputValue, maskedConcreteBinding_outputIndex]
  let value :=
    (coordinateBlock production layout).linearValue assignment
      output.val coordinate.val
  have commitmentEqual :=
    maskedCommitment_coordinate_eq_linearValue (production := production)
      sourceExact output coordinate
  calc
    value = (SeededPhi81RingRefinement.residueNat value).val := by
      rw [SeededPhi81RingRefinement.residueNat_val,
        Nat.mod_eq_of_lt
          ((coordinateBlock production layout).linearValue_lt assignment
            output.val coordinate.val)]
    _ = ((commit (seededMatrix production.setup) coefficientMap
        (maskedWitness fields layout.selected) output).coefficients
        coordinate).val := by
      simpa [value] using congrArg Fin.val commitmentEqual.symm

/-- Main generated-row soundness theorem for the 108 compact output rows. -/
theorem compact_output_exact_of_rows
    {production : ProductionSetup} {layout : Layout}
    {assignment : Nat → Nat} {fields : Fields}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (sourceExact : SourceColumnsExact layout assignment fields)
    (satisfies : Satisfies
      (coordinateBlock production layout).rows assignment)
    (output : Fin verifierRows) (coordinate : Fin ringDegree) :
    assignment (layout.outputColumn (outputIndex output coordinate)) =
      (maskedConcreteBinding production fields layout.selected
        (outputIndex output coordinate)).val := by
  exact compact_output_exact sourceExact
    (SeededPhi81.sound canonical one satisfies) output coordinate

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingOutputRows
