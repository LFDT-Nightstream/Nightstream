import NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
import NightstreamFPrime.Export.Stage1.PiRLCWitnessAction
import Mathlib.Tactic.SplitIfs

/-!
Executable PiRLC combination of one complete witness block. The same stored
54-coefficient type and source scan serve the public partial traces. The final
value is the corresponding block of the existing complete-assignment fold.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCWitnessBlock

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation (Assignment)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Export.Stage1.PiRLCNonzero (SourceCount)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- An exactly zero source leaves the stored accumulator unchanged.
Otherwise materialize the ring action and addition before the next source. -/
def witnessBlockStep (challenges : Fin SourceCount → RingF)
    (sources : Fin SourceCount → MaterializedRingF)
    (current : MaterializedRingF) (source : Fin SourceCount) : MaterializedRingF :=
  let value := sources source
  if ∀ lane : Fin ringDegree, value.toRing lane = 0 then current
  else MaterializedRingF.ofRing <|
    ringFAdd current.toRing (ringFMul (challenges source) value.toRing)

/-- Skipping an exactly zero source preserves the existing ring expression. -/
theorem witnessBlockStep_toRing (challenges : Fin SourceCount → RingF)
    (sources : Fin SourceCount → MaterializedRingF)
    (current : MaterializedRingF) (source : Fin SourceCount) :
    (witnessBlockStep challenges sources current source).toRing =
      ringFAdd current.toRing (ringFMul (challenges source) (sources source).toRing) := by
  dsimp only [witnessBlockStep]
  split_ifs with zero
  · have sourceZero : (sources source).toRing = ringFZero := funext zero
    rw [sourceZero, CarrierAction.ringFMul_zero_right]
    funext lane
    exact (ConcreteCarrier.baseLaws.add_zero _).symm
  · exact MaterializedRingF.toRing_ofRing _

/-- All 17 source prefixes in the existing fresh-then-running order. -/
def witnessBlockPartials (challenges : Fin SourceCount → RingF)
    (sources : Fin SourceCount → MaterializedRingF) : List MaterializedRingF :=
  scan (MaterializedRingF.ofRing ringFZero) (witnessBlockStep challenges sources)

private def semanticStep (challenges : Fin SourceCount → RingF)
    (sources : Fin SourceCount → MaterializedRingF)
    (current : RingF) (source : Fin SourceCount) : RingF :=
  ringFAdd current (ringFMul (challenges source) (sources source).toRing)

private theorem partials_semantics (challenges : Fin SourceCount → RingF)
    (sources : Fin SourceCount → MaterializedRingF) :
    (witnessBlockPartials challenges sources).map MaterializedRingF.toRing =
      scan ringFZero (semanticStep challenges sources) := by
  unfold witnessBlockPartials
  apply scan_map_hom
  · simp
  · intro current source
    exact witnessBlockStep_toRing challenges sources current source

private theorem assignmentBlock_combine_eq_foldr
    {shape : Phi81Relation.Shape}
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    ∀ {count : Nat} (challenges : Fin count → RingF)
      (assignments : Fin count → Assignment shape),
    CarrierAction.assignmentBlock
        (PiRLCFinite.combineAssignments challenges assignments) block =
      (List.finRange count).foldr
        (fun source current => ringFAdd
          (ringFMul (challenges source)
            (CarrierAction.assignmentBlock (assignments source) block)) current)
        ringFZero := by
  intro count
  induction count with
  | zero =>
      intro challenges assignments
      rfl
  | succ count ih =>
      intro challenges assignments
      simp only [PiRLCFinite.combineAssignments, List.finRange_succ,
        List.foldr_cons, List.foldr_map]
      change ringFAdd
        (CarrierAction.assignmentBlock
          (CarrierAction.act (challenges 0) (assignments 0)) block)
        (CarrierAction.assignmentBlock
          (PiRLCFinite.combineAssignments
            (fun index => challenges index.succ)
            (fun index => assignments index.succ)) block) = _
      rw [CarrierAction.assignmentBlock_act, ih]

private theorem ringFAdd_assoc (left middle right : RingF) :
    ringFAdd (ringFAdd left middle) right = ringFAdd left (ringFAdd middle right) := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_assoc _ _ _

private theorem ringFZero_add (value : RingF) : ringFAdd ringFZero value = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.zero_add _

private theorem ringFAdd_zero (value : RingF) : ringFAdd value ringFZero = value := by
  funext lane
  exact ConcreteCarrier.baseLaws.add_zero _

local instance : Std.Associative ringFAdd := ⟨ringFAdd_assoc⟩

local instance : Std.LawfulIdentity ringFAdd ringFZero where
  left_id := ringFZero_add
  right_id := ringFAdd_zero

/-- The executable block result is the exact typed PiRLC assignment block.
No source-validity or zero-tail premise is needed for this arithmetic identity. -/
theorem witnessBlockPartials_getLast?
    {shape : Phi81Relation.Shape}
    (challenges : Fin SourceCount → RingF)
    (assignments : Fin SourceCount → Assignment shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    ((witnessBlockPartials challenges (fun source =>
        MaterializedRingF.ofRing
          (CarrierAction.assignmentBlock (assignments source) block))).map
      MaterializedRingF.toRing).getLast? =
      some (CarrierAction.assignmentBlock
        (PiRLCFinite.combineAssignments challenges assignments) block) := by
  rw [partials_semantics, scan_getLast?]
  apply congrArg some
  unfold semanticStep
  simp only [MaterializedRingF.toRing_ofRing]
  rw [← List.foldl_map, ← List.foldr_eq_foldl, List.foldr_map]
  exact (assignmentBlock_combine_eq_foldr block challenges assignments).symm

/-- Prepare the 17 transcript-derived ring actions once, outside the block scan. -/
def prepareWitnessActions (challenges : Fin SourceCount → RingF) :
    FixedArray (FixedArray MaterializedRingF ringDegree) SourceCount :=
  FixedArray.ofFn fun source =>
    PiRLCWitnessAction.prepare (MaterializedRingF.ofRing (challenges source))

private def productOrFallback (challenge : RingF) (source : MaterializedRingF)
    (candidate : Option MaterializedRingF) : MaterializedRingF :=
  match candidate with
  | some product => product
  | none => MaterializedRingF.ofRing (ringFMul challenge source.toRing)

private def preparedProduct (challenge : RingF)
    (table : FixedArray MaterializedRingF ringDegree)
    (source : MaterializedRingF) : MaterializedRingF :=
  productOrFallback challenge source (PiRLCWitnessAction.multiplySigned table source)

private theorem productOrFallback_toRing (challenge : RingF)
    (source : MaterializedRingF) (candidate : Option MaterializedRingF)
    (accepted : Prop) [Decidable accepted]
    (correct : candidate.map MaterializedRingF.toRing =
      if accepted then some (ringFMul challenge source.toRing) else none) :
    (productOrFallback challenge source candidate).toRing =
      ringFMul challenge source.toRing := by
  cases candidate with
  | none => exact MaterializedRingF.toRing_ofRing _
  | some product =>
      change some product.toRing =
        (if accepted then some (ringFMul challenge source.toRing) else none) at correct
      by_cases success : accepted
      · rw [if_pos success] at correct
        exact Option.some.inj correct
      · rw [if_neg success] at correct
        cases correct

private theorem preparedProduct_toRing (challenge : RingF)
    (source : MaterializedRingF) :
    (preparedProduct challenge
      (PiRLCWitnessAction.prepare (MaterializedRingF.ofRing challenge)) source).toRing =
      ringFMul challenge source.toRing := by
  apply productOrFallback_toRing challenge source _
    (∀ lane : Fin ringDegree,
      source.toRing lane = 0 ∨ source.toRing lane = 1 ∨ source.toRing lane = -1)
  simpa only [MaterializedRingF.toRing_ofRing] using
    PiRLCWitnessAction.multiplySigned_correct (MaterializedRingF.ofRing challenge) source

/-- Use the stored action table for signed sources. The general ring product
is evaluated only in the rejected signed-unit branch of `preparedProduct`. -/
def preparedWitnessBlockStep (challenges : Fin SourceCount → RingF)
    (tables : FixedArray (FixedArray MaterializedRingF ringDegree) SourceCount)
    (sources : Fin SourceCount → MaterializedRingF)
    (current : MaterializedRingF) (source : Fin SourceCount) : MaterializedRingF :=
  let value := sources source
  if ∀ lane : Fin ringDegree, value.toRing lane = 0 then current
  else
    let product := preparedProduct (challenges source) (tables.get source) value
    MaterializedRingF.ofRing (ringFAdd current.toRing product.toRing)

/-- The prepared step has the same ring value as the original total step. -/
theorem preparedWitnessBlockStep_toRing (challenges : Fin SourceCount → RingF)
    (sources : Fin SourceCount → MaterializedRingF)
    (current : MaterializedRingF) (source : Fin SourceCount) :
    (preparedWitnessBlockStep challenges (prepareWitnessActions challenges)
      sources current source).toRing =
      (witnessBlockStep challenges sources current source).toRing := by
  rw [witnessBlockStep_toRing]
  dsimp only [preparedWitnessBlockStep]
  split_ifs with zero
  · have sourceZero : (sources source).toRing = ringFZero := funext zero
    rw [sourceZero, CarrierAction.ringFMul_zero_right]
    funext lane
    exact (ConcreteCarrier.baseLaws.add_zero _).symm
  · simp only [MaterializedRingF.toRing_ofRing, prepareWitnessActions,
      FixedArray.get_ofFn, preparedProduct_toRing]

/-- Replay all source prefixes using the caller's once-prepared tables. -/
def preparedWitnessBlockPartials (challenges : Fin SourceCount → RingF)
    (tables : FixedArray (FixedArray MaterializedRingF ringDegree) SourceCount)
    (sources : Fin SourceCount → MaterializedRingF) : List MaterializedRingF :=
  scan (MaterializedRingF.ofRing ringFZero)
    (preparedWitnessBlockStep challenges tables sources)

/-- Every prepared prefix agrees with the existing executable block scan. -/
theorem preparedWitnessBlockPartials_toRing (challenges : Fin SourceCount → RingF)
    (sources : Fin SourceCount → MaterializedRingF) :
    (preparedWitnessBlockPartials challenges (prepareWitnessActions challenges) sources).map
        MaterializedRingF.toRing =
      (witnessBlockPartials challenges sources).map MaterializedRingF.toRing := by
  rw [partials_semantics]
  unfold preparedWitnessBlockPartials
  apply scan_map_hom
  · simp
  · intro current source
    rw [preparedWitnessBlockStep_toRing, witnessBlockStep_toRing]
    rfl

/-- Prepared replay computes the same complete assignment block, with no
source-validity premise and no supplied expected witness. -/
theorem preparedWitnessBlockPartials_getLast?
    {shape : Phi81Relation.Shape}
    (challenges : Fin SourceCount → RingF)
    (assignments : Fin SourceCount → Assignment shape)
    (block : Fin (Phi81ColumnLayout.blockCount shape.carrierWidth)) :
    ((preparedWitnessBlockPartials challenges (prepareWitnessActions challenges)
      (fun source => MaterializedRingF.ofRing
        (CarrierAction.assignmentBlock (assignments source) block))).map
      MaterializedRingF.toRing).getLast? =
      some (CarrierAction.assignmentBlock
        (PiRLCFinite.combineAssignments challenges assignments) block) := by
  rw [preparedWitnessBlockPartials_toRing]
  exact witnessBlockPartials_getLast? challenges assignments block

end NightstreamFPrime.Export.Stage1.PiRLCWitnessBlock
