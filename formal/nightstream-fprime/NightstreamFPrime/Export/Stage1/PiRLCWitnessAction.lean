import NightstreamFPrime.Export.Stage1.PiRLCPartialTrace
import Mathlib.Tactic.SplitIfs

/-!
Stored Phi81 basis actions for the actual PiRLC challenges. Signed-unit
source blocks use table reads, additions and negations. The table is computed
from the challenge; its result is proved equal to the existing ring product.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiRLCWitnessAction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace

/-- Compute all 54 basis images once for one transcript-derived challenge. -/
def prepare (challenge : MaterializedRingF) : FixedArray MaterializedRingF ringDegree :=
  FixedArray.ofFn fun input =>
    MaterializedRingF.ofRing <|
      ringFMul challenge.toRing (ringFMonomial input.val 1)

private def signedTerm (coefficient value : F) : F :=
  if coefficient = 0 then 0 else if coefficient = 1 then value else -value

private theorem signedTerm_eq_mul (coefficient value : F)
    (signed : coefficient = 0 ∨ coefficient = 1 ∨ coefficient = -1) :
    signedTerm coefficient value = value * coefficient := by
  rcases signed with zero | positive | negative
  · subst coefficient
    simp only [signedTerm]
    exact (ConcreteCarrier.baseLaws.mul_zero value).symm
  · subst coefficient
    simp only [signedTerm, if_neg (show (1 : F) ≠ 0 by decide)]
    exact (ConcreteCarrier.baseLaws.mul_one value).symm
  · subst coefficient
    simp only [signedTerm, if_neg (show (-1 : F) ≠ 0 by decide),
      if_neg (show (-1 : F) ≠ 1 by decide)]
    calc
      -value = -(1 * value) :=
        congrArg (fun entry : F => -entry) (ConcreteCarrier.baseLaws.one_mul value).symm
      _ = (-1) * value := (ConcreteCarrier.baseLaws.neg_mul 1 value).symm
      _ = value * (-1) := ConcreteCarrier.baseLaws.mul_comm _ _

private theorem foldl_eq_sumRange (term : Nat → F) (count : Nat) :
    (List.range count).foldl (fun acc input => acc + term input) 0 =
      sumRange ConcreteCarrier.baseOps count term := by
  induction count with
  | zero => rfl
  | succ count ih =>
      simpa only [List.range_succ, List.foldl_append, List.foldl_cons,
        List.foldl_nil, sumRange] using
          congrArg (fun value : F => value + term count) ih

private def applySigned (table : FixedArray MaterializedRingF ringDegree)
    (source : MaterializedRingF) : MaterializedRingF :=
  MaterializedRingF.ofRing fun output =>
    (List.range ringDegree).foldl (fun acc input =>
      acc + if live : input < ringDegree then
        signedTerm (source.toRing ⟨input, live⟩)
          ((table.get ⟨input, live⟩).toRing output)
      else 0) 0

/-- Reject any coefficient outside {-1, 0, 1} before using the signed action. -/
def multiplySigned (table : FixedArray MaterializedRingF ringDegree)
    (source : MaterializedRingF) : Option MaterializedRingF :=
  if ∀ lane : Fin ringDegree,
      source.toRing lane = 0 ∨ source.toRing lane = 1 ∨ source.toRing lane = -1 then
    some (applySigned table source)
  else none

private theorem applySigned_toRing (challenge source : MaterializedRingF)
    (signed : ∀ lane : Fin ringDegree,
      source.toRing lane = 0 ∨ source.toRing lane = 1 ∨ source.toRing lane = -1) :
    (applySigned (prepare challenge) source).toRing =
      ringFMul challenge.toRing source.toRing := by
  unfold applySigned
  rw [MaterializedRingF.toRing_ofRing]
  funext output
  rw [foldl_eq_sumRange, CarrierAction.ringFMul_apply_eq_rightLinear]
  apply sumRange_congr
  intro input live
  simp only [dif_pos live, prepare, FixedArray.get_ofFn,
    MaterializedRingF.toRing_ofRing, CarrierAction.rightCoefficient]
  exact signedTerm_eq_mul _ _ (signed ⟨input, live⟩)

/-- Exact success and rejection behavior for the computed basis table.
The arithmetic equality uses the existing right-input linearization theorem. -/
theorem multiplySigned_correct (challenge source : MaterializedRingF) :
    (multiplySigned (prepare challenge) source).map MaterializedRingF.toRing =
      if ∀ lane : Fin ringDegree,
          source.toRing lane = 0 ∨ source.toRing lane = 1 ∨ source.toRing lane = -1 then
        some (ringFMul challenge.toRing source.toRing)
      else none := by
  unfold multiplySigned
  split_ifs with signed
  · simp only [Option.map_some, applySigned_toRing challenge source signed]
  · rfl

end NightstreamFPrime.Export.Stage1.PiRLCWitnessAction
