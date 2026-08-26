import NightstreamFPrime.Circuit.Quadratic
import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.PiRLCFinite

/-!
Paper authority: SuperNeo v1.1, Section 7.4, verifier Step 1, evaluation
equations `y_j = sum_i rho_i y_(i,j)`.

This reusable leaf proves that the generic two-cell circuit representation is
exactly the canonical action of `RingF` on `RingK`. `Eval_K` and `Eval_A`
instantiate only the block count; they do not duplicate this arithmetic.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

abbrev cellCount : Nat := 2

structure Interface (blockCount : Nat) where
  challenge : Nat → Fin CombinationFamily.sourceCount → Fin ringDegree → Expr
  input : Nat → Fin CombinationFamily.sourceCount → Fin blockCount →
    Fin ringDegree → KExpr

def kCell (cell : Fin cellCount) (value : K) : F :=
  if cell.val = 0 then value.c0 else value.c1

def ringKCell (cell : Fin cellCount) (value : RingK) : RingF :=
  fun lane => kCell cell (value lane)

def expressionCell (cell : Fin cellCount) (value : KExpr) : Expr :=
  if cell.val = 0 then value.c0 else value.c1

def familyInterface {blockCount : Nat} (interface : Interface blockCount) :
    CombinationFamily.Interface blockCount cellCount where
  challenge := interface.challenge
  input := fun offset source block lane cell =>
    expressionCell cell (interface.input offset source block lane)

def c0Cell : Fin cellCount := ⟨0, by decide⟩
def c1Cell : Fin cellCount := ⟨1, by decide⟩

def output {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (block : Fin blockCount) (lane : Fin ringDegree) : KExpr :=
  ⟨CombinationFamily.output (familyInterface interface) offset
      block lane c0Cell,
    CombinationFamily.output (familyInterface interface) offset
      block lane c1Cell⟩

abbrev Assumptions {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) :=
  CombinationFamily.Assumptions (familyInterface interface) offset env

abbrev SpecHolds {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) :=
  CombinationFamily.CanonicalHolds (familyInterface interface) offset env

def evalChallenges {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) : Fin 17 → RingF :=
  fun source lane =>
    (interface.challenge offset
      (Fin.cast CombinationFamily.sourceCount_eq.symm source) lane).eval env

def evalInputs {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) : Fin 17 → Fin blockCount → RingK :=
  fun source block lane =>
    (interface.input offset
      (Fin.cast CombinationFamily.sourceCount_eq.symm source) block lane).eval env

def evalOutput {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env) : Fin blockCount → RingK :=
  fun block lane => (output interface offset block lane).eval env

@[simp] private theorem kCell_zero (cell : Fin cellCount) :
    kCell cell K.zero = 0 := by
  unfold kCell K.zero
  split <;> rfl

@[simp] private theorem kCell_add (cell : Fin cellCount) (left right : K) :
    kCell cell (K.add left right) = kCell cell left + kCell cell right := by
  unfold kCell K.add
  split <;> rfl

@[simp] private theorem kCell_sub (cell : Fin cellCount) (left right : K) :
    kCell cell (K.sub left right) = kCell cell left - kCell cell right := by
  unfold kCell K.sub
  split <;> rfl

@[simp] private theorem kCell_mul_embed (cell : Fin cellCount)
    (scalar : F) (value : K) :
    kCell cell (K.mul (K.embed scalar) value) =
      scalar * kCell cell value := by
  unfold kCell K.mul K.embed
  split <;> simp

private theorem kCell_mul_coeff (cell : Fin cellCount)
    (challenge : RingF) (value : RingK) (left right : Nat) :
    kCell cell
        (K.mul
          (ringKCoeff (RingKAction.embedChallenge challenge) left)
          (ringKCoeff value right)) =
      ringFCoeff challenge left * ringFCoeff (ringKCell cell value) right := by
  unfold ringKCoeff ringFCoeff RingKAction.embedChallenge ringKCell
  split <;> split
  · exact kCell_mul_embed cell _ _
  · simp
  · unfold kCell K.mul K.zero
    split <;> simp
  · unfold kCell K.mul K.zero
    split <;> simp

private theorem foldl_raw_cell (cell : Fin cellCount)
    (indices : List Nat) (challenge : RingF) (value : RingK)
    (degree : Nat) (initialK : K) (initialF : F)
    (initialEq : kCell cell initialK = initialF) :
    kCell cell
        (indices.foldl (fun accumulated index =>
          if index ≤ degree ∧ degree - index < ringDegree then
            K.add accumulated
              (K.mul
                (ringKCoeff (RingKAction.embedChallenge challenge) index)
                (ringKCoeff value (degree - index)))
          else accumulated) initialK) =
      indices.foldl (fun accumulated index =>
        if index ≤ degree ∧ degree - index < ringDegree then
          accumulated + ringFCoeff challenge index *
            ringFCoeff (ringKCell cell value) (degree - index)
        else accumulated) initialF := by
  induction indices generalizing initialK initialF with
  | nil => exact initialEq
  | cons index rest inductionHypothesis =>
      simp only [List.foldl_cons]
      split
      · apply inductionHypothesis
        rw [kCell_add, kCell_mul_coeff, initialEq]
      · exact inductionHypothesis initialK initialF initialEq

private theorem rawMulCoeffK_cell (cell : Fin cellCount)
    (challenge : RingF) (value : RingK) (degree : Nat) :
    kCell cell
        (rawMulCoeffK (RingKAction.embedChallenge challenge) value degree) =
      rawMulCoeffF challenge (ringKCell cell value) degree := by
  unfold rawMulCoeffK rawMulCoeffF
  exact foldl_raw_cell cell (List.range ringDegree) challenge value degree
    K.zero 0 (kCell_zero cell)

/-- One canonical `RingK` action is exactly the two cellwise circuit actions. -/
theorem ringKMul_cell (cell : Fin cellCount)
    (challenge : RingF) (value : RingK) (lane : Fin ringDegree) :
    ringFMul challenge (ringKCell cell value) lane =
      kCell cell
        (ringKMul (RingKAction.embedChallenge challenge) value lane) := by
  unfold ringFMul ringKMul
  by_cases foldedLow : lane.val < ringMiddleDegree
  · by_cases hasTwice : lane.val + 81 ≤ 106
    · simp [foldedLow, hasTwice, rawMulCoeffK_cell]
    · simp [foldedLow, hasTwice, rawMulCoeffK_cell]
  · by_cases hasTwice : lane.val + 81 ≤ 106
    · simp [foldedLow, hasTwice, rawMulCoeffK_cell]
    · simp [foldedLow, hasTwice, rawMulCoeffK_cell]

private theorem rightCombination_eq_combineEvaluation_cell
    {count : Nat} (challenges : Fin count → RingF)
    (values : Fin count → RingK) (cell : Fin cellCount) :
    CombinationFamily.rightCombination
        (fun source => ringFMul (challenges source)
          (ringKCell cell (values source))) =
      ringKCell cell (PiRLCFinite.combineEvaluation challenges values) := by
  induction count with
  | zero =>
      funext lane
      change 0 = kCell cell K.zero
      exact (kCell_zero cell).symm
  | succ count inductionHypothesis =>
      funext lane
      change
        ringFMul (challenges 0) (ringKCell cell (values 0)) lane +
            CombinationFamily.rightCombination
              (fun source => ringFMul (challenges source.succ)
                (ringKCell cell (values source.succ))) lane =
          kCell cell
            (ringKAdd
              (ringKMul (RingKAction.embedChallenge (challenges 0))
                (values 0))
              (PiRLCFinite.combineEvaluation
                (fun source => challenges source.succ)
                (fun source => values source.succ)) lane)
      change
        ringFMul (challenges 0) (ringKCell cell (values 0)) lane +
            CombinationFamily.rightCombination
              (fun source => ringFMul (challenges source.succ)
                (ringKCell cell (values source.succ))) lane =
          kCell cell
            (K.add
              (ringKMul (RingKAction.embedChallenge (challenges 0))
                (values 0) lane)
              (PiRLCFinite.combineEvaluation
                (fun source => challenges source.succ)
                (fun source => values source.succ) lane))
      rw [kCell_add, ← ringKMul_cell]
      exact congrArg (fun suffix =>
        ringFMul (challenges 0) (ringKCell cell (values 0)) lane + suffix)
        (congrFun
          (inductionHypothesis
            (fun source => challenges source.succ)
            (fun source => values source.succ)) lane)

theorem parentCoverage {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalOutput interface offset env = fun block =>
      PiRLCFinite.combineEvaluation
        (evalChallenges interface offset env)
        (fun source => evalInputs interface offset env source block) := by
  funext block lane
  have c0Result := congrFun (specification block c0Cell) lane
  have c1Result := congrFun (specification block c1Cell) lane
  apply congrArg₂ K.mk
  · calc
      (evalOutput interface offset env block lane).c0 =
          CombinationFamily.orderedCombination (familyInterface interface)
            offset env block c0Cell lane := by
        simpa [evalOutput, output, CombinationFamily.evalOutput] using c0Result
      _ = (PiRLCFinite.combineEvaluation
            (evalChallenges interface offset env)
            (fun source => evalInputs interface offset env source block) lane).c0 := by
        simpa [CombinationFamily.orderedCombination,
          CombinationFamily.term, CombinationFamily.challengeValue,
          CombinationFamily.inputValue, familyInterface, expressionCell,
          evalChallenges, evalInputs, ringKCell, kCell, c0Cell] using
          congrFun (rightCombination_eq_combineEvaluation_cell
            (evalChallenges interface offset env)
            (fun source => evalInputs interface offset env source block)
            c0Cell) lane
  · calc
      (evalOutput interface offset env block lane).c1 =
          CombinationFamily.orderedCombination (familyInterface interface)
            offset env block c1Cell lane := by
        simpa [evalOutput, output, CombinationFamily.evalOutput] using c1Result
      _ = (PiRLCFinite.combineEvaluation
            (evalChallenges interface offset env)
            (fun source => evalInputs interface offset env source block) lane).c1 := by
        simpa [CombinationFamily.orderedCombination,
          CombinationFamily.term, CombinationFamily.challengeValue,
          CombinationFamily.inputValue, familyInterface, expressionCell,
          evalChallenges, evalInputs, ringKCell, kCell, c1Cell] using
          congrFun (rightCombination_eq_combineEvaluation_cell
            (evalChallenges interface offset env)
            (fun source => evalInputs interface offset env source block)
            c1Cell) lane

def circuit {blockCount : Nat} (interface : Interface blockCount) :
    FormalCircuit :=
  CombinationFamily.circuit (familyInterface interface)

theorem soundness {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  CombinationFamily.soundness (familyInterface interface) offset env
    assumptions rows

theorem complete {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  CombinationFamily.complete (familyInterface interface) offset env assumptions

theorem completeness {blockCount : Nat} (interface : Interface blockCount)
    (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  CombinationFamily.completeness (familyInterface interface) offset env
    assumptions specification

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.RingKCombination
