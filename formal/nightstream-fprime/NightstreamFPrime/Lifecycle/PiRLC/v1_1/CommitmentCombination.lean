import NightstreamFPrime.Lifecycle.PiRLC.v1_1.CombinationFamily
import NightstreamFPrime.Spec.Phi81Relation.PiRLCAlgebra.Commitment

/-!
Paper authority: SuperNeo v1.1, Section 7.4, verifier Step 1, equation
`c = sum_i rho_i c_i`.

This leaf instantiates the generic 17-source Phi81 combination for all 22
Ajtai commitment rows. It owns no transcript, public-input, evaluation, or
output-claim wiring.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Phi81Relation

def blockCount : Nat := productionProfile.commitmentWidth
abbrev cellCount : Nat := 1

structure Interface where
  challenge : Nat → Fin CombinationFamily.sourceCount → Fin ringDegree → Expr
  input : Nat → Fin CombinationFamily.sourceCount → Fin blockCount →
    Fin ringDegree → Expr

def familyInterface (interface : Interface) :
    CombinationFamily.Interface blockCount cellCount where
  challenge := interface.challenge
  input := fun offset source block lane _ =>
    interface.input offset source block lane

def cell : Fin cellCount := ⟨0, by decide⟩

def output (interface : Interface) (offset : Nat)
    (row : Fin blockCount) (lane : Fin ringDegree) : Expr :=
  CombinationFamily.output (familyInterface interface) offset row lane cell

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) :=
  CombinationFamily.Assumptions (familyInterface interface) offset env

abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) :=
  CombinationFamily.CanonicalHolds (familyInterface interface) offset env

def evalChallenges (interface : Interface) (offset : Nat) (env : Env) :
    Fin 17 → RingF :=
  fun source lane =>
    (interface.challenge offset
      (Fin.cast CombinationFamily.sourceCount_eq.symm source) lane).eval env

def evalInputs (interface : Interface) (offset : Nat) (env : Env) :
    Fin 17 → PiRLCAlgebra.Commitment.Value blockCount :=
  fun source row lane =>
    (interface.input offset
      (Fin.cast CombinationFamily.sourceCount_eq.symm source) row lane).eval env

def evalOutput (interface : Interface) (offset : Nat) (env : Env) :
    PiRLCAlgebra.Commitment.Value blockCount :=
  fun row lane => (output interface offset row lane).eval env

private theorem rightCombination_eq_combineCommitments
    {count : Nat} (challenges : Fin count → RingF)
    (values : Fin count → PiRLCAlgebra.Commitment.Value blockCount)
    (row : Fin blockCount) :
    CombinationFamily.rightCombination
        (fun source => ringFMul (challenges source) (values source row)) =
      PiRLCAlgebra.Commitment.combineCommitments challenges values row := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      change ringFAdd
          (ringFMul (challenges 0) (values 0 row))
          (CombinationFamily.rightCombination fun source =>
            ringFMul (challenges source.succ) (values source.succ row)) =
        ringFAdd
          (ringFMul (challenges 0) (values 0 row))
          (PiRLCAlgebra.Commitment.combineCommitments
            (fun source => challenges source.succ)
            (fun source => values source.succ) row)
      rw [inductionHypothesis]

/-- The leaf's canonical result is exactly the paper commitment equation. -/
theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalOutput interface offset env =
      PiRLCAlgebra.Commitment.combineCommitments
        (evalChallenges interface offset env)
        (evalInputs interface offset env) := by
  funext row lane
  have familyResult := specification row cell
  have laneResult := congrFun familyResult lane
  calc
    evalOutput interface offset env row lane =
        CombinationFamily.orderedCombination (familyInterface interface)
          offset env row cell lane := by
      simpa [evalOutput, output, CombinationFamily.evalOutput] using laneResult
    _ = PiRLCAlgebra.Commitment.combineCommitments
          (evalChallenges interface offset env)
          (evalInputs interface offset env) row lane := by
      exact congrFun
        (rightCombination_eq_combineCommitments
          (evalChallenges interface offset env)
          (evalInputs interface offset env) row) lane

theorem logicalPrivateCount_eq :
    CombinationFamily.logicalPrivateCount blockCount cellCount = 20196 := by
  rw [CombinationFamily.logicalPrivateCount, CombinationFamily.sourceCount_eq]
  norm_num [CombinationFamily.stepSize, CombinationStep.privateCount, blockCount,
    cellCount, productionProfile, ringDegree]

theorem logicalRowCount_eq :
    CombinationFamily.logicalRowCount blockCount cellCount = 20196 := by
  rw [CombinationFamily.logicalRowCount, logicalPrivateCount_eq]

def circuit (interface : Interface) : FormalCircuit :=
  CombinationFamily.circuit (familyInterface interface)

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 20196 := by
  rw [(circuit interface).privateCount_eq offset]
  exact logicalPrivateCount_eq

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      20196 := by
  rw [(circuit interface).rowCount_eq offset]
  exact logicalRowCount_eq

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow (offset + 20196) := by
  simpa [circuit, logicalPrivateCount_eq] using
    CombinationFamily.flatConstraints_varsBelow (familyInterface interface)
      offset env assumptions

theorem soundness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  CombinationFamily.soundness (familyInterface interface) offset env assumptions rows

theorem complete (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  CombinationFamily.complete (familyInterface interface) offset env assumptions

theorem completeness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  CombinationFamily.completeness (familyInterface interface) offset env assumptions
    specification

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.CommitmentCombination
