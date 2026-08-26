import NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalKCombination

/-!
Paper authority: SuperNeo v1.1, Section 7.4, verifier Step 1, the 14 separate
matrix evaluation equations. This leaf preserves `Eval_A` independently from
the Pad `Eval_K` family.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism

abbrev blockCount : Nat := productionShape.matrixCount

structure Interface where
  challenge : Nat → Fin CombinationFamily.sourceCount → Fin ringDegree → Expr
  input : Nat → Fin CombinationFamily.sourceCount →
    Fin productionShape.matrixCount →
    Fin productionShape.coefficientCount → KExpr

def ringInterface (interface : Interface) :
    RingKCombination.Interface blockCount where
  challenge := interface.challenge
  input := fun offset source matrix lane =>
    interface.input offset source matrix (EvalKCombination.coefficient lane)

def output (interface : Interface) (offset : Nat)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) : KExpr :=
  RingKCombination.output (ringInterface interface) offset matrix
    (Fin.cast EvalKCombination.coefficientCount_eq coefficient)

abbrev Assumptions (interface : Interface) (offset : Nat) (env : Env) :=
  RingKCombination.Assumptions (ringInterface interface) offset env

abbrev SpecHolds (interface : Interface) (offset : Nat) (env : Env) :=
  RingKCombination.SpecHolds (ringInterface interface) offset env

def evalChallenges (interface : Interface) (offset : Nat) (env : Env) :
    Fin 17 → RingF :=
  RingKCombination.evalChallenges (ringInterface interface) offset env

def evalInputs (interface : Interface) (offset : Nat) (env : Env) :
    Fin 17 → Fin productionShape.matrixCount → RingK :=
  fun source matrix lane =>
    (interface.input offset
      (Fin.cast CombinationFamily.sourceCount_eq.symm source) matrix
      (EvalKCombination.coefficient lane)).eval env

def evalOutput (interface : Interface) (offset : Nat) (env : Env) :
    Fin productionShape.matrixCount → RingK :=
  fun matrix lane =>
    (output interface offset matrix
      (EvalKCombination.coefficient lane)).eval env

theorem parentCoverage (interface : Interface) (offset : Nat) (env : Env)
    (specification : SpecHolds interface offset env) :
    evalOutput interface offset env = fun matrix =>
      PiRLCFinite.combineEvaluation
        (evalChallenges interface offset env)
        (fun source => evalInputs interface offset env source matrix) := by
  have all := RingKCombination.parentCoverage (ringInterface interface)
    offset env specification
  simpa [evalOutput, evalInputs, evalChallenges,
    RingKCombination.evalOutput, RingKCombination.evalInputs, output,
    ringInterface, EvalKCombination.coefficient] using all

theorem logicalPrivateCount_eq :
    CombinationFamily.logicalPrivateCount blockCount
      RingKCombination.cellCount = 25704 := by
  rw [CombinationFamily.logicalPrivateCount,
    CombinationFamily.sourceCount_eq]
  norm_num [CombinationFamily.stepSize, CombinationStep.privateCount,
    blockCount, RingKCombination.cellCount, productionShape,
    Phi81MatrixSource.phi81Shape, productionProfile, ringDegree]

theorem logicalRowCount_eq :
    CombinationFamily.logicalRowCount blockCount
      RingKCombination.cellCount = 25704 := by
  rw [CombinationFamily.logicalRowCount, logicalPrivateCount_eq]

def circuit (interface : Interface) : FormalCircuit :=
  RingKCombination.circuit (ringInterface interface)

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 25704 := by
  rw [(circuit interface).privateCount_eq offset]
  exact logicalPrivateCount_eq

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      25704 := by
  rw [(circuit interface).rowCount_eq offset]
  exact logicalRowCount_eq

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow (offset + 25704) := by
  simpa [circuit, RingKCombination.circuit, logicalPrivateCount_eq] using
    CombinationFamily.flatConstraints_varsBelow
      (RingKCombination.familyInterface (ringInterface interface))
      offset env assumptions

theorem soundness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (Circuit.ops (circuit interface).main offset)) :
    SpecHolds interface offset env :=
  RingKCombination.soundness (ringInterface interface) offset env assumptions rows

theorem complete (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  RingKCombination.complete (ringInterface interface) offset env assumptions

theorem completeness (interface : Interface) (offset : Nat) (env : Env)
    (assumptions : Assumptions interface offset env)
    (specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (Circuit.ops (circuit interface).main offset)) ∧
      holdsFlat completed (Circuit.ops (circuit interface).main offset) :=
  RingKCombination.completeness (ringInterface interface) offset env assumptions
    specification

end NightstreamFPrime.Lifecycle.PiRLC.v1_1.EvalACombination
