import Mathlib.Tactic.NormNum.Basic
import NightstreamFPrime.Lifecycle.PiRLC.v1_2.EvalKCombination

/-!
Paper authority: SuperNeo v1.2, Section 7.4, verifier Step 1, the separate
matrix evaluation equations.
-/

namespace NightstreamFPrime.Lifecycle.PiRLC.v1_2.EvalACombination

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.RingRelation.EvaluationHomomorphism

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
    Fin 15 → RingF :=
  RingKCombination.evalChallenges (ringInterface interface) offset env

def evalInputs (interface : Interface) (offset : Nat) (env : Env) :
    Fin 15 → Fin productionShape.matrixCount → RingK :=
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
    ringInterface, EvalKCombination.coefficient] using! all

theorem logicalPrivateCount_eq :
    CombinationFamily.logicalPrivateCount blockCount
      RingKCombination.cellCount = 7680 := by
  rw [CombinationFamily.logicalPrivateCount,
    CombinationFamily.sourceCount_eq]
  norm_num [CombinationFamily.stepSize, CombinationStep.privateCount,
    blockCount, RingKCombination.cellCount, productionShape,
    RingMatrixSource.ringShape, productionProfile, ringDegree]

theorem logicalRowCount_eq :
    CombinationFamily.logicalRowCount blockCount
      RingKCombination.cellCount = 7680 := by
  rw [CombinationFamily.logicalRowCount, logicalPrivateCount_eq]

def circuit (interface : Interface) : FormalCircuit :=
  RingKCombination.circuit (ringInterface interface)

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (Circuit.ops (circuit interface).main offset) = 7680 := by
  rw [(circuit interface).privateCount_eq offset]
  exact logicalPrivateCount_eq

theorem flatConstraints_length (interface : Interface) (offset : Nat) :
    (flatConstraints (Circuit.ops (circuit interface).main offset)).length =
      7680 := by
  rw [(circuit interface).rowCount_eq offset]
  exact logicalRowCount_eq

theorem flatConstraints_varsBelow (interface : Interface) (offset : Nat)
    (env : Env) (assumptions : Assumptions interface offset env) :
    ∀ expression ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      expression.VarsBelow (offset + 7680) := by
  simpa [circuit, RingKCombination.circuit, logicalPrivateCount_eq] using!
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

/-! ### The exporter's blocks (SPEC.md section 4.4) -/

theorem marks_nil (interface : Interface) (offset : Nat) :
    marks (Circuit.ops (circuit interface).main offset) = [] :=
  RingKCombination.marks_nil (ringInterface interface) offset

/-- The rows read the challenges, both components of the inputs and the gadget's own locals. -/
theorem free (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (challenge : ∀ source lane, (interface.challenge offset source lane).VarsSatisfy live)
    (input : ∀ source matrix coefficient,
      (interface.input offset source matrix coefficient).c0.VarsSatisfy live ∧
        (interface.input offset source matrix coefficient).c1.VarsSatisfy live)
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) →
        live column) :
    ∀ row ∈ flatConstraints (Circuit.ops (circuit interface).main offset),
      row.VarsSatisfy live :=
  RingKCombination.free (ringInterface interface) offset challenge
    (fun source matrix lane => input source matrix (EvalKCombination.coefficient lane)) locals

theorem output_live (interface : Interface) (offset : Nat) {live : Nat → Prop}
    (locals : ∀ column, offset ≤ column →
      column < offset + localLength (Circuit.ops (circuit interface).main offset) →
        live column)
    (matrix : Fin productionShape.matrixCount)
    (coefficient : Fin productionShape.coefficientCount) :
    (output interface offset matrix coefficient).c0.VarsSatisfy live ∧
      (output interface offset matrix coefficient).c1.VarsSatisfy live :=
  RingKCombination.output_live (ringInterface interface) offset locals matrix _

end NightstreamFPrime.Lifecycle.PiRLC.v1_2.EvalACombination
