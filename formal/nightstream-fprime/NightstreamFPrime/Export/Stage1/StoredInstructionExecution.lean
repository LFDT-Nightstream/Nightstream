import NightstreamFPrime.Export.Stage1.StoredWitnessExecution
import NightstreamFPrime.Export.RowSemantics

/-!
Stored execution of the existing ordinary witness instruction.
Row satisfaction retains the owner's input-avoidance premises.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.StoredInstructionExecution

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package

private theorem combination_eval (combination : SparseCombination) (env : Env) :
    combination.toR1CS.eval env = combination.eval env := by
  simp [SparseCombination.toR1CS, SparseCombination.eval,
    Layout.R1CS.LinearCombination.eval, List.map_map, Function.comp_def]

@[inline] def execute (instruction : WitnessInstruction) (values : Array F) : Array F :=
  StoredWitnessExecution.write values instruction.target
    (instruction.a.toR1CS.eval (StoredWitnessExecution.asEnv values) *
      instruction.b.toR1CS.eval (StoredWitnessExecution.asEnv values))

theorem execute_size (instruction : WitnessInstruction) (values : Array F) :
    (execute instruction values).size = values.size := by
  unfold execute
  exact StoredWitnessExecution.write_size _ _ _

theorem execute_eq (instruction : WitnessInstruction) (values : Array F)
    (bounded : instruction.target < values.size) :
    StoredWitnessExecution.asEnv (execute instruction values) =
      instruction.execute (StoredWitnessExecution.asEnv values) := by
  unfold execute
  rw [StoredWitnessExecution.asEnv_write _ _ _ bounded]
  unfold WitnessInstruction.execute
  rw [combination_eval, combination_eval]

end NightstreamFPrime.Export.Stage1.StoredInstructionExecution
