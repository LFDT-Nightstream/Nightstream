import NightstreamFPrime.Export.Stage1.Package
import CompressionLean.Terminal.Data

/-!
Owns the projection of terminal-package satisfaction onto the F′ component
families: the prior hash chain, the permutation invocations, the compact
PiRLC rows, and the ordinary rows.

Each theorem only proves membership of an F′ component in the terminal
package. The family lemmas of F′ give the meaning of the rows.
-/

namespace CompressionLean.Terminal

open NightstreamFPrime.Export
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout

theorem circuitPackage_witnessInstructions :
    (circuitPackage ()).witnessInstructions =
      Data.liftPilotInstructions (PilotData.witnessInstructions ()) ++
        Rows.witnessInstructionsTR (arithmeticRows ()) := by
  rfl

theorem circuitPackage_assertionRows :
    (circuitPackage ()).assertionRows =
      priorAssertions () ++ Rows.assertionRowsTR (arithmeticRows ()) := by
  rfl

/-- The pilot column lift, as F′ uses it. -/
def pilotEnv (env : Env) : Env :=
  fun column => env (NightstreamFPrime.Layout.Stage1.Spartan.liftPilotColumn column)

private theorem liftPilotCombination_eval
    (combination : SparseCombination) (env : Env) :
    (Data.liftPilotCombination combination).eval env =
      combination.eval (pilotEnv env) := by
  unfold Data.liftPilotCombination Data.liftPilotTerm
    SparseCombination.eval pilotEnv
  rw [List.map_map]
  rfl

private theorem liftPilotRow_holds (row : SparseRow) (env : Env) :
    (Data.liftPilotRow row).Holds env ↔ row.Holds (pilotEnv env) := by
  unfold Data.liftPilotRow SparseRow.Holds
  rw [liftPilotCombination_eval, liftPilotCombination_eval,
    liftPilotCombination_eval]

private theorem liftPilotInstruction_holds
    (instruction : WitnessInstruction) (env : Env) :
    (Data.liftPilotInstruction instruction).Holds env ↔
      instruction.Holds (pilotEnv env) := by
  unfold Data.liftPilotInstruction WitnessInstruction.Holds
  rw [liftPilotCombination_eval, liftPilotCombination_eval]
  rfl

/-- The prior-state rows of the pilot hold on the lifted columns. -/
theorem priorExtraRows_hold (env : Env)
    (holds : (circuitPackage ()).RowsHold env) :
    R1CS.RowsHold (pilotEnv env)
      ((PilotData.priorExtraRows ()).map Rows.CompiledRow.toR1CS) := by
  apply (Rows.compiledRows_hold_iff (PilotData.priorExtraRows ())
    (pilotEnv env)).mpr
  constructor
  · intro instruction member
    apply (liftPilotInstruction_holds instruction env).mp
    apply holds.2.2.2.1
    rw [circuitPackage_witnessInstructions]
    apply List.mem_append_left
    apply Data.liftPilotInstruction_mem
    rw [PilotData.witnessInstructions, Rows.witnessInstructionsTR_eq]
    exact member
  · intro assertion member
    apply (liftPilotRow_holds assertion env).mp
    apply holds.2.2.2.2
    rw [circuitPackage_assertionRows]
    apply List.mem_append_left
    rw [priorAssertions, Data.liftPilotRows, Rows.assertionRowsTR_eq]
    exact List.mem_map_of_mem member

/-- The ordinary PiCCS and PiRLC sampler rows hold. -/
theorem arithmeticRows_hold (env : Env)
    (holds : (circuitPackage ()).RowsHold env) :
    R1CS.RowsHold env ((arithmeticRows ()).map Rows.CompiledRow.toR1CS) := by
  apply (Rows.compiledRows_hold_iff (arithmeticRows ()) env).mpr
  constructor
  · intro instruction member
    apply holds.2.2.2.1 instruction
    rw [circuitPackage_witnessInstructions]
    apply List.mem_append_right
    rw [Rows.witnessInstructionsTR_eq]
    exact member
  · intro assertion member
    apply holds.2.2.2.2 assertion
    rw [circuitPackage_assertionRows]
    apply List.mem_append_right
    rw [Rows.assertionRowsTR_eq]
    exact member

theorem piCcsArithmeticRows_hold (env : Env)
    (holds : (circuitPackage ()).RowsHold env) :
    R1CS.RowsHold env
      ((PiCCSArithmetic.arithmeticRows Data.logicalWidth Data.publicFits).map
        Rows.CompiledRow.toR1CS) := by
  have combined := arithmeticRows_hold env holds
  rw [arithmeticRows, List.map_append] at combined
  exact ((R1CS.rowsHold_append env _ _).mp combined).1

theorem piRlcSamplerOrdinaryRows_hold (env : Env)
    (holds : (circuitPackage ()).RowsHold env) :
    R1CS.RowsHold env
      ((PiRLCSamplerOrdinaryRows.rows (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)).map Rows.CompiledRow.toR1CS) := by
  have combined := arithmeticRows_hold env holds
  rw [arithmeticRows, List.map_append] at combined
  exact ((R1CS.rowsHold_append env _ _).mp combined).2

/-- Every F′ Stage 1 permutation invocation holds with the one canonical
Poseidon2 template. -/
theorem permutationInvocation_holds (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (invocation : PermutationInvocation)
    (member : invocation ∈ Data.permutationInvocations ()) :
    PermutationInvocationHolds (PilotData.circuitPackage ()) invocation env :=
  holds.2.1 invocation member

/-- The prior hash chain holds with the canonical template. -/
theorem priorChain_holds (env : Env)
    (holds : (circuitPackage ()).RowsHold env) :
    HashChainHolds (PilotData.circuitPackage ()) Data.priorChain env :=
  holds.1 Data.priorChain (List.mem_singleton_self _)

/-- The compact rows of every PiRLC combination invocation hold. -/
theorem compactInvocation_holds (env : Env)
    (holds : (circuitPackage ()).RowsHold env)
    (invocation : CompactRowInvocation)
    (member : invocation ∈ Data.compactRowInvocations ()) :
    CompactRowInvocationHolds (circuitPackage ()) invocation env :=
  holds.2.2.1 invocation member

end CompressionLean.Terminal
