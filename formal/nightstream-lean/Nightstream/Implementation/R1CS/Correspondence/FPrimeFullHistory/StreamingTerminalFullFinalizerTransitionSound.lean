import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerAdvanceChainRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerDecodeRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound

/-!
Contract: compose the exact terminal advance-chain, advance-algebra, close,
and terminal-selector rows into the typed one-step terminal transition.

The complete theorem derives field-to-natural counter no-wrap facts from the
exact 16-bit decoder rows and advance matches. It does not own lifecycle phase
selection, leaf construction, or security reductions.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionSound

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.ProjectionIndexedRows
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerAdvanceChainRowSound

private abbrev artifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact

def roleLane : LinkRole → Fin 3
  | .operations => ⟨0, by decide⟩
  | .memory => ⟨1, by decide⟩

def roleRounds (role : LinkRole) : List Round :=
  (advanceChainLink (roleLane role)).recipe.trace.rounds

def roleConstants (role : LinkRole) : List Nat :=
  (advanceChainLink (roleLane role)).recipe.constantValues

def chainInputValues
    (role : LinkRole) (prior leaf : Digest) : List Nat :=
  roleConstants role ++ digestValues prior ++ digestValues leaf

def emittedChainLink
    (role : LinkRole) (prior leaf : Digest) : Digest := fun output =>
  fieldValue
    (runValueRounds (roleRounds role) (chainInputValues role prior leaf)
      (fun _ => 0)) output.val

def emittedHashSemantics : HashSemantics where
  chainLink := emittedChainLink
  operationsHeader := emittedOperationsHeader
  memoryHeader := emittedMemoryHeader

/-- The three exact leaf-digest column slices consumed by the chain rows. -/
def assignedLeaves (assignment : Nat → Nat) : LeafDigests :=
  fun lane output => fieldValue assignment
    ((advanceChainLink lane).recipe.payloadColumns.getD output.val 0)

theorem chain_constants_role_exact (lane : Fin 3) :
    (advanceChainLink lane).recipe.constantValues =
      roleConstants (chainRole lane) := by
  fin_cases lane <;> rfl

theorem chain_value_schedules_role_exact (lane : Fin 3) :
    valueSchedules (advanceChainLink lane).recipe.trace.rounds =
      valueSchedules (roleRounds (chainRole lane)) := by
  fin_cases lane <;> rfl

theorem chain_input_columns_exact (lane : Fin 3) :
    (advanceChainLink lane).recipe.inputColumns =
      (advanceChainLink lane).recipe.constantColumns ++
      (advanceChainLink lane).recipe.localColumns ++
      (advanceChainLink lane).recipe.payloadColumns := by
  fin_cases lane <;> rfl

theorem chain_local_columns_exact (lane : Fin 3) :
    (advanceChainLink lane).recipe.localColumns =
      List.ofFn fun output : Fin 4 =>
        artifact.openedLaneColumns.getD
          (34 + 4 * lane.val + output.val) 0 := by
  fin_cases lane <;> rfl

theorem chain_output_column_exact (lane : Fin 3) (output : Fin 4) :
    (advanceChainLink lane).recipe.outputColumns.getD output.val 0 =
      artifact.advancedLaneColumns.getD
        (34 + 4 * lane.val + output.val) 0 := by
  fin_cases lane <;> fin_cases output <;> rfl

theorem chain_payload_columns_length (lane : Fin 3) :
    (advanceChainLink lane).recipe.payloadColumns.length = 4 := by
  fin_cases lane <;> rfl

private theorem digest_values_at_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (columns : List Nat) (start : Nat) :
    digestValues (digestAt assignment columns start) =
      List.ofFn fun output : Fin 4 =>
        assignment (columns.getD (start + output.val) 0) := by
  have reduce : ∀ column,
      assignment column % goldilocksModulus = assignment column := by
    intro column
    exact Nat.mod_eq_of_lt (canonical column)
  simp [digestValues, digestAt, fieldAt, fieldValue, columnAt, reduce]

private theorem assigned_leaf_values_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (lane : Fin 3) :
    (advanceChainLink lane).recipe.payloadColumns.map assignment =
      digestValues (assignedLeaves assignment lane) := by
  have reduce : ∀ column,
      assignment column % goldilocksModulus = assignment column := by
    intro column
    exact Nat.mod_eq_of_lt (canonical column)
  let columns := (advanceChainLink lane).recipe.payloadColumns
  change columns.map assignment =
    List.ofFn (fun output : Fin 4 =>
      assignment (columns.getD output.val 0) % goldilocksModulus)
  simp_rw [reduce]
  have lengthExact : columns.length = 4 := by
    exact chain_payload_columns_length lane
  have reindexed :
      columns = List.ofFn (fun output : Fin 4 =>
        columns.get (Fin.cast lengthExact.symm output)) := by
    have transported := List.ofFn_congr lengthExact (List.get columns)
    rw [List.ofFn_get] at transported
    exact transported
  have pointwise :
      (fun output : Fin 4 =>
        assignment (columns.get (Fin.cast lengthExact.symm output))) =
      (fun output : Fin 4 =>
        assignment (columns.getD output.val 0)) := by
    funext output
    congr 1
    symm
    have outputLt : output.val < columns.length := by
      rw [lengthExact]
      exact output.isLt
    rw [List.getD_eq_getElem?_getD,
      getElem?_pos columns output.val outputLt, Option.getD_some]
    change columns.get ⟨output.val, outputLt⟩ =
      columns.get (Fin.cast lengthExact.symm output)
    apply congrArg (List.get columns)
    apply Fin.ext
    rfl
  calc
    columns.map assignment =
        (List.ofFn (fun output : Fin 4 =>
          columns.get (Fin.cast lengthExact.symm output))).map assignment :=
      congrArg (List.map assignment) reindexed
    _ = List.ofFn (fun output : Fin 4 =>
          assignment (columns.get (Fin.cast lengthExact.symm output))) :=
      (List.ofFn_comp'
        (fun output : Fin 4 =>
          columns.get (Fin.cast lengthExact.symm output)) assignment).symm
    _ = List.ofFn (fun output : Fin 4 =>
          assignment (columns.getD output.val 0)) :=
      congrArg List.ofFn pointwise

private theorem chain_local_values_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (lane : Fin 3) :
    (advanceChainLink lane).recipe.localColumns.map assignment =
      digestValues ((openedLane assignment).dSeen lane) := by
  rw [chain_local_columns_exact, List.map_ofFn]
  simpa [openedLane, laneAt] using
    (digest_values_at_exact assignment canonical
      artifact.openedLaneColumns (34 + 4 * lane.val)).symm

private theorem chain_input_values_exact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (lane : Fin 3)
    (sound : Sound lane assignment) :
    inputValues lane assignment =
      chainInputValues (chainRole lane)
        ((openedLane assignment).dSeen lane)
        (assignedLeaves assignment lane) := by
  rw [inputValues, chain_input_columns_exact, List.map_append,
    List.map_append, sound.constants, chain_constants_role_exact]
  rw [chain_local_values_exact assignment canonical lane,
    assigned_leaf_values_exact assignment canonical lane]
  rfl

theorem rows_imply_chain_output
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : ∀ lane, (advanceChainLink lane).Satisfied assignment)
    (lane : Fin 3) :
    (advancedLane assignment).dSeen lane =
      emittedHashSemantics.chainLink (chainRole lane)
        ((openedLane assignment).dSeen lane)
        (assignedLeaves assignment lane) := by
  have sound := rows_sound lane assignment canonical one
    (satisfied lane)
  have inputs := chain_input_values_exact assignment canonical lane sound
  have schedules := runValueRounds_eq_of_schedules
    (chain_value_schedules_role_exact lane)
    (chainInputValues (chainRole lane)
      ((openedLane assignment).dSeen lane)
      (assignedLeaves assignment lane)) (fun _ => 0)
  funext output
  apply Fin.ext
  have rowOutput := congrFun sound.hash output
  rw [assignedDigest, computedDigest, chain_output_column_exact, inputs]
    at rowOutput
  have scheduleOutput := congrFun schedules output.val
  have exactValue := rowOutput.trans scheduleOutput
  simpa [advancedLane, laneAt, digestAt, fieldAt, columnAt,
    emittedHashSemantics, emittedChainLink, fieldValue] using
      congrArg (fun value => value % goldilocksModulus) exactValue

def ChainOutputsExact (assignment : Nat → Nat) : Prop :=
  ∀ lane, (advancedLane assignment).dSeen lane =
    emittedHashSemantics.chainLink (chainRole lane)
      ((openedLane assignment).dSeen lane)
      (assignedLeaves assignment lane)

theorem rows_imply_chainOutputsExact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : ∀ lane, (advanceChainLink lane).Satisfied assignment) :
    ChainOutputsExact assignment := by
  intro lane
  exact rows_imply_chain_output assignment canonical one satisfied lane

private def zeroRow : Row := ⟨[], [], []⟩

private def advanceStepIndexRow : Row :=
  (indexedRowValues artifact.advanceAlgebraRows).getD 18 zeroRow

private theorem advance_step_index_row_exact :
    advanceStepIndexRow =
      outputLastLinearRow (artifact.advancedLaneColumns.getD 6 0)
        [(0, 1), (artifact.openedLaneColumns.getD 6 0, 1)] := by
  rfl

private theorem advance_step_index_row_mem :
    advanceStepIndexRow ∈ indexedRowValues artifact.advanceAlgebraRows := by
  norm_num [advanceStepIndexRow, indexedRowValues, artifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.advanceAlgebraRows]

theorem rows_imply_advance_stepIndex
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.AdvanceAlgebraSatisfied assignment)
    (noWrap :
      assignment (artifact.openedLaneColumns.getD 6 0) + 1 < goldilocksP) :
    (advancedLane assignment).stepIndex =
      (openedLane assignment).stepIndex + 1 := by
  have sourceHolds : RowHolds assignment advanceStepIndexRow :=
    satisfied advanceStepIndexRow advance_step_index_row_mem
  rw [advance_step_index_row_exact] at sourceHolds
  have builderHolds : RowHolds assignment
      (builderLinearRow (artifact.advancedLaneColumns.getD 6 0)
        [(0, 1), (artifact.openedLaneColumns.getD 6 0, 1)]) := by
    apply rowHolds_of_permutationEquivalent
      (source := outputLastLinearRow
        (artifact.advancedLaneColumns.getD 6 0)
        [(0, 1), (artifact.openedLaneColumns.getD 6 0, 1)])
      (reconstructed := builderLinearRow
        (artifact.advancedLaneColumns.getD 6 0)
        [(0, 1), (artifact.openedLaneColumns.getD 6 0, 1)])
    · refine ⟨?_, List.Perm.refl _, List.Perm.refl _⟩
      simpa [outputLastLinearRow, builderLinearRow] using
        (List.Perm.append_comm
          (negateTerms [(0, 1),
            (artifact.openedLaneColumns.getD 6 0, 1)])
          [(artifact.advancedLaneColumns.getD 6 0, 1)])
    · exact sourceHolds
  have equation := builderLinearRow_sound canonical one
    (artifact.advancedLaneColumns.getD 6 0)
    [(0, 1), (artifact.openedLaneColumns.getD 6 0, 1)]
    (by simp [CanonicalTerms, goldilocksP]) builderHolds
  have noWrap' :
      1 + assignment (artifact.openedLaneColumns.getD 6 0) <
        goldilocksP := by
    omega
  change assignment (artifact.advancedLaneColumns.getD 6 0) =
    assignment (artifact.openedLaneColumns.getD 6 0) + 1
  simp [lcEval, one] at equation
  exact equation.trans ((Nat.mod_eq_of_lt noWrap').trans
    (Nat.add_comm 1 _))

theorem rows_imply_advanceExact
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (advanceSatisfied : artifact.AdvanceAlgebraSatisfied assignment)
    (stepNoWrap :
      assignment (artifact.openedLaneColumns.getD 6 0) + 1 < goldilocksP)
    (chainOutputs : ChainOutputsExact assignment) :
    advancedLane assignment =
      advanceLane emittedHashSemantics (openedLane assignment)
        (stepInput assignment) (assignedLeaves assignment) := by
  apply Lane.ext
  · funext output
    fin_cases output <;> rfl
  · rfl
  · rfl
  · exact rows_imply_advance_stepIndex assignment canonical one
      advanceSatisfied stepNoWrap
  · rfl
  · funext index
    fin_cases index <;> rfl
  · funext index
    fin_cases index <;> rfl
  · funext index
    fin_cases index <;> rfl
  · funext lane output
    fin_cases lane <;> fin_cases output <;> rfl
  · funext lane
    exact chainOutputs lane
  · funext output
    fin_cases output <;> rfl

/-- All exact terminal row families imply the complete typed one-step
terminal transition. Decoder-owned 16-bit counters rule out field wrap. -/
theorem rows_imply_terminalTransition
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (decodeSatisfied : artifact.DecodeSatisfied assignment)
    (advanceAlgebraSatisfied : artifact.AdvanceAlgebraSatisfied assignment)
    (advanceChainSatisfied :
      ∀ lane, (advanceChainLink lane).Satisfied assignment)
    (closeSatisfied : artifact.CloseSatisfied assignment)
    (terminalClosed : artifact.TerminalClosedSatisfied assignment) :
    TerminalTransition emittedHashSemantics (openedLane assignment)
      (stepInput assignment) (assignedLeaves assignment)
      (finalLane assignment) := by
  have decodeSound :=
    StreamingTerminalFullFinalizerDecodeRowSound.rows_sound
      assignment canonical one decodeSatisfied
  have counterBounds :=
    StreamingTerminalFullFinalizerDecodeRowSound.counter_words_bound
      assignment decodeSound
  have stepMatches := rows_imply_stepMatches assignment canonical one
    advanceAlgebraSatisfied
  have openedStepBound : (openedLane assignment).stepIndex < 2 ^ 16 := by
    rw [← stepMatches.stepIndex]
    exact counterBounds.2
  have stepNoWrap :
      assignment (artifact.openedLaneColumns.getD 6 0) + 1 <
        goldilocksP := by
    change (openedLane assignment).stepIndex + 1 < goldilocksP
    have rangeBelowField : 2 ^ 16 + 1 < goldilocksP := by
      norm_num [goldilocksP]
    omega
  have advanceExact := rows_imply_advanceExact assignment canonical one
    advanceAlgebraSatisfied stepNoWrap
    (rows_imply_chainOutputsExact assignment canonical one
      advanceChainSatisfied)
  have openedSegmentBound :
      (openedLane assignment).segmentIndex < 2 ^ 16 := by
    rw [← stepMatches.segmentIndex]
    exact counterBounds.1
  have advancedSegmentBound :
      (advancedLane assignment).segmentIndex < 2 ^ 16 := by
    rw [advanceExact]
    simpa [advanceLane] using openedSegmentBound
  have segmentNoWrap :
      assignment (artifact.advancedLaneColumns.getD 5 0) + 1 <
        goldilocksP := by
    change (advancedLane assignment).segmentIndex + 1 < goldilocksP
    have rangeBelowField : 2 ^ 16 + 1 < goldilocksP := by
      norm_num [goldilocksP]
    omega
  refine {
    stepMatches := stepMatches
    closeChecks := ?_
    outputExact := ?_ }
  · rw [← advanceExact]
    exact rows_imply_closeChecks assignment canonical one closeSatisfied
      terminalClosed
  · rw [← advanceExact]
    exact rows_imply_outputExact assignment canonical one closeSatisfied
      terminalClosed segmentNoWrap emittedHashSemantics rfl rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionSound
