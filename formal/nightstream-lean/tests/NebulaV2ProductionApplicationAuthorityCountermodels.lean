import Nightstream.Implementation.NebulaV2.ProductionRecursiveSuccessorRowsFor
import Nightstream.Implementation.NebulaV2.ProductionSemanticAuthority
import tests.NebulaV2WasmState

/-!
Hostile examples for the generated application-relation boundary.

These examples prove that successor output placement does not establish an
application transition or a memory-port link. They also record why a theorem
of the form `forall supplement, ...` cannot establish supplement existence.
-/

set_option autoImplicit false

namespace tests.NebulaV2ProductionApplicationAuthorityCountermodels

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionRecursiveSuccessorRowsFor
open Nightstream.Protocol.NebulaV2.ApplicationBatch
open Nightstream.Protocol.NebulaV2.ProductionProfileCandidates
open Nightstream.Protocol.NebulaV2.WasmState

def identityMachine : Machine Unit where
  step := fun _ state _ => some state

theorem accessRowAccepted :
    identityMachine.semantics.active () tests.NebulaV2WasmState.running
      tests.NebulaV2WasmState.terminalRow
      tests.NebulaV2WasmState.running := by
  refine ⟨tests.NebulaV2WasmState.validRunning,
    tests.NebulaV2WasmState.validRunning, ?_, ?_, ?_, rfl⟩
  · simp [AppStateVector.TerminalReady, tests.NebulaV2WasmState.running]
  · simp [AppStateVector.TerminalReady, tests.NebulaV2WasmState.running]
  · intro halted
    simp [tests.NebulaV2WasmState.running] at halted

/-- The same canonical state and normalized row can satisfy one selected
machine and fail another. Row shape, state placement, and public identifiers
therefore cannot replace a verifier-owned application transition relation. -/
theorem same_row_shape_does_not_imply_selected_machine_transition :
    identityMachine.semantics.active () tests.NebulaV2WasmState.running
        tests.NebulaV2WasmState.terminalRow
        tests.NebulaV2WasmState.running /\
      ¬ (ProductionSemanticAuthority.RejectMachine Unit).semantics.active ()
        tests.NebulaV2WasmState.running tests.NebulaV2WasmState.terminalRow
        tests.NebulaV2WasmState.running := by
  refine ⟨accessRowAccepted, ?_⟩
  intro accepted
  have impossible := accepted.2.2.2.2.2
  simp [ProductionSemanticAuthority.RejectMachine] at impossible

def accessBatch : Batch .e1 identityMachine ()
    tests.NebulaV2WasmState.running tests.NebulaV2WasmState.running where
  rows :=
    [ .active tests.NebulaV2WasmState.terminalRow
    , .active tests.NebulaV2WasmState.terminalRow
    , .active tests.NebulaV2WasmState.terminalRow
    ]
  rowsExact := by decide
  run := .cons (.active accessRowAccepted)
    (.cons (.active accessRowAccepted)
      (.cons (.active accessRowAccepted) (.nil _)))

def layout : Layout 1 where
  successor := { start := 1000, startPositive := by decide }
  successorHashBase := 2000
  preCarryDigestBase := 2500
  applicationRowCountColumn := 1
  applicationStateColumn := fun index => 2 + index.val
  nifsOutputColumn := fun index => 3000 + index.val
  invocationValueBitStart := 4000
  invocationSlackColumn := 5000
  invocationSlackBitStart := 6000
  realRowsValueBitStart := 7000
  realRowsSlackColumn := 8000
  realRowsSlackBitStart := 9000

def assignment (column : Nat) : Nat :=
  if column = 1 then 3 else 0

private theorem runningFieldsZero :
    ProductionWasmStateFields.encode
        (Nightstream.Protocol.NebulaV2.WasmStateEncoding.encode
          tests.NebulaV2WasmState.running) =
      List.replicate 85 0 := by
  decide

/-- The narrow placement record can be manufactured for a semantic batch
without checking any application-compiler row or any memory port. -/
theorem output_placement_accepts_unlinked_accesses :
    ApplicationProducerPlaced layout assignment accessBatch /\
      accesses accessBatch.rows ≠ [] := by
  constructor
  · constructor
    · decide
    · intro index
      have columnNotCount : 2 + index.val ≠ 1 := by omega
      simp only [layout, assignment, columnNotCount, if_false]
      let encoded := ProductionWasmStateFields.encode
        (Nightstream.Protocol.NebulaV2.WasmStateEncoding.encode
          tests.NebulaV2WasmState.running)
      let coordinate : Fin encoded.length := Fin.cast
        (ProductionWasmStateFields.encode_length
          (Nightstream.Protocol.NebulaV2.WasmStateEncoding.encode
            tests.NebulaV2WasmState.running)).symm index
      have valueAt := congrArg
        (fun values : List Nat => values.getD index.val 0) runningFieldsZero
      have getDZero : encoded.getD index.val 0 = 0 := by
        change encoded.getD index.val 0 =
          (List.replicate 85 0).getD index.val 0 at valueAt
        exact valueAt.trans (List.getD_replicate
          (x := 0) (y := 0) (i := index.val) (n := 85) index.isLt)
      calc
        0 = encoded.getD index.val 0 := getDZero.symm
        _ = encoded.get coordinate := by
          simpa [coordinate] using List.getD_eq_get encoded 0 coordinate
  · decide

/-- Universal completion over an empty supplement type is vacuous. It does
not prove that a successor supplement exists. -/
theorem universal_supplement_is_vacuous :
    (∀ _supplement : Empty, False) /\
      ¬ Nonempty Empty := by
  constructor
  · intro supplement
    exact nomatch supplement
  · intro existsSupplement
    exact nomatch existsSupplement

end tests.NebulaV2ProductionApplicationAuthorityCountermodels
