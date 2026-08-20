import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplayExecution

/-!
Contract: bounded physical execution certificate for the terminal Nebula
gamma transcript.

Owns twelve small replay leaves and their composition. It does not own row
satisfaction, semantic transcript refinement, the initial application-domain
state, output muxes, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptExecution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel

def trace : TranscriptCertificate.Trace :=
  rawArtifact.gammaTranscriptTrace

def start : ColumnReplay.Run where
  cursor := {
    lanes := fun lane =>
      rawArtifact.gammaTranscriptPinColumns.getD lane.val 0
    absorbed := ⟨3, by decide⟩
    nextPin := 8
    nextCall := 0 }
  digests := []

def resultOr
    (fallback : ColumnReplay.Run) : Option ColumnReplay.Run → ColumnReplay.Run
  | none => fallback
  | some result => result

def run1 : ColumnReplay.Run :=
  resultOr start (ColumnReplay.execute trace start vkFsAppend)

def run2 : ColumnReplay.Run :=
  resultOr run1 (ColumnReplay.execute trace run1 boundaryAppend)

def run3 : ColumnReplay.Run :=
  resultOr run2 (ColumnReplay.execute trace run2 accumulatorAppend)

def run4 : ColumnReplay.Run :=
  resultOr run3 (ColumnReplay.execute trace run3 stagedLaneAppend)

def run5 : ColumnReplay.Run :=
  resultOr run4 (ColumnReplay.execute trace run4 programBindingAppend)

def run6 : ColumnReplay.Run :=
  resultOr run5 (ColumnReplay.execute trace run5 segmentIndexAppend)

def run7 : ColumnReplay.Run :=
  resultOr run6 (ColumnReplay.execute trace run6 timestampAppend)

def run8 : ColumnReplay.Run :=
  resultOr run7 (ColumnReplay.execute trace run7 dPreOpsAppend)

def run9 : ColumnReplay.Run :=
  resultOr run8 (ColumnReplay.execute trace run8 dPreIsAppend)

def run10 : ColumnReplay.Run :=
  resultOr run9 (ColumnReplay.execute trace run9 dPreFsAppend)

def run11 : ColumnReplay.Run :=
  resultOr run10 (ColumnReplay.execute trace run10 gamma1Challenge)

def result : ColumnReplay.Run :=
  resultOr run11 (ColumnReplay.execute trace run11 gamma2Challenge)

private theorem leaf1 :
    ColumnReplay.execute trace start vkFsAppend = some run1 := by
  rfl

private theorem leaf2 :
    ColumnReplay.execute trace run1 boundaryAppend = some run2 := by
  rfl

private theorem leaf3 :
    ColumnReplay.execute trace run2 accumulatorAppend = some run3 := by
  rfl

private theorem leaf4 :
    ColumnReplay.execute trace run3 stagedLaneAppend = some run4 := by
  rfl

private theorem leaf5 :
    ColumnReplay.execute trace run4 programBindingAppend = some run5 := by
  rfl

private theorem leaf6 :
    ColumnReplay.execute trace run5 segmentIndexAppend = some run6 := by
  rfl

private theorem leaf7 :
    ColumnReplay.execute trace run6 timestampAppend = some run7 := by
  rfl

private theorem leaf8 :
    ColumnReplay.execute trace run7 dPreOpsAppend = some run8 := by
  rfl

private theorem leaf9 :
    ColumnReplay.execute trace run8 dPreIsAppend = some run9 := by
  rfl

private theorem leaf10 :
    ColumnReplay.execute trace run9 dPreFsAppend = some run10 := by
  rfl

private theorem leaf11 :
    ColumnReplay.execute trace run10 gamma1Challenge = some run11 := by
  rfl

private theorem leaf12 :
    ColumnReplay.execute trace run11 gamma2Challenge = some result := by
  rfl

theorem execution :
    ColumnReplay.execute trace start operations = some result := by
  unfold operations operationSegments
  exact ColumnReplay.execute_append leaf1
    (ColumnReplay.execute_append leaf2
      (ColumnReplay.execute_append leaf3
        (ColumnReplay.execute_append leaf4
          (ColumnReplay.execute_append leaf5
            (ColumnReplay.execute_append leaf6
              (ColumnReplay.execute_append leaf7
                (ColumnReplay.execute_append leaf8
                  (ColumnReplay.execute_append leaf9
                    (ColumnReplay.execute_append leaf10
                      (ColumnReplay.execute_append leaf11 leaf12))))))))))

theorem complete_consumption :
    result.cursor.nextPin = 84 ∧
      result.cursor.nextCall = 29 ∧
      result.cursor.absorbed.val = 2 ∧
      result.digests.length = 2 := by
  constructor
  · rfl
  constructor
  · rfl
  constructor <;> rfl

def zeroColumns : Fin 4 → Nat := fun _ => 0

def gamma1DigestColumns : Fin 4 → Nat :=
  result.digests.getD 0 zeroColumns

def gamma2DigestColumns : Fin 4 → Nat :=
  result.digests.getD 1 zeroColumns

theorem gamma1_columns_exact (lane : Fin 2) :
    gamma1DigestColumns ⟨lane.val, by omega⟩ =
      rawArtifact.gamma1Columns.getD lane.val 0 := by
  fin_cases lane <;> rfl

theorem gamma2_columns_exact (lane : Fin 2) :
    gamma2DigestColumns ⟨lane.val, by omega⟩ =
      rawArtifact.gamma2Columns.getD lane.val 0 := by
  fin_cases lane <;> rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptExecution
