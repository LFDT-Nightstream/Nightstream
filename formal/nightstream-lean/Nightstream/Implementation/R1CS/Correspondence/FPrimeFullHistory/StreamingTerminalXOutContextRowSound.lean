import Mathlib.Tactic
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContext

/-!
Contract: the 24 exact terminal XOut context rows copy verifier-owned values
and fixed lifecycle constants into their assigned XOut lanes.

This leaf does not own the authority of the four public source digests, the
phase-semantic lanes, the Nebula-state lanes, or terminal acceptance.

Assurance tier: artifact-checked for
`rust:streaming-terminal-x-out-context/v1`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutContextRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutContext.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutContext
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutContext
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField

abbrev DigestValues := Fin 4 → Nat

def xOutValueFor
    (artifact : RawArtifact) (assignment : Nat → Nat) (index : Nat) : Nat :=
  assignment (artifact.xOutColumns.getD index 0)

abbrev xOutValue := xOutValueFor rawArtifact

def sourceDigest (assignment : Nat → Nat) (columns : List Nat) : DigestValues :=
  fun lane => assignment (columns.getD lane.val 0)

private theorem copy_row_sound
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    {output input : Nat}
    (member : builderLinearRow output [(input, 1)] ∈
      artifact.contextRows) :
    assignment output = assignment input := by
  have holds := satisfied _ member
  have exact := builderLinearRow_sound canonical one output [(input, 1)]
    (by simp [CanonicalTerms, goldilocksP]) holds
  have inputCanonical := canonical input
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul] at exact
  rw [Nat.mod_eq_of_lt inputCanonical] at exact
  exact exact

private theorem constant_row_sound
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    {output value : Nat}
    (valuePositive : 0 < value)
    (valueCanonical : value < goldilocksP)
    (member : builderLinearRow output [(0, value)] ∈
      artifact.contextRows) :
    assignment output = value := by
  have holds := satisfied _ member
  have exact := builderLinearRow_sound canonical one output [(0, value)]
    (by simp [CanonicalTerms, valuePositive, valueCanonical]) holds
  simpa [lcEval, one, Nat.mod_eq_of_lt valueCanonical] using exact

private theorem zero_row_sound
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    {output : Nat}
    (member : builderLinearRow output [] ∈ artifact.contextRows) :
    assignment output = 0 := by
  have holds := satisfied _ member
  have exact := builderLinearRow_sound canonical one output []
    (by simp [CanonicalTerms]) holds
  simpa [lcEval] using exact

structure ContextConstantsCanonical (artifact : RawArtifact) : Prop where
  domainPositive : 0 < artifact.domainTag
  domainCanonical : artifact.domainTag < goldilocksP
  acceptedPositive : 0 < artifact.acceptedWorkItems
  acceptedCanonical : artifact.acceptedWorkItems < goldilocksP
  markerPositive : 0 < artifact.nebulaMarker
  markerCanonical : artifact.nebulaMarker < goldilocksP

structure ContextRowsPresent (artifact : RawArtifact) : Prop where
  domain : builderLinearRow (artifact.xOutColumns.getD 0 0)
    [(0, artifact.domainTag)] ∈ artifact.contextRows
  verifierKey : ∀ lane : Fin 4,
    builderLinearRow (artifact.xOutColumns.getD (1 + lane.val) 0)
      [(artifact.vkFsSourceColumns.getD lane.val 0, 1)] ∈ artifact.contextRows
  piCcsHeader : ∀ lane : Fin 4,
    builderLinearRow (artifact.xOutColumns.getD (5 + lane.val) 0)
      [(artifact.piCcsHeaderSourceColumns.getD lane.val 0, 1)] ∈
        artifact.contextRows
  chunkCountLow : builderLinearRow (artifact.xOutColumns.getD 9 0)
    [(0, artifact.acceptedWorkItems)] ∈ artifact.contextRows
  chunkCountHigh : builderLinearRow (artifact.xOutColumns.getD 10 0) [] ∈
    artifact.contextRows
  stepCountLow : builderLinearRow (artifact.xOutColumns.getD 11 0)
    [(0, artifact.acceptedWorkItems)] ∈ artifact.contextRows
  stepCountHigh : builderLinearRow (artifact.xOutColumns.getD 12 0) [] ∈
    artifact.contextRows
  programCounterLow : builderLinearRow (artifact.xOutColumns.getD 13 0)
    [(0, 1)] ∈ artifact.contextRows
  programCounterHigh : builderLinearRow (artifact.xOutColumns.getD 14 0) [] ∈
    artifact.contextRows
  boundary : ∀ lane : Fin 4,
    builderLinearRow (artifact.xOutColumns.getD (15 + lane.val) 0)
      [(artifact.boundarySourceColumns.getD lane.val 0, 1)] ∈
        artifact.contextRows
  accumulator : ∀ lane : Fin 4,
    builderLinearRow (artifact.xOutColumns.getD (23 + lane.val) 0)
      [(artifact.accumulatorSourceColumns.getD lane.val 0, 1)] ∈
        artifact.contextRows
  nebulaMarker : builderLinearRow (artifact.xOutColumns.getD 27 0)
    [(0, artifact.nebulaMarker)] ∈ artifact.contextRows

structure SoundFor (artifact : RawArtifact) (assignment : Nat → Nat) : Prop where
  domain : xOutValueFor artifact assignment 0 = artifact.domainTag
  verifierKey : ∀ lane,
    xOutValueFor artifact assignment (1 + lane.val) =
      sourceDigest assignment artifact.vkFsSourceColumns lane
  piCcsHeader : ∀ lane,
    xOutValueFor artifact assignment (5 + lane.val) =
      sourceDigest assignment artifact.piCcsHeaderSourceColumns lane
  chunkCountLow : xOutValueFor artifact assignment 9 = artifact.acceptedWorkItems
  chunkCountHigh : xOutValueFor artifact assignment 10 = 0
  stepCountLow : xOutValueFor artifact assignment 11 = artifact.acceptedWorkItems
  stepCountHigh : xOutValueFor artifact assignment 12 = 0
  programCounterLow : xOutValueFor artifact assignment 13 = 1
  programCounterHigh : xOutValueFor artifact assignment 14 = 0
  boundary : ∀ lane,
    xOutValueFor artifact assignment (15 + lane.val) =
      sourceDigest assignment artifact.boundarySourceColumns lane
  accumulator : ∀ lane,
    xOutValueFor artifact assignment (23 + lane.val) =
      sourceDigest assignment artifact.accumulatorSourceColumns lane
  nebulaMarker : xOutValueFor artifact assignment 27 = artifact.nebulaMarker

abbrev Sound := SoundFor rawArtifact

theorem rows_sound_for
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (constants : ContextConstantsCanonical artifact)
    (present : ContextRowsPresent artifact) :
    SoundFor artifact assignment := by
  refine {
    domain := constant_row_sound artifact assignment canonical one satisfied
      constants.domainPositive constants.domainCanonical present.domain
    verifierKey := fun lane =>
      copy_row_sound artifact assignment canonical one satisfied
        (present.verifierKey lane)
    piCcsHeader := fun lane =>
      copy_row_sound artifact assignment canonical one satisfied
        (present.piCcsHeader lane)
    chunkCountLow := constant_row_sound artifact assignment canonical one satisfied
      constants.acceptedPositive constants.acceptedCanonical present.chunkCountLow
    chunkCountHigh := zero_row_sound artifact assignment canonical one satisfied
      present.chunkCountHigh
    stepCountLow := constant_row_sound artifact assignment canonical one satisfied
      constants.acceptedPositive constants.acceptedCanonical present.stepCountLow
    stepCountHigh := zero_row_sound artifact assignment canonical one satisfied
      present.stepCountHigh
    programCounterLow := constant_row_sound artifact assignment canonical one satisfied
      (by decide) (by decide) present.programCounterLow
    programCounterHigh := zero_row_sound artifact assignment canonical one satisfied
      present.programCounterHigh
    boundary := fun lane =>
      copy_row_sound artifact assignment canonical one satisfied
        (present.boundary lane)
    accumulator := fun lane =>
      copy_row_sound artifact assignment canonical one satisfied
        (present.accumulator lane)
    nebulaMarker := constant_row_sound artifact assignment canonical one satisfied
      constants.markerPositive constants.markerCanonical present.nebulaMarker }

private theorem rawArtifact_constants :
    ContextConstantsCanonical rawArtifact := by
  refine {
    domainPositive := by norm_num [rawArtifact]
    domainCanonical := by norm_num [rawArtifact, goldilocksP]
    acceptedPositive := by norm_num [rawArtifact]
    acceptedCanonical := by norm_num [rawArtifact, goldilocksP]
    markerPositive := by norm_num [rawArtifact]
    markerCanonical := by norm_num [rawArtifact, goldilocksP] }

private theorem rawArtifact_rows_present : ContextRowsPresent rawArtifact := by
  refine {
    domain := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
    verifierKey := ?_
    piCcsHeader := ?_
    chunkCountLow := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
    chunkCountHigh := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
    stepCountLow := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
    stepCountHigh := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
    programCounterLow := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
    programCounterHigh := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
    boundary := ?_
    accumulator := ?_
    nebulaMarker := by
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact] }
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]
  · intro lane
    fin_cases lane <;>
      norm_num [RawArtifact.contextRows, copyRows, rawArtifact]

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    Sound assignment := by
  exact rows_sound_for rawArtifact assignment canonical one satisfied
    rawArtifact_constants rawArtifact_rows_present

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutContextRowSound
