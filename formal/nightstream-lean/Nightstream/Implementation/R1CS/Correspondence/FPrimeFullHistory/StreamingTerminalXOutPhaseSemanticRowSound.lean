import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticValidityCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: all exact production terminal phase-semantic rows compute Poseidon2 from
the same phase-local and delayed-payload columns and copy the result to the
terminal XOut semantic lanes.

This leaf does not claim that either input slice is lifecycle-authoritative.

Assurance tier: artifact-checked for
`rust:streaming-terminal-phase-semantic/v2`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPhaseSemantic
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutPhaseSemantic
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField

abbrev DigestValues := Fin 4 → Nat

def inputValuesFor
    (artifact : RawArtifact) (assignment : Nat → Nat) : List Nat :=
  artifact.hashRecipe.inputColumns.map assignment

abbrev inputValues := inputValuesFor rawArtifact

def computedDigestFor
    (artifact : RawArtifact) (assignment : Nat → Nat) : DigestValues :=
  fun lane => runValueRounds artifact.hashRecipe.trace.rounds
    (inputValuesFor artifact assignment) (fun _ => 0) lane.val

abbrev computedDigest := computedDigestFor rawArtifact

def assignedDigestFor
    (artifact : RawArtifact) (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (artifact.hashOutputColumns.getD lane.val 0)

abbrev assignedDigest := assignedDigestFor rawArtifact

def xOutSemanticDigestFor
    (artifact : RawArtifact) (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (artifact.xOutSemanticColumns.getD lane.val 0)

abbrev xOutSemanticDigest := xOutSemanticDigestFor rawArtifact

private theorem all_pieces_satisfied
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (satisfied : artifact.Satisfied assignment) :
    ∀ piece ∈ artifact.programPieces, Satisfies piece assignment := by
  apply (satisfies_flatten_iff artifact.programPieces assignment).mp
  simpa [RawArtifact.Satisfied, RawArtifact.program] using satisfied

private theorem trace_rows_imply_hash
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (traceValid : artifact.hashRecipe.trace.OwnedValid)
    (satisfied : Satisfies artifact.hashRecipe.trace.rows assignment) :
    assignedDigestFor artifact assignment =
      computedDigestFor artifact assignment := by
  funext lane
  exact ownedTrace_values_sound traceValid canonical one satisfied
    lane.val lane.isLt

structure SoundFor (artifact : RawArtifact) (assignment : Nat → Nat) : Prop where
  constants : artifact.hashRecipe.constantColumns.map assignment =
    artifact.hashRecipe.constantValues
  hash : assignedDigestFor artifact assignment =
    computedDigestFor artifact assignment
  xOutLink : xOutSemanticDigestFor artifact assignment =
    assignedDigestFor artifact assignment

abbrev Sound := SoundFor rawArtifact

theorem rows_sound_for
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (traceValid : artifact.hashRecipe.trace.OwnedValid)
    (constantsCanonical :
      ∀ value ∈ artifact.constantValues, value < goldilocksP) :
    SoundFor artifact assignment := by
  have pieces := all_pieces_satisfied artifact assignment satisfied
  have constantsSatisfied :
      Satisfies (constantRows artifact.hashRecipe) assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have traceSatisfied :
      Satisfies artifact.hashRecipe.trace.rows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have equalitySatisfied : Satisfies artifact.equalityRows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  refine {
    constants := constantRows_values artifact.hashRecipe assignment
      canonical one constantsCanonical constantsSatisfied
    hash := trace_rows_imply_hash artifact assignment canonical one
      traceValid traceSatisfied
    xOutLink := ?_ }
  funext lane
  have laneMember : lane.val ∈ List.range digestFields := by
    simp [digestFields, lane.isLt]
  have linkHolds := equalitySatisfied
    (builderLinearRow (artifact.xOutSemanticColumns.getD lane.val 0)
      [(artifact.hashOutputColumns.getD lane.val 0, 1)])
    (List.mem_map.mpr ⟨lane.val, laneMember, by
      simp [RawArtifact.equalityRows]⟩)
  have exact := builderLinearRow_sound canonical one
    (artifact.xOutSemanticColumns.getD lane.val 0)
    [(artifact.hashOutputColumns.getD lane.val 0, 1)]
    (by simp [CanonicalTerms, goldilocksP]) linkHolds
  have sourceCanonical :=
    canonical (artifact.hashOutputColumns.getD lane.val 0)
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul] at exact
  rw [Nat.mod_eq_of_lt sourceCanonical] at exact
  simpa [xOutSemanticDigestFor, assignedDigestFor] using exact

private theorem constants_canonical :
    ∀ value ∈ rawArtifact.constantValues, value < goldilocksP := by
  have valid := rawArtifact_valid
  unfold RawArtifact.Valid at valid
  aesop

theorem rows_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : rawArtifact.Satisfied assignment) :
    Sound assignment := by
  exact rows_sound_for rawArtifact assignment canonical one satisfied
    trace_ownedValid constants_canonical

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPhaseSemanticRowSound
