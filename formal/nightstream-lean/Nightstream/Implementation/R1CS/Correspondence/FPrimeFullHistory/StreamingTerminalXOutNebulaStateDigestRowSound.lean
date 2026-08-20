import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestValidityCertificate
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound

/-!
Contract: all 19,353 exact terminal Nebula-state-digest rows compute the
selected Poseidon2 lane digest and copy it to the terminal XOut state lanes.

The absent and present branches use the same assignment. The open wire is
proved Boolean before the four mux rows select one branch. This leaf does not
claim that the branch inputs equal an authoritative lifecycle lane.

Assurance tier: artifact-checked for
`rust:streaming-terminal-nebula-state-digest/v2`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingVariableHashRecipeConstantSound
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigest
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestLink
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalXOutNebulaStateDigestLink
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField

abbrev DigestValues := Fin 4 → Nat

def inputValues (recipe : VariableHashRecipe)
    (assignment : Nat → Nat) : List Nat :=
  recipe.inputColumns.map assignment

def computedDigest (recipe : VariableHashRecipe)
    (assignment : Nat → Nat) : DigestValues :=
  fun lane => runValueRounds recipe.trace.rounds
    (inputValues recipe assignment) (fun _ => 0) lane.val

def assignedDigest (recipe : VariableHashRecipe)
    (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (recipe.outputColumns.getD lane.val 0)

def selectedDigestFor
    (artifact : RawArtifact) (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (artifact.hashOutputColumns.getD lane.val 0)

abbrev selectedDigest := selectedDigestFor rawArtifact

def computedSelectedDigestFor
    (artifact : RawArtifact) (assignment : Nat → Nat) : DigestValues :=
  if assignment artifact.openColumn = 1 then
    computedDigest artifact.presentRecipe assignment
  else
    computedDigest artifact.absentRecipe assignment

abbrev computedSelectedDigest := computedSelectedDigestFor rawArtifact

def xOutStateDigestFor
    (artifact : RawArtifact) (assignment : Nat → Nat) : DigestValues :=
  fun lane => assignment (artifact.xOutStateColumns.getD lane.val 0)

abbrev xOutStateDigest := xOutStateDigestFor rawArtifact

private theorem all_pieces_satisfied
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (satisfied : artifact.Satisfied assignment) :
    ∀ piece ∈ artifact.programPieces, Satisfies piece assignment := by
  apply (satisfies_flatten_iff artifact.programPieces assignment).mp
  simpa [RawArtifact.Satisfied, RawArtifact.program] using satisfied

private theorem canonical_bitRow_holds
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (holds : RowHolds assignment artifact.bitRow) :
    RowHolds assignment
      (Nightstream.Implementation.R1CS.bitRow artifact.openColumn) := by
  simpa [RawArtifact.bitRow, Nightstream.Implementation.R1CS.bitRow,
    RowHolds, lcEval, negCoeff, Nat.add_comm] using holds

private theorem open_exact
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : RowHolds assignment artifact.bitRow) :
    assignment artifact.openColumn = 0 ∨
      assignment artifact.openColumn = 1 := by
  have bounded := bitRow_le_one goldilocks_euclidPrime
    (canonical artifact.openColumn) one
    (canonical_bitRow_holds artifact assignment holds)
  omega

private theorem cancel_common_neg
    {left right shared : Nat}
    (leftCanonical : left < goldilocksP)
    (rightCanonical : right < goldilocksP)
    (sharedCanonical : shared < goldilocksP)
    (equal :
      ((goldilocksP - 1) * shared + left) % goldilocksP =
        ((goldilocksP - 1) * shared + right) % goldilocksP) :
    left = right := by
  have recover (value : Nat) (valueCanonical : value < goldilocksP) :
      ((((goldilocksP - 1) * shared + value) % goldilocksP + shared) %
          goldilocksP) = value := by
    calc
      ((((goldilocksP - 1) * shared + value) % goldilocksP + shared) %
          goldilocksP) =
          ((((goldilocksP - 1) * shared + value) % goldilocksP +
            shared % goldilocksP) % goldilocksP) := by
              rw [Nat.mod_eq_of_lt sharedCanonical]
      _ = (((goldilocksP - 1) * shared + value + shared) %
          goldilocksP) := by rw [← Nat.add_mod]
      _ = value := by
        have factor :
            (goldilocksP - 1) * shared + value + shared =
              value + goldilocksP * shared := by
          simp only [goldilocksP]
          omega
        rw [factor, Nat.add_mod, Nat.mod_eq_of_lt valueCanonical]
        simp [Nat.mod_eq_of_lt valueCanonical]
  have adjusted := congrArg
    (fun value => (value + shared) % goldilocksP) equal
  change
    ((((goldilocksP - 1) * shared + left) % goldilocksP + shared) %
        goldilocksP) =
      ((((goldilocksP - 1) * shared + right) % goldilocksP + shared) %
        goldilocksP) at adjusted
  rw [recover left leftCanonical, recover right rightCanonical] at adjusted
  exact adjusted

private theorem selectedMuxRow_sound
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    {selector present absent output : Nat}
    (selectorExact : assignment selector = 0 ∨ assignment selector = 1)
    (holds : RowHolds assignment
      (selectedMuxRow selector present absent output)) :
    assignment output =
      if assignment selector = 1 then assignment present else assignment absent := by
  rcases selectorExact with selectorZero | selectorOne
  · have outputCancel :
        ((goldilocksP - 1) * assignment absent + assignment output) %
            goldilocksP = 0 := by
      simpa [selectedMuxRow, RowHolds, lcEval, negCoeff, selectorZero,
        goldilocksP] using holds.symm
    have absentCancel :
        ((goldilocksP - 1) * assignment absent + assignment absent) %
            goldilocksP = 0 := by
      have factor :
          (goldilocksP - 1) * assignment absent + assignment absent =
            goldilocksP * assignment absent := by
        simp only [goldilocksP]
        omega
      rw [factor]
      simp
    have exact := cancel_common_neg
      (canonical output) (canonical absent) (canonical absent)
      (outputCancel.trans absentCancel.symm)
    simpa [selectorZero] using exact
  · have equation :
        ((goldilocksP - 1) * assignment absent + assignment present) %
            goldilocksP =
          ((goldilocksP - 1) * assignment absent + assignment output) %
            goldilocksP := by
      simpa [selectedMuxRow, RowHolds, lcEval, negCoeff, selectorOne,
        goldilocksP] using holds
    have exact := cancel_common_neg
      (canonical present) (canonical output) (canonical absent) equation
    simpa [selectorOne] using exact.symm

private theorem recipe_rows_imply_hash
    (recipe : VariableHashRecipe)
    (valid : recipe.trace.OwnedValid)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies recipe.trace.rows assignment) :
    assignedDigest recipe assignment = computedDigest recipe assignment := by
  funext lane
  exact ownedTrace_values_sound valid canonical one satisfied lane.val lane.isLt

structure SoundFor (artifact : RawArtifact) (assignment : Nat → Nat) : Prop where
  openExact : assignment artifact.openColumn = 0 ∨
    assignment artifact.openColumn = 1
  absentConstants :
    artifact.absentRecipe.constantColumns.map assignment =
      artifact.absentRecipe.constantValues
  presentConstants :
    artifact.presentRecipe.constantColumns.map assignment =
      artifact.presentRecipe.constantValues
  absentHash : assignedDigest artifact.absentRecipe assignment =
    computedDigest artifact.absentRecipe assignment
  presentHash : assignedDigest artifact.presentRecipe assignment =
    computedDigest artifact.presentRecipe assignment
  selectedHash : selectedDigestFor artifact assignment =
    computedSelectedDigestFor artifact assignment
  xOutLink : xOutStateDigestFor artifact assignment =
    selectedDigestFor artifact assignment

abbrev Sound := SoundFor rawArtifact

theorem rows_sound_for
    (artifact : RawArtifact)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : artifact.Satisfied assignment)
    (absentTraceValid : artifact.absentRecipe.trace.OwnedValid)
    (presentTraceValid : artifact.presentRecipe.trace.OwnedValid)
    (absentConstantsCanonical :
      ∀ value ∈ artifact.absentConstantValues, value < goldilocksP)
    (presentConstantsCanonical :
      ∀ value ∈ artifact.presentConstantValues, value < goldilocksP) :
    SoundFor artifact assignment := by
  have pieces := all_pieces_satisfied artifact assignment satisfied
  have bitSatisfied : Satisfies [artifact.bitRow] assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have absentConstantsSatisfied :
      Satisfies (constantRows artifact.absentRecipe) assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have absentTraceSatisfied :
      Satisfies artifact.absentRecipe.trace.rows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have presentConstantsSatisfied :
      Satisfies (constantRows artifact.presentRecipe) assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have presentTraceSatisfied :
      Satisfies artifact.presentRecipe.trace.rows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have muxSatisfied : Satisfies artifact.muxRows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have equalitySatisfied : Satisfies artifact.equalityRows assignment :=
    pieces _ (by simp [RawArtifact.programPieces])
  have bitHolds := bitSatisfied artifact.bitRow (by simp)
  have openExact := open_exact artifact assignment canonical one bitHolds
  have absentHash := recipe_rows_imply_hash artifact.absentRecipe
    absentTraceValid assignment canonical one absentTraceSatisfied
  have presentHash := recipe_rows_imply_hash artifact.presentRecipe
    presentTraceValid assignment canonical one presentTraceSatisfied
  refine {
    openExact := openExact
    absentConstants := constantRows_values artifact.absentRecipe assignment
      canonical one absentConstantsCanonical absentConstantsSatisfied
    presentConstants := constantRows_values artifact.presentRecipe assignment
      canonical one presentConstantsCanonical presentConstantsSatisfied
    absentHash := absentHash
    presentHash := presentHash
    selectedHash := ?_
    xOutLink := ?_ }
  · funext lane
    have laneMember : lane.val ∈ List.range digestFields := by
      simp [digestFields, lane.isLt]
    have muxHolds := muxSatisfied
      (selectedMuxRow artifact.openColumn
        (artifact.presentOutputColumns.getD lane.val 0)
        (artifact.absentOutputColumns.getD lane.val 0)
        (artifact.hashOutputColumns.getD lane.val 0))
      (List.mem_map.mpr ⟨lane.val, laneMember, by
        simp [RawArtifact.muxRows]⟩)
    have selected := selectedMuxRow_sound assignment canonical openExact muxHolds
    by_cases present : assignment artifact.openColumn = 1
    · have selectedPresent :
          assignment (artifact.hashOutputColumns.getD lane.val 0) =
            assignment (artifact.presentOutputColumns.getD lane.val 0) := by
        simpa [present] using selected
      simpa [selectedDigestFor, computedSelectedDigestFor, present,
        assignedDigest] using selectedPresent.trans (congrFun presentHash lane)
    · have absent : assignment artifact.openColumn = 0 := by
        rcases openExact with zero | one
        · exact zero
        · exact False.elim (present one)
      have selectedAbsent :
          assignment (artifact.hashOutputColumns.getD lane.val 0) =
            assignment (artifact.absentOutputColumns.getD lane.val 0) := by
        simpa [present, absent] using selected
      simpa [selectedDigestFor, computedSelectedDigestFor, present, absent,
        assignedDigest] using selectedAbsent.trans (congrFun absentHash lane)
  · funext lane
    have laneMember : lane.val ∈ List.range digestFields := by
      simp [digestFields, lane.isLt]
    have linkHolds := equalitySatisfied
      (builderLinearRow (artifact.xOutStateColumns.getD lane.val 0)
        [(artifact.hashOutputColumns.getD lane.val 0, 1)])
      (List.mem_map.mpr ⟨lane.val, laneMember, by
        simp [RawArtifact.equalityRows]⟩)
    have exact := builderLinearRow_sound canonical one
      (artifact.xOutStateColumns.getD lane.val 0)
      [(artifact.hashOutputColumns.getD lane.val 0, 1)]
      (by simp [CanonicalTerms, goldilocksP]) linkHolds
    have sourceCanonical :=
      canonical (artifact.hashOutputColumns.getD lane.val 0)
    simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul] at exact
    rw [Nat.mod_eq_of_lt sourceCanonical] at exact
    simpa [xOutStateDigestFor, selectedDigestFor] using exact

private theorem absent_constants_canonical :
    ∀ value ∈ rawArtifact.absentConstantValues, value < goldilocksP := by
  have valid := rawArtifact_valid
  unfold RawArtifact.Valid at valid
  aesop

private theorem present_constants_canonical :
    ∀ value ∈ rawArtifact.presentConstantValues, value < goldilocksP := by
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
    absent_trace_ownedValid present_trace_ownedValid
    absent_constants_canonical present_constants_canonical

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutNebulaStateDigestRowSound
