import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPriorStateReplayArtifact

/-!
Pairwise composition of exact prior-state replay slices.

Owns the opaque transport from adjacent Rust source-slice certificates to one
nested Poseidon2 replay. It does not unfold or re-evaluate the source lists.

Assurance tier: artifact-checked.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySliceComposition

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplayTransitionExecutionCertificate
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachineDuplex

/-- The first two exact 256-field full-arm source slices form one sequential
Poseidon2 replay. The proof reuses the two leaf certificates and the exact
shared physical checkpoint. -/
theorem full_pair01_eq_absorbSlices
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice1Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice1Columns.map assignment)
        (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice0Columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice0Start).state)) := by
  calc
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice1Result).state =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice1Columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice1Start).state) :=
      full_slice1_eq_absorbSlice assignment canonical one satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice1Columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice0Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice1Columns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice0Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical
                fullSlice0Start).state)) := by
      rw [full_slice0_eq_absorbSlice assignment canonical one satisfied]

/-- All four exact 256-field full-arm source slices form one sequential
Poseidon2 replay. Each concrete slice remains opaque. -/
theorem full_eq_absorbSlices
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : fullArtifact.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (fullSlice3Columns.map assignment)
        (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice2Columns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice1Columns.map assignment)
            (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
              (fullSlice0Columns.map assignment)
              (toDuplex
                (ColumnReplay.decodeRun assignment canonical
                  fullSlice0Start).state)))) := by
  calc
    toDuplex
        (ColumnReplay.decodeRun assignment canonical fullSlice3Result).state =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice3Start).state) :=
      full_slice3_eq_absorbSlice assignment canonical one satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              fullSlice2Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice2Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical
                fullSlice2Start).state)) := by
      rw [full_slice2_eq_absorbSlice assignment canonical one satisfied]
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice2Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical
                fullSlice1Result).state)) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (fullSlice3Columns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (fullSlice2Columns.map assignment)
            (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
              (fullSlice1Columns.map assignment)
              (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
                (fullSlice0Columns.map assignment)
                (toDuplex
                  (ColumnReplay.decodeRun assignment canonical
                    fullSlice0Start).state)))) := by
      rw [full_pair01_eq_absorbSlices assignment canonical one satisfied]

/-- The two exact 256-field slices and exact ten-field tail form the complete
522-field final-arm replay. Each concrete slice remains opaque. -/
theorem final_eq_absorbSlices
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : finalArtifact.Satisfied assignment) :
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        (finalTailColumns.map assignment)
        (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalSlice1Columns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (finalSlice0Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical
                finalSlice0Start).state))) := by
  calc
    toDuplex
        (ColumnReplay.decodeRun assignment canonical finalTailResult).state =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              finalTailStart).state) :=
      final_tail_eq_absorbSlice assignment canonical one satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (toDuplex
            (ColumnReplay.decodeRun assignment canonical
              finalSlice1Result).state) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (finalSlice1Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical
                finalSlice1Start).state)) := by
      rw [final_slice1_eq_absorbSlice assignment canonical one satisfied]
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (finalSlice1Columns.map assignment)
            (toDuplex
              (ColumnReplay.decodeRun assignment canonical
                finalSlice0Result).state)) := by
      rfl
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (finalTailColumns.map assignment)
          (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
            (finalSlice1Columns.map assignment)
            (Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
              (finalSlice0Columns.map assignment)
              (toDuplex
                (ColumnReplay.decodeRun assignment canonical
                  finalSlice0Start).state))) := by
      rw [final_slice0_eq_absorbSlice assignment canonical one satisfied]

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySliceComposition
