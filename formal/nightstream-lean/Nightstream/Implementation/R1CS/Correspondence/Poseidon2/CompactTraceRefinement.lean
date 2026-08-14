import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Final
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.FirstTerminal
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Partial
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schedule0
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schedule1
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schedule4
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Schedule5
import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.CompactTrace.Tail

/-!
Contract: exact refinement from the compact trace recorded by the production
Rust Poseidon2 emitter to the independent Lean compact permutation relation.

Owns: aggregation of the bounded exact certificates, same-assignment
refinement, and the resulting reference-permutation theorem.

Does not own: inclusion of these rows in a complete recursive artifact,
selector authority, outer field encoding, a call-site column renaming, or
Poseidon2 collision security.

Emits constraints: no.

Assurance tier: Rust-conformant for property
`POSEIDON2-COMPACT-TRACE-REFINEMENT`.
-/

set_option autoImplicit false
set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

namespace Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.LinearSubstitution
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
open Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm

/-- Exact comparison for the 56 schedule inputs whose recursive form stays
small: the first 32 inputs and indices 62 through 85. -/
theorem compact_raw_schedule_exact (index : Fin sboxCount)
    (outsideRecursiveBlock : index.val < 32 ∨ 62 ≤ index.val) :
    ScheduleExactAt index := by
  have indexBound : index.val < 86 := by
    simpa only [sboxCount_eq] using index.isLt
  by_cases first : index.val < 16
  · let offset : Fin 16 := ⟨index.val, first⟩
    have same : shardIndex0 offset = index := by
      apply Fin.ext
      simp [shardIndex0, offset]
    simpa [same] using compact_schedule_exact_0 offset
  · by_cases second : index.val < 32
    · let offset : Fin 16 := ⟨index.val - 16, by omega⟩
      have same : shardIndex1 offset = index := by
        apply Fin.ext
        simp [shardIndex1, offset]
        omega
      simpa [same] using compact_schedule_exact_1 offset
    · have afterRecursive : 62 ≤ index.val :=
        outsideRecursiveBlock.resolve_left second
      by_cases tail : index.val < 64
      · let offset : Fin 2 := ⟨index.val - 62, by omega⟩
        have same : tailArtifactIndex offset = index := by
          apply Fin.ext
          simp [tailArtifactIndex, offset]
          omega
        simpa [same] using compact_tail_schedule_exact offset
      · by_cases fourth : index.val < 80
        · let offset : Fin 16 := ⟨index.val - 64, by omega⟩
          have same : shardIndex4 offset = index := by
            apply Fin.ext
            simp [shardIndex4, offset]
            omega
          simpa [same] using compact_schedule_exact_4 offset
        · let offset : Fin 6 := ⟨index.val - 80, by omega⟩
          have same : shardIndex5 offset = index := by
            apply Fin.ext
            simp [shardIndex5, offset]
            omega
          simpa [same] using compact_schedule_exact_5 offset

/-- Physical equations emitted by selective Poseidon2 lowering, with the
general branch selector already fixed to one by verifier-owned dispatch. -/
structure TraceHolds (physical : Nat → Nat) : Prop where
  constantWire : physical 0 = 1
  sboxes : ∀ index : Fin sboxCount,
    physical (traceSboxOutput index) =
      sbox7 (lcEval physical (traceSboxInput index))
  outputs : ∀ lane : Fin width,
    physical (traceOutputColumn lane) =
      lcEval physical (traceFinalForm lane)

private theorem eval_trace_form_eq
    (physical : Nat → Nat) (index : Fin sboxCount)
    (form : Poseidon2Core.LinComb)
    (exact : traceTerms (terms expansion form) =
      traceTerms (traceSboxInput index)) :
    lcEval (logicalAssignment physical) form =
      lcEval physical (traceSboxInput index) := by
  unfold logicalAssignment
  rw [← lcEval_terms expansion physical]
  calc
    lcEval physical (terms expansion form) =
        lcEval physical (traceTerms (terms expansion form)) :=
      (lcEval_traceTerms physical _).symm
    _ = lcEval physical (traceTerms (traceSboxInput index)) := by
      rw [exact]
    _ = lcEval physical (traceSboxInput index) :=
      lcEval_traceTerms physical (traceSboxInput index)

private theorem eval_schedule_eq
    (physical : Nat → Nat) (index : Fin sboxCount) :
    lcEval (logicalAssignment physical)
        (scheduleOf canonicalLayout Poseidon2CanonicalConstants.selected index) =
      lcEval physical (traceSboxInput index) := by
  have indexBound : index.val < 86 := by
    simpa only [sboxCount_eq] using index.isLt
  by_cases initial : index.val < 32
  · exact eval_trace_form_eq physical index _
      (compact_raw_schedule_exact index (Or.inl initial))
  · by_cases inPartial : index.val < 54
    · let round : Fin partialRounds :=
        ⟨index.val - 32, by simp only [partialRounds]; omega⟩
      have same : partialArtifactIndex round = index := by
        apply Fin.ext
        simp [partialArtifactIndex, round]
        omega
      rw [← same]
      calc
        lcEval (logicalAssignment physical)
            (scheduleOf canonicalLayout Poseidon2CanonicalConstants.selected
              (partialArtifactIndex round)) =
            lcEval (logicalAssignment physical)
              (addConstant
                (Poseidon2CanonicalConstants.selected.internal round.val)
                (partialState canonicalLayout round.val ⟨0, by decide⟩)) := by
          rw [scheduleOf_partial canonicalLayout
            Poseidon2CanonicalConstants.selected (partialArtifactIndex round)
            round.val (by simp [partialArtifactIndex]) round.isLt]
          rfl
        _ = lcEval (logicalAssignment physical) (partialScheduleForm round) := by
          simpa only [partialScheduleForm] using
            (lcEval_addConstant_coefficientForm canonicalLayout round.val
              (Nat.le_of_lt round.isLt) ⟨0, by decide⟩
              (Poseidon2CanonicalConstants.selected.internal round.val)
              (logicalAssignment physical)).symm
        _ = lcEval physical (traceSboxInput (partialArtifactIndex round)) :=
          eval_trace_form_eq physical (partialArtifactIndex round)
            (partialScheduleForm round) (compact_partial_schedule_exact round)
    · by_cases firstTerminal : index.val < 62
      · let lane : Fin width :=
          ⟨index.val - 54, by simp only [width]; omega⟩
        have same : firstTerminalArtifactIndex lane = index := by
          apply Fin.ext
          simp [firstTerminalArtifactIndex, lane]
          omega
        rw [← same]
        calc
          lcEval (logicalAssignment physical)
              (scheduleOf canonicalLayout Poseidon2CanonicalConstants.selected
                (firstTerminalArtifactIndex lane)) =
              lcEval (logicalAssignment physical)
                (addConstant
                  (Poseidon2CanonicalConstants.selected.terminal 0 lane)
                  (partialState canonicalLayout partialRounds lane)) := by
            rw [scheduleOf_terminal canonicalLayout
              Poseidon2CanonicalConstants.selected
              (firstTerminalArtifactIndex lane) 0 lane (by
                simp [firstTerminalArtifactIndex, terminalSboxIndex,
                  halfFullRounds, width, partialRounds]) (by decide)]
            rfl
          _ = lcEval (logicalAssignment physical)
              (firstTerminalScheduleForm lane) := by
            simpa only [firstTerminalScheduleForm] using
              (lcEval_addConstant_coefficientForm canonicalLayout partialRounds
                (Nat.le_refl _) lane
                (Poseidon2CanonicalConstants.selected.terminal 0 lane)
                (logicalAssignment physical)).symm
          _ = lcEval physical
              (traceSboxInput (firstTerminalArtifactIndex lane)) :=
            eval_trace_form_eq physical (firstTerminalArtifactIndex lane)
              (firstTerminalScheduleForm lane)
              (compact_first_terminal_schedule_exact lane)
      · exact eval_trace_form_eq physical index _
          (compact_raw_schedule_exact index
            (Or.inr (Nat.le_of_not_gt firstTerminal)))

private theorem eval_final_eq
    (physical : Nat → Nat) (lane : Fin width) :
    lcEval (logicalAssignment physical) (finalState canonicalLayout lane) =
      lcEval physical (traceFinalForm lane) := by
  unfold logicalAssignment
  rw [← lcEval_terms expansion physical]
  calc
    lcEval physical (terms expansion (finalState canonicalLayout lane)) =
        lcEval physical
          (traceTerms (terms expansion (finalState canonicalLayout lane))) :=
      (lcEval_traceTerms physical _).symm
    _ = lcEval physical (traceTerms (traceFinalForm lane)) := by
      rw [compact_final_exact lane]
    _ = lcEval physical (traceFinalForm lane) :=
      lcEval_traceTerms physical (traceFinalForm lane)

private theorem logical_singleton
    (physical : Nat → Nat)
    (canonical : ∀ column, physical column < goldilocksP)
    (logical physicalColumn : Nat)
    (expanded : expansion logical = [(physicalColumn, 1)]) :
    logicalAssignment physical logical = physical physicalColumn := by
  unfold logicalAssignment LinearSubstitution.assignment
  rw [expanded]
  exact Poseidon2RoundInduction.lcEval_singleton physical physicalColumn
    (canonical physicalColumn)

/-- The exact Rust compact equations satisfy the independent Lean compact
relation on the same input, S-box-output, and output values. -/
theorem trace_refines_compact
    (physical : Nat → Nat)
    (canonical : ∀ column, physical column < goldilocksP)
    (holds : TraceHolds physical) :
    Poseidon2Compact.Holds canonicalLayout
      Poseidon2CanonicalConstants.selected (logicalAssignment physical) := by
  refine ⟨?_, ?_, ?_⟩
  · exact (logical_singleton physical canonical 0 0 expansion_constant).trans
      holds.constantWire
  · intro index
    rw [logical_singleton physical canonical _ _ (expansion_sboxOutput index),
      holds.sboxes index, eval_schedule_eq physical index]
  · intro lane
    rw [logical_singleton physical canonical _ _ (expansion_output lane),
      holds.outputs lane, eval_final_eq physical lane]

/-- Headline Rust-conformance result: the compact physical trace forces the
same selected reference permutation for every canonical assignment. -/
theorem trace_computes_reference
    (physical : Nat → Nat)
    (canonical : ∀ column, physical column < goldilocksP)
    (holds : TraceHolds physical)
    (lane : Fin width) :
    physical (traceOutputColumn lane) =
      referencePermutation Poseidon2CanonicalConstants.selected
        (fun inputLane => physical (traceInputColumn inputLane)) lane := by
  have refined := trace_refines_compact physical canonical holds
  have sound := Poseidon2Compact.computes_reference canonicalLayout
    Poseidon2CanonicalConstants.selected (logicalAssignment physical)
    (fun column => Nat.mod_lt _ (by decide)) refined lane
  rw [logical_singleton physical canonical _ _ (expansion_output lane)] at sound
  apply sound.trans
  congr 2
  funext inputLane
  exact logical_singleton physical canonical _ _ (expansion_input inputLane)

end Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
