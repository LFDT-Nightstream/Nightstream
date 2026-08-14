import Nightstream.Implementation.R1CS.Artifacts.Poseidon2
import Nightstream.Implementation.R1CS.Canonical.Poseidon2CanonicalConstants
import Nightstream.Implementation.R1CS.Canonical.Poseidon2Compact
import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm
import Nightstream.Implementation.R1CS.Core.LinearSubstitution

/-!
Contract: compact Rust Poseidon2 trace schema and comparison normal form.

Owns: physical-to-logical column expansion, exact artifact shape and coverage,
semantics-preserving sparse-term sorting, and the bounded certificate partition.

Does not own: concrete schedule certificates or end-to-end refinement.

Emits constraints: no.

Assurance tier: Rust-conformant schema for property
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
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Layout
open Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm

namespace Artifact

abbrev inputColumns := Poseidon2Permutation.inputColumns
abbrev outputColumns := Poseidon2Permutation.outputColumns
abbrev sboxInputs := Poseidon2Permutation.compactSboxInputTerms
abbrev sboxOutputs := Poseidon2Permutation.compactSboxOutputColumns
abbrev finalForms := Poseidon2Permutation.compactOutputLinearForms

end Artifact

def traceInputColumn (lane : Fin width) : Nat :=
  Artifact.inputColumns.getD (lane.val + 1) 0

def traceOutputColumn (lane : Fin width) : Nat :=
  Artifact.outputColumns.getD lane.val 0

def traceSboxInput (index : Fin sboxCount) : Poseidon2Core.LinComb :=
  Artifact.sboxInputs.getD index.val []

def traceSboxOutput (index : Fin sboxCount) : Nat :=
  Artifact.sboxOutputs.getD index.val 0

def traceFinalForm (lane : Fin width) : Poseidon2Core.LinComb :=
  Artifact.finalForms.getD lane.val []

/-- Expand each active logical column into its exact physical Rust column.
The three omitted intermediate S-box slots have an empty expansion because no
compact schedule or final form reads them. -/
def expansion : ColumnExpansion := fun logical =>
  if logical = 0 then [(0, 1)]
  else if logical < 9 then
    [(Artifact.inputColumns.getD logical 0, 1)]
  else if logical < 17 then
    [(Artifact.outputColumns.getD (logical - 9) 0, 1)]
  else if 20 ≤ logical ∧ (logical - 20) % 4 = 0 ∧
      (logical - 20) / 4 < sboxCount then
    [(Artifact.sboxOutputs.getD ((logical - 20) / 4) 0, 1)]
  else []

def logicalAssignment (physical : Nat → Nat) : Nat → Nat :=
  LinearSubstitution.assignment expansion physical

/-- Compact Rust references have one of 95 physical owners: constant one,
eight inputs, or 86 S-box outputs. The check covers exactly 94 linear forms. -/
def ownedPhysicalColumns : List Nat :=
  Artifact.inputColumns ++ Artifact.sboxOutputs

def compactForms : List Poseidon2Core.LinComb :=
  Artifact.sboxInputs ++ Artifact.finalForms

theorem generated_shape_exact :
    Artifact.inputColumns.length = 9 ∧
      Artifact.outputColumns.length = 8 ∧
      Artifact.sboxInputs.length = 86 ∧
      Artifact.sboxOutputs.length = 86 ∧
      Artifact.finalForms.length = 8 := by
  native_decide

theorem compact_reference_coverage :
    compactForms.length = 94 ∧
      ∀ form ∈ compactForms, ∀ term ∈ form,
        term.1 ∈ ownedPhysicalColumns := by
  native_decide

theorem expansion_constant : expansion 0 = [(0, 1)] := by
  native_decide

theorem expansion_input : ∀ lane : Fin width,
    expansion (canonicalLayout.inputPort lane) =
      [(traceInputColumn lane, 1)] := by
  native_decide

theorem expansion_output : ∀ lane : Fin width,
    expansion (canonicalLayout.outputPort lane) =
      [(traceOutputColumn lane, 1)] := by
  native_decide

theorem expansion_sboxOutput : ∀ index : Fin sboxCount,
    expansion (sboxOutput canonicalLayout index.val) =
      [(traceSboxOutput index, 1)] := by
  native_decide

/-- Insert one normalized term into ascending physical-column order. This is
only a comparison form. It emits no circuit column or constraint. -/
def insertByColumn (term : Nat × Nat) :
    List (Nat × Nat) → List (Nat × Nat)
  | [] => [term]
  | head :: tail =>
      if term.1 ≤ head.1 then term :: head :: tail
      else head :: insertByColumn term tail

theorem insertByColumn_perm (term : Nat × Nat)
    (form : Poseidon2Core.LinComb) :
    (insertByColumn term form).Perm (term :: form) := by
  induction form with
  | nil => simp [insertByColumn]
  | cons head tail hypothesis =>
      simp only [insertByColumn]
      split
      · exact List.Perm.refl _
      · exact (List.Perm.cons head hypothesis).trans
          (List.Perm.swap term head tail)

/-- Stable ascending-column order for a sparse linear combination. -/
def sortByColumn (form : Poseidon2Core.LinComb) : Poseidon2Core.LinComb :=
  form.foldr insertByColumn []

theorem sortByColumn_perm (form : Poseidon2Core.LinComb) :
    (sortByColumn form).Perm form := by
  induction form with
  | nil => exact List.Perm.refl []
  | cons head tail hypothesis =>
      exact (insertByColumn_perm head (sortByColumn tail)).trans
        (List.Perm.cons head hypothesis)

/-- Field-normalized terms in one deterministic physical-column order. -/
def traceTerms (form : Poseidon2Core.LinComb) : Poseidon2Core.LinComb :=
  sortByColumn (fieldNormalize form)

theorem lcEval_traceTerms (assignment : Nat → Nat)
    (form : Poseidon2Core.LinComb) :
    lcEval assignment (traceTerms form) = lcEval assignment form := by
  calc
    lcEval assignment (traceTerms form) =
        lcEval assignment (fieldNormalize form) :=
      Program.lcEval_eq_of_perm assignment
        (by simpa [traceTerms] using sortByColumn_perm (fieldNormalize form))
    _ = lcEval assignment form := lcEval_fieldNormalize assignment form

/-- Exact comparison for one S-box input after both sparse combinations use
the same comparison order. -/
def ScheduleExactAt (index : Fin sboxCount) : Prop :=
    traceTerms
        (terms expansion
          (scheduleOf canonicalLayout Poseidon2CanonicalConstants.selected index)) =
      traceTerms (traceSboxInput index)

def shardIndex0 (offset : Fin 16) : Fin sboxCount :=
  ⟨offset.val, by rw [sboxCount_eq]; omega⟩

def shardIndex1 (offset : Fin 16) : Fin sboxCount :=
  ⟨16 + offset.val, by rw [sboxCount_eq]; omega⟩

def shardIndex4 (offset : Fin 16) : Fin sboxCount :=
  ⟨64 + offset.val, by rw [sboxCount_eq]; omega⟩

def shardIndex5 (offset : Fin 6) : Fin sboxCount :=
  ⟨80 + offset.val, by rw [sboxCount_eq]; omega⟩

def partialArtifactIndex (round : Fin partialRounds) : Fin sboxCount :=
  ⟨32 + round.val, by
    have roundBound := round.isLt
    simp only [partialRounds] at roundBound
    rw [sboxCount_eq]
    omega⟩

def firstTerminalArtifactIndex (lane : Fin width) : Fin sboxCount :=
  ⟨54 + lane.val, by
    have laneBound := lane.isLt
    simp only [width] at laneBound
    rw [sboxCount_eq]
    omega⟩

def tailArtifactIndex (offset : Fin 2) : Fin sboxCount :=
  ⟨62 + offset.val, by rw [sboxCount_eq]; omega⟩

def partialScheduleForm (round : Fin partialRounds) : Poseidon2Core.LinComb :=
  addConstant (Poseidon2CanonicalConstants.selected.internal round.val)
    (coefficientForm canonicalLayout round.val ⟨0, by decide⟩)

def firstTerminalScheduleForm (lane : Fin width) : Poseidon2Core.LinComb :=
  addConstant (Poseidon2CanonicalConstants.selected.terminal 0 lane)
    (coefficientForm canonicalLayout partialRounds lane)

/-- Compact comparison for one partial-round input. It evaluates the certified
coefficient table and never expands the recursive sparse expression. -/
def PartialScheduleExactAt (round : Fin partialRounds) : Prop :=
  traceTerms (terms expansion (partialScheduleForm round)) =
    traceTerms (traceSboxInput (partialArtifactIndex round))

/-- Compact comparison for the first terminal full round, whose input is the
30-term final partial state. -/
def FirstTerminalScheduleExactAt (lane : Fin width) : Prop :=
  traceTerms (terms expansion (firstTerminalScheduleForm lane)) =
    traceTerms (traceSboxInput (firstTerminalArtifactIndex lane))

def partialShardIndex0 (offset : Fin 4) : Fin partialRounds :=
  ⟨offset.val, by simp only [partialRounds]; omega⟩

def partialShardIndex1 (offset : Fin 4) : Fin partialRounds :=
  ⟨4 + offset.val, by simp only [partialRounds]; omega⟩

def partialShardIndex2 (offset : Fin 4) : Fin partialRounds :=
  ⟨8 + offset.val, by simp only [partialRounds]; omega⟩

def partialShardIndex3 (offset : Fin 4) : Fin partialRounds :=
  ⟨12 + offset.val, by simp only [partialRounds]; omega⟩

def partialShardIndex4 (offset : Fin 4) : Fin partialRounds :=
  ⟨16 + offset.val, by simp only [partialRounds]; omega⟩

def partialShardIndex5 (offset : Fin 2) : Fin partialRounds :=
  ⟨20 + offset.val, by simp only [partialRounds]; omega⟩

/-- The actual native certificate partition covers all 86 S-box inputs once.
The recursive partial block is represented by six compact table shards. -/
def nativeCertificateShardIndices : List Nat :=
  (List.range 16).map (0 + ·) ++
    (List.range 16).map (16 + ·) ++
    (List.range 4).map (32 + ·) ++
    (List.range 4).map (36 + ·) ++
    (List.range 4).map (40 + ·) ++
    (List.range 4).map (44 + ·) ++
    (List.range 4).map (48 + ·) ++
    (List.range 2).map (52 + ·) ++
    (List.range 8).map (54 + ·) ++
    (List.range 2).map (62 + ·) ++
    (List.range 16).map (64 + ·) ++
    (List.range 6).map (80 + ·)

theorem native_certificate_shards_exact :
    nativeCertificateShardIndices = List.range sboxCount ∧
      nativeCertificateShardIndices.Nodup ∧
      [16, 16, 4, 4, 4, 4, 4, 2, 8, 2, 16, 6].sum = sboxCount ∧
      ∀ length ∈ [16, 16, 4, 4, 4, 4, 4, 2, 8, 2, 16, 6],
        length ≤ 16 := by
  native_decide

end Nightstream.Implementation.R1CS.Poseidon2CompactTraceRefinement
