import Batteries.Data.List.Perm
import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: compact sparse forms for the Poseidon2 partial-round state.

Owns: reconstruction of each partial state from the certified 8-by-30
coefficient recurrence and semantic equality with the recursive schedule.

Does not own: Rust column placement, round-constant conformance, or trace-row
refinement.

Emits constraints: no.

Assurance tier: model-level for property
`POSEIDON2-PARTIAL-COEFFICIENT-FORM`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate
open Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge

/-- One entry per active support label. The table already stores canonical
Goldilocks coefficients, so this list never expands a recursive linear form. -/
def coefficientForm (layout : Layout) (round : Nat) (lane : Fin width) :
    Poseidon2Core.LinComb :=
  (List.range (width + round)).map fun label =>
    (coefficientColumn layout label, tableValue (tableAt round) lane label)

theorem coefficientForm_length (layout : Layout) (round : Nat)
    (lane : Fin width) :
    (coefficientForm layout round lane).length = width + round := by
  simp [coefficientForm]

theorem coefficientForm_nodup (layout : Layout) (round : Nat)
    (lane : Fin width) :
    (coefficientForm layout round lane).Nodup := by
  unfold coefficientForm
  exact nodup_map (List.range (width + round)) _
    (fun left right image =>
      coefficientColumn_injective layout left right
        (congrArg Prod.fst image))
    List.nodup_range

/-- Every certified table entry is the exact field-normalized coefficient of
the recursive partial state at the same column. -/
theorem coefficientForm_entry_mem_fieldNormalize
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) (label : Nat) (labelBound : label < width + round) :
    (coefficientColumn layout label,
        tableValue (tableAt round) lane label) ∈
      fieldNormalize (partialState layout round lane) := by
  have valueNonzero :=
    tableAt_nonzero round roundBound lane label labelBound
  have sourceEval :=
    partialState_basis_eval layout round roundBound lane label labelBound
  have mentioned :
      Mentions (partialState layout round lane)
        (coefficientColumn layout label) := by
    by_cases present : Mentions (partialState layout round lane)
        (coefficientColumn layout label)
    · exact present
    · have zero := lcEval_basis_not_mentions
        (partialState layout round lane) (coefficientColumn layout label) present
      exfalso
      apply valueNonzero
      rw [← sourceEval, zero]
  have normalizedMention :
      Mentions (normalize (partialState layout round lane))
        (coefficientColumn layout label) :=
    (mentions_normalize _ _).2 mentioned
  rcases List.mem_map.1 normalizedMention with
    ⟨entry, entryMember, entryColumn⟩
  have coefficientRead :=
    lcEval_basis_normalized_entry (partialState layout round lane)
      entry entryMember
  rw [entryColumn, sourceEval] at coefficientRead
  have entryNonzero :=
    partialState_normalized_coefficients_nonzero layout round roundBound lane
      entry entryMember
  apply List.mem_filterMap.2
  refine ⟨entry, entryMember, ?_⟩
  simp [reduceTerm, entryNonzero, entryColumn, coefficientRead]

theorem coefficientForm_subset_fieldNormalize
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) :
    coefficientForm layout round lane ⊆
      fieldNormalize (partialState layout round lane) := by
  intro entry member
  rcases List.mem_map.1 member with ⟨label, labelMember, rfl⟩
  exact coefficientForm_entry_mem_fieldNormalize layout round roundBound lane
    label (List.mem_range.1 labelMember)

/-- The compact table form is a permutation of the field-normalized recursive
state. Exact length prevents a missing or duplicate coefficient from passing. -/
theorem coefficientForm_perm_fieldNormalize
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) :
    (coefficientForm layout round lane).Perm
      (fieldNormalize (partialState layout round lane)) := by
  have subperm := List.subperm_of_subset
    (coefficientForm_nodup layout round lane)
    (coefficientForm_subset_fieldNormalize layout round roundBound lane)
  apply subperm.perm_of_length_le
  rw [partialState_fieldNormalize_length layout round roundBound lane,
    coefficientForm_length]
  exact Nat.le_refl _

/-- The compact form evaluates exactly like the recursive partial state for
every assignment. -/
theorem lcEval_coefficientForm
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) (assignment : Nat → Nat) :
    lcEval assignment (coefficientForm layout round lane) =
      lcEval assignment (partialState layout round lane) := by
  calc
    lcEval assignment (coefficientForm layout round lane) =
        lcEval assignment (fieldNormalize (partialState layout round lane)) :=
      Program.lcEval_eq_of_perm assignment
        (coefficientForm_perm_fieldNormalize layout round roundBound lane)
    _ = lcEval assignment (partialState layout round lane) :=
      lcEval_fieldNormalize assignment _

/-- Adding the same round constant preserves the compact-state equality. -/
theorem lcEval_addConstant_coefficientForm
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) (constant : Nat) (assignment : Nat → Nat) :
    lcEval assignment (addConstant constant (coefficientForm layout round lane)) =
      lcEval assignment (addConstant constant (partialState layout round lane)) := by
  have stateEval :=
    lcEval_coefficientForm layout round roundBound lane assignment
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum] at stateEval
  rw [lcEval_eq_rawSum, lcEval_eq_rawSum]
  simp only [addConstant, rawSum_cons]
  calc
    (constant * assignment 0 +
          rawSum assignment (coefficientForm layout round lane)) % goldilocksP =
        (constant * assignment 0 % goldilocksP +
          rawSum assignment (coefficientForm layout round lane) % goldilocksP) %
            goldilocksP := Nat.add_mod _ _ _
    _ = (constant * assignment 0 % goldilocksP +
          rawSum assignment (partialState layout round lane) % goldilocksP) %
            goldilocksP := by rw [stateEval]
    _ = (constant * assignment 0 +
          rawSum assignment (partialState layout round lane)) % goldilocksP :=
      (Nat.add_mod _ _ _).symm

end Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientForm
