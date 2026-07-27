import Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate

/-!
Contract: relate the compact 8-by-30 partial-round coefficient certificate to
the actual sparse combinations emitted by the canonical Poseidon2 schedule.

Owns: the label-to-column map, basis-evaluation induction for every partial
round, and survival of every normalized partial-state coefficient.

Does not own: the closed finite arithmetic check (the certificate module), row
emission, or whole-program coefficient accounting.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Eval
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Matrices
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Support
open Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientCertificate

/-- Label `0..7` names the last initial-full-round outputs; every later label
names the corresponding partial-round output.  Both families are consecutive
S-box outputs, so one arithmetic map owns their exact column order. -/
def coefficientColumn (layout : Layout) (label : Nat) : Nat :=
  sboxOutput layout
    (initialSboxIndex (halfFullRounds - 1) 0 + label)

theorem coefficientColumn_injective (layout : Layout) :
    ∀ left right, coefficientColumn layout left = coefficientColumn layout right →
      left = right := by
  intro left right equal
  exact Nat.add_left_cancel
    (sboxOutput_injective layout _ _ equal)

theorem coefficientColumn_initial
    (layout : Layout) (lane : Fin width) :
    coefficientColumn layout lane.val =
      sboxOutput layout
        (initialSboxIndex (halfFullRounds - 1) lane.val) := by
  apply congrArg (sboxOutput layout)
  unfold initialSboxIndex
  omega

theorem coefficientColumn_partial
    (layout : Layout) (round : Nat) :
    coefficientColumn layout (width + round) =
      sboxOutput layout (partialSboxIndex round) := by
  apply congrArg (sboxOutput layout)
  unfold initialSboxIndex partialSboxIndex
  simp only [halfFullRounds, width]
  omega

theorem partialSupportList_has_label
    (layout : Layout) (round : Nat) (column : Nat)
    (member : column ∈ partialSupportList layout round) :
    ∃ label, label < width + round ∧
      column = coefficientColumn layout label := by
  induction round with
  | zero =>
      simp only [partialSupportList, fullRoundOutputs, List.mem_map] at member
      rcases member with ⟨source, _, image⟩
      exact ⟨source.val, by simpa using source.isLt, by
        rw [coefficientColumn_initial]
        exact image.symm⟩
  | succ previous hypothesis =>
      simp only [partialSupportList, List.mem_cons] at member
      rcases member with newest | older
      · exact ⟨width + previous, by omega, by
          rw [coefficientColumn_partial]
          exact newest⟩
      · rcases hypothesis older with ⟨label, bound, image⟩
        exact ⟨label, by omega, image⟩

private theorem sum_ite_zero_of_not_mem
    {α : Type} [DecidableEq α] (items : List α) (target : α)
    (f : α → Nat) (absent : target ∉ items) :
    (items.map (fun item => if item = target then f item else 0)).sum = 0 := by
  induction items with
  | nil => rfl
  | cons head tail hypothesis =>
      have headNe : head ≠ target := by
        intro same
        apply absent
        exact List.mem_cons.2 (Or.inl same.symm)
      have tailAbsent : target ∉ tail := by
        intro member
        exact absent (List.mem_cons_of_mem _ member)
      simp [headNe, hypothesis tailAbsent]

private theorem sum_ite_single
    {α : Type} [DecidableEq α] (items : List α) (target : α)
    (f : α → Nat) (nodup : items.Nodup) (member : target ∈ items) :
    (items.map (fun item => if item = target then f item else 0)).sum =
      f target := by
  induction items with
  | nil => simp at member
  | cons head tail hypothesis =>
      rw [List.nodup_cons] at nodup
      rcases List.mem_cons.1 member with same | inTail
      · subst target
        simp [sum_ite_zero_of_not_mem tail head f nodup.1]
      · have headNe : head ≠ target := by
          intro same
          exact nodup.1 (same ▸ inTail)
        simp [headNe, hypothesis nodup.2 inTail]

private theorem applyMatrixValues_basis
    (matrix : Fin width → Fin width → Nat)
    (target source : Fin width)
    (bounded : matrix target source < goldilocksP) :
    applyMatrixValues matrix
      (fun current => if current = source then 1 else 0) target =
        matrix target source := by
  unfold applyMatrixValues
  rw [show
      (fun current : Fin width =>
        matrix target current * (if current = source then 1 else 0)) =
      (fun current : Fin width =>
        if current = source then matrix target current else 0) by
      funext current
      split <;> simp_all]
  rw [sum_ite_single (List.finRange width) source
    (fun current => matrix target current) (by decide)
    (List.mem_finRange source)]
  exact Nat.mod_eq_of_lt bounded

private theorem basis_initial_source
    (layout : Layout) (source label : Fin width) :
    lcEval (basisAssignment (coefficientColumn layout label.val))
      [(sboxOutput layout
        (initialSboxIndex (halfFullRounds - 1) source.val), 1)] =
      if source = label then 1 else 0 := by
  rw [← coefficientColumn_initial layout source, lcEval_basis_singleton]
  by_cases same : source = label
  · subst label
    simp [Nat.mod_eq_of_lt (by decide : 1 < goldilocksP)]
  · have columnsNe :
        coefficientColumn layout source.val ≠
          coefficientColumn layout label.val := by
      intro equal
      exact same (Fin.ext (coefficientColumn_injective layout _ _ equal))
    simp [columnsNe, same]

theorem partialState_basis_eval
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) (label : Nat) (labelBound : label < width + round) :
    lcEval (basisAssignment (coefficientColumn layout label))
        (partialState layout round lane) =
      tableValue (tableAt round) lane label := by
  induction round generalizing lane label with
  | zero =>
      have labelLt : label < width := by simpa using labelBound
      let labelFin : Fin width := ⟨label, labelLt⟩
      change
        lcEval (basisAssignment (coefficientColumn layout label))
            (applyMatrix externalMatrix
              (fun source =>
                [(sboxOutput layout
                  (initialSboxIndex (halfFullRounds - 1) source.val), 1)])
              lane) =
          tableValue (tableAt 0) lane label
      rw [lcEval_applyMatrix]
      have sources :
          (fun source : Fin width =>
            lcEval (basisAssignment (coefficientColumn layout label))
              [(sboxOutput layout
                (initialSboxIndex (halfFullRounds - 1) source.val), 1)]) =
          (fun source => if source = labelFin then 1 else 0) := by
        funext source
        exact basis_initial_source layout source labelFin
      rw [sources, tableAt_zero]
      have labelSupport : label < supportWidth := by
        unfold supportWidth
        omega
      let supportLabel : Fin supportWidth := ⟨label, labelSupport⟩
      rw [show label = supportLabel.val from rfl,
        tableValue_initial lane supportLabel]
      rw [dif_pos labelLt]
      simpa only [labelFin] using
        applyMatrixValues_basis externalMatrix lane labelFin
          (externalMatrix_lt lane labelFin)
  | succ previous hypothesis =>
      have previousBound : previous ≤ partialRounds := by omega
      have labelSupport : label < supportWidth := by
        unfold supportWidth
        omega
      let supportLabel : Fin supportWidth := ⟨label, labelSupport⟩
      rw [partialState, lcEval_applyMatrix, tableAt_succ,
        show label = supportLabel.val from rfl,
        tableValue_next previous (tableAt previous) lane supportLabel]
      congr 2
      funext source
      by_cases sourceZero : source.val = 0
      · simp [sourceZero]
        have sourceEq : source = ⟨0, by decide⟩ := Fin.ext sourceZero
        subst source
        change
          lcEval (basisAssignment (coefficientColumn layout label))
              [(sboxOutput layout (partialSboxIndex previous), 1)] =
            if label = width + previous then 1 else 0
        rw [← coefficientColumn_partial, lcEval_basis_singleton]
        by_cases newest : label = width + previous
        · subst label
          simp [Nat.mod_eq_of_lt (by decide : 1 < goldilocksP)]
        · have columnsNe :
              coefficientColumn layout (width + previous) ≠
                coefficientColumn layout label := by
            intro equal
            exact newest
              (coefficientColumn_injective layout _ _ equal).symm
          simp [newest, columnsNe]
      · simp [sourceZero]
        by_cases newest : label = width + previous
        · subst label
          rw [tableAt_inactive_zero previous source (width + previous)
            (by omega) labelSupport]
          change
            lcEval
                (basisAssignment
                  (coefficientColumn layout (width + previous)))
                (partialState layout previous source) = 0
          apply lcEval_basis_not_mentions
          intro mentioned
          have member := partialState_mentions_subset layout previous source
            _ mentioned
          rcases partialSupportList_index layout previous _ member with
            ⟨index, indexBound, image⟩
          rw [coefficientColumn_partial] at image
          have sameIndex := sboxOutput_injective layout _ _ image.symm
          simp only [partialSboxIndex] at sameIndex
          omega
        · have oldBound : label < width + previous := by omega
          change
            lcEval (basisAssignment (coefficientColumn layout label))
                (partialState layout previous source) =
              tableValue (tableAt previous) source label
          exact hypothesis previousBound source label oldBound

theorem partialState_normalized_coefficients_nonzero
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) (entry : Nat × Nat)
    (member : entry ∈ normalize (partialState layout round lane)) :
    entry.2 % goldilocksP ≠ 0 := by
  have mentioned : Mentions (partialState layout round lane) entry.1 := by
    exact (mentions_normalize _ _).1
      (List.mem_map.2 ⟨entry, member, rfl⟩)
  have supportMember :=
    partialState_mentions_subset layout round lane entry.1 mentioned
  have labelled :=
    partialSupportList_has_label layout round entry.1 supportMember
  rcases labelled with ⟨label, labelBound, columnEq⟩
  have coefficientRead :=
    lcEval_basis_normalized_entry (partialState layout round lane) entry member
  have tableRead :=
    partialState_basis_eval layout round roundBound lane label labelBound
  rw [columnEq] at coefficientRead
  rw [tableRead] at coefficientRead
  rw [← coefficientRead]
  exact tableAt_nonzero round roundBound lane label labelBound

/-- Every coefficient in a partial-round state survives field normalization,
so its exact nonzero count is the structural support count `8 + round`. -/
theorem partialState_fieldNormalize_length
    (layout : Layout) (round : Nat) (roundBound : round ≤ partialRounds)
    (lane : Fin width) :
    (fieldNormalize (partialState layout round lane)).length = width + round := by
  have survives :
      ∀ entry ∈ normalize (partialState layout round lane),
        entry.2 % goldilocksP ≠ 0 :=
    partialState_normalized_coefficients_nonzero layout round roundBound lane
  rw [fieldNormalize_length_of_nonzero _ survives,
    partialState_normalize_length layout round lane]

end Nightstream.Implementation.R1CS.Canonical.Poseidon2PartialCoefficientBridge
