import NightstreamFPrime.Layout.ProductionRelation
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-!
Counted sparse-form construction and duplicate-aware coefficient lookup.
The loops retain stored entries and charge their actual traversals. Bounds
use entry-list length, not the number of distinct columns. These are named
operation clocks; source generation and machine-runtime costs are separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.MatrixProgram.SparseWork

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Layout.ProductionRelation
open _root_.NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork (Result)

private def reverseOnto {columns : Nat} :
    List (SparseEntry columns) → List (SparseEntry columns) → Nat → Result (List (SparseEntry columns))
  | [], suffix, clock => ⟨suffix, clock + 2⟩
  | entry :: rest, suffix, clock => reverseOnto rest (entry :: suffix) (clock + 5)

private theorem reverseOnto_value {columns : Nat} (entries suffix : List (SparseEntry columns))
    (clock : Nat) : (reverseOnto entries suffix clock).value = entries.reverse ++ suffix := by
  induction entries generalizing suffix clock with
  | nil => rfl
  | cons entry rest ih =>
      simp only [reverseOnto, ih, List.reverse_cons, List.append_assoc, List.singleton_append]

private theorem reverseOnto_work {columns : Nat} (entries suffix : List (SparseEntry columns))
    (clock : Nat) : (reverseOnto entries suffix clock).work = clock + 5 * entries.length + 2 := by
  induction entries generalizing suffix clock with
  | nil => simp [reverseOnto]
  | cons entry rest ih =>
      simp only [reverseOnto, ih, List.length_cons, Nat.mul_add, Nat.mul_one]
      omega

/-- Construct the empty list, form and result. -/
def empty {columns : Nat} (_ : Unit) : Result (SparseForm columns) := ⟨⟨[]⟩, 3⟩

theorem empty_value {columns : Nat} : (empty (columns := columns) ()).value = SparseForm.empty := rfl

theorem empty_work {columns : Nat} : (empty (columns := columns) ()).work = 3 := rfl

/-- Construct the entry, empty tail, list cell, form and result. -/
def singleton {columns : Nat} (column : Fin columns) (value : F) : Result (SparseForm columns) :=
  ⟨⟨[⟨column, value⟩]⟩, 5⟩

theorem singleton_value {columns : Nat} (column : Fin columns) (value : F) :
    (singleton column value).value = SparseForm.singleton column value := rfl

theorem singleton_work {columns : Nat} (column : Fin columns) (value : F) :
    (singleton column value).work = 5 := rfl

/-- Two tail-recursive passes copy the left entries and share the right
tail. Each step charges dispatch, head/tail reads, cons and the next call.
The wrapper charges list/result reads, calls, and constructors. -/
def add {columns : Nat} (left right : SparseForm columns) : Result (SparseForm columns) :=
  let reversed := reverseOnto left.entries [] 0
  let result := reverseOnto reversed.value right.entries reversed.work
  ⟨⟨result.value⟩, result.work + 9⟩

theorem add_value {columns : Nat} (left right : SparseForm columns) :
    (add left right).value = SparseForm.add left right := by
  simp only [add, reverseOnto_value, List.append_nil, List.reverse_reverse, SparseForm.add]

theorem add_work {columns : Nat} (left right : SparseForm columns) :
    (add left right).work = 10 * left.entries.length + 13 := by
  simp only [add, reverseOnto_work, reverseOnto_value, List.length_append,
    List.length_reverse, List.length_nil, Nat.add_zero, Nat.zero_add]
  omega

private def scaleLoop {columns : Nat} (scalar : F) :
    List (SparseEntry columns) → List (SparseEntry columns) → Nat → Result (List (SparseEntry columns))
  | [], reversed, clock =>
      let result := reverseOnto reversed [] clock
      ⟨result.value, result.work + 5⟩
  | entry :: rest, reversed, clock =>
      scaleLoop scalar rest (⟨entry.column, scalar * entry.coefficient⟩ :: reversed) (clock + 9)

private theorem scaleLoop_value {columns : Nat} (scalar : F)
    (entries reversed : List (SparseEntry columns)) (clock : Nat) :
    (scaleLoop scalar entries reversed clock).value = reversed.reverse ++
      entries.map (fun entry => ⟨entry.column, scalar * entry.coefficient⟩) := by
  induction entries generalizing reversed clock with
  | nil => simp [scaleLoop, reverseOnto_value]
  | cons entry rest ih =>
      simp only [scaleLoop, ih, List.reverse_cons, List.map_cons,
        List.append_assoc, List.singleton_append]

private theorem scaleLoop_work {columns : Nat} (scalar : F)
    (entries reversed : List (SparseEntry columns)) (clock : Nat) :
    (scaleLoop scalar entries reversed clock).work =
      clock + 9 * entries.length + 5 * (reversed.length + entries.length) + 7 := by
  induction entries generalizing reversed clock with
  | nil => simp [scaleLoop, reverseOnto_work]
  | cons entry rest ih =>
      simp only [scaleLoop, ih, List.length_cons, Nat.mul_add, Nat.mul_one]
      omega

/-- Each mapped entry charges list dispatch/reads, both entry fields, field
multiplication, the entry and list constructors, and the recursive call.
The final reverse and outer form construction are included. -/
def scale {columns : Nat} (scalar : F) (form : SparseForm columns) : Result (SparseForm columns) :=
  let result := scaleLoop scalar form.entries [] 0
  ⟨⟨result.value⟩, result.work + 6⟩

theorem scale_value {columns : Nat} (scalar : F) (form : SparseForm columns) :
    (scale scalar form).value = SparseForm.scale scalar form := by
  simp only [scale, scaleLoop_value, List.reverse_nil, List.nil_append, SparseForm.scale]

theorem scale_work {columns : Nat} (scalar : F) (form : SparseForm columns) :
    (scale scalar form).work = 14 * form.entries.length + 13 := by
  simp only [scale, scaleLoop_work, List.length_nil, Nat.zero_add]
  omega

private def coefficientLoop {columns : Nat} (wanted : Nat) :
    List (SparseEntry columns) → F → Nat → Result F
  | [], accumulated, clock => ⟨accumulated, clock + 2⟩
  | entry :: rest, accumulated, clock =>
      if entry.column.val = wanted then
        coefficientLoop wanted rest (accumulated + entry.coefficient) (clock + 10)
      else coefficientLoop wanted rest accumulated (clock + 8)

private theorem coefficientLoop_value {columns : Nat} (column : Fin columns)
    (entries : List (SparseEntry columns)) (accumulated : F) (clock : Nat) :
    (coefficientLoop column.val entries accumulated clock).value =
      entries.foldl (fun total entry =>
        if entry.column = column then total + entry.coefficient else total) accumulated := by
  induction entries generalizing accumulated clock with
  | nil => rfl
  | cons entry rest ih =>
      by_cases same : entry.column.val = column.val <;>
        simp only [coefficientLoop, same, ↓reduceIte, ih, List.foldl_cons, Fin.ext_iff]

private theorem coefficientLoop_work_le {columns : Nat} (wanted : Nat)
    (entries : List (SparseEntry columns)) (accumulated : F) (clock : Nat) :
    (coefficientLoop wanted entries accumulated clock).work ≤ clock + 10 * entries.length + 2 := by
  induction entries generalizing accumulated clock with
  | nil => simp [coefficientLoop]
  | cons entry rest ih =>
      have hit := ih (accumulated + entry.coefficient) (clock + 10)
      have miss := ih accumulated (clock + 8)
      simp only [coefficientLoop, List.length_cons, Nat.mul_add, Nat.mul_one]
      split <;> omega

/-- Read the wanted index once. Every stored entry is tested; all matches,
including repeated columns, contribute to the coefficient in list order. -/
def coefficient {columns : Nat} (form : SparseForm columns) (column : Fin columns) : Result F :=
  let result := coefficientLoop column.val form.entries 0 0
  ⟨result.value, result.work + 6⟩

theorem coefficient_value {columns : Nat} (form : SparseForm columns) (column : Fin columns) :
    (coefficient form column).value = form.coefficient column :=
  coefficientLoop_value column form.entries 0 0

theorem coefficient_work_le {columns : Nat} (form : SparseForm columns) (column : Fin columns) :
    (coefficient form column).work ≤ 10 * form.entries.length + 8 := by
  have bounded := coefficientLoop_work_le column.val form.entries 0 0
  change (coefficientLoop column.val form.entries 0 0).work + 6 ≤ _
  omega

end NightstreamFPrime.Export.MatrixProgram.SparseWork
