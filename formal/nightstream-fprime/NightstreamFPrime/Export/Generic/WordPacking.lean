import NightstreamFPrime.Export.Generic.BoundedWord

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.WordPacking

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open Fin.CommRing

inductive Bit where
  | zero
  | one
  | column (index : Nat)
  deriving DecidableEq

def Bit.form : Bit → R1CS.LinearCombination
  | .zero => .zero
  | .one => .one
  | .column index => .ofVar index

abbrev Pair := Bit × Bit

def values (env : Env) (pairs : List Pair) : List (F × F) :=
  pairs.map fun pair => (pair.1.form.eval env, pair.2.form.eval env)

def linear : List Pair → R1CS.LinearCombination
  | [] => .zero
  | pair :: rest =>
      (pair.1.form.add (pair.2.form.scale (-1))).add ((linear rest).scale 2)

theorem linear_eval (pairs : List Pair) (env : Env) :
    (linear pairs).eval env = BoundedWord.pack (values env pairs) := by
  induction pairs with
  | nil => rfl
  | cons pair rest ih =>
    simp only [linear, R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_scale,
      ih, values, List.map_cons, BoundedWord.pack, sub_eq_add_neg]
    ring

def row (guard : R1CS.LinearCombination) (pairs : List Pair) : R1CS.Row :=
  ⟨guard, linear pairs, .zero⟩

def Equal (guard : F) (env : Env) (pair : Pair) : Prop :=
  guard * (pair.1.form.eval env - pair.2.form.eval env) = 0

theorem row_iff (guard : R1CS.LinearCombination) (pairs : List Pair) (env : Env)
    (count : pairs.length ≤ 63) (bits : BoundedWord.Bits (values env pairs)) :
    (row guard pairs).Holds env ↔ ∀ pair ∈ pairs, Equal (guard.eval env) env pair := by
  simpa only [row, R1CS.Row.Holds, R1CS.LinearCombination.eval_zero, linear_eval,
    values, List.length_map, List.forall_mem_map, Equal]
    using BoundedWord.common_guard_iff (guard.eval env) (values env pairs)
      (by simpa [values] using count) bits

theorem equal_swap (guard : F) (env : Env) (pair : Pair) :
    Equal guard env pair.swap ↔ Equal guard env pair := by
  unfold Equal
  simp only [Prod.fst_swap, Prod.snd_swap]
  constructor <;> intro holds <;> linear_combination -holds

theorem scaled_guard (guard residual scale : F) (nonzero : scale ≠ 0) :
    (scale * guard) * residual = 0 ↔ guard * residual = 0 := by
  rw [mul_assoc]
  exact ⟨fun holds => (GoldilocksPrime.baseFieldNoZeroDivisors _ _ holds).resolve_left nonzero,
    fun holds => by rw [holds, mul_zero]⟩

structure Source where
  pairs : List Pair
  count : pairs.length ≤ 32
  scale : F
  nonzero : scale ≠ 0
  swapped : Bool

def Source.row (source : Source) (guard : R1CS.LinearCombination) : R1CS.Row :=
  if source.swapped then ⟨linear source.pairs, guard.scale source.scale, .zero⟩
  else ⟨guard.scale source.scale, linear source.pairs, .zero⟩

theorem source_row_iff (source : Source) (guard : R1CS.LinearCombination) (env : Env)
    (bits : BoundedWord.Bits (values env source.pairs)) :
    (source.row guard).Holds env ↔
      ∀ pair ∈ source.pairs, Equal (guard.eval env) env pair := by
  have scaled := scaled_guard (guard.eval env) ((linear source.pairs).eval env)
    source.scale source.nonzero
  have unscaled := row_iff guard source.pairs env (by have := source.count; omega) bits
  unfold Source.row
  split <;> simp only [R1CS.Row.Holds, R1CS.LinearCombination.eval_scale,
    R1CS.LinearCombination.eval_zero]
  · rw [mul_comm, scaled]
    exact unscaled
  · rw [scaled]
    exact unscaled

def Same (left right : Pair) : Prop := left = right ∨ left = right.swap

theorem same_equal {left right : Pair} (same : Same left right) (guard : F) (env : Env) :
    Equal guard env left ↔ Equal guard env right := by
  rcases same with rfl | rfl
  · rfl
  · exact equal_swap guard env right

structure Group where
  guard : R1CS.LinearCombination
  sources : List Source
  chunks : List (List Pair)
  counts : ∀ chunk ∈ chunks, chunk.length ≤ 63
  forward : ∀ source ∈ sources, ∀ pair ∈ source.pairs, pair.1 ≠ pair.2 →
    ∃ chunk ∈ chunks, ∃ target ∈ chunk, Same pair target
  backward : ∀ chunk ∈ chunks, ∀ pair ∈ chunk,
    ∃ source ∈ sources, ∃ original ∈ source.pairs, Same pair original

def Group.original (group : Group) : List R1CS.Row :=
  group.sources.map fun source => source.row group.guard

def Group.target (group : Group) : List R1CS.Row := group.chunks.map (row group.guard)

theorem group_iff (group : Group) (env : Env)
    (sourceBits : ∀ source ∈ group.sources, BoundedWord.Bits (values env source.pairs))
    (targetBits : ∀ chunk ∈ group.chunks, BoundedWord.Bits (values env chunk)) :
    R1CS.RowsHold env group.target ↔ R1CS.RowsHold env group.original := by
  constructor
  · intro holds original member
    obtain ⟨source, sourceMember, rfl⟩ := List.mem_map.mp member
    apply (source_row_iff source group.guard env (sourceBits source sourceMember)).mpr
    intro pair pairMember
    by_cases identical : pair.1 = pair.2
    · simp [Equal, identical]
    obtain ⟨chunk, chunkMember, target, targetMember, same⟩ :=
      group.forward source sourceMember pair pairMember identical
    have targetHolds := holds (row group.guard chunk) (List.mem_map.mpr ⟨chunk, chunkMember, rfl⟩)
    have equal := (row_iff group.guard chunk env (group.counts chunk chunkMember)
      (targetBits chunk chunkMember)).mp targetHolds target targetMember
    exact (same_equal same (group.guard.eval env) env).mpr equal
  · intro holds target member
    obtain ⟨chunk, chunkMember, rfl⟩ := List.mem_map.mp member
    apply (row_iff group.guard chunk env (group.counts chunk chunkMember)
      (targetBits chunk chunkMember)).mpr
    intro pair pairMember
    obtain ⟨source, sourceMember, original, originalMember, same⟩ :=
      group.backward chunk chunkMember pair pairMember
    have originalHolds := holds (source.row group.guard) (List.mem_map.mpr ⟨source, sourceMember, rfl⟩)
    have equal := (source_row_iff source group.guard env (sourceBits source sourceMember)).mp
      originalHolds original originalMember
    exact (same_equal same (group.guard.eval env) env).mpr equal

inductive Certificate (retained : List R1CS.Row) (column : Nat) : Type
  | forced (row : R1CS.Row) (member : row ∈ retained)
      (checked : SlotLayout.forcedColumn? row = some column)
  | packed (pairs : List PackedBoolean.Pair) (count : pairs.length ≤ PackedBoolean.capacity)
      (member : PackedBoolean.row pairs ∈ retained)
      (companion : Nat) (source : (column, companion) ∈ pairs)

def Certificate.Norm {retained : List R1CS.Row} {column : Nat}
    (certificate : Certificate retained column) (env : Env) : Prop :=
  match certificate with
  | .forced .. => True
  | .packed pairs .. => PackedBoolean.Norm (PackedBoolean.values env pairs)

theorem Certificate.valid {retained : List R1CS.Row} {column : Nat}
    (certificate : Certificate retained column) (env : Env) (one : env 0 = 1)
    (norm : certificate.Norm env) (holds : R1CS.RowsHold env retained) :
    LowNormSlot.Valid .bit (env column) := by
  cases certificate with
  | forced row member checked => exact NativeBlock.forced_valid checked one (holds row member)
  | packed pairs count member companion source =>
    have packed := (PackedBoolean.row_holds pairs env).mp (holds _ member)
    exact (PackedBoolean.packed_implies_pairs (PackedBoolean.values env pairs)
      (by simpa [PackedBoolean.values] using count) norm packed
      (env column, env companion) (List.mem_map.mpr ⟨(column, companion), source, rfl⟩)).1

def Certified (retained : List R1CS.Row) (env : Env) (bit : Bit) : Prop :=
  match bit with
  | .zero | .one => True
  | .column column => ∃ certificate : Certificate retained column, certificate.Norm env

theorem certified_valid {retained : List R1CS.Row} {env : Env} {bit : Bit}
    (certified : Certified retained env bit) (one : env 0 = 1)
    (holds : R1CS.RowsHold env retained) :
    bit.form.eval env = 0 ∨ bit.form.eval env = 1 := by
  cases bit with
  | zero => exact Or.inl rfl
  | one => exact Or.inr rfl
  | column column =>
    obtain ⟨certificate, norm⟩ := certified
    simpa only [Bit.form, R1CS.LinearCombination.eval_ofVar, LowNormSlot.Valid]
      using certificate.valid env one norm holds

theorem certified_bits {retained : List R1CS.Row} {env : Env} {pairs : List Pair}
    (certified : ∀ pair ∈ pairs, Certified retained env pair.1 ∧ Certified retained env pair.2)
    (one : env 0 = 1) (holds : R1CS.RowsHold env retained) :
    BoundedWord.Bits (values env pairs) := by
  intro value member
  obtain ⟨pair, inside, rfl⟩ := List.mem_map.mp member
  exact ⟨certified_valid (certified pair inside).1 one holds,
    certified_valid (certified pair inside).2 one holds⟩

theorem retained_iff (groups : List Group) (retained : List R1CS.Row) (env : Env)
    (one : env 0 = 1)
    (sourceCertificates : ∀ group ∈ groups, ∀ source ∈ group.sources, ∀ pair ∈ source.pairs,
      Certified retained env pair.1 ∧ Certified retained env pair.2)
    (targetCertificates : ∀ group ∈ groups, ∀ chunk ∈ group.chunks, ∀ pair ∈ chunk,
      Certified retained env pair.1 ∧ Certified retained env pair.2) :
    R1CS.RowsHold env (retained ++ groups.flatMap Group.target) ↔
      R1CS.RowsHold env (retained ++ groups.flatMap Group.original) := by
  rw [R1CS.rowsHold_append, R1CS.rowsHold_append]
  constructor
  · rintro ⟨kept, lowered⟩
    refine ⟨kept, ?_⟩
    intro row member
    obtain ⟨group, member, rowMember⟩ := List.mem_flatMap.mp member
    apply (group_iff group env
      (fun source inside => certified_bits (sourceCertificates group member source inside) one kept)
      (fun chunk inside => certified_bits (targetCertificates group member chunk inside) one kept)).mp
      (fun row inside => lowered row (List.mem_flatMap.mpr ⟨group, member, inside⟩)) row rowMember
  · rintro ⟨kept, original⟩
    refine ⟨kept, ?_⟩
    intro row member
    obtain ⟨group, member, rowMember⟩ := List.mem_flatMap.mp member
    apply (group_iff group env
      (fun source inside => certified_bits (sourceCertificates group member source inside) one kept)
      (fun chunk inside => certified_bits (targetCertificates group member chunk inside) one kept)).mpr
      (fun row inside => original row (List.mem_flatMap.mpr ⟨group, member, inside⟩)) row rowMember

def Certificate.Slots {width : Nat} {retained : List R1CS.Row} {column : Nat}
    (certificate : Certificate retained column) (slots : List (LowNormAssignment.Slot width)) : Prop :=
  match certificate with
  | .forced .. => True
  | .packed pairs .. => ∀ pair ∈ pairs,
      PackedBoolean.BitSource slots pair.1 ∧ PackedBoolean.BitSource slots pair.2

theorem Certificate.logical_norm {width : Nat} {retained : List R1CS.Row} {column : Nat}
    (certificate : Certificate retained column) (slots : List (LowNormAssignment.Slot width))
    (sources : certificate.Slots slots)
    (assignment : Assignment F (ProductionAssignment.logicalWidth slots))
    (norm : ∀ coordinate, centeredMagnitude (assignment coordinate) < 2) :
    certificate.Norm (LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment) := by
  cases certificate with
  | forced row member checked => trivial
  | packed pairs count member companion source =>
    intro value inside
    obtain ⟨pair, pairMember, rfl⟩ := List.mem_map.mp inside
    exact ⟨PackedBoolean.source_slot_norm slots (sources pair pairMember).1 assignment norm,
      PackedBoolean.source_slot_norm slots (sources pair pairMember).2 assignment norm⟩

def Source.matches (source : Source) (guard : R1CS.LinearCombination) (original : R1CS.Row) : Bool :=
  decide (original = source.row guard)

theorem matched_source_iff (source : Source) (guard : R1CS.LinearCombination)
    (original : R1CS.Row) (matched : source.matches guard original = true) (env : Env)
    (bits : BoundedWord.Bits (values env source.pairs)) :
    original.Holds env ↔ ∀ pair ∈ source.pairs, Equal (guard.eval env) env pair := by
  have same : original = source.row guard := of_decide_eq_true matched
  rw [same]
  exact source_row_iff source guard env bits

theorem rows_iff_of_members (left right : List R1CS.Row)
    (same : ∀ row, row ∈ left ↔ row ∈ right) (env : Env) :
    R1CS.RowsHold env left ↔ R1CS.RowsHold env right := by
  exact ⟨fun holds row member => holds row ((same row).mpr member),
    fun holds row member => holds row ((same row).mp member)⟩

theorem ordered_iff (groups : List Group) (retained original target : List R1CS.Row)
    (sourceOrder : ∀ row, row ∈ original ↔ row ∈ retained ++ groups.flatMap Group.original)
    (targetOrder : ∀ row, row ∈ target ↔ row ∈ retained ++ groups.flatMap Group.target)
    (env : Env) (one : env 0 = 1)
    (sourceCertificates : ∀ group ∈ groups, ∀ source ∈ group.sources, ∀ pair ∈ source.pairs,
      Certified retained env pair.1 ∧ Certified retained env pair.2)
    (targetCertificates : ∀ group ∈ groups, ∀ chunk ∈ group.chunks, ∀ pair ∈ chunk,
      Certified retained env pair.1 ∧ Certified retained env pair.2) :
    R1CS.RowsHold env target ↔ R1CS.RowsHold env original := by
  exact (rows_iff_of_members target _ targetOrder env).trans
    ((retained_iff groups retained env one sourceCertificates targetCertificates).trans
      (rows_iff_of_members original _ sourceOrder env).symm)

end NightstreamFPrime.Export.Generic.WordPacking
