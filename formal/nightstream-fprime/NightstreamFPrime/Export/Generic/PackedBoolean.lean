import NightstreamFPrime.Export.Generic.BooleanInference
import NightstreamFPrime.Layout.ProductionRelation.AcceptedOpening

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.PackedBoolean

open NightstreamFPrime.Spec NightstreamFPrime.Circuit NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Export.BoundedDepth.SlotCompleteness
open Fin.CommRing

def capacity : Nat := 31

def target : Nat → Nat
  | 0 => 0
  | n + 1 => 3 + 4 * target n

def packNat : List Nat → Nat
  | [] => 0
  | x :: xs => x + 4 * packNat xs

theorem target_mono : Monotone target := by
  apply monotone_nat_of_le_succ
  intro n
  simp only [target]
  omega

theorem packNat_bound (digits : List Nat) (small : ∀ d ∈ digits, d ≤ 4) :
    packNat digits ≤ 2 * target digits.length := by
  induction digits with
  | nil => simp [packNat, target]
  | cons d ds ih =>
    have head := small d List.mem_cons_self
    have tail := ih (fun x member => small x (List.mem_cons_of_mem _ member))
    simp only [packNat, List.length_cons, target]
    omega

theorem target_capacity_lt_modulus : 2 * target capacity < goldilocksModulus := by
  decide

theorem packNat_below_modulus (digits : List Nat) (count : digits.length ≤ capacity)
    (small : ∀ d ∈ digits, d ≤ 4) : packNat digits < goldilocksModulus := by
  have bound := packNat_bound digits small
  have limit := target_mono count
  have maximum := target_capacity_lt_modulus
  omega

theorem packNat_eq_target (digits : List Nat) (small : ∀ d ∈ digits, d ≤ 4)
    (equal : packNat digits = target digits.length) : ∀ d ∈ digits, d = 3 := by
  induction digits with
  | nil => simp
  | cons d ds ih =>
    have head := small d List.mem_cons_self
    simp only [packNat, List.length_cons, target] at equal
    have three : d = 3 := by omega
    have rest : packNat ds = target ds.length := by omega
    intro value member
    rcases List.mem_cons.mp member with rfl | member
    · exact three
    · exact ih (fun x inside => small x (List.mem_cons_of_mem _ inside)) rest value member

def unitDigit (value : F) : Nat :=
  if value = -1 then 0 else if value = 0 then 1 else 2

theorem unit_cases {value : F} (norm : centeredMagnitude value < 2) :
    value = -1 ∨ value = 0 ∨ value = 1 := by
  rcases (NormRange.strictNormTwo_iff_representedRoot value).mp norm with negative | zero | one
  · exact Or.inl (negative.trans NativeBlock.representedMinusOne_eq)
  · exact Or.inr (Or.inl zero)
  · exact Or.inr (Or.inr one)

theorem unitDigit_le (value : F) : unitDigit value ≤ 2 := by
  unfold unitDigit
  split
  · omega
  · split <;> omega

private theorem unitDigit_neg_one : unitDigit (-1 : F) = 0 := by decide
private theorem unitDigit_zero : unitDigit (0 : F) = 1 := by decide
private theorem unitDigit_one : unitDigit (1 : F) = 2 := by decide

theorem unitDigit_field {value : F} (norm : centeredMagnitude value < 2) :
    fieldOfNat (unitDigit value) = value + 1 := by
  rcases unit_cases norm with rfl | rfl | rfl <;> decide

def pack : List (F × F) → F
  | [] => 0
  | (x, y) :: rest => x + y - 1 + 4 * pack rest

def digit (pair : F × F) : Nat := unitDigit pair.1 + unitDigit pair.2

def Norm (pairs : List (F × F)) : Prop :=
  ∀ pair ∈ pairs, centeredMagnitude pair.1 < 2 ∧ centeredMagnitude pair.2 < 2

theorem digit_le (pair : F × F) : digit pair ≤ 4 := by
  have left := unitDigit_le pair.1
  have right := unitDigit_le pair.2
  unfold digit
  omega

theorem digit_field {pair : F × F}
    (norm : centeredMagnitude pair.1 < 2 ∧ centeredMagnitude pair.2 < 2) :
    fieldOfNat (digit pair) = pair.1 + pair.2 + 2 := by
  rw [digit, fieldOfNat_add, unitDigit_field norm.1, unitDigit_field norm.2]
  ring

theorem pack_field (pairs : List (F × F)) (norm : Norm pairs) :
    fieldOfNat (packNat (pairs.map digit)) = pack pairs + fieldOfNat (target pairs.length) := by
  induction pairs with
  | nil => simp [packNat, pack, target]
  | cons pair rest ih =>
    have first := digit_field (norm pair List.mem_cons_self)
    have remaining := ih (fun p member => norm p (List.mem_cons_of_mem _ member))
    simp only [List.map_cons, List.length_cons, packNat, target, fieldOfNat_add, fieldOfNat_mul]
    rw [first, remaining]
    rcases pair with ⟨x, y⟩
    change x + y + 2 + 4 * (pack rest + fieldOfNat (target rest.length)) =
      x + y - 1 + 4 * pack rest + (3 + 4 * fieldOfNat (target rest.length))
    ring

theorem digit_eq_three {pair : F × F}
    (norm : centeredMagnitude pair.1 < 2 ∧ centeredMagnitude pair.2 < 2)
    (three : digit pair = 3) : LowNormSlot.Valid .bit pair.1 ∧ pair.2 = 1 - pair.1 := by
  rcases pair with ⟨x, y⟩
  rcases unit_cases norm.1 with rfl | rfl | rfl <;>
    rcases unit_cases norm.2 with rfl | rfl | rfl <;>
    norm_num only [digit, unitDigit_neg_one, unitDigit_zero, unitDigit_one] at three
  all_goals unfold LowNormSlot.Valid; decide

theorem packed_implies_pairs (pairs : List (F × F)) (count : pairs.length ≤ capacity)
    (norm : Norm pairs) (zero : pack pairs = 0) :
    ∀ pair ∈ pairs, LowNormSlot.Valid .bit pair.1 ∧ pair.2 = 1 - pair.1 := by
  have small : ∀ d ∈ pairs.map digit, d ≤ 4 := by
    intro d member
    obtain ⟨p, _, rfl⟩ := List.mem_map.mp member
    exact digit_le p
  have packedBound := packNat_below_modulus (pairs.map digit) (by simpa using count) small
  have targetBound : target pairs.length < goldilocksModulus := by
    have limit := target_mono count
    have maximum := target_capacity_lt_modulus
    omega
  have field := pack_field pairs norm
  rw [zero, zero_add] at field
  have equal : packNat (pairs.map digit) = target pairs.length := by
    have values := congrArg Fin.val field
    simpa only [fieldOfNat, Nat.mod_eq_of_lt packedBound, Nat.mod_eq_of_lt targetBound] using values
  have allThree := packNat_eq_target (pairs.map digit) small (by simpa using equal)
  intro pair member
  exact digit_eq_three (norm pair member) (allThree _ (List.mem_map.mpr ⟨pair, member, rfl⟩))

theorem pairs_imply_packed (pairs : List (F × F))
    (complements : ∀ pair ∈ pairs, pair.2 = 1 - pair.1) : pack pairs = 0 := by
  induction pairs with
  | nil => rfl
  | cons pair rest ih =>
    have first := complements pair List.mem_cons_self
    have remaining := ih (fun p member => complements p (List.mem_cons_of_mem _ member))
    rcases pair with ⟨x, y⟩
    change y = 1 - x at first
    simp only [pack, remaining, mul_zero, add_zero]
    rw [first]
    ring

theorem packed_iff_pairs (pairs : List (F × F)) (count : pairs.length ≤ capacity)
    (norm : Norm pairs) : pack pairs = 0 ↔
      ∀ pair ∈ pairs, LowNormSlot.Valid .bit pair.1 ∧ pair.2 = 1 - pair.1 := by
  exact ⟨packed_implies_pairs pairs count norm,
    fun pairsHold => pairs_imply_packed pairs (fun p member => (pairsHold p member).2)⟩

theorem honest_pairs_norm (values : List F)
    (bits : ∀ value ∈ values, LowNormSlot.Valid .bit value) :
    Norm (values.map fun value => (value, 1 - value)) := by
  intro pair member
  obtain ⟨value, inside, rfl⟩ := List.mem_map.mp member
  rcases bits value inside with rfl | rfl <;> decide

theorem honest_pairs_packed (values : List F) :
    pack (values.map fun value => (value, 1 - value)) = 0 := by
  apply pairs_imply_packed
  intro pair member
  obtain ⟨value, _, rfl⟩ := List.mem_map.mp member
  rfl

abbrev Pair := Nat × Nat

def values (env : Env) (pairs : List Pair) : List (F × F) :=
  pairs.map fun pair => (env pair.1, env pair.2)

def linear : List Pair → R1CS.LinearCombination
  | [] => .zero
  | pair :: rest =>
      .add (.add (.add (.ofVar pair.1) (.ofVar pair.2)) (.const (-1)))
        (.scale 4 (linear rest))

def row (pairs : List Pair) : R1CS.Row := ⟨linear pairs, .one, .zero⟩

theorem linear_eval (pairs : List Pair) (env : Env) :
    (linear pairs).eval env = pack (values env pairs) := by
  induction pairs with
  | nil => rfl
  | cons pair rest ih =>
    simp only [linear, R1CS.LinearCombination.eval_add, R1CS.LinearCombination.eval_ofVar,
      R1CS.LinearCombination.eval_const, R1CS.LinearCombination.eval_scale, ih]
    change env pair.1 + env pair.2 + -1 + 4 * pack (values env rest) =
      env pair.1 + env pair.2 - 1 + 4 * pack (values env rest)
    rw [sub_eq_add_neg]

theorem row_holds (pairs : List Pair) (env : Env) :
    (row pairs).Holds env ↔ pack (values env pairs) = 0 := by
  simp only [row, R1CS.Row.Holds, R1CS.LinearCombination.eval_one,
    R1CS.LinearCombination.eval_zero, mul_one, linear_eval]

def BitSource {width : Nat} (slots : List (LowNormAssignment.Slot width)) (column : Nat) : Prop :=
  ∃ (inside : column < width) (index : Fin slots.length),
    LogicalPlan.publicCount ≤ column ∧
      SlotLayout.slotIndex slots ⟨column, inside⟩ = some index ∧
      (slots.get index).kind = .bit

theorem source_slot_norm {width : Nat} (slots : List (LowNormAssignment.Slot width))
    {column : Nat} (source : BitSource slots column)
    (assignment : Assignment F (ProductionAssignment.logicalWidth slots))
    (norm : ∀ coordinate, centeredMagnitude (assignment coordinate) < 2) :
    centeredMagnitude (LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment column) < 2 := by
  obtain ⟨inside, index, outside, found, bit⟩ := source
  have notPublic : ¬ column < LogicalPlan.publicCount := by omega
  simp only [LogicalPlan.sourceEnv, inside, dite_true, SlotLayout.layout, notPublic,
    dite_false, found, NativeBlock.bitSlot_form_eval slots index bit]
  exact norm _

theorem source_slot_norm_wide {width wide : Nat} (slots : List (LowNormAssignment.Slot width))
    (le : ProductionAssignment.logicalWidth slots ≤ wide)
    {column : Nat} (source : BitSource slots column)
    (assignment : Assignment F wide)
    (norm : ∀ coordinate, centeredMagnitude (assignment coordinate) < 2) :
    centeredMagnitude (LogicalPlan.sourceEnv (WideLayout.widen (SlotLayout.layout slots) le)
      assignment column) < 2 := by
  rw [WideLayout.sourceEnv_widen_restrict]
  exact source_slot_norm slots source _ (fun coordinate => norm (Fin.castLE le coordinate))

theorem forced_holds {original : R1CS.Row} {column : Nat}
    (forced : SlotLayout.forcedColumn? original = some column) {env : Env}
    (one : env 0 = 1) (bit : LowNormSlot.Valid .bit (env column)) : original.Holds env := by
  unfold SlotLayout.forcedColumn? at forced
  cases found : (original.a.terms ++ original.b.terms ++ original.c.terms).find?
      (fun term => term.1 != 0) with
  | none => simp only [found] at forced; exact absurd forced (by simp)
  | some first =>
    obtain ⟨first, coefficient⟩ := first
    simp only [found] at forced
    split at forced
    · rename_i checks
      obtain ⟨reads, _, atZero, atOne⟩ := checks
      have columnEq := Option.some.inj forced
      subst columnEq
      have nonzero : first ≠ 0 := by simpa using List.find?_some found
      simp only [List.all_eq_true, Bool.or_eq_true, beq_iff_eq] at reads
      have side : ∀ side : R1CS.LinearCombination, (∀ term ∈ side.terms,
          term ∈ original.a.terms ++ original.b.terms ++ original.c.terms) →
          side.eval env = (side.constant + SlotLayout.coefficientAt side 0) +
            SlotLayout.coefficientAt side first * env first :=
        fun side inside => NativeBlock.side_eval_two one nonzero
          (fun term member => reads term (inside term member))
      change original.a.eval env * original.b.eval env = original.c.eval env
      rw [side original.a (fun _ member => by simp [member]),
        side original.b (fun _ member => by simp [member]),
        side original.c (fun _ member => by simp [member])]
      rcases bit with zero | one
      · simpa only [zero, mul_zero, add_zero] using atZero
      · simpa only [one, mul_one] using atOne
    · exact absurd forced (by simp)

def Covers (original kept : List R1CS.Row) (groups : List (List Pair)) : Prop :=
  ∀ originalRow ∈ original, originalRow ∈ kept ∨
    ∃ group ∈ groups, ∃ pair ∈ group,
      SlotLayout.forcedColumn? originalRow = some pair.1

theorem restore_rows (original kept : List R1CS.Row) (groups : List (List Pair))
    (cover : Covers original kept groups)
    (counts : ∀ group ∈ groups, group.length ≤ capacity)
    {env : Env} (one : env 0 = 1)
    (norm : ∀ group ∈ groups, ∀ pair ∈ group,
      centeredMagnitude (env pair.1) < 2 ∧ centeredMagnitude (env pair.2) < 2)
    (holds : R1CS.RowsHold env (kept ++ groups.map row)) : R1CS.RowsHold env original := by
  obtain ⟨keptHold, packedHold⟩ := (R1CS.rowsHold_append env kept (groups.map row)).mp holds
  intro originalRow member
  rcases cover originalRow member with keptMember | ⟨group, groupMember, pair, pairMember, forced⟩
  · exact keptHold originalRow keptMember
  · have packed := (row_holds group env).mp
      (packedHold (row group) (List.mem_map.mpr ⟨group, groupMember, rfl⟩))
    have groupNorm : Norm (values env group) := by
      intro point pointMember
      obtain ⟨source, sourceMember, rfl⟩ := List.mem_map.mp pointMember
      exact norm group groupMember source sourceMember
    have valid := packed_implies_pairs (values env group)
      (by simpa [values] using counts group groupMember) groupNorm packed
      (env pair.1, env pair.2) (List.mem_map.mpr ⟨pair, pairMember, rfl⟩)
    exact forced_holds forced one valid.1

theorem restore_logical_rows {width : Nat}
    (slots : List (LowNormAssignment.Slot width))
    (original kept : List R1CS.Row) (groups : List (List Pair))
    (cover : Covers original kept groups)
    (counts : ∀ group ∈ groups, group.length ≤ capacity)
    (sources : ∀ group ∈ groups, ∀ pair ∈ group,
      BitSource slots pair.1 ∧ BitSource slots pair.2)
    (fits : LogicalPlan.Fits (kept ++ groups.map row) width)
    (publicInside : LogicalPlan.publicCount ≤ width)
    (assignment : Assignment F (ProductionAssignment.logicalWidth slots))
    (one : assignment (SlotLayout.layout slots).one = 1)
    (norm : ∀ coordinate, centeredMagnitude (assignment coordinate) < 2)
    (holds : (LogicalPlan.plan (kept ++ groups.map row) width fits
      (SlotLayout.layout slots)).RowsZero assignment) :
    R1CS.RowsHold (LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment) original := by
  have oneSource : LogicalPlan.sourceEnv (SlotLayout.layout slots) assignment 0 = 1 := by
    rw [LogicalPlan.sourceEnv_public _ _ 0 (by decide) (by
      have positive : 0 < LogicalPlan.publicCount := by decide
      omega)]
    exact one
  apply restore_rows original kept groups cover counts oneSource
  · intro group groupMember pair pairMember
    have source := sources group groupMember pair pairMember
    exact ⟨source_slot_norm slots source.1 assignment norm,
      source_slot_norm slots source.2 assignment norm⟩
  · exact (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ assignment one).mp holds

theorem restore_logical_rows_wide {width wide : Nat}
    (slots : List (LowNormAssignment.Slot width))
    (le : ProductionAssignment.logicalWidth slots ≤ wide)
    (original kept : List R1CS.Row) (groups : List (List Pair))
    (cover : Covers original kept groups)
    (counts : ∀ group ∈ groups, group.length ≤ capacity)
    (sources : ∀ group ∈ groups, ∀ pair ∈ group,
      BitSource slots pair.1 ∧ BitSource slots pair.2)
    (fits : LogicalPlan.Fits (kept ++ groups.map row) width)
    (publicInside : LogicalPlan.publicCount ≤ width)
    (assignment : Assignment F wide)
    (one : assignment (WideLayout.widen (SlotLayout.layout slots) le).one = 1)
    (norm : ∀ coordinate, centeredMagnitude (assignment coordinate) < 2)
    (holds : (LogicalPlan.plan (kept ++ groups.map row) width fits
      (WideLayout.widen (SlotLayout.layout slots) le)).RowsZero assignment) :
    R1CS.RowsHold (LogicalPlan.sourceEnv (WideLayout.widen (SlotLayout.layout slots) le)
      assignment) original := by
  have oneSource : LogicalPlan.sourceEnv (WideLayout.widen (SlotLayout.layout slots) le)
      assignment 0 = 1 := by
    rw [LogicalPlan.sourceEnv_public _ _ 0 (by decide) (by
      have positive : 0 < LogicalPlan.publicCount := by decide
      omega)]
    exact one
  apply restore_rows original kept groups cover counts oneSource
  · intro group groupMember pair pairMember
    have source := sources group groupMember pair pairMember
    exact ⟨source_slot_norm_wide slots le source.1 assignment norm,
      source_slot_norm_wide slots le source.2 assignment norm⟩
  · exact (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ assignment one).mp holds

theorem complete_rows (kept : List R1CS.Row) (groups : List (List Pair)) {env : Env}
    (holds : R1CS.RowsHold env kept)
    (complements : ∀ group ∈ groups, ∀ pair ∈ group, env pair.2 = 1 - env pair.1) :
    R1CS.RowsHold env (kept ++ groups.map row) := by
  apply (R1CS.rowsHold_append env kept (groups.map row)).mpr
  refine ⟨holds, ?_⟩
  intro packedRow member
  obtain ⟨group, groupMember, rfl⟩ := List.mem_map.mp member
  apply (row_holds group env).mpr
  apply pairs_imply_packed
  intro point pointMember
  obtain ⟨pair, pairMember, rfl⟩ := List.mem_map.mp pointMember
  exact complements group groupMember pair pairMember

def extendSource (start : Nat) (sources : List Nat) (env : Env) : Env :=
  fun column =>
    if column < start then env column
    else if inside : column - start < sources.length then
      1 - env (sources.get ⟨column - start, inside⟩)
    else env column

def companions (start : Nat) (sources : List Nat) : List Pair :=
  List.ofFn fun index : Fin sources.length => (sources.get index, start + index.val)

theorem extendSource_before (start : Nat) (sources : List Nat) (env : Env)
    {column : Nat} (before : column < start) : extendSource start sources env column = env column := by
  simp [extendSource, before]

theorem extendSource_companion (start : Nat) (sources : List Nat) (env : Env)
    (index : Fin sources.length) :
    extendSource start sources env (start + index.val) = 1 - env (sources.get index) := by
  have after : ¬start + index.val < start := by omega
  simp only [extendSource, after, ite_false, Nat.add_sub_cancel_left, index.isLt, dite_true]

theorem companion_value (start : Nat) (sources : List Nat)
    (before : ∀ source ∈ sources, source < start) (env : Env)
    {pair : Pair} (member : pair ∈ companions start sources) :
    extendSource start sources env pair.2 = 1 - extendSource start sources env pair.1 := by
  obtain ⟨index, rfl⟩ := List.mem_ofFn.mp member
  rw [extendSource_companion, extendSource_before start sources env
    (before _ (List.get_mem sources index))]

theorem extended_rows (original kept : List R1CS.Row) (groups : List (List Pair))
    (start : Nat) (sources : List Nat) (env : Env)
    (before : ∀ source ∈ sources, source < start)
    (keptRows : ∀ r ∈ kept, r ∈ original)
    (scope : ∀ r ∈ kept, r.VarsBelow start)
    (paired : ∀ group ∈ groups, ∀ pair ∈ group, pair ∈ companions start sources)
    (holds : R1CS.RowsHold env original) :
    R1CS.RowsHold (extendSource start sources env) (kept ++ groups.map row) := by
  apply complete_rows
  · apply R1CS.rowsHold_of_agree_below kept start env (extendSource start sources env) scope
    · intro column below
      exact extendSource_before start sources env below
    · exact fun r member => holds r (keptRows r member)
  · intro group groupMember pair pairMember
    exact companion_value start sources before env (paired group groupMember pair pairMember)

def SourceForced (original : List R1CS.Row) (source : Nat) : Prop :=
  ∃ r ∈ original, SlotLayout.forcedColumn? r = some source

theorem sourceForced_valid {original : List R1CS.Row} {source : Nat}
    (forced : SourceForced original source) {env : Env}
    (one : env 0 = 1) (holds : R1CS.RowsHold env original) :
    LowNormSlot.Valid .bit (env source) := by
  obtain ⟨r, member, found⟩ := forced
  exact NativeBlock.forced_valid found one (holds r member)

theorem complement_valid {value : F} (bit : LowNormSlot.Valid .bit value) :
    LowNormSlot.Valid .bit (1 - value) := by
  rcases bit with rfl | rfl
  · exact Or.inr (by decide)
  · exact Or.inl (by decide)

def SlotKind {width : Nat} (original : List R1CS.Row) (start : Nat) (sources : List Nat)
    (slot : LowNormAssignment.Slot width) : Prop :=
  slot.kind = .field ∨ slot.kind = .bit ∧
    ((slot.source.val < start ∧ BooleanInference.Derived original slot.source.val) ∨
      ∃ index : Fin sources.length, slot.source.val = start + index.val)

theorem extended_slots_valid {width : Nat} (slots : List (LowNormAssignment.Slot width))
    (original : List R1CS.Row) (start : Nat) (sources : List Nat) (env : Env)
    (forced : ∀ source ∈ sources, SourceForced original source)
    (kinds : ∀ slot ∈ slots, SlotKind original start sources slot)
    (one : env 0 = 1) (holds : R1CS.RowsHold env original) :
    ∀ slot ∈ slots, slot.Valid (fun column => extendSource start sources env column.val) := by
  intro slot member
  rcases kinds slot member with field | ⟨bit, old | fresh⟩
  · simp only [LowNormAssignment.Slot.Valid, field, LowNormSlot.Valid]
  · have valid := BooleanInference.derived_valid old.2 one holds
    simpa only [LowNormAssignment.Slot.Valid, bit,
      extendSource_before start sources env old.1] using valid
  · obtain ⟨index, indexEq⟩ := fresh
    have valid := complement_valid
      (sourceForced_valid (forced _ (List.get_mem sources index)) one holds)
    simpa only [LowNormAssignment.Slot.Valid, bit, indexEq, extendSource_companion] using valid

theorem honest_extended_layout {width : Nat}
    (slots : List (LowNormAssignment.Slot width))
    (original kept : List R1CS.Row) (groups : List (List Pair))
    (start : Nat) (sources : List Nat) (env : Env)
    (publicBefore : LogicalPlan.publicCount ≤ start)
    (before : ∀ source ∈ sources, source < start)
    (keptRows : ∀ r ∈ kept, r ∈ original)
    (scope : ∀ r ∈ kept, r.VarsBelow start)
    (paired : ∀ group ∈ groups, ∀ pair ∈ group, pair ∈ companions start sources)
    (forced : ∀ source ∈ sources, SourceForced original source)
    (kinds : ∀ slot ∈ slots, SlotKind original start sources slot)
    (one : env 0 = 1) (holds : R1CS.RowsHold env original)
    (publicNorm : ∀ column, column < LogicalPlan.publicCount → centeredMagnitude (env column) < 2)
    (fits : LogicalPlan.Fits (kept ++ groups.map row) width)
    (covered : ∀ r ∈ kept ++ groups.map row, r.VarsSatisfy (fun column => column < width ∧
      (column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column))) :
    (LogicalPlan.plan (kept ++ groups.map row) width fits (SlotLayout.layout slots)).RowsZero
        (honestAssignment slots (extendSource start sources env)) ∧
      ∀ column, centeredMagnitude (honestAssignment slots (extendSource start sources env) column) < 2 := by
  have publicSame : ∀ column, column < LogicalPlan.publicCount →
      extendSource start sources env column = env column :=
    fun column inside => extendSource_before start sources env (lt_of_lt_of_le inside publicBefore)
  have oneExtended : extendSource start sources env 0 = 1 :=
    (publicSame 0 (by decide)).trans one
  have oneAt : honestAssignment slots (extendSource start sources env) (SlotLayout.layout slots).one = 1 :=
    (ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨0, by decide⟩).trans oneExtended
  constructor
  · apply (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ _ oneAt).mpr
    apply R1CS.rowsHold_of_agree (kept ++ groups.map row) _ (extendSource start sources env)
      _ covered
    · intro column coverage
      exact sourceEnv_honest slots _ coverage.1 coverage.2
    · exact extended_rows original kept groups start sources env before keptRows scope paired holds
  · apply ProductionAssignment.logicalAssignment_norm
    · intro column
      rw [publicSame column.val column.isLt]
      exact publicNorm column.val column.isLt
    · exact extended_slots_valid slots original start sources env forced kinds one holds

theorem rowsHold_move_middle (left middle right : List R1CS.Row) (env : Env) :
    R1CS.RowsHold env (left ++ middle ++ right) ↔
      R1CS.RowsHold env ((left ++ right) ++ middle) := by
  simp only [R1CS.rowsHold_append]
  tauto

theorem extended_ordered_rows (original left right : List R1CS.Row) (groups : List (List Pair))
    (start : Nat) (sources : List Nat) (env : Env)
    (before : ∀ source ∈ sources, source < start)
    (keptRows : ∀ r ∈ left ++ right, r ∈ original)
    (scope : ∀ r ∈ left ++ right, r.VarsBelow start)
    (paired : ∀ group ∈ groups, ∀ pair ∈ group, pair ∈ companions start sources)
    (holds : R1CS.RowsHold env original) :
    R1CS.RowsHold (extendSource start sources env) (left ++ groups.map row ++ right) := by
  apply (rowsHold_move_middle left (groups.map row) right _).mpr
  exact extended_rows original (left ++ right) groups start sources env before keptRows scope paired holds

theorem honest_extended_ordered_layout {width : Nat}
    (slots : List (LowNormAssignment.Slot width))
    (original left right : List R1CS.Row) (groups : List (List Pair))
    (start : Nat) (sources : List Nat) (env : Env)
    (publicBefore : LogicalPlan.publicCount ≤ start)
    (before : ∀ source ∈ sources, source < start)
    (keptRows : ∀ r ∈ left ++ right, r ∈ original)
    (scope : ∀ r ∈ left ++ right, r.VarsBelow start)
    (paired : ∀ group ∈ groups, ∀ pair ∈ group, pair ∈ companions start sources)
    (forced : ∀ source ∈ sources, SourceForced original source)
    (kinds : ∀ slot ∈ slots, SlotKind original start sources slot)
    (one : env 0 = 1) (holds : R1CS.RowsHold env original)
    (publicNorm : ∀ column, column < LogicalPlan.publicCount → centeredMagnitude (env column) < 2)
    (fits : LogicalPlan.Fits (left ++ groups.map row ++ right) width)
    (covered : ∀ r ∈ left ++ groups.map row ++ right,
      r.VarsSatisfy (fun column => column < width ∧
        (column < LogicalPlan.publicCount ∨ ∃ slot ∈ slots, slot.source.val = column))) :
    (LogicalPlan.plan (left ++ groups.map row ++ right) width fits (SlotLayout.layout slots)).RowsZero
        (honestAssignment slots (extendSource start sources env)) ∧
      ∀ column, centeredMagnitude (honestAssignment slots (extendSource start sources env) column) < 2 := by
  have publicSame : ∀ column, column < LogicalPlan.publicCount →
      extendSource start sources env column = env column :=
    fun column inside => extendSource_before start sources env (lt_of_lt_of_le inside publicBefore)
  have oneExtended : extendSource start sources env 0 = 1 :=
    (publicSame 0 (by decide)).trans one
  have oneAt : honestAssignment slots (extendSource start sources env) (SlotLayout.layout slots).one = 1 :=
    (ProductionAssignment.logicalAssignment_publicColumn _ slots _ ⟨0, by decide⟩).trans oneExtended
  constructor
  · apply (LogicalPlan.rowsZero_iff_rowsHold _ _ _ _ _ oneAt).mpr
    apply R1CS.rowsHold_of_agree (left ++ groups.map row ++ right) _ (extendSource start sources env)
      _ covered
    · intro column coverage
      exact sourceEnv_honest slots _ coverage.1 coverage.2
    · exact extended_ordered_rows original left right groups start sources env before keptRows scope paired holds
  · apply ProductionAssignment.logicalAssignment_norm
    · intro column
      rw [publicSame column.val column.isLt]
      exact publicNorm column.val column.isLt
    · exact extended_slots_valid slots original start sources env forced kinds one holds

theorem freshHolds_implies_original_rows {width wide : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth wide}
    (slots : List (LowNormAssignment.Slot width))
    (le : ProductionAssignment.logicalWidth slots ≤ wide)
    (original kept : List R1CS.Row) (groups : List (List Pair))
    (cover : Covers original kept groups)
    (counts : ∀ group ∈ groups, group.length ≤ capacity)
    (sources : ∀ group ∈ groups, ∀ pair ∈ group,
      BitSource slots pair.1 ∧ BitSource slots pair.2)
    (fits : LogicalPlan.Fits (kept ++ groups.map row) width)
    (publicInside : LogicalPlan.publicCount ≤ width)
    (extra : ProductionRelation.Plan wide)
    (rowBound : (LogicalPlan.plan (kept ++ groups.map row) width fits
      (WideLayout.widen (SlotLayout.layout slots) le)).rowCount + extra.rowCount ≤ 2 ^ cubeVariables)
    (cubeFits : Phi81CarrierLayout.carrierWidth wide ≤ 2 ^ cubeVariables)
    (key : PaperAlgebra.AjtaiKey (logicalWidth := wide) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := wide) (publicFits := publicFits))
    (assignment : Assignment F (Phi81CarrierLayout.carrierWidth wide))
    (one : Plan.logicalAssignment assignment (WideLayout.widen (SlotLayout.layout slots) le).one = 1)
    (accepted : CCS.Holds (PaperAlgebra.semantics key) productionGlobalParams
      (freshStatement ((Plan.append
        (LogicalPlan.plan (kept ++ groups.map row) width fits
          (WideLayout.widen (SlotLayout.layout slots) le)) extra rowBound).logicalRelation
            (publicFits := publicFits) cubeFits) fresh) assignment) :
    R1CS.RowsHold (LogicalPlan.sourceEnv (WideLayout.widen (SlotLayout.layout slots) le)
      (Plan.logicalAssignment assignment)) original ∧
      extra.RowsZero (Plan.logicalAssignment assignment) ∧
      Phi81Relation.projectPublicInput (shape := PaperAlgebra.FullShape wide publicFits)
        (Phi81CarrierLayout.extendAssignment 0 (Plan.logicalAssignment assignment)) =
          fresh.publicInputs ⟨0, by decide⟩ := by
  obtain ⟨allRows, publicInput, norm⟩ := Plan.freshHolds_implies_rowsPublicAndNorm
    (Plan.append (LogicalPlan.plan (kept ++ groups.map row) width fits
      (WideLayout.widen (SlotLayout.layout slots) le)) extra rowBound)
    cubeFits key fresh assignment (le_trans (SlotLayout.publicFits slots) le) accepted
  obtain ⟨ordinary, extraHold⟩ := (Plan.append_rowsZero_iff _ _ _ _).mp allRows
  exact ⟨restore_logical_rows_wide slots le original kept groups cover counts sources fits
    publicInside _ one norm ordinary, extraHold, publicInput⟩

end NightstreamFPrime.Export.Generic.PackedBoolean
