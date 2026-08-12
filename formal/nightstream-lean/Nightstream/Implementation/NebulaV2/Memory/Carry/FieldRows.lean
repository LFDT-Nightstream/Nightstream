import Nightstream.Implementation.NebulaV2.Core.CanonicalFieldSchemaRows
import Nightstream.Implementation.NebulaV2.Memory.Carry.Codec

/-!
Contract: exact canonical-u64 row coverage for all 52 Goldilocks limbs in a
V2 recursive memory carry.

Assurance tier: implementation model.

Owns the challenge, product, and root slot order; their exact bit offsets in
the 3,433-bit carry codec; 6,916 generated rows; fail-closed native decoding;
and equality between every parsed typed limb and its circuit wire.

Does not own carry counter and closed-state rows, the state-hash permutation,
container bytes, or the final recursive relation manifest.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryCarryFieldRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.SuperNeo.Concrete

/-- One slot for each authority-bearing 64-bit Goldilocks word. -/
inductive Slot where
  | challenge (repetition coordinate limb : Fin 2)
  | product (repetition : Fin 2) (role : ProductRole) (limb : Fin 2)
  | root (source : RootSource) (lane : Fin 4)
deriving DecidableEq, Repr

def challengeSlots : List Slot :=
  (List.ofFn fun repetition : Fin 2 =>
    (List.ofFn fun coordinate : Fin 2 =>
      List.ofFn fun limb : Fin 2 =>
        Slot.challenge repetition coordinate limb).flatten).flatten

def productSlots : List Slot :=
  (List.ofFn fun repetition : Fin 2 =>
    (productRoles.map fun role =>
      List.ofFn fun limb : Fin 2 =>
        Slot.product repetition role limb).flatten).flatten

def rootSlots : List Slot :=
  (rootSources.map fun source =>
    List.ofFn fun lane : Fin 4 => Slot.root source lane).flatten

/-- Exact suffix order after the seven carry counters. -/
def Slot.all : List Slot := challengeSlots ++ productSlots ++ rootSlots

def Slot.tag : Slot → MemoryCarryCodec.FieldTag
  | .challenge repetition coordinate limb =>
      .challenge repetition coordinate limb
  | .product repetition role limb => .product repetition role limb
  | .root source lane => .root source lane

theorem Slot.all_length_exact : Slot.all.length = 52 := by decide

theorem Slot.all_nodup : Slot.all.Nodup := by decide

theorem Slot.tags_exact : Slot.all.map Slot.tag =
    MemoryCarryCodec.challengeSchema ++ MemoryCarryCodec.productSchema ++
      MemoryCarryCodec.rootSchema := by
  decide

theorem Slot.mem_all (slot : Slot) : slot ∈ Slot.all := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
        simp [Slot.all, challengeSlots]
  | product repetition role limb =>
      fin_cases repetition <;> cases role <;> fin_cases limb <;>
        simp [Slot.all, productSlots, productRoles]
  | root source lane =>
      cases source with
      | memory =>
          fin_cases lane <;> simp [Slot.all, rootSlots, rootSources]
      | precommit role =>
          cases role <;> fin_cases lane <;>
            simp [Slot.all, rootSlots, rootSources]
      | seen role =>
          cases role <;> fin_cases lane <;>
            simp [Slot.all, rootSlots, rootSources]

/-- The seven carry counters occupy exactly the first 105 carry bits. -/
def fieldBitStart : Nat :=
  (counterSchema.map MemoryCarryCodec.FieldTag.bitWidth).sum

theorem fieldBitStart_exact : fieldBitStart = 105 := by decide

def Slot.position (slot : Slot) : Nat := Slot.all.idxOf slot

def Slot.bitOffset (slot : Slot) : Nat :=
  fieldBitStart + CanonicalFieldBits.bitCount * slot.position

theorem first_bit_exact :
    (Slot.challenge 0 0 0).bitOffset = 105 := by decide

theorem last_bit_end_exact :
    (Slot.root .memory 3).bitOffset + CanonicalFieldBits.bitCount = 3433 := by
  decide

theorem Slot.bitOffset_eq_tag (slot : Slot) :
    slot.bitOffset = slot.tag.bitOffset := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
        decide
  | product repetition role limb =>
      fin_cases repetition <;> cases role <;> fin_cases limb <;> decide
  | root source lane =>
      cases source with
      | memory => fin_cases lane <;> decide
      | precommit role => cases role <;> fin_cases lane <;> decide
      | seen role => cases role <;> fin_cases lane <;> decide

theorem Slot.tag_width (slot : Slot) :
    slot.tag.bitWidth = CanonicalFieldBits.bitCount := by
  cases slot <;> rfl

structure Layout where
  publicBitStart : Nat
  columnMap : Slot → List Nat
  mapsConstantOne : ∀ slot, Relabel.column (columnMap slot) 0 = 0

def Layout.rawColumns (layout : Layout) (slot : Slot) : List Nat :=
  (List.range CanonicalFieldBits.bitCount).map fun index =>
    layout.publicBitStart + slot.bitOffset + index

def Layout.schema (layout : Layout) : CanonicalFieldSchemaRows.Layout Slot where
  columnMap := layout.columnMap
  rawColumns := layout.rawColumns
  rawColumnsLength := by intro slot; simp [Layout.rawColumns]
  mapsConstantOne := layout.mapsConstantOne

def rows (layout : Layout) : List Row :=
  CanonicalFieldSchemaRows.schemaRows Slot.all layout.schema

abbrev RawWords := CanonicalFieldSchemaRows.RawWords Slot

def Places (layout : Layout) (assignment : Nat → Nat)
    (raw : RawWords) : Prop :=
  CanonicalFieldSchemaRows.Places layout.schema assignment raw

private def canonicalOfF (value : F) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨value.val, by
    simpa [ShiftedTernary41V1.modulus, goldilocksModulus] using value.isLt⟩

private def kLimb (value : K) : Fin 2 → F
  | 0 => value.c0
  | _ => value.c1

private def challengeField (value : Value)
    (repetition coordinate limb : Fin 2) : F :=
  if coordinate = 0 then kLimb (value.challenges repetition).gamma1 limb
  else kLimb (value.challenges repetition).gamma2 limb

private def productField (value : Value) (repetition : Fin 2)
    (role : ProductRole) (limb : Fin 2) : F :=
  match role with
  | .initialSnapshot => kLimb (value.products repetition).initialSnapshot limb
  | .writes => kLimb (value.products repetition).writes limb
  | .reads => kLimb (value.products repetition).reads limb
  | .finalSnapshot => kLimb (value.products repetition).finalSnapshot limb

private def rootField (value : Value) (source : RootSource)
    (lane : Fin 4) : ShiftedTernary41V1.CanonicalGoldilocks :=
  match source with
  | .precommit .operations => value.dPre.operations.lanes lane
  | .precommit .initialSnapshot => value.dPre.initialSnapshot.lanes lane
  | .precommit .finalSnapshot => value.dPre.finalSnapshot.lanes lane
  | .seen .operations => value.dSeen.operations.lanes lane
  | .seen .initialSnapshot => value.dSeen.initialSnapshot.lanes lane
  | .seen .finalSnapshot => value.dSeen.finalSnapshot.lanes lane
  | .memory => value.memoryRoot.lanes lane

def Slot.canonicalValue (slot : Slot) (value : Value) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  match slot with
  | .challenge repetition coordinate limb =>
      canonicalOfF (challengeField value repetition coordinate limb)
  | .product repetition role limb =>
      canonicalOfF (productField value repetition role limb)
  | .root source lane => rootField value source lane

theorem Slot.canonicalValue_val (slot : Slot) (value : Value) :
    (slot.canonicalValue value).val = value.fieldValue slot.tag := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;> rfl
  | product repetition role limb =>
      cases role <;> fin_cases limb <;> rfl
  | root source lane =>
      cases source with
      | memory => rfl
      | precommit role => cases role <;> rfl
      | seen role => cases role <;> rfl

/-- Exact result required from the fail-closed native carry parser. This is a
named parser boundary, not a row-satisfaction conclusion. -/
def NativeParses (raw : RawWords) (value : Value) : Prop :=
  ∀ slot, FieldCodec.nativeDecode (raw slot) =
    some (slot.canonicalValue value)

/-- Rows alone reject every noncanonical carry field word. -/
theorem rows_force_native_acceptance
    {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw) :
    ∀ slot, ∃ value,
      FieldCodec.nativeDecode (raw slot) = some value ∧
        value.val = assignment
          (Relabel.column (layout.columnMap slot) CanonicalU64.varCol) := by
  intro slot
  exact CanonicalFieldSchemaRows.slot_sound canonical one satisfies placed
    slot.mem_all

/-- The deterministic `q` encoding cannot occupy any listed carry slot in a
satisfying assignment. -/
theorem modulus_alias_impossible
    {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw)
    (slot : Slot)
    (aliasEq : raw slot = CanonicalFieldBits.modulusWord) : False := by
  rcases rows_force_native_acceptance canonical one satisfies placed slot with
    ⟨decodedValue, decoded, _⟩
  rw [aliasEq, FieldCodec.rejects_zero_modulus_alias.2] at decoded
  simp at decoded

/-- Exact native parsing plus exact rows forces all 52 typed carry limbs into
their matching circuit value columns. -/
theorem typed_columns_of_rows
    {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    {value : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw)
    (parsed : NativeParses raw value) :
    ∀ slot,
      assignment
          (Relabel.column (layout.columnMap slot) CanonicalU64.varCol) =
        value.fieldValue slot.tag := by
  intro slot
  rcases rows_force_native_acceptance canonical one satisfies placed slot with
    ⟨decoded, decodedRaw, decodedWire⟩
  have decodedEqual : decoded = slot.canonicalValue value :=
    FieldCodec.nativeDecode_unique decodedRaw (parsed slot)
  calc
    assignment
        (Relabel.column (layout.columnMap slot) CanonicalU64.varCol) =
        decoded.val := decodedWire.symm
    _ = (slot.canonicalValue value).val := congrArg Subtype.val decodedEqual
    _ = value.fieldValue slot.tag := slot.canonicalValue_val value

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 6916 := by
  rw [rows, CanonicalFieldSchemaRows.schemaRows_length,
    Slot.all_length_exact]

end Nightstream.Implementation.NebulaV2.MemoryCarryFieldRows
