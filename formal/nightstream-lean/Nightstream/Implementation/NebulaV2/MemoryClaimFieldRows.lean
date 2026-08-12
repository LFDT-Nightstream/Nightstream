import Nightstream.Implementation.NebulaV2.CanonicalFieldSchemaRows
import Nightstream.Implementation.NebulaV2.MemoryClaimCodec

/-!
Contract: exact canonical-u64 row coverage for all 76 Goldilocks limbs in a
V2 fresh-claim memory block.

Assurance tier: implementation model.

Owns the challenge, product, and root slot order; their exact bit offsets in
the 4,980-bit claim codec; 10,108 generated rows; fail-closed native decoding;
and equality between every parsed typed limb and its circuit wire.

Does not own the seven narrow counter blocks, the enclosing full CCS claim,
container bytes, or the final generated relation manifest.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.SuperNeo.Concrete

/-- One slot for each authority-bearing 64-bit Goldilocks word. -/
inductive Slot where
  | challenge (repetition coordinate limb : Fin 2)
  | product (side repetition : Fin 2) (role : ProductRole) (limb : Fin 2)
  | root (stage : RootStage) (role : RootRole) (lane : Fin 4)
deriving DecidableEq, Repr

def challengeSlots : List Slot :=
  (List.ofFn fun repetition : Fin 2 =>
    (List.ofFn fun coordinate : Fin 2 =>
      List.ofFn fun limb : Fin 2 =>
        Slot.challenge repetition coordinate limb).flatten).flatten

def productSlots : List Slot :=
  (List.ofFn fun side : Fin 2 =>
    (List.ofFn fun repetition : Fin 2 =>
      (productRoles.map fun role =>
        List.ofFn fun limb : Fin 2 =>
          Slot.product side repetition role limb).flatten).flatten).flatten

def rootSlots : List Slot :=
  (rootStages.map fun stage =>
    (rootRoles.map fun role =>
      List.ofFn fun lane : Fin 4 =>
        Slot.root stage role lane).flatten).flatten

/-- Exact suffix order after the seven narrow counters. -/
def Slot.all : List Slot := challengeSlots ++ productSlots ++ rootSlots

def Slot.tag : Slot → MemoryClaimCodec.FieldTag
  | .challenge repetition coordinate limb =>
      .challenge repetition coordinate limb
  | .product side repetition role limb =>
      .product side repetition role limb
  | .root stage role lane => .root stage role lane

theorem Slot.all_length_exact : Slot.all.length = 76 := by decide

theorem Slot.all_nodup : Slot.all.Nodup := by decide

theorem Slot.tags_exact : Slot.all.map Slot.tag =
    challengeSchema ++ productSchema ++ rootSchema := by
  decide

theorem Slot.mem_all (slot : Slot) : slot ∈ Slot.all := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
        simp [Slot.all, challengeSlots]
  | product side repetition role limb =>
      fin_cases side <;> fin_cases repetition <;> cases role <;>
        fin_cases limb <;>
          simp [Slot.all, productSlots, productRoles]
  | root stage role lane =>
      cases stage <;> cases role <;> fin_cases lane <;>
        simp [Slot.all, rootSlots, rootStages, rootRoles]

/-- The seven narrow counters occupy exactly the first 116 claim bits. -/
def fieldBitStart : Nat :=
  (counterSchema.map MemoryClaimCodec.FieldTag.bitWidth).sum

theorem fieldBitStart_exact : fieldBitStart = 116 := by decide

def Slot.position (slot : Slot) : Nat := Slot.all.idxOf slot

def Slot.bitOffset (slot : Slot) : Nat :=
  fieldBitStart + CanonicalFieldBits.bitCount * slot.position

theorem first_bit_exact :
    (Slot.challenge 0 0 0).bitOffset = 116 := by decide

theorem last_bit_end_exact :
    (Slot.root .seenAfter .finalSnapshot 3).bitOffset +
      CanonicalFieldBits.bitCount = 4980 := by decide

theorem Slot.bitOffset_eq_tag (slot : Slot) :
    slot.bitOffset = slot.tag.bitOffset := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases repetition <;> fin_cases coordinate <;> fin_cases limb <;>
        decide
  | product side repetition role limb =>
      fin_cases side <;> fin_cases repetition <;> cases role <;>
        fin_cases limb <;> decide
  | root stage role lane =>
      cases stage <;> cases role <;> fin_cases lane <;> decide

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

private def challengeField (claim : Claim)
    (repetition coordinate limb : Fin 2) : F :=
  if coordinate = 0 then kLimb (claim.challenge repetition).gamma1 limb
  else kLimb (claim.challenge repetition).gamma2 limb

private def productField (claim : Claim) (side repetition : Fin 2)
    (role : ProductRole) (limb : Fin 2) : F :=
  let products :=
    if side = 0 then claim.productsBefore repetition
    else claim.productsAfter repetition
  match role with
  | .initialSnapshot => kLimb products.initialSnapshot limb
  | .writes => kLimb products.writes limb
  | .reads => kLimb products.reads limb
  | .finalSnapshot => kLimb products.finalSnapshot limb

private def selectedRoots (claim : Claim) : RootStage → Roots Digest.Value
  | .precommit => claim.dPre
  | .seenBefore => claim.dSeenBefore
  | .seenAfter => claim.dSeenAfter

private def rootField (claim : Claim) (stage : RootStage)
    (role : RootRole) (lane : Fin 4) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  match role with
  | .operations => (selectedRoots claim stage).operations.lanes lane
  | .initialSnapshot =>
      (selectedRoots claim stage).initialSnapshot.lanes lane
  | .finalSnapshot => (selectedRoots claim stage).finalSnapshot.lanes lane

def Slot.canonicalValue (slot : Slot) (claim : Claim) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  match slot with
  | .challenge repetition coordinate limb =>
      canonicalOfF (challengeField claim repetition coordinate limb)
  | .product side repetition role limb =>
      canonicalOfF (productField claim side repetition role limb)
  | .root stage role lane => rootField claim stage role lane

theorem Slot.canonicalValue_val (slot : Slot) (claim : Claim) :
    (slot.canonicalValue claim).val = claim.fieldValue slot.tag := by
  cases slot with
  | challenge repetition coordinate limb =>
      fin_cases coordinate <;> fin_cases limb <;> rfl
  | product side repetition role limb =>
      fin_cases side <;> cases role <;> fin_cases limb <;> rfl
  | root stage role lane =>
      cases stage <;> cases role <;> rfl

/-- Exact result required from the fail-closed native parser. This is a named
parser boundary, not a row-satisfaction conclusion. -/
def NativeParses (raw : RawWords) (claim : Claim) : Prop :=
  ∀ slot, FieldCodec.nativeDecode (raw slot) =
    some (slot.canonicalValue claim)

/-- Rows alone reject every noncanonical field word, including the modulus
encoding of zero. No parser-to-typed-value premise is needed. -/
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

/-- The deterministic `q` encoding cannot occupy any listed claim slot in a
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
    ⟨value, decoded, _⟩
  rw [aliasEq, FieldCodec.rejects_zero_modulus_alias.2] at decoded
  simp at decoded

/-- Exact native parsing plus exact generated rows forces every typed field
limb into the matching circuit value column. The equality is a conclusion. -/
theorem typed_columns_of_rows
    {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw)
    (parsed : NativeParses raw claim) :
    ∀ slot,
      assignment
          (Relabel.column (layout.columnMap slot) CanonicalU64.varCol) =
        claim.fieldValue slot.tag := by
  intro slot
  rcases rows_force_native_acceptance canonical one satisfies placed slot with
    ⟨decoded, decodedRaw, decodedWire⟩
  have decodedEqual : decoded = slot.canonicalValue claim :=
    FieldCodec.nativeDecode_unique decodedRaw (parsed slot)
  calc
    assignment
        (Relabel.column (layout.columnMap slot) CanonicalU64.varCol) =
        decoded.val := decodedWire.symm
    _ = (slot.canonicalValue claim).val := congrArg Subtype.val decodedEqual
    _ = claim.fieldValue slot.tag := slot.canonicalValue_val claim

theorem rows_length_exact (layout : Layout) :
    (rows layout).length = 10108 := by
  rw [rows, CanonicalFieldSchemaRows.schemaRows_length,
    Slot.all_length_exact]

end Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows
