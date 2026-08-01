import Nightstream.Implementation.Lowering.Nebula.Layout
import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment

/-!
Exact fresh-public carrier for the Nebula-enabled 42-times-6 deployment.

Assurance tier: model-level.

This file owns the public-coordinate order required by the delayed Nebula
transition. It preserves the existing 257 Boolean F-prime link coordinates,
then appends the 1,400 Boolean memory-state coordinates, one segment-open
bit, and twelve 64-bit `D_pre` field limbs. Four fixed zeros complete the
45th Phi81 ring.

It does not own private-column placement, row emission, the delayed lane
transition, transcript derivation, a recursive fixed point, Rust, or a
security reduction.

Every live coordinate is a bit. The carrier intentionally does not pack
several bits into one assignment coordinate: fresh SuperNeo assignments are
strictly bounded by `b = 2`, so a packed field value would not be a valid
fresh witness coordinate.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaFreshCarrier

open Nightstream.Implementation.Lowering.Nebula
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- Existing F-prime public link: affine one plus 256 digest bits. -/
def linkWidth : Nat :=
  FPrimeCarrier270.legacyPublicWidth

/-- Public memory state owned by `S_mem`. -/
def memoryWidth : Nat :=
  Layout.publicInputBits

/-- One Boolean command that opens a new memory segment. -/
def openWidth : Nat := 1

/-- Three lane chains, each represented by a four-field Poseidon2 digest. -/
def chainCount : Nat := 3
def digestFieldCount : Nat := 4
def fieldBits : Nat := 64

/-- Canonical bit width of the three claimed precommitment-chain digests. -/
def dPreWidth : Nat :=
  chainCount * digestFieldCount * fieldBits

/-- All live public coordinates before ring alignment. -/
def logicalPublicWidth : Nat :=
  linkWidth + memoryWidth + openWidth + dPreWidth

/-- The least whole-ring public width containing every live coordinate. -/
def publicRingColumns : Nat :=
  (logicalPublicWidth + ringDegree - 1) / ringDegree

def alignedPublicWidth : Nat :=
  ringDegree * publicRingColumns

def fixedPaddingWidth : Nat :=
  alignedPublicWidth - logicalPublicWidth

theorem dimensions_exact :
    linkWidth = 257 /\
      memoryWidth = 1400 /\
      dPreWidth = 768 /\
      logicalPublicWidth = 2426 /\
      publicRingColumns = 45 /\
      alignedPublicWidth = 2430 /\
      fixedPaddingWidth = 4 := by
  decide

def memoryStart : Nat := linkWidth
def openIndex : Nat := memoryStart + memoryWidth
def dPreStart : Nat := openIndex + openWidth
def paddingStart : Nat := logicalPublicWidth

theorem offsets_exact :
    memoryStart = 257 /\
      openIndex = 1657 /\
      dPreStart = 1658 /\
      paddingStart = 2426 := by
  decide

/-- Public column occupied by one memory-state bit. -/
def memoryColumn (bit : Fin memoryWidth) : Fin alignedPublicWidth :=
  ⟨memoryStart + bit.val, by
    have bitBound := bit.isLt
    simp only [memoryStart, memoryWidth, linkWidth, alignedPublicWidth,
      publicRingColumns, logicalPublicWidth, openWidth, dPreWidth,
      chainCount, digestFieldCount, fieldBits, Layout.publicInputBits,
      Layout.segmentIndexBits, Layout.stepIndexBits, Layout.timestampBits,
      Layout.extensionBits, Layout.extensionLimbBits, ringDegree] at bitBound ⊢
    omega⟩

/-- Public column occupied by the segment-open bit. -/
def segmentOpenColumn : Fin alignedPublicWidth :=
  ⟨openIndex, by decide⟩

/-- Flatten one `(chain, digest field, bit)` position. -/
def dPreOffset
    (chain : Fin chainCount)
    (field : Fin digestFieldCount)
    (bit : Fin fieldBits) : Fin dPreWidth :=
  ⟨(chain.val * digestFieldCount + field.val) * fieldBits + bit.val, by
    have chainBound := chain.isLt
    have fieldBound := field.isLt
    have bitBound := bit.isLt
    simp only [chainCount] at chainBound
    simp only [digestFieldCount] at fieldBound
    simp only [fieldBits] at bitBound
    simp only [dPreWidth, chainCount, digestFieldCount, fieldBits]
    omega⟩

/-- Public column occupied by one canonical `D_pre` bit. -/
def dPreColumn
    (chain : Fin chainCount)
    (field : Fin digestFieldCount)
    (bit : Fin fieldBits) : Fin alignedPublicWidth :=
  ⟨dPreStart + (dPreOffset chain field bit).val, by
    have offsetBound := (dPreOffset chain field bit).isLt
    simp only [dPreStart, openIndex, openWidth, memoryStart, memoryWidth,
      linkWidth, alignedPublicWidth, publicRingColumns, logicalPublicWidth,
      dPreWidth, chainCount, digestFieldCount, fieldBits,
      Layout.publicInputBits, Layout.segmentIndexBits, Layout.stepIndexBits,
      Layout.timestampBits, Layout.extensionBits,
      Layout.extensionLimbBits, ringDegree] at offsetBound ⊢
    omega⟩

/-- One of the four verifier-fixed public zeros. -/
def paddingColumn (offset : Fin fixedPaddingWidth) :
    Fin alignedPublicWidth :=
  ⟨paddingStart + offset.val, by
    have offsetBound := offset.isLt
    simp only [paddingStart, fixedPaddingWidth, alignedPublicWidth,
      publicRingColumns, logicalPublicWidth, linkWidth, memoryWidth,
      openWidth, dPreWidth, chainCount, digestFieldCount, fieldBits,
      Layout.publicInputBits, Layout.segmentIndexBits, Layout.stepIndexBits,
      Layout.timestampBits, Layout.extensionBits,
      Layout.extensionLimbBits, ringDegree] at offsetBound ⊢
    omega⟩

@[simp] theorem memoryColumn_val (bit : Fin memoryWidth) :
    (memoryColumn bit).val = memoryStart + bit.val :=
  rfl

@[simp] theorem segmentOpenColumn_val :
    segmentOpenColumn.val = openIndex :=
  rfl

@[simp] theorem dPreColumn_val
    (chain : Fin chainCount)
    (field : Fin digestFieldCount)
    (bit : Fin fieldBits) :
    (dPreColumn chain field bit).val =
      dPreStart +
        ((chain.val * digestFieldCount + field.val) * fieldBits + bit.val) :=
  rfl

@[simp] theorem paddingColumn_val (offset : Fin fixedPaddingWidth) :
    (paddingColumn offset).val = paddingStart + offset.val :=
  rfl

theorem memoryColumn_injective : Function.Injective memoryColumn := by
  intro left right equal
  apply Fin.ext
  have values := congrArg Fin.val equal
  simp only [memoryColumn_val] at values
  omega

theorem dPreOffset_injective :
    Function.Injective
      (fun item : Fin chainCount × Fin digestFieldCount × Fin fieldBits =>
        dPreOffset item.1 item.2.1 item.2.2) := by
  rintro ⟨leftChain, leftField, leftBit⟩
    ⟨rightChain, rightField, rightBit⟩ equal
  have values := congrArg Fin.val equal
  have leftChainBound := leftChain.isLt
  have rightChainBound := rightChain.isLt
  have leftFieldBound := leftField.isLt
  have rightFieldBound := rightField.isLt
  have leftBitBound := leftBit.isLt
  have rightBitBound := rightBit.isLt
  simp only [dPreOffset, chainCount, digestFieldCount, fieldBits] at values
  simp only [chainCount] at leftChainBound rightChainBound
  simp only [digestFieldCount] at leftFieldBound rightFieldBound
  simp only [fieldBits] at leftBitBound rightBitBound
  have bitEqual : leftBit.val = rightBit.val := by omega
  have fieldEqual : leftField.val = rightField.val := by omega
  have chainEqual : leftChain.val = rightChain.val := by omega
  cases Fin.ext chainEqual
  cases Fin.ext fieldEqual
  cases Fin.ext bitEqual
  rfl

/-- A typed complete public carrier. `padding` is present in the type so its
zero condition remains an explicit verifier obligation. -/
structure Carrier where
  link : Fin linkWidth -> F
  memory : Fin memoryWidth -> F
  segmentOpen : F
  dPre : Fin dPreWidth -> F
  padding : Fin fixedPaddingWidth -> F

def Carrier.coordinates (carrier : Carrier) : List F :=
  List.ofFn carrier.link ++
    List.ofFn carrier.memory ++
    [carrier.segmentOpen] ++
    List.ofFn carrier.dPre ++
    List.ofFn carrier.padding

theorem Carrier.coordinates_length (carrier : Carrier) :
    carrier.coordinates.length = alignedPublicWidth := by
  unfold Carrier.coordinates
  simp only [List.length_append, List.length_ofFn, List.length_cons,
    List.length_nil]
  decide

/-- Every live coordinate and every alignment coordinate is Boolean. This is
the exact fresh-`b = 2` carrier condition; row emission proves it later. -/
def Carrier.Binary (carrier : Carrier) : Prop :=
  (∀ index, carrier.link index = 0 ∨ carrier.link index = 1) ∧
    (∀ index, carrier.memory index = 0 ∨ carrier.memory index = 1) ∧
    (carrier.segmentOpen = 0 ∨ carrier.segmentOpen = 1) ∧
    (∀ index, carrier.dPre index = 0 ∨ carrier.dPre index = 1) ∧
    (∀ index, carrier.padding index = 0)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NebulaFreshCarrier
