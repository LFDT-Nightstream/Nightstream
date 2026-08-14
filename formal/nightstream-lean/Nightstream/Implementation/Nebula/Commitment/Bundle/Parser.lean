import Nightstream.Implementation.Nebula.Commitment.Bundle.FieldRows
import Nightstream.Implementation.Nebula.Core.FixedBits

/-!
Contract: executable fail-closed parser for the mandatory V2 commitment
bundle.

Assurance tier: implementation-model refinement.

Owns exact 64-bit field slicing, strict Goldilocks canonicality, reconstruction
of all four commitment components, rejection of modulo aliases, and exact
successful-parser re-encoding.

Does not own byte-container framing, generated parser rows, Ajtai evaluation,
terminal same-witness opening, Rust conformance, or cryptographic soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 500000

namespace Nightstream.Implementation.Nebula.CommitmentBundleParser

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.CommitmentBundleCodec
open Nightstream.Implementation.Nebula.CommitmentBundleFieldRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.Protocol.Nebula.MemoryWireGeometry

abbrev Block := FixedBits.Word mandatoryBundleBits

theorem field_slice_fits (slot : Slot) :
    slot.bitOffset + CanonicalFieldBits.bitCount ≤ mandatoryBundleBits := by
  have positionBound := slot.position_lt
  norm_num [Slot.bitOffset, CanonicalFieldBits.bitCount,
    mandatoryBundleBits_exact] at *
  omega

/-- The exact little-endian field word at one authority-bearing bundle
position. -/
def fieldWord (block : Block) (slot : Slot) : CanonicalFieldBits.Word :=
  FixedBits.slice block slot.bitOffset CanonicalFieldBits.bitCount
    (field_slice_fits slot)

/-- Executable strict-canonicality check over all 3,888 field words. -/
def fieldsCanonical (block : Block) : Bool :=
  Slot.all.all fun slot =>
    decide (CanonicalFieldBits.decode (fieldWord block slot) <
      ShiftedTernary41V1.modulus)

theorem field_canonical_of_all
    {block : Block} (allCanonical : fieldsCanonical block = true)
    (slot : Slot) : CanonicalFieldBits.Canonical (fieldWord block slot) := by
  have every := List.all_eq_true.mp allCanonical slot slot.mem_all
  simpa [CanonicalFieldBits.Canonical] using of_decide_eq_true every

def decodedField (block : Block)
    (allCanonical : fieldsCanonical block = true) (slot : Slot) :
    ShiftedTernary41V1.CanonicalGoldilocks :=
  ⟨CanonicalFieldBits.decode (fieldWord block slot),
    field_canonical_of_all allCanonical slot⟩

/-- Reconstruction uses the same explicit component/coordinate tags as the
authority-bearing codec. -/
def decodedBundle (block : Block)
    (allCanonical : fieldsCanonical block = true) : Value :=
  fun component coordinate =>
    decodedField block allCanonical { component, coordinate }

/-- Canonical typed bundles embed as exact parser inputs. -/
def blockOfBundle (bundle : Value) : Block :=
  ⟨encode bundle, encode_length bundle,
    fun digit member => encode_binary bundle digit member⟩

theorem fieldWord_blockOfBundle (bundle : Value) (slot : Slot) :
    fieldWord (blockOfBundle bundle) slot = expectedWord bundle slot := by
  apply Subtype.ext
  change
    ((encode bundle).drop slot.bitOffset).take
        CanonicalFieldBits.bitCount =
      (expectedWord bundle slot).val
  exact encoded_word_exact bundle slot

theorem fieldsCanonical_blockOfBundle (bundle : Value) :
    fieldsCanonical (blockOfBundle bundle) = true := by
  rw [fieldsCanonical, List.all_eq_true]
  intro slot _member
  apply decide_eq_true
  rw [fieldWord_blockOfBundle]
  exact CanonicalFieldBits.encode_is_canonical _

theorem decodedField_blockOfBundle
    (bundle : Value)
    (allCanonical : fieldsCanonical (blockOfBundle bundle) = true)
    (slot : Slot) :
    decodedField (blockOfBundle bundle) allCanonical slot =
      bundle slot.component slot.coordinate := by
  apply Subtype.ext
  change CanonicalFieldBits.decode
      (fieldWord (blockOfBundle bundle) slot) =
    (bundle slot.component slot.coordinate).val
  rw [fieldWord_blockOfBundle]
  exact CanonicalFieldBits.decode_encode _

theorem decodedBundle_blockOfBundle
    (bundle : Value)
    (allCanonical : fieldsCanonical (blockOfBundle bundle) = true) :
    decodedBundle (blockOfBundle bundle) allCanonical = bundle := by
  funext component coordinate
  exact decodedField_blockOfBundle bundle allCanonical
    { component, coordinate }

/-- The parser uses both strict field canonicality and exact complete-block
re-encoding. The second check makes the success refinement mechanically
direct and fail-closed. -/
def parse (block : Block) : Option Value :=
  if allCanonical : fieldsCanonical block = true then
    if _exact : (blockOfBundle
        (decodedBundle block allCanonical)).val = block.val then
      some (decodedBundle block allCanonical)
    else
      none
  else
    none

theorem parse_success
    {block : Block} {bundle : Value} (accepted : parse block = some bundle) :
    ∃ allCanonical : fieldsCanonical block = true,
      bundle = decodedBundle block allCanonical ∧
        (blockOfBundle bundle).val = block.val := by
  unfold parse at accepted
  split at accepted
  next allCanonical =>
    split at accepted
    next exactEncoding =>
      have bundleEqual := Option.some.inj accepted.symm
      subst bundle
      exact ⟨allCanonical, rfl, exactEncoding⟩
    next notExact => simp at accepted
  next notCanonical => simp at accepted

/-- Every successful parse is the exact canonical encoding of its output. -/
theorem parse_success_reencodes
    {block : Block} {bundle : Value} (accepted : parse block = some bundle) :
    encode bundle = block.val := by
  have exact := (parse_success accepted).choose_spec.2
  exact exact

/-- Parser completeness for every typed four-component bundle. -/
theorem parse_blockOfBundle (bundle : Value) :
    parse (blockOfBundle bundle) = some bundle := by
  have canonical := fieldsCanonical_blockOfBundle bundle
  unfold parse
  rw [dif_pos canonical]
  rw [decodedBundle_blockOfBundle bundle canonical]
  rw [dif_pos rfl]

/-- A word equal to the Goldilocks modulus is rejected at every component and
coordinate. In particular, `0` and `q` cannot alias in a parsed bundle. -/
theorem rejects_modulus_alias
    (block : Block) (slot : Slot)
    (aliasEq : fieldWord block slot = CanonicalFieldBits.modulusWord) :
    parse block = none := by
  have notCanonical : fieldsCanonical block ≠ true := by
    intro allCanonical
    have canonical := field_canonical_of_all allCanonical slot
    rw [aliasEq] at canonical
    exact CanonicalFieldBits.modulusWord_not_canonical canonical
  simp [parse, notCanonical]

end Nightstream.Implementation.Nebula.CommitmentBundleParser
