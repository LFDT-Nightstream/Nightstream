import Nightstream.Implementation.Lowering.Nebula.ConstraintSemantics

/-!
Compact field-to-memory-tuple bridge for the Lean-owned Nebula relation.

Assurance tier: model-level.

Owns only the generic three-field tuple decoder and the exact packed/value
field equalities. Keeping this leaf separate prevents concrete 64-bit word
combinations from unfolding during the packing proof.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.Nebula.SourceSemantics

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula
open Nightstream.Implementation.Lowering.Nebula.Layout
open Nightstream.Implementation.Lowering.Nebula.Rows
open Nightstream.Implementation.Lowering.Nebula.Compiler
open Nightstream.Implementation.Lowering.Nebula.ProductSemantics
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.Fingerprint

private abbrev Lin := Rows.LinearCombination

def fieldValue (assignment : Nat → F) (value : Lin) : F :=
  Rows.LinearCombination.eval assignment value

def challenges (assignment : Nat → F) : Challenges where
  gamma1 := evaluatePair assignment (gammaWord 0 0) (gammaWord 0 1)
  gamma2 := evaluatePair assignment (gammaWord 1 0) (gammaWord 1 1)

def entryOfFields
    (timestamp globalIndex value : F) : MemTuple where
  timestamp := timestamp.val
  globalIndex := globalIndex.val
  value := value.val

def operationEntry
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) : MemTuple :=
  entryOfFields
    (fieldValue assignment
      (if write then operationWriteTimestamp params slot
        else operationReadTimestamp params slot))
    (fieldValue assignment (operationGlobalIndex params slot))
    (fieldValue assignment
      (if write then operationWriteValue params slot
        else operationReadValue params slot))

def scanEntry
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) : MemTuple :=
  entryOfFields
    (fieldValue assignment (scanTimestamp params final slot))
    (fieldValue assignment (scanGlobalIndex params slot))
    (fieldValue assignment (scanValue params final slot))

private theorem fieldTwoPower_timestamp_exact :
    Rows.LinearCombination.fieldTwoPower Layout.timestampBits =
      Compiler.fieldOfNat Fingerprint.timestampRadix := by
  decide

private theorem fieldOfNat_val (value : Nat) :
    (Compiler.fieldOfNat value).val = value % goldilocksModulus := by
  rfl

theorem entryOfFields_packed
    (timestamp globalIndex value : F) :
    Fingerprint.packed (entryOfFields timestamp globalIndex value) =
      timestamp +
        Rows.LinearCombination.fieldTwoPower Layout.timestampBits *
          globalIndex := by
  apply Fin.eq_of_val_eq
  simp only [Fingerprint.packed, Fingerprint.packedNat, entryOfFields,
    Fingerprint.timestampRadix, Fingerprint.timestampBits,
    Fin.val_add, Fin.val_mul]
  rw [fieldTwoPower_timestamp_exact, fieldOfNat_val]
  simp only [Fingerprint.timestampRadix, Fingerprint.timestampBits]
  rw [Nat.mod_eq_of_lt (by decide : 2 ^ 44 < goldilocksModulus)]
  exact (Nat.add_mod_mod _ _ _).symm

theorem entryOfFields_valueField
    (timestamp globalIndex value : F) :
    Fingerprint.valueField (entryOfFields timestamp globalIndex value) =
      value := by
  apply Fin.eq_of_val_eq
  change value.val % goldilocksModulus = value.val
  exact Nat.mod_eq_of_lt value.isLt

theorem operationEntry_packed
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) :
    Fingerprint.packed (operationEntry assignment params slot write) =
      fieldValue assignment
          (if write then operationWriteTimestamp params slot
            else operationReadTimestamp params slot) +
        Rows.LinearCombination.fieldTwoPower Layout.timestampBits *
          fieldValue assignment (operationGlobalIndex params slot) :=
  entryOfFields_packed _ _ _

theorem scanEntry_packed
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) :
    Fingerprint.packed (scanEntry assignment params final slot) =
      fieldValue assignment (scanTimestamp params final slot) +
        Rows.LinearCombination.fieldTwoPower Layout.timestampBits *
          fieldValue assignment (scanGlobalIndex params slot) :=
  entryOfFields_packed _ _ _

theorem operationEntry_valueField
    (assignment : Nat → F) (params : Params)
    (slot : Nat) (write : Bool) :
    Fingerprint.valueField (operationEntry assignment params slot write) =
      fieldValue assignment
        (if write then operationWriteValue params slot
          else operationReadValue params slot) :=
  entryOfFields_valueField _ _ _

theorem scanEntry_valueField
    (assignment : Nat → F) (params : Params)
    (final : Bool) (slot : Nat) :
    Fingerprint.valueField (scanEntry assignment params final slot) =
      fieldValue assignment (scanValue params final slot) :=
  entryOfFields_valueField _ _ _

end Nightstream.Implementation.Lowering.Nebula.SourceSemantics
