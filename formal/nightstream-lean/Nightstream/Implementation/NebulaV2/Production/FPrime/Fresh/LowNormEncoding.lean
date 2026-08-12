import Mathlib.Logic.Equiv.Fin.Basic
import Nightstream.Implementation.NebulaV2.NIFS.Core.PaperAlgebra
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.CenteredTernary
import Nightstream.Protocol.NebulaV2.CompactCommit
import Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
import Nightstream.Protocol.NebulaV2.Encoding

/-!
Contract: exact low-norm assignment encoding for one V2 fresh source.

The first 540 source coordinates are public bits. The next 13,824 coordinates
are the exact operations, initial-snapshot, and final-snapshot lane bits. This
complete 266-ring direct prefix remains unchanged. Every later private field
becomes its exact 41-coordinate `ShiftedTernary41V1` centered word. Canonical
zero padding completes the payload to whole Phi81 rings.

The source width is `14,364 + privateWidth` by construction. No truncated
subtraction or proof-selected prefix width occurs in the authority path.

Owns the direct lane prefix, private-word encoding, ring completion, decoding,
and strict fresh norm bound. Does not own source-row generation, commitment
keys, NIFS extraction, Rust, or cryptographic binding.

Assurance tier: implementation model.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding

open Nightstream.Implementation.NebulaV2
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.ConcreteLaneGeometry
open Nightstream.Protocol.NebulaV2.ShiftedTernary41V1
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

def publicWidth : Nat := 540
def laneWidth : Nat := 13824
def directWidth : Nat := 14364

theorem laneWidth_eq_blockWidth : laneWidth = blockWidth := by
  norm_num [laneWidth, blockWidth_exact]

theorem directWidth_eq_public_lane :
    directWidth = publicWidth + laneWidth := by
  decide

theorem publicWidth_le_directWidth : publicWidth <= directWidth := by
  decide

theorem directWidth_aligned : LaneLayout.Aligned directWidth := by
  norm_num [LaneLayout.Aligned, directWidth, LaneLayout.ringDegree]

/-- Exact total source width. -/
def sourceWidth (privateWidth : Nat) : Nat := directWidth + privateWidth

/-- Non-padding low-norm payload width. -/
def payloadWidth (privateWidth : Nat) : Nat :=
  directWidth + privateWidth * digitCount

theorem directWidth_le_payloadWidth (privateWidth : Nat) :
    directWidth <= payloadWidth privateWidth := by
  simp [payloadWidth]

theorem publicWidth_le_payloadWidth (privateWidth : Nat) :
    publicWidth <= payloadWidth privateWidth :=
  Nat.le_trans publicWidth_le_directWidth
    (directWidth_le_payloadWidth privateWidth)

/-- Authority-bearing logical width. -/
def logicalWidth (privateWidth : Nat) : Nat :=
  Phi81CarrierLayout.carrierWidth (payloadWidth privateWidth)

theorem payloadWidth_le_logicalWidth (privateWidth : Nat) :
    payloadWidth privateWidth <= logicalWidth privateWidth :=
  Phi81CarrierLayout.logicalWidth_le_carrierWidth _

theorem publicWidth_le_logicalWidth (privateWidth : Nat) :
    publicWidth <= logicalWidth privateWidth :=
  Nat.le_trans (publicWidth_le_payloadWidth privateWidth)
    (payloadWidth_le_logicalWidth privateWidth)

/-- Ring completion is idempotent. -/
theorem logicalWidth_carrier_exact (privateWidth : Nat) :
    Phi81CarrierLayout.carrierWidth (logicalWidth privateWidth) =
      logicalWidth privateWidth := by
  calc
    Phi81CarrierLayout.carrierWidth (logicalWidth privateWidth) =
        Phi81ColumnLayout.blockCount (logicalWidth privateWidth) * ringDegree :=
      rfl
    _ = Phi81ColumnLayout.blockCount (payloadWidth privateWidth) * ringDegree := by
      rw [logicalWidth, Phi81CarrierLayout.blockCount_carrierWidth]
    _ = logicalWidth privateWidth := rfl

theorem publicFits (privateWidth : Nat) :
    publicWidth <= Phi81CarrierLayout.carrierWidth
      (logicalWidth privateWidth) :=
  Nat.le_trans (publicWidth_le_logicalWidth privateWidth)
    (Phi81CarrierLayout.logicalWidth_le_carrierWidth _)

abbrev SourceAssignment (privateWidth : Nat) :=
  Fin (sourceWidth privateWidth) -> F

def directSourceColumn {privateWidth : Nat}
    (column : Fin directWidth) : Fin (sourceWidth privateWidth) :=
  Fin.castLE (by simp [sourceWidth]) column

def publicSourceColumn {privateWidth : Nat}
    (column : Fin publicWidth) : Fin (sourceWidth privateWidth) :=
  directSourceColumn (Fin.castLE publicWidth_le_directWidth column)

def laneSourceColumn {privateWidth : Nat}
    (column : Fin laneWidth) : Fin (sourceWidth privateWidth) :=
  directSourceColumn
    ⟨publicWidth + column.val, by
      have bounded := column.isLt
      norm_num [publicWidth, directWidth, laneWidth] at bounded ⊢
      omega⟩

def privateSourceColumn {privateWidth : Nat}
    (column : Fin privateWidth) : Fin (sourceWidth privateWidth) :=
  ⟨directWidth + column.val, by simp [sourceWidth, column.isLt]⟩

def canonical (value : F) : CanonicalGoldilocks :=
  ⟨value.val, by simpa [modulus, goldilocksModulus] using value.isLt⟩

theorem fieldDigit_lt_modulus {trit : Nat} (bounded : trit < 3) :
    fieldDigit trit < modulus := by
  interval_cases trit <;> simp [fieldDigit, modulus]

def centeredFieldDigit
    (value : CanonicalGoldilocks) (index : Fin digitCount) : F :=
  match CompactCommit.tritAt value index with
  | 0 => -1
  | 1 => 0
  | 2 => 1
  | _ => 0

@[simp] theorem centeredFieldDigit_val
    (value : CanonicalGoldilocks) (index : Fin digitCount) :
    (centeredFieldDigit value index).val =
      fieldDigit (CompactCommit.tritAt value index) := by
  have bounded := CompactCommit.tritAt_lt_three value index
  have alternatives :
      CompactCommit.tritAt value index = 0 \/
      CompactCommit.tritAt value index = 1 \/
      CompactCommit.tritAt value index = 2 := by
    omega
  rcases alternatives with equal | equal | equal <;>
    unfold centeredFieldDigit <;> rw [equal] <;> rfl

theorem protocolTrit_eq_compilerTrit
    (value : CanonicalGoldilocks) (index : Fin digitCount) :
    CompactCommit.tritAt value index =
      Nightstream.Implementation.R1CS.CenteredTernaryField.encodeTrit
        value.val index.val := by
  rw [CompactCommit.tritAt,
    Nightstream.Protocol.NebulaV2.ShiftedTernary41V1.trits_get]
  rfl

theorem centeredFieldDigit_eq_compilerDigit
    (value : CanonicalGoldilocks) (index : Fin digitCount) :
    (centeredFieldDigit value index).val =
      Nightstream.Implementation.R1CS.CenteredTernaryField.encodeDigit
        value.val index.val := by
  rw [centeredFieldDigit_val, protocolTrit_eq_compilerTrit value index]
  have bounded :=
    Nightstream.Implementation.R1CS.CenteredTernaryField.encodeTrit_lt_three
      value.val index.val
  have alternatives :
      Nightstream.Implementation.R1CS.CenteredTernaryField.encodeTrit
          value.val index.val = 0 \/
      Nightstream.Implementation.R1CS.CenteredTernaryField.encodeTrit
          value.val index.val = 1 \/
      Nightstream.Implementation.R1CS.CenteredTernaryField.encodeTrit
          value.val index.val = 2 := by
    omega
  rcases alternatives with equal | equal | equal <;>
    simp [fieldDigit,
      Nightstream.Implementation.R1CS.CenteredTernaryField.encodeDigit,
      equal, modulus,
      Nightstream.Implementation.R1CS.goldilocksP]

theorem centeredFieldDigit_norm
    (value : CanonicalGoldilocks) (index : Fin digitCount) :
    centeredMagnitude (centeredFieldDigit value index) < 2 := by
  have bounded := CompactCommit.tritAt_lt_three value index
  have alternatives :
      CompactCommit.tritAt value index = 0 \/
      CompactCommit.tritAt value index = 1 \/
      CompactCommit.tritAt value index = 2 := by
    omega
  rcases alternatives with equal | equal | equal <;>
    unfold centeredFieldDigit <;> rw [equal] <;> decide

/-- Every direct public or memory-lane coordinate is a bit. -/
def DirectBinary {privateWidth : Nat}
    (source : SourceAssignment privateWidth) : Prop :=
  forall column : Fin directWidth,
    source (directSourceColumn column) = 0 \/
      source (directSourceColumn column) = 1

def encodePayload {privateWidth : Nat}
    (source : SourceAssignment privateWidth) :
    Fin (payloadWidth privateWidth) -> F :=
  fun column =>
    match finSumFinEquiv.symm column with
    | Sum.inl directColumn => source (directSourceColumn directColumn)
    | Sum.inr privateFlat =>
        let pair := (finProdFinEquiv
          (m := privateWidth) (n := digitCount)).symm privateFlat
        centeredFieldDigit
          (canonical (source (privateSourceColumn pair.1))) pair.2

@[simp] theorem encodePayload_direct {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin directWidth) :
    encodePayload source
        (finSumFinEquiv (Sum.inl column :
          Fin directWidth ⊕ Fin (privateWidth * digitCount))) =
      source (directSourceColumn column) := by
  simp [encodePayload]

@[simp] theorem encodePayload_private {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin privateWidth)
    (index : Fin digitCount) :
    encodePayload source
        (finSumFinEquiv (Sum.inr (finProdFinEquiv (column, index)) :
          Fin directWidth ⊕ Fin (privateWidth * digitCount))) =
      centeredFieldDigit
        (canonical (source (privateSourceColumn column))) index := by
  simp [encodePayload]

def payloadColumn {privateWidth : Nat}
    (column : Fin (payloadWidth privateWidth)) :
    Fin (logicalWidth privateWidth) :=
  Phi81CarrierLayout.embedLogical column

def encodeLogical {privateWidth : Nat}
    (source : SourceAssignment privateWidth) :
    Fin (logicalWidth privateWidth) -> F :=
  Phi81CarrierLayout.extendAssignment 0 (encodePayload source)

def encodeCarrier {privateWidth : Nat}
    (source : SourceAssignment privateWidth) :
    ProductPaperAlgebra.Assignment
      (logicalWidth privateWidth) (publicFits privateWidth) :=
  Phi81CarrierLayout.extendAssignment 0 (encodeLogical source)

def carrierToLogical {privateWidth : Nat}
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (logicalWidth privateWidth))) : Fin (logicalWidth privateWidth) :=
  Fin.cast (logicalWidth_carrier_exact privateWidth) column

def logicalToCarrier {privateWidth : Nat}
    (column : Fin (logicalWidth privateWidth)) :
    Fin (Phi81CarrierLayout.carrierWidth (logicalWidth privateWidth)) :=
  Fin.cast (logicalWidth_carrier_exact privateWidth).symm column

@[simp] theorem carrierToLogical_val {privateWidth : Nat}
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (logicalWidth privateWidth))) :
    (carrierToLogical column).val = column.val := rfl

@[simp] theorem logicalToCarrier_val {privateWidth : Nat}
    (column : Fin (logicalWidth privateWidth)) :
    (logicalToCarrier column).val = column.val := rfl

@[simp] theorem carrierToLogical_logicalToCarrier {privateWidth : Nat}
    (column : Fin (logicalWidth privateWidth)) :
    carrierToLogical (logicalToCarrier column) = column := by
  apply Fin.ext
  rfl

@[simp] theorem logicalToCarrier_carrierToLogical {privateWidth : Nat}
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (logicalWidth privateWidth))) :
    logicalToCarrier (carrierToLogical column) = column := by
  apply Fin.ext
  rfl

theorem embedLogical_carrierToLogical {privateWidth : Nat}
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (logicalWidth privateWidth))) :
    Phi81CarrierLayout.embedLogical (carrierToLogical column) = column := by
  apply Fin.ext
  rfl

theorem encodeCarrier_eq_encodeLogical {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (column : Fin (Phi81CarrierLayout.carrierWidth
      (logicalWidth privateWidth))) :
    encodeCarrier source column = encodeLogical source (carrierToLogical column) := by
  rw [← embedLogical_carrierToLogical column]
  exact Phi81CarrierLayout.extendAssignment_embedLogical 0
    (encodeLogical source) (carrierToLogical column)

@[simp] theorem encodeLogical_direct {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin directWidth) :
    encodeLogical source
        (payloadColumn (finSumFinEquiv (Sum.inl column :
          Fin directWidth ⊕ Fin (privateWidth * digitCount)))) =
      source (directSourceColumn column) := by
  rw [encodeLogical, payloadColumn,
    Phi81CarrierLayout.extendAssignment_embedLogical]
  exact encodePayload_direct source column

@[simp] theorem encodeLogical_public {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin publicWidth) :
    encodeLogical source
        (payloadColumn (finSumFinEquiv (Sum.inl
          (Fin.castLE publicWidth_le_directWidth column) :
          Fin directWidth ⊕ Fin (privateWidth * digitCount)))) =
      source (publicSourceColumn column) :=
  encodeLogical_direct source (Fin.castLE publicWidth_le_directWidth column)

@[simp] theorem encodeLogical_private {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin privateWidth)
    (index : Fin digitCount) :
    encodeLogical source
        (payloadColumn (finSumFinEquiv (Sum.inr
          (finProdFinEquiv (column, index)) :
          Fin directWidth ⊕ Fin (privateWidth * digitCount)))) =
      centeredFieldDigit
        (canonical (source (privateSourceColumn column))) index := by
  rw [encodeLogical, payloadColumn,
    Phi81CarrierLayout.extendAssignment_embedLogical]
  exact encodePayload_private source column index

/-- The direct lane block is the exact complete assignment interval that starts
after the ten public rings. -/
theorem encodeCarrier_lane {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin laneWidth) :
    encodeCarrier source
        (Phi81CarrierLayout.embedLogical
          (payloadColumn (finSumFinEquiv (Sum.inl
            (⟨publicWidth + column.val, by
              have bounded := column.isLt
              norm_num [publicWidth, directWidth, laneWidth] at bounded ⊢
              omega⟩ : Fin directWidth) :
            Fin directWidth ⊕ Fin (privateWidth * digitCount))))) =
      source (laneSourceColumn column) := by
  rw [encodeCarrier, Phi81CarrierLayout.extendAssignment_embedLogical]
  exact encodeLogical_direct source
    (⟨publicWidth + column.val, by
      have bounded := column.isLt
      norm_num [publicWidth, directWidth, laneWidth] at bounded ⊢
      omega⟩ : Fin directWidth)

theorem encodePayload_norm {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (directBinary : DirectBinary source) :
    forall column, centeredMagnitude (encodePayload source column) < 2 := by
  intro column
  let split := finSumFinEquiv.symm column
  have columnEq : column = finSumFinEquiv split :=
    (finSumFinEquiv.apply_symm_apply column).symm
  obtain directColumn | privateFlat := split
  · rw [show column = finSumFinEquiv (Sum.inl directColumn) by exact columnEq]
    rw [encodePayload_direct]
    rcases directBinary directColumn with zero | one
    · rw [zero]
      decide
    · rw [one]
      decide
  · let pair := (finProdFinEquiv
      (m := privateWidth) (n := digitCount)).symm privateFlat
    rw [show column = finSumFinEquiv (Sum.inr privateFlat) by exact columnEq]
    rw [show privateFlat = finProdFinEquiv pair by
      exact ((finProdFinEquiv
        (m := privateWidth) (n := digitCount)).apply_symm_apply
          privateFlat).symm]
    rw [encodePayload_private]
    exact centeredFieldDigit_norm
      (canonical (source (privateSourceColumn pair.1))) pair.2

theorem encodeLogical_norm {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (directBinary : DirectBinary source) :
    forall column, centeredMagnitude (encodeLogical source column) < 2 := by
  intro column
  by_cases payload : column.val < payloadWidth privateWidth
  · let sourceColumn : Fin (payloadWidth privateWidth) := ⟨column.val, payload⟩
    have columnEq : column = payloadColumn sourceColumn := by
      apply Fin.ext
      rfl
    rw [columnEq, encodeLogical, payloadColumn,
      Phi81CarrierLayout.extendAssignment_embedLogical]
    exact encodePayload_norm source directBinary sourceColumn
  · have tail : payloadWidth privateWidth <= column.val := Nat.le_of_not_gt payload
    rw [encodeLogical,
      Phi81CarrierLayout.extendAssignment_tail_zero 0 _ column tail]
    decide

theorem encodeCarrier_norm {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (directBinary : DirectBinary source) :
    Phi81Relation.assignmentNormBounded 2 (encodeCarrier source) := by
  intro column
  by_cases logical : column.val < logicalWidth privateWidth
  · let sourceColumn : Fin (logicalWidth privateWidth) := ⟨column.val, logical⟩
    have columnEq : column = Phi81CarrierLayout.embedLogical sourceColumn := by
      apply Fin.ext
      rfl
    rw [columnEq]
    have exactValue :
        encodeCarrier source (Phi81CarrierLayout.embedLogical sourceColumn) =
          encodeLogical source sourceColumn :=
      Phi81CarrierLayout.extendAssignment_embedLogical 0
        (encodeLogical source) sourceColumn
    calc
      centeredMagnitude
          (encodeCarrier source
            (Phi81CarrierLayout.embedLogical sourceColumn)) =
          centeredMagnitude (encodeLogical source sourceColumn) :=
        congrArg centeredMagnitude exactValue
      _ < 2 := encodeLogical_norm source directBinary sourceColumn
  · have tail : logicalWidth privateWidth <= column.val := Nat.le_of_not_gt logical
    rw [encodeCarrier,
      Phi81CarrierLayout.extendAssignment_tail_zero 0 _ column tail]
    decide

theorem encodeCarrier_public {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin publicWidth) :
    Phi81Relation.projectPublicInput (encodeCarrier source) column =
      source (publicSourceColumn column) := by
  let logicalColumn : Fin (logicalWidth privateWidth) :=
    payloadColumn (finSumFinEquiv (Sum.inl
      (Fin.castLE publicWidth_le_directWidth column) :
      Fin directWidth ⊕ Fin (privateWidth * digitCount)))
  have carrierEq :
      (ProductPaperAlgebra.fullShape
        (logicalWidth privateWidth) (publicFits privateWidth)).publicColumn column =
        Phi81CarrierLayout.embedLogical logicalColumn := by
    rfl
  unfold Phi81Relation.projectPublicInput
  rw [carrierEq]
  have exactValue :
      encodeCarrier source (Phi81CarrierLayout.embedLogical logicalColumn) =
        encodeLogical source logicalColumn :=
    Phi81CarrierLayout.extendAssignment_embedLogical 0
      (encodeLogical source) logicalColumn
  exact exactValue.trans (encodeLogical_public source column)

theorem projectPublicInput_encodeCarrier {privateWidth : Nat}
    (source : SourceAssignment privateWidth) :
    Phi81Relation.projectPublicInput (encodeCarrier source) =
      fun column => source (publicSourceColumn column) := by
  funext column
  exact encodeCarrier_public source column

def privateTritWord {privateWidth : Nat}
    (source : SourceAssignment privateWidth)
    (column : Fin privateWidth) : List Nat :=
  trits (canonical (source (privateSourceColumn column)))

theorem privateTritWord_length {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin privateWidth) :
    (privateTritWord source column).length = digitCount :=
  trits_length _

theorem decode_privateTritWord {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin privateWidth) :
    decode (privateTritWord source column) =
      (source (privateSourceColumn column)).val :=
  decode_encode _

theorem encodeLogical_private_word_exact {privateWidth : Nat}
    (source : SourceAssignment privateWidth) (column : Fin privateWidth)
    (index : Fin digitCount) :
    (encodeLogical source
      (payloadColumn (finSumFinEquiv (Sum.inr
        (finProdFinEquiv (column, index)) :
        Fin directWidth ⊕ Fin (privateWidth * digitCount))))).val =
      fieldDigit ((privateTritWord source column).get
        ⟨index.val, by rw [privateTritWord_length]; exact index.isLt⟩) := by
  rw [encodeLogical_private]
  simp only [centeredFieldDigit_val]
  rfl

end Nightstream.Implementation.NebulaV2.ProductionFreshLowNormEncoding
