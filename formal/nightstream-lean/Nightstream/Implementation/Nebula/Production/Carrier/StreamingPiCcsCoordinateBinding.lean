import Nightstream.Protocol.Nebula.CompactCommit
import Nightstream.Protocol.Nebula.AjtaiBinding

/-!
Contract: coordinate-preserving Ajtai binding for the variable part of the
production PiCCS statement.

Assurance tier: model-level Module-SIS reduction boundary.

Owns the exact 21,220-field geometry, the standard 41-coordinate
shifted-ternary packing, its 28-coordinate zero tail, injectivity and unit
bound of the packed witness, additive phase masking, and recovery of equal
field vectors or one rank-two Module-SIS kernel witness.

Does not own the concrete seeded matrix, generated rows, claim-chunk
selection, PiCCS transcript rows, hardness of Module-SIS, Rust refinement, or
recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding

open scoped BigOperators
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

def fieldCount : Nat := 21220
def messageColumnCount : Nat := 16112
def verifierRows : Nat := 2

theorem exact_geometry :
    fieldCount * digitCount = 870020 /\
      messageColumnCount * ringDegree = 870048 /\
      messageColumnCount * ringDegree - fieldCount * digitCount = 28 /\
      messageColumnCount <= 50371 := by
  decide

def shape : Shape where
  rows := verifierRows
  columns := messageColumnCount
  degree := ringDegree

theorem exact_shape :
    shape.rows = 2 /\ shape.columns = 16112 /\ shape.degree = 54 := by
  decide

abbrev Fields := FieldVector fieldCount

def flatIndex
    (column : Fin messageColumnCount) (coefficient : Fin ringDegree) : Nat :=
  coefficient.val * messageColumnCount + column.val

def wordIndex
    (field : Fin fieldCount) (digit : Fin digitCount) : Nat :=
  field.val * digitCount + digit.val

theorem wordIndex_lt (field : Fin fieldCount) (digit : Fin digitCount) :
    wordIndex field digit < fieldCount * digitCount := by
  unfold wordIndex fieldCount digitCount
  omega

def messagePosition
    (field : Fin fieldCount) (digit : Fin digitCount) :
    Fin messageColumnCount × Fin ringDegree :=
  (⟨wordIndex field digit % messageColumnCount,
      Nat.mod_lt _ (by decide)⟩,
   ⟨wordIndex field digit / messageColumnCount, by
      have indexBound := wordIndex_lt field digit
      norm_num [fieldCount, digitCount, messageColumnCount, ringDegree]
        at indexBound ⊢
      omega⟩)

theorem flatIndex_messagePosition
    (field : Fin fieldCount) (digit : Fin digitCount) :
    flatIndex (messagePosition field digit).1
        (messagePosition field digit).2 =
      wordIndex field digit := by
  unfold flatIndex messagePosition
  simpa [Nat.mul_comm, Nat.add_comm] using
    Nat.div_add_mod (wordIndex field digit) messageColumnCount

/-- Exact Rust row-major message. The last 28 matrix coordinates are zero. -/
def coordinateWitness (fields : Fields) : Witness shape :=
  fun column coefficient =>
    let index := flatIndex column coefficient
    if valid : index < fieldCount * digitCount then
      signedDigit
        (fields ⟨index / digitCount, by
          unfold fieldCount digitCount at valid ⊢
          omega⟩)
        ⟨index % digitCount, Nat.mod_lt _ (by decide)⟩
    else
      0

theorem coordinateWitness_at
    (fields : Fields) (field : Fin fieldCount) (digit : Fin digitCount) :
    coordinateWitness fields (messagePosition field digit).1
        (messagePosition field digit).2 =
      signedDigit (fields field) digit := by
  have valid := wordIndex_lt field digit
  have quotient : wordIndex field digit / digitCount = field.val := by
    unfold wordIndex
    rw [Nat.mul_comm]
    rw [Nat.mul_add_div (by decide : 0 < digitCount),
      Nat.div_eq_of_lt digit.isLt, Nat.add_zero]
  have remainder : wordIndex field digit % digitCount = digit.val := by
    unfold wordIndex
    exact Nat.mul_add_mod_of_lt digit.isLt
  simp only [coordinateWitness, flatIndex_messagePosition, dif_pos valid,
    quotient, remainder]

theorem coordinateWitness_injective : Function.Injective coordinateWitness := by
  intro left right equal
  funext field
  apply signedDigits_injective
  funext digit
  have atCoordinate := congrFun
    (congrFun equal (messagePosition field digit).1)
      (messagePosition field digit).2
  rw [coordinateWitness_at, coordinateWitness_at] at atCoordinate
  exact atCoordinate

theorem coordinateWitness_unit_bound
    (fields : Fields) (column : Fin shape.columns)
    (coefficient : Fin shape.degree) :
    (coordinateWitness fields column coefficient).natAbs ≤ 1 := by
  simp only [coordinateWitness]
  split
  · simpa [signedDigits] using signedDigits_unit_bound _ _
  · simp

def bindingMap
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (fields : Fields) : Commitment RingType shape :=
  commit matrix coefficientMap (coordinateWitness fields)

set_option maxRecDepth 262144 in
def refinement
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType) :
    MapRefinement Fields (Commitment RingType shape) RingType shape
      (bindingMap matrix coefficientMap) where
  matrix := matrix
  coefficientMap := coefficientMap
  witness := coordinateWitness
  witnessInjective := coordinateWitness_injective
  outputEquiv := Equiv.refl _
  correct := fun input => by
    change commit matrix coefficientMap (coordinateWitness input) =
      commit matrix coefficientMap (coordinateWitness input)
    rfl

/-- Exact failure exposed by two different canonical field vectors with the
same coordinate commitment. The bound is strict three because both packed
witnesses use only signed-unit coefficients. -/
def BindingFailure
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType) : Prop :=
  Nonempty (KernelWitness matrix coefficientMap 3)

theorem equal_binding_recovers_fields_or_failure
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (left right : Fields)
    (equal : bindingMap matrix coefficientMap left =
      bindingMap matrix coefficientMap right) :
    left = right \/ BindingFailure matrix coefficientMap := by
  by_cases same : left = right
  · exact Or.inl same
  · apply Or.inr
    exact signed_unit_collision_to_kernel
      (refinement matrix coefficientMap) same equal
      (fun fields => coordinateWitness_unit_bound fields)

/-- One phase keeps the fixed global word positions and replaces every word
outside its mask with zero. -/
def maskedWitness
    (fields : Fields) (selected : Fin fieldCount -> Bool) : Witness shape :=
  fun column coefficient =>
    let index := flatIndex column coefficient
    if valid : index < fieldCount * digitCount then
      let field : Fin fieldCount := ⟨index / digitCount, by
        unfold fieldCount digitCount at valid ⊢
        omega⟩
      if selected field then
        signedDigit (fields field)
          ⟨index % digitCount, Nat.mod_lt _ (by decide)⟩
      else
        0
    else
      0

theorem maskedWitness_at
    (fields : Fields) (selected : Fin fieldCount → Bool)
    (field : Fin fieldCount) (digit : Fin digitCount) :
    maskedWitness fields selected (messagePosition field digit).1
        (messagePosition field digit).2 =
      if selected field then signedDigit (fields field) digit else 0 := by
  have valid := wordIndex_lt field digit
  have quotient : wordIndex field digit / digitCount = field.val := by
    unfold wordIndex
    rw [Nat.mul_comm]
    rw [Nat.mul_add_div (by decide : 0 < digitCount),
      Nat.div_eq_of_lt digit.isLt, Nat.add_zero]
  have remainder : wordIndex field digit % digitCount = digit.val := by
    unfold wordIndex
    exact Nat.mul_add_mod_of_lt digit.isLt
  simp only [maskedWitness, flatIndex_messagePosition, dif_pos valid,
    quotient, remainder]

theorem maskedWitness_partition
    (fields : Fields) (selected : Fin fieldCount -> Bool) :
    (fun column coefficient =>
      maskedWitness fields selected column coefficient +
        maskedWitness fields (fun field => !(selected field))
          column coefficient) =
      coordinateWitness fields := by
  funext column coefficient
  simp only [maskedWitness, coordinateWitness]
  split
  · rename_i valid
    split <;> simp_all
  · rfl

theorem commit_mask_add_complement
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (fields : Fields) (selected : Fin fieldCount -> Bool) :
    (fun row =>
      commit matrix coefficientMap (maskedWitness fields selected) row +
        commit matrix coefficientMap
          (maskedWitness fields (fun field => !(selected field))) row) =
      commit matrix coefficientMap (coordinateWitness fields) := by
  funext row
  unfold commit
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro column _
  rw [← add_mul, ← map_add]
  have partition := congrFun
    (congrFun (maskedWitness_partition fields selected) column)
  have columnPartition :
      maskedWitness fields selected column +
          maskedWitness fields (fun field => !(selected field)) column =
        coordinateWitness fields column :=
    funext partition
  rw [columnPartition]

def productionChunkSourceRows : Nat :=
  digitCount + 1024 * 124 + 2 + verifierRows * ringDegree

def productionChunkSourceColumns : Nat :=
  1 + 1024 + digitCount + 1024 * 122 + 2 + verifierRows * ringDegree

theorem exact_production_chunk_source_shape :
    productionChunkSourceRows = 127127 /\
      productionChunkSourceColumns = 126104 /\
      productionChunkSourceRows < 2 ^ 24 /\
      productionChunkSourceColumns < 2 ^ 24 := by
  decide

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
