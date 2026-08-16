import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLC
import Nightstream.Protocol.Nebula.CompactCommit
import Nightstream.Protocol.Nebula.AjtaiBinding

/-!
Contract: fixed-position algebraic binding for the complete production PiRLC
input vector.

Assurance tier: model-level Module-SIS reduction boundary.

Owns the exact 89,100-field family-major geometry, its standard 41-coordinate
signed-ternary packing, the exact 67,650-column Phi81 message, the bijection
between family/source/lane coordinates and vector positions, and recovery of
equal PiRLC inputs or one rank-two Module-SIS kernel witness.

Does not own a concrete seeded matrix, generated rows, PiCCS output placement,
family-phase residual updates, Rust conformance, or Module-SIS hardness.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding

open scoped BigOperators
open Nightstream.Implementation.Nebula.ProductPiRlcAlgebraRows
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlc
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

abbrev Family := ProductionStreamingPiRlc.Family
abbrev Source := ProductionStreamingPiRlc.Source
abbrev InputRings := ProductionStreamingPiRlc.InputRings

def familyCount : Nat := 110
def sourceCount : Nat := 15
def laneCount : Nat := 54
def fieldsPerFamily : Nat := sourceCount * laneCount
def fieldCount : Nat := familyCount * fieldsPerFamily
def messageColumnCount : Nat := fieldCount * digitCount / ringDegree
def verifierRows : Nat := 2

theorem exact_geometry :
    fieldsPerFamily = 810 /\
      fieldCount = 89100 /\
      fieldCount * digitCount = 3653100 /\
      messageColumnCount = 67650 /\
      messageColumnCount * ringDegree = fieldCount * digitCount := by
  decide

def shape : Shape where
  rows := verifierRows
  columns := messageColumnCount
  degree := ringDegree

theorem exact_shape :
    shape.rows = 2 /\ shape.columns = 67650 /\ shape.degree = 54 := by
  decide

abbrev Fields := FieldVector fieldCount

/-! ## Exact family-major coordinate map -/

def componentAt : Fin 4 -> Nightstream.Protocol.Nebula.CommitmentBundle.Component
  | 0 => .full
  | 1 => .operations
  | 2 => .initialSnapshot
  | _ => .finalSnapshot

@[simp] theorem componentIndex_componentAt (index : Fin 4) :
    componentIndex (componentAt index) = index := by
  fin_cases index <;> rfl

/-- Inverse of the verifier-owned family ordinal on `0, ..., 109`. -/
def familyAtOrdinal (ordinal : Fin familyCount) : Family :=
  if commitment : ordinal.val < 72 then
    .commitment
      (componentAt ⟨ordinal.val / 18, by
        have bound := ordinal.isLt
        simp only [familyCount] at bound
        omega⟩)
      ⟨ordinal.val % 18, Nat.mod_lt _ (by decide)⟩
  else if publicInput : ordinal.val < 82 then
    .publicInput ⟨ordinal.val - 72, by omega⟩
  else
    .evaluation
      ⟨(ordinal.val - 82) / 2, by
        have bound := ordinal.isLt
        simp only [familyCount] at bound
        change (ordinal.val - 82) / 2 < 14
        omega⟩
      ⟨(ordinal.val - 82) % 2, Nat.mod_lt _ (by decide)⟩

@[simp] theorem familyOrdinal_familyAtOrdinal (ordinal : Fin familyCount) :
    familyOrdinal (familyAtOrdinal ordinal) = ordinal.val := by
  unfold familyAtOrdinal
  split
  · simp only [familyOrdinal, componentIndex_componentAt]
    omega
  · split
    · simp only [familyOrdinal]
      omega
    · simp only [familyOrdinal]
      have bound := ordinal.isLt
      simp only [familyCount] at bound
      omega

@[simp] theorem familyAtOrdinal_familyOrdinal (family : Family) :
    familyAtOrdinal
        ⟨familyOrdinal family, by
          simpa [familyCount] using familyOrdinal_lt family⟩ =
      family := by
  apply familyOrdinal_injective
  exact familyOrdinal_familyAtOrdinal _

/-- Canonical finite index of one verifier-owned PiRLC family. -/
def familyIndex (family : Family) : Fin familyCount :=
  ⟨familyOrdinal family, by
    simpa [familyCount] using familyOrdinal_lt family⟩

@[simp] theorem familyIndex_val (family : Family) :
    (familyIndex family).val = familyOrdinal family := rfl

@[simp] theorem familyAtOrdinal_familyIndex (family : Family) :
    familyAtOrdinal (familyIndex family) = family := by
  exact familyAtOrdinal_familyOrdinal family

@[simp] theorem familyIndex_familyAtOrdinal (ordinal : Fin familyCount) :
    familyIndex (familyAtOrdinal ordinal) = ordinal := by
  apply Fin.ext
  exact familyOrdinal_familyAtOrdinal ordinal

def familyOffset (source : Source) (lane : Fin laneCount) : Nat :=
  source.val * laneCount + lane.val

theorem familyOffset_lt (source : Source) (lane : Fin laneCount) :
    familyOffset source lane < fieldsPerFamily := by
  have sourceBound := source.isLt
  have laneBound := lane.isLt
  change source.val < 15 at sourceBound
  change lane.val < 54 at laneBound
  change source.val * 54 + lane.val < 810
  omega

def familyInputPosition
    (family : Family) (source : Source) (lane : Fin laneCount) :
    Fin fieldCount :=
  ⟨familyOrdinal family * fieldsPerFamily + familyOffset source lane, by
    have familyBound := familyOrdinal_lt family
    have offsetBound := familyOffset_lt source lane
    change familyOffset source lane < 810 at offsetBound
    change familyOrdinal family * 810 + familyOffset source lane < 89100
    omega⟩

def positionOrdinal (position : Fin fieldCount) : Fin familyCount :=
  ⟨position.val / fieldsPerFamily, by
    have bound := position.isLt
    simp only [fieldCount] at bound
    exact Nat.div_lt_iff_lt_mul (by decide : 0 < fieldsPerFamily) |>.2 bound⟩

def positionWithinFamily (position : Fin fieldCount) : Nat :=
  position.val % fieldsPerFamily

def positionFamily (position : Fin fieldCount) : Family :=
  familyAtOrdinal (positionOrdinal position)

def positionSource (position : Fin fieldCount) : Source :=
  ⟨positionWithinFamily position / laneCount, by
    have within := Nat.mod_lt position.val (by decide : 0 < fieldsPerFamily)
    simp only [positionWithinFamily, fieldsPerFamily, sourceCount] at within ⊢
    exact Nat.div_lt_iff_lt_mul (by decide : 0 < laneCount) |>.2 within⟩

def positionLane (position : Fin fieldCount) : Fin laneCount :=
  ⟨positionWithinFamily position % laneCount,
    Nat.mod_lt _ (by decide)⟩

private theorem position_div_family
    (family : Family) (source : Source) (lane : Fin laneCount) :
    (familyInputPosition family source lane).val / fieldsPerFamily =
      familyOrdinal family := by
  simp only [familyInputPosition]
  rw [Nat.mul_comm (familyOrdinal family) fieldsPerFamily]
  rw [Nat.mul_add_div (by decide : 0 < fieldsPerFamily)]
  rw [Nat.div_eq_of_lt (familyOffset_lt source lane), Nat.add_zero]

@[simp] theorem positionOrdinal_familyInputPosition
    (family : Family) (source : Source) (lane : Fin laneCount) :
    positionOrdinal (familyInputPosition family source lane) =
      familyIndex family := by
  apply Fin.ext
  exact position_div_family family source lane

private theorem position_mod_family
    (family : Family) (source : Source) (lane : Fin laneCount) :
    (familyInputPosition family source lane).val % fieldsPerFamily =
      familyOffset source lane := by
  simp only [familyInputPosition]
  exact Nat.mul_add_mod_of_lt (familyOffset_lt source lane)

private theorem offset_div_source (source : Source) (lane : Fin laneCount) :
    familyOffset source lane / laneCount = source.val := by
  unfold familyOffset
  rw [Nat.mul_comm source.val laneCount]
  rw [Nat.mul_add_div (by decide : 0 < laneCount)]
  rw [Nat.div_eq_of_lt lane.isLt, Nat.add_zero]

private theorem offset_mod_lane (source : Source) (lane : Fin laneCount) :
    familyOffset source lane % laneCount = lane.val := by
  exact Nat.mul_add_mod_of_lt lane.isLt

@[simp] theorem positionFamily_familyInputPosition
    (family : Family) (source : Source) (lane : Fin laneCount) :
    positionFamily (familyInputPosition family source lane) = family := by
  apply familyOrdinal_injective
  simp [positionFamily, positionOrdinal, position_div_family]

@[simp] theorem positionSource_familyInputPosition
    (family : Family) (source : Source) (lane : Fin laneCount) :
    positionSource (familyInputPosition family source lane) = source := by
  apply Fin.ext
  simp [positionSource, positionWithinFamily, position_mod_family,
    offset_div_source]

@[simp] theorem positionLane_familyInputPosition
    (family : Family) (source : Source) (lane : Fin laneCount) :
    positionLane (familyInputPosition family source lane) = lane := by
  apply Fin.ext
  simp [positionLane, positionWithinFamily, position_mod_family,
    offset_mod_lane]

/-- The family/source/lane coordinates reconstructed from a global position
return that exact global position. -/
@[simp] theorem familyInputPosition_positionCoordinates
    (position : Fin fieldCount) :
    familyInputPosition (positionFamily position) (positionSource position)
        (positionLane position) =
      position := by
  apply Fin.ext
  have ordinalValue :
      familyOrdinal (positionFamily position) =
        (positionOrdinal position).val := by
    unfold positionFamily
    exact familyOrdinal_familyAtOrdinal _
  have positionOrdinalValue :
      (positionOrdinal position).val = position.val / fieldsPerFamily := rfl
  have positionSourceValue :
      (positionSource position).val =
        position.val % fieldsPerFamily / laneCount := rfl
  have positionLaneValue :
      (positionLane position).val =
        position.val % fieldsPerFamily % laneCount := rfl
  have familySplit := Nat.div_add_mod position.val fieldsPerFamily
  have sourceSplit :=
    Nat.div_add_mod (position.val % fieldsPerFamily) laneCount
  have familySplit' :
      position.val / fieldsPerFamily * fieldsPerFamily +
          position.val % fieldsPerFamily =
        position.val := by
    simpa only [Nat.mul_comm] using familySplit
  have sourceSplit' :
      position.val % fieldsPerFamily / laneCount * laneCount +
          position.val % fieldsPerFamily % laneCount =
        position.val % fieldsPerFamily := by
    simpa only [Nat.mul_comm] using sourceSplit
  have familyTerm :
      familyOrdinal (positionFamily position) * fieldsPerFamily +
          familyOffset (positionSource position) (positionLane position) =
        (positionOrdinal position).val * fieldsPerFamily +
          familyOffset (positionSource position) (positionLane position) :=
    congrArg
      (fun ordinal => ordinal * fieldsPerFamily +
        familyOffset (positionSource position) (positionLane position))
      ordinalValue
  have offsetTerm :
      familyOffset (positionSource position) (positionLane position) =
        position.val % fieldsPerFamily / laneCount * laneCount +
          position.val % fieldsPerFamily % laneCount := by
    unfold familyOffset
    rw [positionSourceValue, positionLaneValue]
  calc
    (familyInputPosition (positionFamily position) (positionSource position)
        (positionLane position)).val =
        familyOrdinal (positionFamily position) * fieldsPerFamily +
          familyOffset (positionSource position) (positionLane position) := rfl
    _ = (positionOrdinal position).val * fieldsPerFamily +
        familyOffset (positionSource position) (positionLane position) :=
      familyTerm
    _ = position.val / fieldsPerFamily * fieldsPerFamily +
        (position.val % fieldsPerFamily / laneCount * laneCount +
          position.val % fieldsPerFamily % laneCount) := by
      rw [positionOrdinalValue, offsetTerm]
    _ = position.val := by rw [sourceSplit', familySplit']

/-- Exact family-major field vector consumed by the binding map. -/
def inputVector (inputs : InputRings) : Fields := fun position =>
  let value := inputs (positionSource position) (positionFamily position)
    (positionLane position)
  ⟨value.val, by
    simpa [Nightstream.Protocol.Nebula.ShiftedTernary41V1.modulus,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using value.isLt⟩

@[simp] theorem inputVector_familyInputPosition
    (inputs : InputRings) (family : Family) (source : Source)
    (lane : Fin laneCount) :
    inputVector inputs (familyInputPosition family source lane) =
      ⟨(inputs source family lane).val, by
        simpa [Nightstream.Protocol.Nebula.ShiftedTernary41V1.modulus,
          Nightstream.SuperNeo.Concrete.goldilocksModulus] using
          (inputs source family lane).isLt⟩ := by
  apply Subtype.ext
  simp [inputVector]

theorem inputVector_injective : Function.Injective inputVector := by
  intro left right equal
  funext source family lane
  have atPosition := congrFun equal (familyInputPosition family source lane)
  rw [inputVector_familyInputPosition, inputVector_familyInputPosition]
    at atPosition
  apply Fin.ext
  exact congrArg Subtype.val atPosition

/-! ## Fixed-position signed-ternary binding -/

def flatIndex
    (column : Fin messageColumnCount) (coefficient : Fin ringDegree) : Nat :=
  coefficient.val * messageColumnCount + column.val

def wordIndex (field : Fin fieldCount) (digit : Fin digitCount) : Nat :=
  field.val * digitCount + digit.val

theorem wordIndex_lt (field : Fin fieldCount) (digit : Fin digitCount) :
    wordIndex field digit < fieldCount * digitCount := by
  have fieldBound := field.isLt
  have digitBound := digit.isLt
  simp only [wordIndex, fieldCount, familyCount, fieldsPerFamily,
    sourceCount, laneCount, digitCount] at fieldBound digitBound ⊢
  omega

def messagePosition
    (field : Fin fieldCount) (digit : Fin digitCount) :
    Fin messageColumnCount × Fin ringDegree :=
  (⟨wordIndex field digit % messageColumnCount,
      Nat.mod_lt _ (by
        norm_num [messageColumnCount, fieldCount, familyCount,
          fieldsPerFamily, sourceCount, laneCount, digitCount, ringDegree])⟩,
   ⟨wordIndex field digit / messageColumnCount, by
      have indexBound := wordIndex_lt field digit
      norm_num [fieldCount, familyCount, fieldsPerFamily, sourceCount,
        laneCount, digitCount, messageColumnCount, ringDegree]
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

theorem flatIndex_lt
    (column : Fin messageColumnCount) (coefficient : Fin ringDegree) :
    flatIndex column coefficient < fieldCount * digitCount := by
  have columnBound := column.isLt
  have coefficientBound := coefficient.isLt
  norm_num [flatIndex, fieldCount, familyCount, fieldsPerFamily, sourceCount,
    laneCount, digitCount, messageColumnCount, ringDegree]
    at columnBound coefficientBound ⊢
  omega

def coordinateWitness (fields : Fields) : Witness shape :=
  fun column coefficient =>
    let index := flatIndex column coefficient
    let field : Fin fieldCount := ⟨index / digitCount, by
      apply Nat.div_lt_iff_lt_mul (by decide : 0 < digitCount) |>.2
      exact flatIndex_lt column coefficient⟩
    signedDigit (fields field)
      ⟨index % digitCount, Nat.mod_lt _ (by decide)⟩

theorem coordinateWitness_at
    (fields : Fields) (field : Fin fieldCount) (digit : Fin digitCount) :
    coordinateWitness fields (messagePosition field digit).1
        (messagePosition field digit).2 =
      signedDigit (fields field) digit := by
  have quotient : wordIndex field digit / digitCount = field.val := by
    unfold wordIndex
    rw [Nat.mul_comm field.val digitCount]
    rw [Nat.mul_add_div (by decide : 0 < digitCount)]
    rw [Nat.div_eq_of_lt digit.isLt, Nat.add_zero]
  have remainder : wordIndex field digit % digitCount = digit.val := by
    unfold wordIndex
    exact Nat.mul_add_mod_of_lt digit.isLt
  simp only [coordinateWitness, flatIndex_messagePosition, quotient, remainder]

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
    (coordinateWitness fields column coefficient).natAbs <= 1 := by
  simpa [coordinateWitness, signedDigits] using
    signedDigits_unit_bound
      (fields ⟨flatIndex column coefficient / digitCount, by
        apply Nat.div_lt_iff_lt_mul (by decide : 0 < digitCount) |>.2
        exact flatIndex_lt column coefficient⟩)
      ⟨flatIndex column coefficient % digitCount,
        Nat.mod_lt _ (by decide)⟩

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
  correct := fun _ => rfl

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

def inputBindingMap
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (inputs : InputRings) : Commitment RingType shape :=
  bindingMap matrix coefficientMap (inputVector inputs)

theorem equal_input_binding_recovers_inputs_or_failure
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (left right : InputRings)
    (equal : inputBindingMap matrix coefficientMap left =
      inputBindingMap matrix coefficientMap right) :
    left = right \/ BindingFailure matrix coefficientMap := by
  rcases equal_binding_recovers_fields_or_failure matrix coefficientMap
      (inputVector left) (inputVector right) equal with fieldsEqual | failure
  · exact Or.inl (inputVector_injective fieldsEqual)
  · exact Or.inr failure

/-- Exact handwritten source footprint for one 810-field family opening under
the fixed-position binding. It excludes arithmetic, selectors, and lifecycle
glue. -/
def perFamilySourceRows : Nat :=
  digitCount + fieldsPerFamily * 124 + 2 + verifierRows * ringDegree

def perFamilySourceColumns : Nat :=
  1 + fieldsPerFamily + digitCount + fieldsPerFamily * 122 + 2 +
    verifierRows * ringDegree

/-- A direct full-vector binding also fits the required local source domain.
Normalized selective-CCS geometry is a separate obligation. -/
def fullSourceRows : Nat :=
  digitCount + fieldCount * 124 + 2 + verifierRows * ringDegree

def fullSourceColumns : Nat :=
  1 + fieldCount + digitCount + fieldCount * 122 + 2 +
    verifierRows * ringDegree

theorem exact_source_geometry :
    perFamilySourceRows = 100591 /\
      perFamilySourceColumns = 99782 /\
      fullSourceRows = 11048551 /\
      fullSourceColumns = 10959452 /\
      perFamilySourceRows < 2 ^ 24 /\
      perFamilySourceColumns < 2 ^ 24 /\
      fullSourceRows < 2 ^ 24 /\
      fullSourceColumns < 2 ^ 24 := by
  decide

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
