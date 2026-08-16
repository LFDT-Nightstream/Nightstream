import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCInputBinding

/-!
Contract: additive family residual for the complete production PiRLC input
binding.

Assurance tier: model-level exact refinement and Module-SIS reduction
boundary.

Owns the disjoint 110-family partition of the fixed 89,100-field Ajtai
witness, the local 810-field phase opening at unchanged global positions,
the exact sum of all family commitments, and recovery of the authoritative
PiCCS inputs from a zero-terminal aggregate residual or one named Module-SIS
failure.

Does not own a concrete seeded matrix, generated rows, local-transition
telescoping, Rust conformance, recursive state placement, or Module-SIS
hardness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 262144

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual

open scoped BigOperators
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputBinding
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.ShiftedTernary41V1

abbrev Family := ProductionStreamingPiRlcInputBinding.Family
abbrev Source := ProductionStreamingPiRlcInputBinding.Source
abbrev InputRings := ProductionStreamingPiRlcInputBinding.InputRings
abbrev RingF := Nightstream.SuperNeo.Concrete.RingF

/-! ## Exact family partition -/

/-- Field position carried by one physical Ajtai matrix coordinate. -/
def coordinateField
    (column : Fin shape.columns) (coefficient : Fin shape.degree) :
    Fin fieldCount :=
  ⟨flatIndex column coefficient / digitCount, by
    apply Nat.div_lt_iff_lt_mul (by decide : 0 < digitCount) |>.2
    exact flatIndex_lt column coefficient⟩

/-- Signed-ternary digit carried by one physical Ajtai matrix coordinate. -/
def coordinateDigit
    (column : Fin shape.columns) (coefficient : Fin shape.degree) :
    Fin digitCount :=
  ⟨flatIndex column coefficient % digitCount,
    Nat.mod_lt _ (by decide)⟩

theorem coordinateWitness_eq
    (fields : Fields) (column : Fin shape.columns)
    (coefficient : Fin shape.degree) :
    coordinateWitness fields column coefficient =
      signedDigit (fields (coordinateField column coefficient))
        (coordinateDigit column coefficient) := by
  rfl

/-- One family keeps its fixed global coordinates and zeros all other
coordinates. -/
def familyMaskedWitness
    (fields : Fields) (ordinal : Fin familyCount) : Witness shape :=
  fun column coefficient =>
    if positionOrdinal (coordinateField column coefficient) = ordinal then
      signedDigit (fields (coordinateField column coefficient))
        (coordinateDigit column coefficient)
    else
      0

/-- The 110 verifier-owned family masks are disjoint and complete. -/
theorem familyMaskedWitness_sum (fields : Fields) :
    (fun column coefficient =>
      ∑ ordinal : Fin familyCount,
        familyMaskedWitness fields ordinal column coefficient) =
      coordinateWitness fields := by
  classical
  funext column coefficient
  rw [coordinateWitness_eq]
  rw [Fintype.sum_eq_single
    (positionOrdinal (coordinateField column coefficient))]
  · simp [familyMaskedWitness]
  · intro other different
    simp [familyMaskedWitness, Ne.symm different]

def familyBinding
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (fields : Fields) (ordinal : Fin familyCount) :
    Commitment RingType shape :=
  commit matrix coefficientMap (familyMaskedWitness fields ordinal)

/-- The sum of all 110 family commitments is the direct full-vector
commitment. -/
theorem familyBindings_sum
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (fields : Fields) :
    (fun row =>
      ∑ ordinal : Fin familyCount,
        familyBinding matrix coefficientMap fields ordinal row) =
      bindingMap matrix coefficientMap fields := by
  funext row
  unfold familyBinding bindingMap commit
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro column _
  rw [← Finset.sum_mul, ← map_sum]
  apply congrArg (fun value => value * matrix row column)
  apply congrArg coefficientMap
  funext coefficient
  exact congrFun (congrFun (familyMaskedWitness_sum fields) column)
    coefficient

/-! ## Local 810-field opening -/

def canonicalInput
    (value : Nightstream.SuperNeo.Concrete.F) : CanonicalGoldilocks :=
  ⟨value.val, by
    simpa [Nightstream.Protocol.Nebula.ShiftedTernary41V1.modulus,
      Nightstream.SuperNeo.Concrete.goldilocksModulus] using value.isLt⟩

/-- One phase witness uses only the fifteen supplied source rings for the
selected family. Its nonzero coordinates stay at their global positions. -/
def phaseWitness
    (family : Family) (inputs : Source → RingF) : Witness shape :=
  fun column coefficient =>
    if positionOrdinal (coordinateField column coefficient) =
        familyIndex family then
      signedDigit
        (canonicalInput
          (inputs (positionSource (coordinateField column coefficient))
            (positionLane (coordinateField column coefficient))))
        (coordinateDigit column coefficient)
    else
      0

private theorem canonicalInput_eq_inputVector
    (inputs : InputRings) (family : Family) (field : Fin fieldCount)
    (selected : positionOrdinal field = familyIndex family) :
    canonicalInput
        (inputs (positionSource field) family (positionLane field)) =
      inputVector inputs field := by
  have familyExact : positionFamily field = family := by
    unfold positionFamily
    rw [selected, familyAtOrdinal_familyIndex]
  apply Subtype.ext
  simp [canonicalInput, inputVector, familyExact]

theorem phaseWitness_eq_familyMaskedWitness
    (inputs : InputRings) (family : Family) :
    phaseWitness family (fun source => inputs source family) =
      familyMaskedWitness (inputVector inputs) (familyIndex family) := by
  funext column coefficient
  by_cases selected : positionOrdinal (coordinateField column coefficient) =
      familyIndex family
  · simp [phaseWitness, familyMaskedWitness, selected,
      canonicalInput_eq_inputVector inputs family
        (coordinateField column coefficient) selected]
  · simp [phaseWitness, familyMaskedWitness, selected]

def phaseBinding
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (family : Family) (inputs : Source → RingF) :
    Commitment RingType shape :=
  commit matrix coefficientMap (phaseWitness family inputs)

theorem phaseBinding_eq_familyBinding
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (inputs : InputRings) (family : Family) :
    phaseBinding matrix coefficientMap family
        (fun source => inputs source family) =
      familyBinding matrix coefficientMap (inputVector inputs)
        (familyIndex family) := by
  unfold phaseBinding familyBinding
  rw [phaseWitness_eq_familyMaskedWitness inputs family]

/-- The sum of the 110 local 810-field openings is the full PiRLC input
binding. -/
theorem phaseBindings_sum
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (inputs : InputRings) :
    (fun row =>
      ∑ ordinal : Fin familyCount,
        phaseBinding matrix coefficientMap (familyAtOrdinal ordinal)
          (fun source => inputs source (familyAtOrdinal ordinal)) row) =
      inputBindingMap matrix coefficientMap inputs := by
  funext row
  calc
    (∑ ordinal : Fin familyCount,
        phaseBinding matrix coefficientMap (familyAtOrdinal ordinal)
          (fun source => inputs source (familyAtOrdinal ordinal)) row) =
        ∑ ordinal : Fin familyCount,
          familyBinding matrix coefficientMap (inputVector inputs) ordinal row := by
      apply Finset.sum_congr rfl
      intro ordinal _
      rw [phaseBinding_eq_familyBinding]
      simp
    _ = bindingMap matrix coefficientMap (inputVector inputs) row :=
      congrFun (familyBindings_sum matrix coefficientMap (inputVector inputs))
        row
    _ = inputBindingMap matrix coefficientMap inputs row := rfl

/-! ## Aggregate carried residual -/

def addResidual
    {RingType : Type} [CommRing RingType]
    (left right : Commitment RingType shape) : Commitment RingType shape :=
  fun row => left row + right row

def zeroResidual
    {RingType : Type} [CommRing RingType] : Commitment RingType shape :=
  fun _ => 0

/-- Exact local update that generated family glue must prove. -/
def ResidualTransition
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (before after : Commitment RingType shape)
    (family : Family) (inputs : Source → RingF) : Prop :=
  before = addResidual
    (phaseBinding matrix coefficientMap family inputs) after

/-- Aggregate meaning of a complete carried-residual run. Later generated
glue must derive this equation by telescoping all local transitions. -/
def CompleteResidualRun
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (start finish : Commitment RingType shape)
    (inputs : InputRings) : Prop :=
  start = addResidual
    (fun row =>
      ∑ ordinal : Fin familyCount,
        phaseBinding matrix coefficientMap (familyAtOrdinal ordinal)
          (fun source => inputs source (familyAtOrdinal ordinal)) row)
    finish

theorem honest_completeResidualRun
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (inputs : InputRings) :
    CompleteResidualRun matrix coefficientMap
      (inputBindingMap matrix coefficientMap inputs) zeroResidual inputs := by
  unfold CompleteResidualRun
  funext row
  simp only [addResidual, zeroResidual, add_zero]
  exact (congrFun (phaseBindings_sum matrix coefficientMap inputs) row).symm

/-- A zero-terminal residual that starts at the authoritative PiCCS binding
recovers every supplied PiRLC input, or exposes the named rank-two Module-SIS
failure. -/
theorem complete_zero_residual_recovers_inputs_or_failure
    {RingType : Type} [CommRing RingType]
    (matrix : Matrix RingType shape)
    (coefficientMap : CoefficientVector shape →+ RingType)
    (authoritative supplied : InputRings)
    (start : Commitment RingType shape)
    (startAuthoritative :
      start = inputBindingMap matrix coefficientMap authoritative)
    (run : CompleteResidualRun matrix coefficientMap start zeroResidual
      supplied) :
    Or (supplied = authoritative)
      (BindingFailure matrix coefficientMap) := by
  apply equal_input_binding_recovers_inputs_or_failure matrix coefficientMap
    supplied authoritative
  calc
    inputBindingMap matrix coefficientMap supplied =
        (fun row =>
          ∑ ordinal : Fin familyCount,
            phaseBinding matrix coefficientMap (familyAtOrdinal ordinal)
              (fun source => supplied source (familyAtOrdinal ordinal)) row) :=
      (phaseBindings_sum matrix coefficientMap supplied).symm
    _ = start := by
      have exactRun :
          start =
            (fun row =>
              ∑ ordinal : Fin familyCount,
                phaseBinding matrix coefficientMap (familyAtOrdinal ordinal)
                  (fun source =>
                    supplied source (familyAtOrdinal ordinal)) row) := by
        funext row
        have atRow := congrFun run row
        simpa [CompleteResidualRun, addResidual, zeroResidual] using atRow
      exact exactRun.symm
    _ = inputBindingMap matrix coefficientMap authoritative :=
      startAuthoritative

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputResidual
