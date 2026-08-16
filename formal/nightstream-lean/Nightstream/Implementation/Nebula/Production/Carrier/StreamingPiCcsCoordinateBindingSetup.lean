import Mathlib.Algebra.Ring.MinimalAxioms
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBinding
import Nightstream.Implementation.R1CS.Core.SeededAjtai
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules

/-!
Contract: concrete seeded Phi81 setup for the production PiCCS variable
coordinate binding.

Assurance tier: implementation-to-security-reduction bridge.

Owns a type-class wrapper for the executable Phi81 quotient ring, the exact
integer-coefficient embedding, the rank-two seeded matrix selected by one
`SeededAjtai.Setup`, the fixed Rust seed identity, flattening of two ring
outputs to 108 canonical Goldilocks fields, and recovery of equal input fields
or the named Module-SIS failure for that concrete setup.

Does not own Rust `rand_chacha` conformance, sampler liveness for a deployed
seed, generated R1CS rows, phase selection, public-state placement, Module-SIS
hardness, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBinding
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.CompactCommit
open Nightstream.Protocol.Nebula.AjtaiBinding
open Nightstream.Protocol.Nebula.ShiftedTernary41V1
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

local instance : CommRing F :=
  CommRing.ofMinimalAxioms
    ConcreteCarrier.baseLaws.add_assoc
    ConcreteCarrier.baseLaws.zero_add
    Lean.Grind.Fin.neg_add_cancel
    ConcreteCarrier.baseLaws.mul_assoc
    ConcreteCarrier.baseLaws.mul_comm
    ConcreteCarrier.baseLaws.one_mul
    ConcreteCarrier.baseLaws.left_distrib

namespace ExecutablePhi81

/-- The executable coefficient vector with quotient-ring multiplication.
The wrapper prevents Lean's pointwise function multiplication from being used
at the Module-SIS boundary. -/
structure Ring where
  coefficients : RingF
deriving DecidableEq

@[ext]
theorem Ring.ext {left right : Ring}
    (equal : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  simp_all

instance : Zero Ring := ⟨⟨ringFZero⟩⟩
instance : One Ring := ⟨⟨ringFOne⟩⟩
instance : Add Ring :=
  ⟨fun left right => ⟨ringFAdd left.coefficients right.coefficients⟩⟩
instance : Neg Ring :=
  ⟨fun value =>
    ⟨Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules.ringFNeg
      value.coefficients⟩⟩
instance : Mul Ring :=
  ⟨fun left right => ⟨ringFMul left.coefficients right.coefficients⟩⟩

private abbrev laws :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules.ringFLaws

instance : CommRing Ring :=
  CommRing.ofMinimalAxioms
    (by
      intro left middle right
      apply Ring.ext
      exact laws.add_assoc left.coefficients middle.coefficients
        right.coefficients)
    (by
      intro value
      apply Ring.ext
      exact laws.zero_add value.coefficients)
    (by
      intro value
      apply Ring.ext
      calc
        ringFAdd
            (Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules.ringFNeg
              value.coefficients)
            value.coefficients =
            ringFAdd value.coefficients
              (Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules.ringFNeg
                value.coefficients) :=
          laws.add_comm _ _
        _ = ringFZero := laws.add_neg value.coefficients)
    (by
      intro left middle right
      apply Ring.ext
      exact laws.mul_assoc left.coefficients middle.coefficients
        right.coefficients)
    (by
      intro left right
      apply Ring.ext
      exact laws.mul_comm left.coefficients right.coefficients)
    (by
      intro value
      apply Ring.ext
      exact laws.one_mul value.coefficients)
    (by
      intro left middle right
      apply Ring.ext
      exact laws.left_distrib left.coefficients middle.coefficients
        right.coefficients)

@[simp]
theorem zero_coefficients : (0 : Ring).coefficients = ringFZero := rfl

@[simp]
theorem one_coefficients : (1 : Ring).coefficients = ringFOne := rfl

@[simp]
theorem add_coefficients (left right : Ring) :
    (left + right).coefficients =
      ringFAdd left.coefficients right.coefficients := rfl

@[simp]
theorem neg_coefficients (value : Ring) :
    (-value).coefficients =
      Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PaperForkModules.ringFNeg
        value.coefficients := rfl

@[simp]
theorem mul_coefficients (left right : Ring) :
    (left * right).coefficients =
      ringFMul left.coefficients right.coefficients := rfl

end ExecutablePhi81

/-- Canonical Goldilocks residue of one integer coefficient. -/
def integerResidue (value : Int) : F :=
  Int.cast (R := F) value

@[simp]
theorem integerResidue_zero_val : (integerResidue 0).val = 0 := by
  rfl

set_option maxRecDepth 10000 in
/-- The integer embedding and the canonical R1CS centered-digit encoding use
the same Goldilocks residue for every protocol digit. -/
theorem integerResidue_signedDigit
    (value : CanonicalGoldilocks) (digit : Fin digitCount) :
    (integerResidue (signedDigit value digit)).val =
      fieldDigit (tritAt value digit) := by
  have bound := tritAt_lt_three value digit
  have alternatives :
      tritAt value digit = 0 ∨ tritAt value digit = 1 ∨
        tritAt value digit = 2 := by
    omega
  rcases alternatives with equal | equal | equal
  · simp [integerResidue, signedDigit, equal, fieldDigit, Fin.val_neg,
      Nightstream.SuperNeo.Concrete.goldilocksModulus, modulus]
  · simp [integerResidue, signedDigit, equal, fieldDigit]
  · simp [integerResidue, signedDigit, equal, fieldDigit]
    norm_num [Nightstream.SuperNeo.Concrete.goldilocksModulus]

/-- Coefficientwise integer reduction into the executable Phi81 ring. -/
def coefficientMap : CoefficientVector shape →+ ExecutablePhi81.Ring where
  toFun vector := ⟨fun lane => integerResidue (vector lane)⟩
  map_zero' := by
    apply ExecutablePhi81.Ring.ext
    funext lane
    change Int.cast (R := F) 0 = 0
    exact Int.cast_zero (R := F)
  map_add' := by
    intro left right
    apply ExecutablePhi81.Ring.ext
    funext lane
    change Int.cast (R := F) (left lane + right lane) =
      Int.cast (R := F) (left lane) + Int.cast (R := F) (right lane)
    exact Int.cast_add (R := F) (left lane) (right lane)

@[simp]
theorem coefficientMap_coefficients
    (vector : CoefficientVector shape) (lane : Fin shape.degree) :
    (coefficientMap vector).coefficients lane =
      integerResidue (vector lane) := rfl

/-- Exact rank-two matrix selected by the verifier-owned seeded setup. -/
def seededMatrix
    (setup : SeededAjtai.Setup verifierRows messageColumnCount) :
    Matrix ExecutablePhi81.Ring shape :=
  fun row column => ⟨setup.verifierKey row column⟩

@[simp]
theorem seededMatrix_coefficients
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (row : Fin shape.rows) (column : Fin shape.columns) :
    (seededMatrix setup row column).coefficients =
      setup.verifierKey row column := rfl

/-- Master seed used by Rust's
`PI_CCS_VARIABLE_COORDINATE_SIS_CONFIG`. -/
def rustSeedBytes : List Nat := List.replicate 32 200

def rustDomain : Nat := 0x5049_4356_4152_4244

theorem exact_rust_identity :
    rustSeedBytes.length = 32 /\
      (∀ byte ∈ rustSeedBytes, byte < 256) /\
      rustDomain = 5785229234076271172 := by
  constructor
  · simp [rustSeedBytes]
  constructor
  · intro byte member
    have : byte = 200 := by
      simpa [rustSeedBytes] using member
    omega
  · rfl

/-- A production setup fixes the Rust master seed and supplies the successful
bounded sampler execution required by `SeededAjtai.Setup`. The rejection fuel
is verifier-owned setup data; it is not prover advice. -/
structure ProductionSetup where
  setup : SeededAjtai.Setup verifierRows messageColumnCount
  seed_eq : setup.seed.bytes = rustSeedBytes

theorem exact_chunk_geometry :
    SeededAjtai.chunkSize messageColumnCount = 16112 /\
      SeededAjtai.chunkCount messageColumnCount = 1 := by
  decide

abbrev OutputFields := FieldVector (shape.rows * shape.degree)

def outputPair
    (output : Fin (shape.rows * shape.degree)) :
    Fin shape.rows × Fin shape.degree :=
  (finProdFinEquiv (m := shape.rows) (n := shape.degree)).symm output

/-- Canonical row-major flattening of one commitment row/lane pair. -/
def outputIndex
    (row : Fin shape.rows) (lane : Fin shape.degree) :
    Fin (shape.rows * shape.degree) :=
  finProdFinEquiv (row, lane)

@[simp]
theorem outputPair_outputIndex
    (row : Fin shape.rows) (lane : Fin shape.degree) :
    outputPair (outputIndex row lane) = (row, lane) := by
  exact Equiv.symm_apply_apply _ _

theorem outputIndex_val
    (row : Fin shape.rows) (lane : Fin shape.degree) :
    (outputIndex row lane).val = row.val * shape.degree + lane.val := by
  unfold outputIndex
  change lane.val + shape.degree * row.val =
    row.val * shape.degree + lane.val
  ac_rfl

/-- Canonical field view of the two degree-54 ring outputs. -/
def flattenCommitment
    (commitment : Commitment ExecutablePhi81.Ring shape) : OutputFields :=
  fun output =>
    let pair := outputPair output
    let value := (commitment pair.1).coefficients pair.2
    ⟨value.val, by
      simpa [Nightstream.SuperNeo.Concrete.goldilocksModulus,
        Nightstream.Protocol.Nebula.ShiftedTernary41V1.modulus] using
        value.isLt⟩

@[simp]
theorem flattenCommitment_outputIndex
    (commitment : Commitment ExecutablePhi81.Ring shape)
    (row : Fin shape.rows) (lane : Fin shape.degree) :
    (flattenCommitment commitment (outputIndex row lane)).val =
      ((commitment row).coefficients lane).val := by
  unfold flattenCommitment
  simp only [outputPair_outputIndex]

theorem flattenCommitment_injective :
    Function.Injective flattenCommitment := by
  intro left right equal
  funext row
  apply ExecutablePhi81.Ring.ext
  funext lane
  apply Fin.ext
  let output :=
    finProdFinEquiv (m := shape.rows) (n := shape.degree) (row, lane)
  have atOutput := congrFun equal output
  have pairEq : outputPair output = (row, lane) := by
    exact Equiv.symm_apply_apply _ _
  simp only [flattenCommitment] at atOutput
  have valueEq := congrArg Subtype.val atOutput
  change
    ((left (outputPair output).1).coefficients
        (outputPair output).2).val =
      ((right (outputPair output).1).coefficients
        (outputPair output).2).val at valueEq
  rw [pairEq] at valueEq
  exact valueEq

/-- Concrete verifier-key map carried as 108 canonical field coordinates. -/
def concreteBinding
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (fields : Fields) : OutputFields :=
  flattenCommitment
    (bindingMap (seededMatrix setup) coefficientMap fields)

def ConcreteBindingFailure
    (setup : SeededAjtai.Setup verifierRows messageColumnCount) : Prop :=
  BindingFailure (seededMatrix setup) coefficientMap

theorem equal_concrete_binding_recovers_fields_or_failure
    (setup : SeededAjtai.Setup verifierRows messageColumnCount)
    (left right : Fields)
    (equal : concreteBinding setup left = concreteBinding setup right) :
    left = right \/ ConcreteBindingFailure setup := by
  apply equal_binding_recovers_fields_or_failure
    (seededMatrix setup) coefficientMap left right
  exact flattenCommitment_injective equal

theorem exact_output_width :
    shape.rows * shape.degree = 108 := by
  decide

end Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingSetup
