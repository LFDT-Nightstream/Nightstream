import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.PaddedRowIdentityPoseidon2

/-!
Contract: versioned canonical transport codec for `PaddedRowIdentity`.

Owns:
- exact field order for running claims, fresh claims, and the NIFS proof;
- exact finite widths for every selected carrier;
- admissibility and injectivity through the shared Goldilocks codec laws;
- distinct public-input and proof envelope tags; and
- one canonical 64-bit word for every Goldilocks coordinate.

Does not own: a Rust decoder, a generated golden vector, network framing,
Poseidon2 permutation security, or R1CS rows.

Assurance tier: model-level. Rust-conformant and byte-vector status starts only
after the Rust codec exists and its emitted artifacts are checked here.

Wire rule: fields occur in the order below. Every field is its canonical
Goldilocks residue represented as one unsigned 64-bit word. A byte transport
must emit each word in little-endian order. No Serde, enum-layout, or bincode
choice is part of this contract.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCodec

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCodecCore
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentity
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityConcreteAlgebra

abbrev SelectedCommitment :=
  PaddedRowIdentityConcreteAlgebra.Commitment
abbrev SelectedPublicInput :=
  PaddedRowIdentityConcreteAlgebra.PublicInput
abbrev SelectedEvaluation :=
  PaddedRowIdentityConcreteAlgebra.Evaluation
abbrev SelectedRunning :=
  Running K SelectedCommitment SelectedPublicInput shape
abbrev SelectedFresh := Fresh SelectedCommitment SelectedPublicInput shape
abbrev SelectedProof := Proof K SelectedCommitment shape 9

/-! ## Selected carrier codecs -/

/-- Matrix-major, then coefficient-major, then low/high extension limbs. -/
noncomputable def evaluationCodec : Codec SelectedEvaluation :=
  Codec.finFunction shape.matrixCount
    (Codec.finFunction shape.coefficientCount kCodec)

@[simp] theorem evaluationCodec_width :
    evaluationCodec.width = 1512 := by
  rfl

theorem evaluationCodec_admissible (value : SelectedEvaluation) :
    evaluationCodec.Admissible value := by
  intro matrix coefficient
  exact kCodec_admissible (value matrix coefficient)

/-- Constant-first coefficients with the exact declared degree. -/
def fixedPolynomialData (value : FixedPolynomial K 9) : List K :=
  value.coefficients

theorem fixedPolynomialData_injective :
    Function.Injective fixedPolynomialData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def fixedPolynomialCodec : Codec (FixedPolynomial K 9) :=
  Codec.pullback (Codec.fixedList 10 K.zero kCodec)
    fixedPolynomialData fixedPolynomialData_injective

@[simp] theorem fixedPolynomialCodec_width :
    fixedPolynomialCodec.width = 20 := by
  rfl

theorem fixedPolynomialCodec_admissible (value : FixedPolynomial K 9) :
    fixedPolynomialCodec.Admissible value := by
  constructor
  · exact value.coefficients_length
  · intro index
    exact kCodec_admissible _

/-- Source-major, matrix-major, coefficient-major, then low/high limbs. -/
def fullOutputData
    (value : FullOutputCoordinates.FullOutput K shape) :=
  value.coordinate

theorem fullOutputData_injective :
    Function.Injective fullOutputData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def fullOutputCodec :
    Codec (FullOutputCoordinates.FullOutput K shape) :=
  Codec.pullback
    (Codec.finFunction shape.sourceCount
      (Codec.finFunction shape.matrixCount
        (Codec.finFunction shape.coefficientCount kCodec)))
    fullOutputData fullOutputData_injective

@[simp] theorem fullOutputCodec_width :
    fullOutputCodec.width = 22680 := by
  rfl

theorem fullOutputCodec_admissible
    (value : FullOutputCoordinates.FullOutput K shape) :
    fullOutputCodec.Admissible value := by
  intro source matrix coefficient
  exact kCodec_admissible (value.coordinate source matrix coefficient)

/-! ## Public claim codecs -/

def runningData (value : SelectedRunning) :=
  (value.point,
    (value.commitments, (value.publicInputs, value.evaluations)))

theorem runningData_injective : Function.Injective runningData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def runningCodec : Codec SelectedRunning :=
  Codec.pullback
    (Codec.product (pointCodec rowVariables)
      (Codec.product
        (Codec.finFunction shape.runningCount
          (commitmentCodec verifierRows))
        (Codec.product
          (Codec.finFunction shape.runningCount
            (publicInputCodec relationShape.publicWidth))
          (Codec.finFunction shape.runningCount evaluationCodec))))
    runningData runningData_injective

@[simp] theorem runningCodec_width : runningCodec.width = 38604 := by
  rfl

theorem runningCodec_admissible (value : SelectedRunning) :
    runningCodec.Admissible value := by
  exact ⟨pointCodec_admissible value.point,
    (fun index => commitmentCodec_admissible (value.commitments index)),
    (fun index => publicInputCodec_admissible (value.publicInputs index)),
    (fun index => evaluationCodec_admissible (value.evaluations index))⟩

def freshData (value : SelectedFresh) :=
  (value.commitments, value.publicInputs)

theorem freshData_injective : Function.Injective freshData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def freshCodec : Codec SelectedFresh :=
  Codec.pullback
    (Codec.product
      (Codec.finFunction shape.freshCount (commitmentCodec verifierRows))
      (Codec.finFunction shape.freshCount
        (publicInputCodec relationShape.publicWidth)))
    freshData freshData_injective

@[simp] theorem freshCodec_width : freshCodec.width = 1242 := by
  rfl

theorem freshCodec_admissible (value : SelectedFresh) :
    freshCodec.Admissible value := by
  exact ⟨
    (fun index => commitmentCodec_admissible (value.commitments index)),
    (fun index => publicInputCodec_admissible (value.publicInputs index))⟩

noncomputable def publicClaimsCodec :
    Codec (SelectedRunning × SelectedFresh) :=
  Codec.product runningCodec freshCodec

@[simp] theorem publicClaimsCodec_width :
    publicClaimsCodec.width = 39846 := by
  rfl

theorem publicClaimsCodec_admissible
    (value : SelectedRunning × SelectedFresh) :
    publicClaimsCodec.Admissible value :=
  ⟨runningCodec_admissible value.1, freshCodec_admissible value.2⟩

/-! ## Proof codec -/

def proofData (value : SelectedProof) :=
  (value.piCcsRounds,
    (value.piCcsOutput,
      (value.piDecCommitments, value.piDecEvaluations)))

theorem proofData_injective : Function.Injective proofData := by
  intro left right equal
  cases left
  cases right
  cases equal
  rfl

noncomputable def proofCodec : Codec SelectedProof :=
  Codec.pullback
    (Codec.product
      (Codec.finFunction shape.cubeVariables fixedPolynomialCodec)
      (Codec.product fullOutputCodec
        (Codec.product
          (Codec.finFunction shape.runningCount
            (commitmentCodec verifierRows))
          (Codec.finFunction shape.runningCount evaluationCodec))))
    proofData proofData_injective

@[simp] theorem proofCodec_width : proofCodec.width = 57936 := by
  rfl

theorem proofCodec_admissible (value : SelectedProof) :
    proofCodec.Admissible value := by
  exact ⟨
    (fun round => fixedPolynomialCodec_admissible (value.piCcsRounds round)),
    fullOutputCodec_admissible value.piCcsOutput,
    (fun index => commitmentCodec_admissible (value.piDecCommitments index)),
    (fun index => evaluationCodec_admissible (value.piDecEvaluations index))⟩

/-! ## Versioned wire envelopes -/

def fieldOfNat (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

def publicEnvelopeTag : Nat := 1001
def proofEnvelopeTag : Nat := 1002
def codecVersion : Nat := 1

def profileHeader : List F :=
  [ fieldOfNat rowVariables,
    fieldOfNat shape.freshCount,
    fieldOfNat shape.runningCount,
    fieldOfNat shape.matrixCount,
    fieldOfNat shape.coefficientCount,
    fieldOfNat assignmentColumns,
    fieldOfNat verifierRows,
    fieldOfNat relationShape.publicWidth,
    fieldOfNat 9 ]

noncomputable def publicWireFields
    (value : SelectedRunning × SelectedFresh) : List F :=
  [fieldOfNat publicEnvelopeTag, fieldOfNat codecVersion] ++
    profileHeader ++ publicClaimsCodec.encode value

noncomputable def proofWireFields (value : SelectedProof) : List F :=
  [fieldOfNat proofEnvelopeTag, fieldOfNat codecVersion] ++
    profileHeader ++ proofCodec.encode value

@[simp] theorem profileHeader_length : profileHeader.length = 9 := by
  rfl

@[simp] theorem publicWireFields_length
    (value : SelectedRunning × SelectedFresh) :
    (publicWireFields value).length = 39857 := by
  simp [publicWireFields, publicClaimsCodec.encode_length]

@[simp] theorem proofWireFields_length (value : SelectedProof) :
    (proofWireFields value).length = 57947 := by
  simp [proofWireFields, proofCodec.encode_length]

theorem publicWireFields_injective_on_admissible
    {left right : SelectedRunning × SelectedFresh}
    (equal : publicWireFields left = publicWireFields right) :
    left = right := by
  apply publicClaimsCodec.encode_injective_of_admissible
    (publicClaimsCodec_admissible left) (publicClaimsCodec_admissible right)
  have tails := congrArg (List.drop 11) equal
  simpa [publicWireFields] using tails

theorem proofWireFields_injective
    {left right : SelectedProof}
    (equal : proofWireFields left = proofWireFields right) :
    left = right := by
  apply proofCodec.encode_injective_of_admissible
    (proofCodec_admissible left) (proofCodec_admissible right)
  have tails := congrArg (List.drop 11) equal
  simpa [proofWireFields] using tails

/-! ## Canonical 64-bit field words -/

/-- Every Goldilocks residue fits in one unsigned 64-bit word. -/
def fieldWord (value : F) : BitVec 64 := BitVec.ofNat 64 value.val

theorem fieldValue_lt_twoPow64 (value : F) : value.val < 2 ^ 64 := by
  exact Nat.lt_trans value.isLt (by decide)

theorem fieldWord_toNat (value : F) :
    (fieldWord value).toNat = value.val := by
  simp only [fieldWord, BitVec.toNat_ofNat]
  exact Nat.mod_eq_of_lt (fieldValue_lt_twoPow64 value)

theorem fieldWord_injective : Function.Injective fieldWord := by
  intro left right equal
  apply Fin.ext
  have values := congrArg BitVec.toNat equal
  simpa [fieldWord_toNat] using values

/-- Word order is exactly field order. A byte transport writes each word in
little-endian order without any extra length or enum metadata. -/
def wireWords (fields : List F) : List (BitVec 64) :=
  fields.map fieldWord

@[simp] theorem wireWords_length (fields : List F) :
    (wireWords fields).length = fields.length := by
  simp [wireWords]

theorem wireWords_injective : Function.Injective wireWords := by
  intro left right equal
  exact (List.map_inj_right fieldWord_injective).mp equal

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.PaddedRowIdentityCodec
