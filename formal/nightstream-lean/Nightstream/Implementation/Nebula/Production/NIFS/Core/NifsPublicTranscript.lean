import Nightstream.Implementation.Nebula.NIFS.Core.Poseidon2
import Nightstream.Implementation.Nebula.Production.Carrier.FieldNativeFullClaim
import Nightstream.Implementation.Nebula.NIFS.Running.RunningParser

/-!
Contract: candidate-specific public-input transcript for field-native paper
NIFS.

The transcript absorbs the complete sixteen-running state, mandatory fresh
bundle, and 540-coordinate CCS public input once. Its resulting duplex state
can be reused by the outer F-prime state binding. The prefix fixes the
successor profile version and checked-step factor.

Equal transcript states recover the direct NIFS authority and the memory
batch, or expose an exact memory-digest or Poseidon2 transcript collision.
The application image is verifier-owned and remains outside this NIFS frame.

Does not own generated transcript rows, the remaining paper-NIFS messages,
Poseidon2 security, application-statement checks, terminal verification,
candidate selection, or a verifier key.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductionProductNifsPublicTranscript

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionFieldNativeFullClaim
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Encoding
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.Protocol.Nebula.WasmStatement
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev State := ProductPoseidon2.State
abbrev StatementId := ProductPoseidon2.StatementId

/-- ASCII `NSNF`. -/
def publicInputTag : Nat := 0x4e534e46
def frameVersion : Nat := 1
def profileNameTag : Nat := 3
def commitmentEncodingTag : Nat := 1

def fixedPrefix
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (degreeBound : Nat) : List Nat :=
  [publicInputTag, frameVersion, profileNameTag, version candidate,
    checkedStepsPerFreshClaim candidate, commitmentEncodingTag] ++
  ProductPoseidon2.shapeFields
      (ProductNifsCodec.shapeFor fullShape.rowVariables) ++
  [fullShape.logicalWidth, fullShape.carrierWidth,
    Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.blockCount
      fullShape.carrierWidth,
    ProductCommitmentAlgebra.Rank, fullShape.publicWidth, degreeBound]

theorem fixedPrefix_length
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (degreeBound : Nat) :
    (fixedPrefix candidate fullShape degreeBound).length = 17 := rfl

theorem fixedPrefix_candidate_injective
    (fullShape : Phi81Relation.Shape) (degreeBound : Nat) :
    Function.Injective
      (fun candidate => fixedPrefix candidate fullShape degreeBound) := by
  intro left right equal
  cases left <;> cases right <;>
    simp [fixedPrefix, version, checkedStepsPerFreshClaim] at equal ⊢

def nativeValues (values : List F) : List Nat := values.map Fin.val

theorem nativeValues_length (values : List F) :
    (nativeValues values).length = values.length := by
  simp [nativeValues]

/-- Coordinate selection commutes with canonical representative extraction.
The bounds are explicit so callers do not need to normalize dependent
`Fin.cast` proofs for a large fixed-size codec. -/
theorem nativeValues_getElem
    (values : List F) (index : Nat)
    (nativeBound : index < (nativeValues values).length)
    (fieldBound : index < values.length) :
    (nativeValues values)[index]'nativeBound =
      (values[index]'fieldBound).val := by
  simp only [nativeValues, List.getElem_map]

theorem nativeValues_injective : Function.Injective nativeValues := by
  intro left right equal
  exact (List.map_injective_iff.mpr
    (fun _ _ selected => Fin.ext selected)) equal

/-- Candidate-specific paper-NIFS serialization as a function of the exact
paper inputs. This is the function installed in the successor verifier key. -/
noncomputable def publicNifsFields
    (candidate : Id) {fullShape : Phi81Relation.Shape}
    (degreeBound : Nat)
    (running : Running fullShape)
    (fresh : Fresh fullShape) : List Nat :=
  fixedPrefix candidate fullShape degreeBound ++
    nativeValues (runningFields running) ++
    nativeValues
      (ProductNifsCodec.bundleCodec.encode
        (fresh.commitments ⟨0, by simp [ProductNifsCodec.shapeFor]⟩)) ++
    nativeValues
      ((NifsCanonicalCodec.publicInputCodec fullShape.publicWidth).encode
        (fresh.publicInputs ⟨0, by simp [ProductNifsCodec.shapeFor]⟩))

theorem publicNifsFields_lengthFor
    (candidate : Id) {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (degreeBound : Nat)
    (running : Running fullShape)
    (fresh : Fresh fullShape) :
    (publicNifsFields candidate degreeBound running fresh).length =
      17 + ProductNifsCodec.runningFieldCountFor fullShape.rowVariables +
        3888 + 540 := by
  simp only [publicNifsFields, List.length_append, fixedPrefix_length,
    nativeValues_length, runningFields_lengthFor contract,
    ProductNifsCodec.bundleCodec.encode_length]
  rw [(NifsCanonicalCodec.publicInputCodec
    fullShape.publicWidth).encode_length, contract.publicWidth]
  rfl

theorem publicNifsFields_length
    (candidate : Id) {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (degreeBound : Nat)
    (running : Running fullShape)
    (fresh : Fresh fullShape) :
    (publicNifsFields candidate degreeBound running fresh).length = 99535 := by
  rw [publicNifsFields_lengthFor candidate contract.toSelected,
    contract.rowVariables]
  decide

noncomputable def blocks
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (degreeBound : Nat) (value : Value candidate fullShape) :
    List (List Nat) :=
  [ fixedPrefix candidate fullShape degreeBound
  , nativeValues (runningFields value.recursiveState)
  , nativeValues (bundleFields value.commitmentBundle)
  , value.ccsPublic.val
  ]

theorem blocks_lengths
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (degreeBound : Nat) (value : Value candidate fullShape) :
    (blocks degreeBound value).map List.length = [17, 95090, 3888, 540] := by
  simpa [contract.rowVariables, ProductNifsCodec.runningFieldCountFor] using
    (show (blocks degreeBound value).map List.length =
        [17, ProductNifsCodec.runningFieldCountFor fullShape.rowVariables,
          3888, 540] by
      simp [blocks, fixedPrefix_length, nativeValues_length,
        runningFields_lengthFor contract.toSelected, bundleFields_length,
        value.ccsPublic.property.1, MemoryBoundCcsPublic.coordinateCount])

theorem blocks_lengthsFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (degreeBound : Nat) (value : Value candidate fullShape) :
    (blocks degreeBound value).map List.length =
      [17, ProductNifsCodec.runningFieldCountFor fullShape.rowVariables,
        3888, 540] := by
  simp [blocks, fixedPrefix_length, nativeValues_length,
    runningFields_lengthFor contract, bundleFields_length,
    value.ccsPublic.property.1, MemoryBoundCcsPublic.coordinateCount]

noncomputable def frame
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (degreeBound : Nat) (value : Value candidate fullShape) : List Nat :=
  (blocks degreeBound value).flatten

private theorem fieldOfBit_value_of_binary
    {digit : Nat} (binary : digit < 2) :
    (ProductNifsCodec.fieldOfBit digit).val = digit := by
  interval_cases digit <;> rfl

/-- The canonical paper public-input field vector is exactly the original
540-bit CCS word. This closes the only non-definitional bridge between the
typed fresh claim and the successor transcript frame. -/
theorem publicInputOfFor_native_encoding
    {rowVariables : Nat} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor rowVariables fullShape)
    (word : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    nativeValues
        ((NifsCanonicalCodec.publicInputCodec
          fullShape.publicWidth).encode
          (ProductNifsCodec.publicInputOfFor contract word)) =
      word.val := by
  apply List.ext_get
  · rw [nativeValues_length,
      (NifsCanonicalCodec.publicInputCodec
        fullShape.publicWidth).encode_length,
      NifsCanonicalCodec.publicInputCodec_width,
      contract.publicWidth, word.property.1]
  · intro index leftBound rightBound
    let encoded : List F :=
      (NifsCanonicalCodec.publicInputCodec
        fullShape.publicWidth).encode
        (ProductNifsCodec.publicInputOfFor contract word)
    have encodedBound :
        index < encoded.length := by
      simpa [nativeValues_length] using leftBound
    let wordIndex : Fin MemoryBoundCcsPublic.coordinateCount :=
      ⟨index, by simpa [word.property.1] using rightBound⟩
    let column : Fin fullShape.publicWidth :=
      Fin.cast contract.publicWidth.symm wordIndex
    have selected := ProductNifsRunningParser.publicInputCodec_getD
      (ProductNifsCodec.publicInputOfFor contract word) column
    have encodedGetD :
        encoded.getD index 0 = encoded.get ⟨index, encodedBound⟩ := by
      simp [List.getD_eq_getElem?_getD, encodedBound]
    have wordGetD : word.val.getD index 0 = word.val[index]'rightBound := by
      simp [List.getD_eq_getElem?_getD, rightBound]
    have selectedAtIndex :
        encoded.getD index 0 =
          ProductNifsCodec.publicInputOfFor contract word column := by
      simpa [encoded, column] using selected
    change
      (nativeValues encoded).get ⟨index, leftBound⟩ =
        word.val.get ⟨index, rightBound⟩
    calc
      (nativeValues encoded).get ⟨index, leftBound⟩ =
          (encoded.get ⟨index, encodedBound⟩).val := by
            simp [nativeValues]
      _ = (ProductNifsCodec.publicInputOfFor contract word column).val := by
            rw [← encodedGetD, selectedAtIndex]
      _ = (ProductNifsCodec.fieldOfBit (word.val.getD index 0)).val := by
            rfl
      _ = word.val.get ⟨index, rightBound⟩ := by
            rw [wordGetD]
            exact fieldOfBit_value_of_binary
              (word.property.2 _ (List.getElem_mem rightBound))

theorem publicInputOf_native_encoding
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (word : FixedBits.Word MemoryBoundCcsPublic.coordinateCount) :
    nativeValues
        ((NifsCanonicalCodec.publicInputCodec
          fullShape.publicWidth).encode
          (ProductNifsCodec.publicInputOf contract word)) =
      word.val := by
  simpa [ProductNifsCodec.publicInputOf,
    ProductNifsCodec.publicInputOfFor] using
      publicInputOfFor_native_encoding contract.toSelected word

theorem publicNifsFields_of_value
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (degreeBound : Nat) (value : Value candidate fullShape) :
    publicNifsFields candidate degreeBound value.recursiveState
        (freshOfValue contract value) =
      frame degreeBound value := by
  simp [publicNifsFields, frame, blocks, freshOfValue,
    ProductionFieldNativeFullClaim.bundleFields,
    ProductNifsCodec.freshOfFor,
    publicInputOfFor_native_encoding contract]

/-- Candidate separation occurs in the first seventeen public-input fields,
before any running or fresh value and before Poseidon2. -/
theorem publicNifsFields_ne_of_candidate_ne
    {leftCandidate rightCandidate : Id}
    (different : leftCandidate ≠ rightCandidate)
    {fullShape : Phi81Relation.Shape} {degreeBound : Nat}
    (leftRunning : Running fullShape)
    (leftFresh : Fresh fullShape)
    (rightRunning : Running fullShape)
    (rightFresh : Fresh fullShape) :
    publicNifsFields leftCandidate degreeBound leftRunning leftFresh ≠
      publicNifsFields rightCandidate degreeBound rightRunning rightFresh := by
  intro equal
  have prefixes := congrArg (List.take 17) equal
  have prefixEqual :
      fixedPrefix leftCandidate fullShape degreeBound =
        fixedPrefix rightCandidate fullShape degreeBound := by
    simpa [publicNifsFields, fixedPrefix_length] using prefixes
  exact different
    (fixedPrefix_candidate_injective fullShape degreeBound prefixEqual)

theorem frame_length
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (degreeBound : Nat) (value : Value candidate fullShape) :
    (frame degreeBound value).length = 99535 := by
  rw [frame, List.length_flatten, blocks_lengths contract]
  decide

theorem frame_lengthFor
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    (degreeBound : Nat) (value : Value candidate fullShape) :
    (frame degreeBound value).length =
      17 + ProductNifsCodec.runningFieldCountFor fullShape.rowVariables +
        3888 + 540 := by
  rw [frame, List.length_flatten, blocks_lengthsFor contract]
  simp
  omega

private theorem block_fields_equal
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    {degreeBound : Nat} {left right : Value candidate fullShape}
    (equal : blocks degreeBound left = blocks degreeBound right) :
    nativeValues (runningFields left.recursiveState) =
        nativeValues (runningFields right.recursiveState) /\
      nativeValues (bundleFields left.commitmentBundle) =
        nativeValues (bundleFields right.commitmentBundle) /\
      left.ccsPublic.val = right.ccsPublic.val := by
  simpa [blocks] using equal

/-- Equal pre-hash NIFS frames recover all direct NIFS inputs and the
complete memory batch, except for the named CCS memory-digest collision. -/
theorem frame_eq_recovers_direct_authority_or_memory_collision
    {candidate : Id}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {degreeBound : Nat} {left right : Value candidate fullShape}
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (leftMemoryBound : left.MemoryBound)
    (rightMemoryBound : right.MemoryBound)
    (equal : frame degreeBound left = frame degreeBound right) :
    (left.recursiveState = right.recursiveState /\
      left.commitmentBundle = right.commitmentBundle /\
      left.ccsPublic = right.ccsPublic /\
      left.memory = right.memory) \/
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate := by
  have blockEqual : blocks degreeBound left = blocks degreeBound right :=
    WasmResultCodec.flatten_injective_of_lengths
      (blocks_lengthsFor contract degreeBound left)
      (blocks_lengthsFor contract degreeBound right) equal
  have fields := block_fields_equal blockEqual
  have runningEncoded : runningFields left.recursiveState =
      runningFields right.recursiveState := nativeValues_injective fields.1
  have runningEqual : left.recursiveState = right.recursiveState := by
    apply (ProductNifsCodec.runningCodecFor fullShape.rowVariables
      fullShape).encode_injective_of_admissible
      (ProductNifsCodec.runningCodecFor_admissible left.recursiveState)
      (ProductNifsCodec.runningCodecFor_admissible right.recursiveState)
    exact runningEncoded
  have bundleEncoded : bundleFields left.commitmentBundle =
      bundleFields right.commitmentBundle :=
    nativeValues_injective fields.2.1
  have bundleEqual : left.commitmentBundle = right.commitmentBundle := by
    apply ProductNifsCodec.codecBundle_injective
    apply ProductNifsCodec.bundleCodec.encode_injective_of_admissible
      (ProductNifsCodec.bundleCodec_admissible
        (ProductNifsCodec.codecBundle left.commitmentBundle))
      (ProductNifsCodec.bundleCodec_admissible
        (ProductNifsCodec.codecBundle right.commitmentBundle))
    exact bundleEncoded
  have ccsEqual : left.ccsPublic = right.ccsPublic :=
    Subtype.ext fields.2.2
  have rightMatchesLeft :
      ProductionMemoryBoundCcsPublic.MemoryMatches
        left.ccsPublic right.memory := by
    rw [ccsEqual]
    exact rightMemoryBound
  rcases ProductionMemoryBoundCcsPublic.matched_batch_eq_or_collision
      leftCanonical.memoryCanonical rightCanonical.memoryCanonical
      leftMemoryBound rightMatchesLeft with
    memoryEqual | collision
  · exact Or.inl ⟨runningEqual, bundleEqual, ccsEqual, memoryEqual⟩
  · exact Or.inr collision

/-- Candidate separation occurs before Poseidon2. -/
theorem frames_ne_of_candidate_ne
    {leftCandidate rightCandidate : Id}
    (different : leftCandidate ≠ rightCandidate)
    {fullShape : Phi81Relation.Shape} {degreeBound : Nat}
    (left : Value leftCandidate fullShape)
    (right : Value rightCandidate fullShape) :
    frame degreeBound left ≠ frame degreeBound right := by
  intro equal
  have prefixes := congrArg (List.take 17) equal
  have prefixEqual :
      fixedPrefix leftCandidate fullShape degreeBound =
        fixedPrefix rightCandidate fullShape degreeBound := by
    simpa [frame, blocks, fixedPrefix_length] using prefixes
  exact different (fixedPrefix_candidate_injective fullShape degreeBound
    prefixEqual)

noncomputable def publicState
    {candidate : Id} {fullShape : Phi81Relation.Shape}
    (statementId : StatementId) (degreeBound : Nat)
    (value : Value candidate fullShape) : State :=
  Poseidon2Duplex.absorbList ProductPoseidon2.constants
    (frame degreeBound value)
    (ProductPoseidon2.initialStateForStatement statementId)

structure CanonicalValue
    (candidate : Id) (fullShape : Phi81Relation.Shape) where
  value : Value candidate fullShape
  canonical : value.Canonical
  memoryBound : value.MemoryBound

def PublicTranscriptCollision
    (candidate : Id) (fullShape : Phi81Relation.Shape)
    (statementId : StatementId) (degreeBound : Nat) : Prop :=
  ∃ left right : CanonicalValue candidate fullShape,
    frame degreeBound left.value ≠ frame degreeBound right.value /\
      publicState statementId degreeBound left.value =
        publicState statementId degreeBound right.value

/-- Equal reused transcript states have one explicit interpretation. -/
theorem equal_publicState_recovers_authority_or_named_failure
    {candidate : Id}
    {fullShape : Phi81Relation.Shape}
    (contract : ProductNifsCodec.FullShapeContractFor
      fullShape.rowVariables fullShape)
    {statementId : StatementId} {degreeBound : Nat}
    (left right : CanonicalValue candidate fullShape)
    (equal : publicState statementId degreeBound left.value =
      publicState statementId degreeBound right.value) :
    (left.value.recursiveState = right.value.recursiveState /\
      left.value.commitmentBundle = right.value.commitmentBundle /\
      left.value.ccsPublic = right.value.ccsPublic /\
      left.value.memory = right.value.memory) \/
      ProductionMemoryBatchPoseidonBinding.PoseidonCollision candidate \/
      PublicTranscriptCollision candidate fullShape statementId
        degreeBound := by
  by_cases frameEqual :
      frame degreeBound left.value = frame degreeBound right.value
  · rcases frame_eq_recovers_direct_authority_or_memory_collision contract
      left.canonical right.canonical left.memoryBound right.memoryBound
      frameEqual with direct | memoryCollision
    · exact Or.inl direct
    · exact Or.inr (Or.inl memoryCollision)
  · exact Or.inr (Or.inr ⟨left, right, frameEqual, equal⟩)

end Nightstream.Implementation.Nebula.ProductionProductNifsPublicTranscript
