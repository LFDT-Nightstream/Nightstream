import Nightstream.Implementation.Nebula.NIFS.Running.Codec
import Nightstream.Implementation.Nebula.NIFS.Core.PaperSelection

/-!
Contract: exact specification decoder from one complete V2 claim envelope to
the product-commitment paper NIFS input.

Assurance tier: implementation model.

Owns the fixed V2 claim widths, the verifier-owned WASM statement section,
the complete running-claim section, the fresh-claim projection, fail-closed
decoding, and a concrete `ClaimDecoder` for the selected paper verifier.

Does not own the generated executable parser, recursive verifier rows, the
final relation matrices, Rust decoding, or cryptographic soundness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Nebula.ProductFullClaimDecoder

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Implementation.Nebula.ProductPaperNifsSelection
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

local instance concreteKZero : Zero K := ⟨K.zero⟩

/-- Exact authority-bearing V2 section widths. These widths select the
single-fresh, fourteen-running product NIFS profile. -/
def widths : CompilerWidths where
  ccsPublicBits := MemoryBoundCcsPublic.coordinateCount
  applicationPublicBits := 7868
  recursiveStateBits := runningBitCount
  ccsPublicPositive := by decide
  applicationPublicPositive := by decide
  recursiveStatePositive := by decide

@[simp] theorem widths_ccsPublicBits :
    widths.ccsPublicBits = MemoryBoundCcsPublic.coordinateCount := rfl

@[simp] theorem widths_applicationPublicBits :
    widths.applicationPublicBits = 7868 := rfl

@[simp] theorem widths_recursiveStateBits :
    widths.recursiveStateBits = runningBitCount := rfl

theorem widths_totalBits : widths.totalBits = 5587724 := by
  simp [widths, CompilerWidths.totalBits, runningBitCount,
    runningFieldCount, fieldBitWidth,
    MemoryBoundCcsPublic.coordinateCount,
    MemoryWireGeometry.mandatoryBundleBits_exact,
    MemoryWireGeometry.stepPublicBits_exact,
    WasmPublicStatementEncoding.profileSerializedBitCount]

/-- Exact bit image of the public WASM statement selected by the verifier. -/
def applicationWord (image : PublicImage) :
    FixedBits.Word widths.applicationPublicBits :=
  ⟨WasmPublicStatementCodec.encode image,
    by simpa [widths] using WasmPublicStatementCodec.encode_length image,
    fun digit member =>
      WasmPublicStatementCodec.encode_binary image digit member⟩

abbrev Running (fullShape : Phi81Relation.Shape) :=
  ProductNifsCodec.Running fullShape

abbrev Fresh (fullShape : Phi81Relation.Shape) :=
  ProductNifsCodec.Fresh fullShape

/-- The strong language of a complete claim. It states only codec and
verifier-owned statement facts. It does not assume NIFS acceptance, a paper
transition, memory balance, or execution. -/
structure WellFormed
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (value : Value widths) : Prop where
  canonical : value.Canonical
  applicationExact : value.applicationPublic =
    applicationWord expectedApplication
  memoryCarrierExact :
    MemoryBoundCcsPublic.MemoryMatches value.ccsPublic value.memory
  runningDecodes : ∃ running : Running fullShape,
    decodeRunning contract value.recursiveState = some running

def freshOfValue
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Value widths) : Fresh fullShape :=
  ProductNifsCodec.freshOf contract value.commitmentBundle value.ccsPublic

/-- Total interpretation required by `ClaimDecoder`. Malformed values take a
fixed zero value, but the selected decoder rejects them before this value can
reach the paper verifier. -/
def zeroRunning (fullShape : Phi81Relation.Shape) : Running fullShape where
  point := ⟨List.replicate ProductNifsCodec.shape.cubeVariables K.zero,
    by simp⟩
  commitments := fun _ _ _ _ => 0
  publicInputs := fun _ _ => 0
  evaluations := fun _ _ _ => K.zero

noncomputable def runningOfValue
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Value widths) : Running fullShape :=
  match decodeRunning contract value.recursiveState with
  | some running => running
  | none => zeroRunning fullShape

theorem runningOfValue_eq
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    {value : Value widths} {running : Running fullShape}
    (decoded : decodeRunning contract value.recursiveState = some running) :
    runningOfValue contract value = running := by
  simp [runningOfValue, decoded]

/-- Canonical semantic inverse of the complete envelope. It rejects every
bit string that is not the exact image of one strong well-formed value. The
release parser must be executable and must refine this definition. -/
noncomputable def decodeValue
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : FixedBits.Word widths.totalBits) : Option (Value widths) :=
  letI : Decidable
      (∃ value : Value widths,
        WellFormed contract expectedApplication value ∧ value.block = block) :=
    Classical.propDecidable _
  if existsValue :
      ∃ value : Value widths,
        WellFormed contract expectedApplication value ∧ value.block = block then
    some (Classical.choose existsValue)
  else
    none

theorem decodeValue_block
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (value : Value widths)
    (wellFormed : WellFormed contract expectedApplication value) :
    decodeValue contract expectedApplication value.block = some value := by
  let existsValue :
      ∃ candidate : Value widths,
        WellFormed contract expectedApplication candidate ∧
          candidate.block = value.block :=
    ⟨value, wellFormed, rfl⟩
  rw [decodeValue, dif_pos existsValue]
  apply congrArg some
  let candidate := Classical.choose existsValue
  have candidateSpec := Classical.choose_spec existsValue
  apply Value.encode_injective_on_canonical
    candidateSpec.1.canonical wellFormed.canonical
  exact congrArg Subtype.val candidateSpec.2

theorem decodeValue_success
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    {block : FixedBits.Word widths.totalBits}
    {value : Value widths}
    (decoded : decodeValue contract expectedApplication block = some value) :
    WellFormed contract expectedApplication value ∧ value.block = block := by
  unfold decodeValue at decoded
  split at decoded
  next existsValue =>
    have chosen : Classical.choose existsValue = value :=
      Option.some.inj decoded
    simpa [← chosen] using Classical.choose_spec existsValue
  next noValue => contradiction

noncomputable def decode
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : FixedBits.Word widths.totalBits) :
    Option (Running fullShape × Fresh fullShape) :=
  match decodeValue contract expectedApplication block with
  | none => none
  | some value =>
      match decodeRunning contract value.recursiveState with
      | none => none
      | some running => some (running, freshOfValue contract value)

theorem decode_block
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (value : Value widths)
    (wellFormed : WellFormed contract expectedApplication value) :
    decode contract expectedApplication value.block =
      some (runningOfValue contract value, freshOfValue contract value) := by
  rw [decode, decodeValue_block contract expectedApplication value wellFormed]
  rcases wellFormed.runningDecodes with ⟨running, runningDecoded⟩
  change
    (match decodeRunning contract value.recursiveState with
      | none => none
      | some decodedRunning =>
          some (decodedRunning, freshOfValue contract value)) =
      some (runningOfValue contract value, freshOfValue contract value)
  rw [runningDecoded]
  rw [runningOfValue_eq contract runningDecoded]

theorem decode_success
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    {block : FixedBits.Word widths.totalBits}
    {running : Running fullShape} {fresh : Fresh fullShape}
    (decoded : decode contract expectedApplication block =
      some (running, fresh)) :
    ∃ value : Value widths,
      WellFormed contract expectedApplication value ∧
        value.block = block ∧
        running = runningOfValue contract value ∧
        fresh = freshOfValue contract value := by
  cases decodedValue : decodeValue contract expectedApplication block with
  | none => simp [decode, decodedValue] at decoded
  | some value =>
      have semantic := decodeValue_success contract expectedApplication
        decodedValue
      cases runningDecoded : decodeRunning contract value.recursiveState with
      | none => simp [decode, decodedValue, runningDecoded] at decoded
      | some candidate =>
          have pairEqual :
              (candidate, freshOfValue contract value) = (running, fresh) :=
            Option.some.inj (by
              simpa [decode, decodedValue, runningDecoded] using decoded)
          refine ⟨value, semantic.1, semantic.2, ?_, ?_⟩
          · exact (congrArg Prod.fst pairEqual).symm.trans
              (runningOfValue_eq contract runningDecoded).symm
          · exact (congrArg Prod.snd pairEqual).symm

/-- Exact decoder instance used by the selected paper NIFS verifier. -/
noncomputable def claimDecoder
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage) :
    ClaimDecoder widths fullShape ProductNifsCodec.shape where
  WellFormed := WellFormed contract expectedApplication
  decode := decode contract expectedApplication
  runningOf := runningOfValue contract
  freshOf := freshOfValue contract
  decode_block := decode_block contract expectedApplication

end Nightstream.Implementation.Nebula.ProductFullClaimDecoder
