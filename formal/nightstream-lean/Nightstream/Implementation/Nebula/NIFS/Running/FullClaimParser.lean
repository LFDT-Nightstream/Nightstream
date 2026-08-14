import Nightstream.Implementation.Nebula.Commitment.Bundle.Parser
import Nightstream.Implementation.Nebula.Memory.Claim.Parser
import Nightstream.Implementation.Nebula.NIFS.Running.FullClaimDecoder
import Nightstream.Implementation.Nebula.NIFS.Running.RunningParserCorrect

/-!
Contract: executable fail-closed parser for the complete V2 paper-NIFS claim
envelope.

Assurance tier: implementation-model refinement.

Owns exact full-envelope slicing, fixed-profile checking, verifier-owned WASM
statement checking, strict mandatory-bundle parsing, direct running-claim
field-layout parsing, strict memory-suffix parsing, complete re-encoding, and
the selected paper `ClaimDecoder` instance.

Does not own byte-container framing, generated parser rows, recursive-size
closure, Rust conformance, paper-NIFS soundness, or cryptographic reductions.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.Nebula.ProductFullClaimParser

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.ProductFullClaimDecoder
open Nightstream.Implementation.Nebula.ProductNifsCodec
open Nightstream.Implementation.Nebula.ProductPaperNifsSelection
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.WasmPublicStatementEncoding
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

abbrev Block := FixedBits.Word widths.totalBits

/-- Exact safe slice for one full-claim section. -/
def sectionWord (block : Block) (part : Section) :
    FixedBits.Word (part.width widths) :=
  FixedBits.slice block (part.bitOffset widths) (part.width widths)
    (part.slice_fits widths)

theorem sectionWord_block (value : Value widths) (part : Section) :
    (sectionWord value.block part).val = value.sectionBits part := by
  change
    (value.encode.drop (part.bitOffset widths)).take
        (part.width widths) = value.sectionBits part
  exact value.encode_slice part

def profileWord : FixedBits.Word
    Nightstream.Protocol.Nebula.WasmPublicStatementEncoding.profileSerializedBitCount :=
  ⟨WasmPublicStatementCodec.encodeProfile Profile.v2,
    WasmPublicStatementCodec.encodeProfile_length Profile.v2,
    fun digit member =>
      WasmPublicStatementCodec.encodeProfile_binary Profile.v2 digit member⟩

def profileMatches (block : Block) : Prop :=
  (sectionWord block .profile).val = profileWord.val

def applicationMatches (expectedApplication : PublicImage)
    (block : Block) : Prop :=
  (sectionWord block .applicationPublic).val =
    (applicationWord expectedApplication).val

instance profileMatchesDecidable (block : Block) :
    Decidable (profileMatches block) := by
  unfold profileMatches
  infer_instance

instance applicationMatchesDecidable (expectedApplication : PublicImage)
    (block : Block) : Decidable (applicationMatches expectedApplication block) :=
  by
    unfold applicationMatches
    infer_instance

/-- Candidate assembled only from verifier-owned constants and independently
parsed sections. The recursive-state word remains the exact input slice; its
typed interpretation is the separately returned `running` value. -/
def valueOf
    (block : Block)
    (bundle : CommitmentBundleCodec.Value)
    (memory : MemoryClaimCodec.Claim) : Value widths where
  profile := Profile.v2
  ccsPublic := sectionWord block .ccsPublic
  applicationPublic := sectionWord block .applicationPublic
  commitmentBundle := bundle
  recursiveState := sectionWord block .recursiveState
  memory := memory

theorem profileMatches_block
    (value : Value widths) (canonical : value.Canonical) :
    profileMatches value.block := by
  unfold profileMatches profileWord
  rw [sectionWord_block]
  change WasmPublicStatementCodec.encodeProfile value.profile =
    WasmPublicStatementCodec.encodeProfile Profile.v2
  rw [canonical.profileExact]

theorem applicationMatches_block
    (expectedApplication : PublicImage)
    (value : Value widths)
    (applicationExact : value.applicationPublic =
      applicationWord expectedApplication) :
    applicationMatches expectedApplication value.block := by
  unfold applicationMatches
  rw [sectionWord_block]
  exact congrArg Subtype.val applicationExact

theorem sectionWord_bundle_block (value : Value widths) :
    sectionWord value.block .commitmentBundle =
      CommitmentBundleParser.blockOfBundle value.commitmentBundle := by
  apply Subtype.ext
  rw [sectionWord_block]
  rfl

theorem sectionWord_recursive_block (value : Value widths) :
    sectionWord value.block .recursiveState = value.recursiveState := by
  apply Subtype.ext
  exact sectionWord_block value .recursiveState

theorem sectionWord_memory_block_of_canonical
    (value : Value widths) (canonical : value.memory.Canonical) :
    sectionWord value.block .memory =
      MemoryClaimParser.blockOfClaim value.memory canonical := by
  apply Subtype.ext
  rw [sectionWord_block]
  rfl


theorem valueOf_block
    (value : Value widths) (canonical : value.Canonical) :
    valueOf value.block value.commitmentBundle value.memory = value := by
  apply Value.ext
  · exact canonical.profileExact.symm
  · apply Subtype.ext
    exact sectionWord_block value .ccsPublic
  · apply Subtype.ext
    exact sectionWord_block value .applicationPublic
  · rfl
  · apply Subtype.ext
    exact sectionWord_block value .recursiveState
  · rfl

/-- Executable full-envelope parser. The final equality is a computed check
over the complete canonical envelope, not a caller-supplied premise. -/
def parseValue
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) : Option
      (Value widths × ProductNifsCodec.Running fullShape) :=
  if _profileExact : profileMatches block then
    if _applicationExact : applicationMatches expectedApplication block then
      match
        CommitmentBundleParser.parse
          (sectionWord block .commitmentBundle),
        ProductNifsRunningParser.parse contract
          (sectionWord block .recursiveState),
        MemoryClaimParser.parse (sectionWord block .memory) with
      | some bundle, some running, some memory =>
          let value := valueOf block bundle memory
          if _memoryCarrierExact :
              MemoryBoundCcsPublic.MemoryMatches value.ccsPublic value.memory then
            if _exact : value.block.val = block.val then
              some (value, running)
            else
              none
          else
            none
      | _, _, _ => none
    else
      none
  else
    none

/-- The exact executable codec language. It contains no NIFS-acceptance,
memory-balance, application-execution, or cryptographic conclusion. -/
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
  runningDecodes : ∃ running : ProductNifsCodec.Running fullShape,
    ProductNifsRunningParser.parse contract value.recursiveState = some running

def runningOfValue
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Value widths) : ProductNifsCodec.Running fullShape :=
  match ProductNifsRunningParser.parse contract value.recursiveState with
  | some running => running
  | none => zeroRunning fullShape

theorem runningOfValue_eq
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (value : Value widths)
    {running : ProductNifsCodec.Running fullShape}
    (parsed : ProductNifsRunningParser.parse contract value.recursiveState =
      some running) :
    runningOfValue contract value = running := by
  simp [runningOfValue, parsed]

/-- Parser completeness for every value in the exact executable codec
language. -/
theorem parseValue_block
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (value : Value widths)
    (wellFormed : WellFormed contract expectedApplication value) :
    parseValue contract expectedApplication value.block =
      some (value, runningOfValue contract value) := by
  rcases wellFormed.runningDecodes with ⟨running, runningParsed⟩
  unfold parseValue
  rw [dif_pos (profileMatches_block value wellFormed.canonical)]
  rw [dif_pos (applicationMatches_block expectedApplication value
    wellFormed.applicationExact)]
  rw [sectionWord_bundle_block value,
    CommitmentBundleParser.parse_blockOfBundle]
  rw [sectionWord_recursive_block value, runningParsed]
  rw [sectionWord_memory_block_of_canonical value
      wellFormed.canonical.memoryCanonical,
    MemoryClaimParser.parse_blockOfClaim]
  change
    (if memoryCarrierExact : MemoryBoundCcsPublic.MemoryMatches
        (valueOf value.block value.commitmentBundle value.memory).ccsPublic
        (valueOf value.block value.commitmentBundle value.memory).memory then
      if exact :
          (valueOf value.block value.commitmentBundle value.memory).block.val =
            value.block.val then
        some
          (valueOf value.block value.commitmentBundle value.memory, running)
      else none
    else none) = some (value, runningOfValue contract value)
  rw [valueOf_block value wellFormed.canonical]
  rw [dif_pos wellFormed.memoryCarrierExact]
  rw [dif_pos rfl]
  rw [runningOfValue_eq contract value runningParsed]

/-- Parser soundness at the codec boundary. Successful parsing proves the
fixed profile, verifier-owned application section, canonical memory suffix,
direct running-state interpretation, and equality with every input bit. -/
theorem parseValue_success
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    {block : Block} {value : Value widths}
    {running : ProductNifsCodec.Running fullShape}
    (accepted : parseValue contract expectedApplication block =
      some (value, running)) :
    WellFormed contract expectedApplication value ∧
      value.block = block ∧
      ProductNifsRunningParser.parse contract value.recursiveState =
        some running := by
  unfold parseValue at accepted
  split at accepted
  next profileExact =>
    split at accepted
    next applicationExact =>
      cases bundleResult : CommitmentBundleParser.parse
          (sectionWord block .commitmentBundle) with
      | none => simp [bundleResult] at accepted
      | some bundle =>
          cases runningResult : ProductNifsRunningParser.parse contract
              (sectionWord block .recursiveState) with
          | none => simp [bundleResult, runningResult] at accepted
          | some parsedRunning =>
              cases memoryResult : MemoryClaimParser.parse
                  (sectionWord block .memory) with
              | none =>
                  simp [bundleResult, runningResult, memoryResult] at accepted
              | some memory =>
                  simp only [bundleResult, runningResult, memoryResult] at accepted
                  split at accepted
                  next memoryCarrierExact =>
                    split at accepted
                    next exactEncoding =>
                      have pairEqual := Option.some.inj accepted
                      have valueEqual := congrArg Prod.fst pairEqual
                      have runningEqual := congrArg Prod.snd pairEqual
                      change valueOf block bundle memory = value at valueEqual
                      change parsedRunning = running at runningEqual
                      subst value
                      subst running
                      refine ⟨?_, ?_, ?_⟩
                      · refine ⟨?_, ?_, memoryCarrierExact, ?_⟩
                        · constructor
                          · rfl
                          · exact MemoryClaimParser.parse_claim_canonical
                              memoryResult
                        · apply Subtype.ext
                          exact applicationExact
                        · exact ⟨parsedRunning, by
                            simpa [valueOf] using runningResult⟩
                      · apply Subtype.ext
                        exact exactEncoding
                      · simpa [valueOf] using runningResult
                    next notExact => simp at accepted
                  next notMemoryCarrier => simp at accepted
    next notApplication => simp at accepted
  next notProfile => simp at accepted

def decode
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) : Option
      (ProductNifsCodec.Running fullShape × ProductNifsCodec.Fresh fullShape) :=
  match parseValue contract expectedApplication block with
  | none => none
  | some (value, running) =>
      some (running, freshOfValue contract value)

theorem decode_block
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (value : Value widths)
    (wellFormed : WellFormed contract expectedApplication value) :
    decode contract expectedApplication value.block =
      some (runningOfValue contract value, freshOfValue contract value) := by
  rw [decode, parseValue_block contract expectedApplication value wellFormed]

theorem decode_success
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    {block : Block}
    {running : ProductNifsCodec.Running fullShape}
    {fresh : ProductNifsCodec.Fresh fullShape}
    (accepted : decode contract expectedApplication block =
      some (running, fresh)) :
    ∃ value : Value widths,
      WellFormed contract expectedApplication value ∧
        value.block = block ∧
        ProductNifsRunningParser.parse contract value.recursiveState =
          some running ∧
        fresh = freshOfValue contract value := by
  unfold decode at accepted
  cases parsed : parseValue contract expectedApplication block with
  | none => simp [parsed] at accepted
  | some result =>
      rcases result with ⟨value, parsedRunning⟩
      have components : parsedRunning = running ∧
          freshOfValue contract value = fresh := by
        simpa [parsed] using accepted
      have runningEqual := components.1
      have freshEqual := components.2
      subst running
      subst fresh
      have sound := parseValue_success contract expectedApplication parsed
      exact ⟨value, sound.1, sound.2.1, sound.2.2, rfl⟩

theorem parseValue_rejects_profile_mismatch
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) (mismatch : ¬ profileMatches block) :
    parseValue contract expectedApplication block = none := by
  simp [parseValue, mismatch]

theorem parseValue_rejects_application_mismatch
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) (profileExact : profileMatches block)
    (mismatch : ¬ applicationMatches expectedApplication block) :
    parseValue contract expectedApplication block = none := by
  simp [parseValue, profileExact, mismatch]

theorem parseValue_rejects_bundle_failure
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) (profileExact : profileMatches block)
    (applicationExact : applicationMatches expectedApplication block)
    (bundleFailure : CommitmentBundleParser.parse
      (sectionWord block .commitmentBundle) = none) :
    parseValue contract expectedApplication block = none := by
  simp [parseValue, profileExact, applicationExact, bundleFailure]

theorem parseValue_rejects_running_failure
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) (profileExact : profileMatches block)
    (applicationExact : applicationMatches expectedApplication block)
    (runningFailure : ProductNifsRunningParser.parse contract
      (sectionWord block .recursiveState) = none) :
    parseValue contract expectedApplication block = none := by
  simp [parseValue, profileExact, applicationExact, runningFailure]

theorem parseValue_rejects_memory_failure
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) (profileExact : profileMatches block)
    (applicationExact : applicationMatches expectedApplication block)
    (memoryFailure : MemoryClaimParser.parse
      (sectionWord block .memory) = none) :
    parseValue contract expectedApplication block = none := by
  simp [parseValue, profileExact, applicationExact, memoryFailure]

theorem parseValue_rejects_memory_carrier_mismatch
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage)
    (block : Block) (profileExact : profileMatches block)
    (applicationExact : applicationMatches expectedApplication block)
    (bundle : CommitmentBundleCodec.Value)
    (running : ProductNifsCodec.Running fullShape)
    (memory : MemoryClaimCodec.Claim)
    (bundleParsed : CommitmentBundleParser.parse
      (sectionWord block .commitmentBundle) = some bundle)
    (runningParsed : ProductNifsRunningParser.parse contract
      (sectionWord block .recursiveState) = some running)
    (memoryParsed : MemoryClaimParser.parse
      (sectionWord block .memory) = some memory)
    (mismatch : ¬ MemoryBoundCcsPublic.MemoryMatches
      (valueOf block bundle memory).ccsPublic
      (valueOf block bundle memory).memory) :
    parseValue contract expectedApplication block = none := by
  simp [parseValue, profileExact, applicationExact, bundleParsed,
    runningParsed, memoryParsed, mismatch]

/-- Exact decoder instance used by the selected paper NIFS verifier. -/
def claimDecoder
    {fullShape : Phi81Relation.Shape}
    (contract : FullShapeContract fullShape)
    (expectedApplication : PublicImage) :
    ClaimDecoder widths fullShape ProductNifsCodec.shape where
  WellFormed := WellFormed contract expectedApplication
  decode := decode contract expectedApplication
  runningOf := runningOfValue contract
  freshOf := freshOfValue contract
  decode_block := decode_block contract expectedApplication

end Nightstream.Implementation.Nebula.ProductFullClaimParser
