import Nightstream.Implementation.NebulaV2.FPrime.State.AuthoritativeOutputRows
import Nightstream.Implementation.NebulaV2.Commitment.Bundle.ForwardingRows
import Nightstream.Implementation.NebulaV2.Commitment.Compact.CheckedStepChainRows
import Nightstream.Implementation.NebulaV2.Commitment.Compact.ChainHeaderRows
import Nightstream.Implementation.NebulaV2.FPrime.Claim.EnvelopeRows
import Nightstream.Implementation.NebulaV2.FPrime.Claim.NifsReceipt
import Nightstream.Implementation.NebulaV2.Memory.Carry.PublicRows
import Nightstream.Implementation.NebulaV2.Memory.Claim.Rows
import Nightstream.Implementation.NebulaV2.Memory.Product.BalanceRows
import Nightstream.Implementation.NebulaV2.Memory.Segment.ContinuationRows
import Nightstream.Implementation.NebulaV2.Memory.Segment.SourceRows
import Nightstream.Implementation.NebulaV2.Memory.Transition.TransitionRows
import Nightstream.Implementation.NebulaV2.FPrime.State.PriorLinkRows
import Nightstream.Implementation.NebulaV2.NIFS.Running.RunningFieldRows
import Nightstream.Implementation.NebulaV2.NIFS.Running.FullClaimDecoder
import Nightstream.Implementation.NebulaV2.FPrime.State.AuthorityBoundaryRows
import Nightstream.Implementation.R1CS.Canonical.ColumnWindows

/-!
Contract: parametric generated-row manifest schema for the V2 recursive branch.

Assurance tier: implementation schema.

Owns mandatory row-family order, exact row cover by construction, the known
row counts, nonempty opaque compiler families, exact full-claim memory and
bundle source links, the complete checked-step memory-source and product
relation, and non-overlapping target-column ownership windows.

Does not provide final compiler-selected widths, absolute rows or columns,
NIFS row semantics, typed lane-value refinement for the product chains,
recursive-size closure, or a verifier-key digest. A generated V2 artifact
must instantiate this schema.

Emits constraints: through the contained local row programs.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.RecursiveManifestSchema

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.ColumnWindows
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Program

/-- Canonical recursive-branch row owners. -/
inductive Owner where
  | prelude
  | priorStateBinding
  | fullClaimLink
  | nifsVerifier
  | memoryClaimValidation
  | memoryTransition
  | compactChain
  | bundleForwarding
  | intermediateCarryValidation
  | segmentContinuation
  | memoryCarryValidation
  | stateOutput
  | accumulator
  | counterAndControl
deriving DecidableEq, Fintype, Repr

def ownerOrder : List Owner :=
  [.prelude, .priorStateBinding, .fullClaimLink, .nifsVerifier,
    .memoryClaimValidation, .memoryTransition, .compactChain, .bundleForwarding,
    .intermediateCarryValidation, .accumulator, .segmentContinuation,
    .memoryCarryValidation, .stateOutput, .counterAndControl]

theorem ownerOrder_nodup : ownerOrder.Nodup := by decide

theorem Owner.mem_order (owner : Owner) : owner ∈ ownerOrder := by
  cases owner <;> simp [ownerOrder]

/-- Rows not yet owned by an independent V2 local gadget. Each required
semantic family is nonempty in a generated recursive artifact. -/
structure OpaqueRows where
  prelude : List Row
  priorStateBinding : List Row
  nifsVerifier : List Row
  accumulator : List Row
  counterAndControl : List Row
  priorStateBindingNonempty : priorStateBinding ≠ []
  nifsVerifierNonempty : nifsVerifier ≠ []
  accumulatorNonempty : accumulator ≠ []
  counterAndControlNonempty : counterAndControl ≠ []

/-- Exact local layouts that the generated artifact must select. -/
structure Layouts (widths : CompilerWidths) where
  fullClaim : FullClaimEnvelopeRows.Layout widths
  priorMemoryCarry : MemoryCarryPublicRows.Layout
  priorStateLink : PriorStateLinkRows.Layout widths
  priorPublicStateDigestColumn : Fin 4 → Nat
  priorStateBoundary : StateAuthorityBoundaryRows.Layout
  runningClaim : ProductNifsRunningFieldRows.Layout
  memoryClaim : MemoryClaimRows.Layout
  memorySource : MemorySourceRows.Layout
  memoryTransition : MemoryTransitionRows.Layout
  memoryBalance : MemoryProductBalanceRows.Layout
  compactHeaders : CompactChainHeaderRows.Layout
  compactChain : CompactCheckedStepChainRows.Layout
  bundleForwarding : BundleForwardingRows.Layout
  intermediateMemoryCarry : MemoryCarryPublicRows.Layout
  segmentContinuation : MemorySegmentContinuationRows.Layout
  memoryCarry : MemoryCarryPublicRows.Layout
  stateOutput : AuthoritativeStateOutputRows.Layout

structure Layouts.Valid {widths : CompilerWidths}
    (layouts : Layouts widths) : Prop where
  stateOutputValid : layouts.stateOutput.Valid
  priorStateLinkValid : layouts.priorStateLink.Valid
  priorStateLinkUsesFullClaim :
    layouts.priorStateLink.fullClaim = layouts.fullClaim
  priorStateLinkUsesPriorCarryBits :
    layouts.priorStateLink.stateOutput.hash.carry.frame.packing.publicBits =
      layouts.priorMemoryCarry.publicBits
  priorStateLinkUsesMemoryClaim :
    layouts.priorStateLink.memoryDigest.frame.claim = layouts.memoryClaim
  priorBoundaryUsesPublicState :
    layouts.priorStateBoundary.outgoingColumn =
      layouts.priorPublicStateDigestColumn
  priorBoundaryUsesRecomputedState :
    layouts.priorStateBoundary.incomingColumn = fun lane =>
      layouts.priorStateLink.stateOutput.hash.stateOutput.trace.outputColumns.getD
        lane.val 0
  runningClaimFromFullClaim :
    layouts.runningClaim.publicBitStart =
      layouts.fullClaim.claimBitStart +
        Section.recursiveState.bitOffset widths
  memoryClaimFromFullClaim :
    layouts.memoryClaim.publicBitStart =
      layouts.fullClaim.claimBitStart + Section.memory.bitOffset widths
  memorySourceUsesMemoryClaim :
    layouts.memorySource.product.claim = layouts.memoryClaim
  transitionUsesPriorCarry :
    layouts.memoryTransition.before = layouts.priorMemoryCarry
  transitionUsesMemoryClaim :
    layouts.memoryTransition.claim = layouts.memoryClaim
  transitionUsesIntermediateCarry :
    layouts.memoryTransition.after = layouts.intermediateMemoryCarry
  memoryBalanceUsesMemoryClaim :
    layouts.memoryBalance.claim = layouts.memoryClaim
  memoryBalanceUsesIntermediatePhase :
    layouts.memoryBalance.closePhaseColumn =
      layouts.intermediateMemoryCarry.carry.fieldColumn .phase
  segmentContinuationValid : layouts.segmentContinuation.Valid
  segmentContinuationUsesIntermediateCarry :
    layouts.segmentContinuation.intermediate = layouts.intermediateMemoryCarry
  segmentContinuationUsesOutgoingCarry :
    layouts.segmentContinuation.outgoing = layouts.memoryCarry
  compactChainUsesMemoryClaim :
    layouts.compactChain.memoryClaim = layouts.memoryClaim
  compactOperationsHeaderUsesPriorCarry :
    layouts.compactHeaders.operations.digestColumn =
      layouts.priorMemoryCarry.carry.headerColumn .operations
  compactMemoryHeaderUsesPriorInitial :
    layouts.compactHeaders.memory.digestColumn =
      layouts.priorMemoryCarry.carry.headerColumn .initialSnapshot
  compactMemoryHeaderUsesPriorFinal :
    layouts.compactHeaders.memory.digestColumn =
      layouts.priorMemoryCarry.carry.headerColumn .finalSnapshot
  compactOperationsHeaderUsesOutgoingCarry :
    layouts.compactHeaders.operations.digestColumn =
      layouts.memoryCarry.carry.headerColumn .operations
  compactMemoryHeaderUsesOutgoingInitial :
    layouts.compactHeaders.memory.digestColumn =
      layouts.memoryCarry.carry.headerColumn .initialSnapshot
  compactMemoryHeaderUsesOutgoingFinal :
    layouts.compactHeaders.memory.digestColumn =
      layouts.memoryCarry.carry.headerColumn .finalSnapshot
  compactChainBundleFromFullClaim :
    layouts.compactChain.bundleFields.publicBitStart =
      layouts.fullClaim.claimBitStart +
        Section.commitmentBundle.bitOffset widths
  bundleInputFromFullClaim :
    layouts.bundleForwarding.inputStart =
      layouts.fullClaim.claimBitStart +
        Section.commitmentBundle.bitOffset widths
  carryBitsSharedWithStateOutput :
    layouts.memoryCarry.publicBits =
      layouts.stateOutput.hash.carry.frame.packing.publicBits

/-- One schema-level recursive artifact. `targetColumnsCovered` is a generated
column-conservation certificate. Reads can cross owner windows; row targets
cannot. -/
structure Artifact (widths : CompilerWidths) where
  profile : Nightstream.Protocol.NebulaV2.Profile.Identity
  profileExact : profile = Nightstream.Protocol.NebulaV2.Profile.v2
  rowVariableCount : Nat
  /-- The V2 reference's known rows rule out every cube below 25 variables.
  Generation may select a larger exponent after all omitted row families are
  present; this schema must not freeze a planning value as a key identity. -/
  rowVariableCountMinimum : 25 ≤ rowVariableCount
  verifierKeyDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  relationManifestDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  seedManifest : SeedSchedule.Manifest
  seedManifestProfile : seedManifest.profile = profile
  other : OpaqueRows
  layouts : Layouts widths
  layoutsValid : layouts.Valid
  compactHeadersValid : layouts.compactHeaders.Valid seedManifest
  compactChainValid : layouts.compactChain.Valid seedManifest
  columnWidths : List Nat
  columnWidthCount : columnWidths.length = ownerOrder.length
  columnWidthsPositive : ∀ width ∈ columnWidths, 0 < width
  targetColumnsCovered :
    ∀ row ∈
        (ownerOrder.flatMap fun owner =>
          match owner with
          | .prelude => other.prelude
          | .priorStateBinding =>
              other.priorStateBinding ++
                MemoryCarryPublicRows.rows layouts.priorMemoryCarry ++
                PriorStateLinkRows.rows layouts.priorStateLink ++
                StateAuthorityBoundaryRows.rows layouts.priorStateBoundary
          | .fullClaimLink => FullClaimEnvelopeRows.rows layouts.fullClaim
          | .nifsVerifier =>
              ProductNifsRunningFieldRows.rows layouts.runningClaim ++
                other.nifsVerifier
          | .memoryClaimValidation => MemoryClaimRows.rows layouts.memoryClaim
          | .memoryTransition =>
              MemorySourceRows.checkedRows layouts.memorySource ++
                MemoryTransitionRows.rows layouts.memoryTransition ++
                MemoryProductBalanceRows.rows layouts.memoryBalance
          | .compactChain =>
              CompactChainHeaderRows.rows seedManifest layouts.compactHeaders ++
                CompactCheckedStepChainRows.rows seedManifest
                  layouts.compactChain
          | .bundleForwarding =>
              BundleForwardingRows.rows layouts.bundleForwarding
          | .intermediateCarryValidation =>
              MemoryCarryPublicRows.rows layouts.intermediateMemoryCarry
          | .segmentContinuation =>
              MemorySegmentContinuationRows.rows layouts.segmentContinuation
          | .memoryCarryValidation =>
              MemoryCarryPublicRows.rows layouts.memoryCarry
          | .stateOutput => AuthoritativeStateOutputRows.rows layouts.stateOutput
          | .accumulator => other.accumulator
          | .counterAndControl => other.counterAndControl),
      ∀ column, Mentions row.c column →
        column = 0 ∨
          ∃ window ∈ windowsOf 0 columnWidths, window.owns column

/-- The setup-selected NIFS verifier and the recursive manifest have one
profile, verifier key, and relation identity. The digests are identifiers; a
release proof must separately recompute them from authoritative manifests. -/
structure Artifact.MatchesSelected
    {widths : CompilerWidths} (artifact : Artifact widths)
    (selected : FullClaimNifsReceipt.SelectedVerifier widths) : Prop where
  profile : artifact.profile = selected.profile
  verifierKeyDigest :
    artifact.verifierKeyDigest = selected.verifierKeyDigest
  relationManifestDigest :
    artifact.relationManifestDigest = selected.relationManifestDigest

def Artifact.partRows {widths : CompilerWidths}
    (artifact : Artifact widths) : Owner → List Row
  | .prelude => artifact.other.prelude
  | .priorStateBinding =>
      artifact.other.priorStateBinding ++
        MemoryCarryPublicRows.rows artifact.layouts.priorMemoryCarry ++
        PriorStateLinkRows.rows artifact.layouts.priorStateLink ++
        StateAuthorityBoundaryRows.rows artifact.layouts.priorStateBoundary
  | .fullClaimLink => FullClaimEnvelopeRows.rows artifact.layouts.fullClaim
  | .nifsVerifier =>
      ProductNifsRunningFieldRows.rows artifact.layouts.runningClaim ++
        artifact.other.nifsVerifier
  | .memoryClaimValidation => MemoryClaimRows.rows artifact.layouts.memoryClaim
  | .memoryTransition =>
      MemorySourceRows.checkedRows artifact.layouts.memorySource ++
        MemoryTransitionRows.rows artifact.layouts.memoryTransition ++
        MemoryProductBalanceRows.rows artifact.layouts.memoryBalance
  | .compactChain =>
      CompactChainHeaderRows.rows artifact.seedManifest
          artifact.layouts.compactHeaders ++
        CompactCheckedStepChainRows.rows artifact.seedManifest
          artifact.layouts.compactChain
  | .bundleForwarding =>
      BundleForwardingRows.rows artifact.layouts.bundleForwarding
  | .intermediateCarryValidation =>
      MemoryCarryPublicRows.rows artifact.layouts.intermediateMemoryCarry
  | .segmentContinuation =>
      MemorySegmentContinuationRows.rows artifact.layouts.segmentContinuation
  | .memoryCarryValidation =>
      MemoryCarryPublicRows.rows artifact.layouts.memoryCarry
  | .stateOutput => AuthoritativeStateOutputRows.rows artifact.layouts.stateOutput
  | .accumulator => artifact.other.accumulator
  | .counterAndControl => artifact.other.counterAndControl

def Artifact.programRows {widths : CompilerWidths}
    (artifact : Artifact widths) : List Row :=
  ownerOrder.flatMap artifact.partRows

/-- The complete generated recursive relation fits the verifier-key-selected
SuperNeo row cube. Known-family lower bounds do not replace this release
certificate. -/
def Artifact.FitsGeneratedDomain {widths : CompilerWidths}
    (artifact : Artifact widths) : Prop :=
  artifact.programRows.length ≤ 2 ^ artifact.rowVariableCount

def Artifact.columnWindows {widths : CompilerWidths}
    (artifact : Artifact widths) : List Window :=
  windowsOf 0 artifact.columnWidths

theorem Artifact.columnWindows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    artifact.columnWindows.length = ownerOrder.length := by
  rw [Artifact.columnWindows, windowsOf_length, artifact.columnWidthCount]

theorem Artifact.owner_rows_in_program {widths : CompilerWidths}
    (artifact : Artifact widths) (owner : Owner) :
    ∀ row ∈ artifact.partRows owner, row ∈ artifact.programRows := by
  intro row member
  exact List.mem_flatMap.mpr ⟨owner, owner.mem_order, member⟩

theorem Artifact.owner_rows_included {widths : CompilerWidths}
    (artifact : Artifact widths) (owner : Owner) :
    rowsIncluded (artifact.partRows owner) artifact.programRows = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true (artifact.owner_rows_in_program owner row member)

theorem Artifact.owner_satisfied {widths : CompilerWidths}
    {artifact : Artifact widths} {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) (owner : Owner) :
    Satisfies (artifact.partRows owner) assignment := by
  intro row member
  exact satisfies row
    (rowsIncluded_sound (artifact.owner_rows_included owner) row member)

theorem Artifact.fullClaim_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .fullClaimLink).length = widths.totalBits :=
  FullClaimEnvelopeRows.rows_length artifact.layouts.fullClaim

theorem Artifact.memoryClaim_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .memoryClaimValidation).length = 10244 :=
  MemoryClaimRows.rows_length_exact artifact.layouts.memoryClaim

theorem Artifact.priorMemoryCarry_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (MemoryCarryPublicRows.rows artifact.layouts.priorMemoryCarry).length =
      7094 :=
  MemoryCarryPublicRows.rows_length_exact artifact.layouts.priorMemoryCarry

theorem Artifact.priorStateLink_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (PriorStateLinkRows.rows artifact.layouts.priorStateLink).length =
      40090 :=
  PriorStateLinkRows.rows_length_exact
    artifact.layoutsValid.priorStateLinkValid

theorem Artifact.priorStateBoundary_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (StateAuthorityBoundaryRows.rows
      artifact.layouts.priorStateBoundary).length = 4 :=
  StateAuthorityBoundaryRows.rows_length_exact
    artifact.layouts.priorStateBoundary

theorem Artifact.bundleForwarding_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .bundleForwarding).length =
      Nightstream.Protocol.NebulaV2.MemoryWireGeometry.mandatoryBundleBits :=
  BundleForwardingRows.rows_length artifact.layouts.bundleForwarding

theorem Artifact.memoryCarry_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .memoryCarryValidation).length = 7094 :=
  MemoryCarryPublicRows.rows_length_exact artifact.layouts.memoryCarry

theorem Artifact.intermediateMemoryCarry_rows_length
    {widths : CompilerWidths} (artifact : Artifact widths) :
    (artifact.partRows .intermediateCarryValidation).length = 7094 :=
  MemoryCarryPublicRows.rows_length_exact
    artifact.layouts.intermediateMemoryCarry

theorem Artifact.segmentContinuation_rows_length
    {widths : CompilerWidths} (artifact : Artifact widths) :
    (artifact.partRows .segmentContinuation).length = 38065 :=
  MemorySegmentContinuationRows.rows_length_exact
    artifact.layoutsValid.segmentContinuationValid

theorem Artifact.stateOutput_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .stateOutput).length = 24497 :=
  AuthoritativeStateOutputRows.rows_length_exact
    artifact.layoutsValid.stateOutputValid

theorem Artifact.nifs_rows_nonempty {widths : CompilerWidths}
    (artifact : Artifact widths) :
    artifact.partRows .nifsVerifier ≠ [] :=
  by
    intro empty
    have lengths := congrArg List.length empty
    simp only [Artifact.partRows, List.length_append, List.length_nil,
      ProductNifsRunningFieldRows.rows_length_exact] at lengths
    omega

theorem Artifact.runningClaim_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (ProductNifsRunningFieldRows.rows artifact.layouts.runningClaim).length =
      11066930 :=
  ProductNifsRunningFieldRows.rows_length_exact artifact.layouts.runningClaim

/-- The complete strict bit-to-field bridge is a mandatory prefix of the
paper-NIFS verifier owner. The remaining opaque rows own the paper verifier
arithmetic only. -/
theorem Artifact.runningClaim_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (ProductNifsRunningFieldRows.rows artifact.layouts.runningClaim)
      assignment := by
  have nifs := artifact.owner_satisfied satisfies .nifsVerifier
  intro row member
  exact nifs row (List.mem_append_left _ member)

theorem Artifact.memoryTransition_rows_nonempty {widths : CompilerWidths}
    (artifact : Artifact widths) :
    artifact.partRows .memoryTransition ≠ [] :=
  by
    change MemorySourceRows.checkedRows artifact.layouts.memorySource ++
      MemoryTransitionRows.rows artifact.layouts.memoryTransition ++
        MemoryProductBalanceRows.rows artifact.layouts.memoryBalance ≠ []
    intro empty
    have lengths := congrArg List.length empty
    simp only [List.length_append, List.length_nil,
      MemorySourceRows.checkedRows_length_exact] at lengths
    omega

theorem Artifact.memoryTransition_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .memoryTransition).length = 26977 := by
  simp [Artifact.partRows, MemorySourceRows.checkedRows_length_exact,
    MemoryTransitionRows.rows_length_exact,
    MemoryProductBalanceRows.rows_length_exact]

theorem Artifact.memorySource_rows_length
    {widths : CompilerWidths} (artifact : Artifact widths) :
    (MemorySourceRows.rows artifact.layouts.memorySource).length = 22664 :=
  MemorySourceRows.rows_length_exact artifact.layouts.memorySource

theorem Artifact.memoryCheckedStep_rows_length
    {widths : CompilerWidths} (artifact : Artifact widths) :
    (MemorySourceRows.checkedRows artifact.layouts.memorySource).length =
      26736 :=
  MemorySourceRows.checkedRows_length_exact artifact.layouts.memorySource

theorem Artifact.memoryProductUpdate_rows_length
    {widths : CompilerWidths} (artifact : Artifact widths) :
    (MemoryProductUpdateRows.rows
      artifact.layouts.memorySource.product).length = 4072 :=
  MemoryProductUpdateRows.rows_length_exact
    artifact.layouts.memorySource.product

theorem Artifact.exactMemoryTransition_rows_length
    {widths : CompilerWidths} (artifact : Artifact widths) :
    (MemoryTransitionRows.rows artifact.layouts.memoryTransition).length =
      225 :=
  MemoryTransitionRows.rows_length_exact artifact.layouts.memoryTransition

theorem Artifact.memoryBalance_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (MemoryProductBalanceRows.rows artifact.layouts.memoryBalance).length = 16 :=
  MemoryProductBalanceRows.rows_length_exact artifact.layouts.memoryBalance

theorem Artifact.compactChain_rows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    (artifact.partRows .compactChain).length = 962271 := by
  simp [Artifact.partRows,
    CompactChainHeaderRows.rows_length_exact artifact.compactHeadersValid,
    CompactCheckedStepChainRows.rows_length_exact artifact.compactChainValid]

theorem Artifact.compactHeaders_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (CompactChainHeaderRows.rows artifact.seedManifest
        artifact.layouts.compactHeaders) assignment := by
  have compact := artifact.owner_satisfied satisfies .compactChain
  intro row member
  exact compact row (List.mem_append_left _ member)

/-- The shared bundle decoder and all three compact-chain updates are a
mandatory recursive row family. -/
theorem Artifact.compactChain_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (CompactCheckedStepChainRows.rows artifact.seedManifest
        artifact.layouts.compactChain) assignment := by
  have compact := artifact.owner_satisfied satisfies .compactChain
  intro row member
  exact compact row (List.mem_append_right _ member)

/-- The 16 concrete product-balance rows are a mandatory suffix of the memory
transition family and are therefore satisfied by every artifact assignment. -/
theorem Artifact.memoryBalance_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryProductBalanceRows.rows artifact.layouts.memoryBalance)
      assignment := by
  have transition := artifact.owner_satisfied satisfies .memoryTransition
  intro row member
  exact transition row (by
    simp [Artifact.partRows, member])

/-- The exact 225-row transition program is a mandatory middle block of the
memory-transition owner. -/
theorem Artifact.exactMemoryTransition_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryTransitionRows.rows artifact.layouts.memoryTransition)
      assignment := by
  have transition := artifact.owner_satisfied satisfies .memoryTransition
  intro row member
  exact transition row (by
    simp [Artifact.partRows, member])

/-- The complete 26,736-row source and product program is a mandatory prefix
of the memory-transition owner. -/
theorem Artifact.memoryCheckedStep_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemorySourceRows.checkedRows artifact.layouts.memorySource)
      assignment := by
  have transition := artifact.owner_satisfied satisfies .memoryTransition
  intro row member
  exact transition row (by
    simp [Artifact.partRows, member])

/-- The challenge-independent 22,664 source rows are a mandatory subblock. -/
theorem Artifact.memorySource_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (MemorySourceRows.rows artifact.layouts.memorySource)
      assignment := by
  have checked := artifact.memoryCheckedStep_satisfied satisfies
  intro row member
  exact checked row (List.mem_append_left _ member)

/-- The fixed eight-chain product program is also a mandatory subblock. -/
theorem Artifact.memoryProductUpdate_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryProductUpdateRows.rows artifact.layouts.memorySource.product)
      assignment := by
  have checked := artifact.memoryCheckedStep_satisfied satisfies
  intro row member
  exact checked row (List.mem_append_right _ member)

/-- Every satisfying recursive artifact derives a concrete, row-sourced
checked-step record block and the exact update of all eight products. No
premise supplies a source-refinement relation or a product endpoint equality. -/
theorem Artifact.memoryCheckedStep_product_update
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat} {claim : MemoryClaimCodec.Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (parsed : MemoryClaimRows.ParsedColumnsMatch
      artifact.layouts.memoryClaim assignment claim)
    (satisfies : Satisfies artifact.programRows assignment) :
    ∃ derived : MemorySourceRows.Sound artifact.layouts.memorySource
        assignment claim,
      MemoryClaimProductUpdate.mapState claim.productsAfter =
        Nightstream.Protocol.NebulaV2.ProductState.update
          Nightstream.Implementation.NebulaV2.ConcreteField.encode
          (MemoryClaimProductUpdate.mapChallenges claim.challenge)
          (MemoryClaimProductUpdate.mapState claim.productsBefore)
          derived.records.chunk := by
  have sourceParsed : MemoryClaimRows.ParsedColumnsMatch
      artifact.layouts.memorySource.product.claim assignment claim := by
    rw [artifact.layoutsValid.memorySourceUsesMemoryClaim]
    exact parsed
  have checked := artifact.memoryCheckedStep_satisfied satisfies
  have sourceRows := artifact.memorySource_satisfied satisfies
  let derived := MemorySourceRows.sound canonical one sourceParsed sourceRows
  refine ⟨derived, ?_⟩
  exact MemorySourceRows.product_update canonical one sourceParsed checked derived

/-- The incoming carry validator is a mandatory suffix of the prior-state
binding family. Thus every satisfying artifact assignment gives a canonical
typed row interpretation for both sides of the memory transition. This
theorem does not link the incoming bits to a prior-state digest. -/
theorem Artifact.priorMemoryCarry_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryCarryPublicRows.rows artifact.layouts.priorMemoryCarry)
      assignment := by
  have prior := artifact.owner_satisfied satisfies .priorStateBinding
  intro row member
  exact prior row (by
    simp [Artifact.partRows, member])

/-- The intermediate carry is validated before the nonterminal continuation
selects either exact copy or exact segment reopening. -/
theorem Artifact.intermediateMemoryCarry_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryCarryPublicRows.rows artifact.layouts.intermediateMemoryCarry)
      assignment := by
  exact artifact.owner_satisfied satisfies .intermediateCarryValidation

/-- The complete 38,065-row nonterminal continuation block is mandatory. -/
theorem Artifact.segmentContinuation_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemorySegmentContinuationRows.rows
        artifact.layouts.segmentContinuation) assignment := by
  exact artifact.owner_satisfied satisfies .segmentContinuation

/-- The outgoing active carry validator is a mandatory owner. -/
theorem Artifact.memoryCarry_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (MemoryCarryPublicRows.rows artifact.layouts.memoryCarry)
      assignment := by
  exact artifact.owner_satisfied satisfies .memoryCarryValidation

/-- The outgoing typed state and its two-stage Poseidon2 hash are a mandatory
owner in every recursive artifact. -/
theorem Artifact.stateOutput_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (AuthoritativeStateOutputRows.rows artifact.layouts.stateOutput)
      assignment := by
  exact artifact.owner_satisfied satisfies .stateOutput

/-- The exact Poseidon2 prior-state and memory links and selected 540-coordinate
carrier are mandatory suffix rows of the prior-state owner. -/
theorem Artifact.priorStateLink_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (PriorStateLinkRows.rows artifact.layouts.priorStateLink)
      assignment := by
  have prior := artifact.owner_satisfied satisfies .priorStateBinding
  intro row member
  exact prior row (by
    simp [Artifact.partRows, member])

/-- Four generated wrapper rows equate the prior public-state digest with the
typed digest recomputed from the exact prior carry and state payload. -/
theorem Artifact.priorStateBoundary_satisfied
    {widths : CompilerWidths} {artifact : Artifact widths}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (StateAuthorityBoundaryRows.rows artifact.layouts.priorStateBoundary)
      assignment := by
  have prior := artifact.owner_satisfied satisfies .priorStateBinding
  intro row member
  exact prior row (by
    simp [Artifact.partRows, member])

/-- One half-open range in the exact program-row order. -/
structure RowRange where
  owner : Owner
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

def rangesFrom {widths : CompilerWidths} (artifact : Artifact widths) :
    Nat → List Owner → List RowRange
  | _, [] => []
  | cursor, owner :: rest =>
      { owner := owner
        start := cursor
        stop := cursor + (artifact.partRows owner).length } ::
      rangesFrom artifact
        (cursor + (artifact.partRows owner).length) rest

def Covers : Nat → Nat → List RowRange → Prop
  | cursor, finish, [] => cursor = finish
  | cursor, finish, range :: rest =>
      range.start = cursor ∧ range.start ≤ range.stop ∧
        Covers range.stop finish rest

private theorem rangesFrom_cover {widths : CompilerWidths}
    (artifact : Artifact widths) (cursor : Nat) :
    ∀ owners,
      Covers cursor
        (cursor + (owners.map fun owner =>
          (artifact.partRows owner).length).sum)
        (rangesFrom artifact cursor owners) := by
  intro owners
  induction owners generalizing cursor with
  | nil => simp [rangesFrom, Covers]
  | cons owner rest inductionHypothesis =>
      simp only [rangesFrom, Covers, List.map_cons, List.sum_cons]
      refine ⟨trivial, by omega, ?_⟩
      have tail := inductionHypothesis
        (cursor + (artifact.partRows owner).length)
      convert tail using 1 <;> omega

def Artifact.rowRanges {widths : CompilerWidths}
    (artifact : Artifact widths) : List RowRange :=
  rangesFrom artifact 0 ownerOrder

theorem Artifact.programRows_length {widths : CompilerWidths}
    (artifact : Artifact widths) :
    artifact.programRows.length =
      (ownerOrder.map fun owner => (artifact.partRows owner).length).sum := by
  simp [Artifact.programRows, List.length_flatMap]

private theorem knownOwnerLengthSum_lower_bound
    (lengthOf : Owner → Nat)
    (prior : 47188 ≤ lengthOf .priorStateBinding)
    (fullClaim : lengthOf .fullClaimLink = 5587724)
    (nifs : 11066930 ≤ lengthOf .nifsVerifier)
    (memoryClaim : lengthOf .memoryClaimValidation = 10244)
    (memoryTransition : lengthOf .memoryTransition = 26977)
    (compactChain : lengthOf .compactChain = 962271)
    (bundle : lengthOf .bundleForwarding = 248832)
    (intermediate : lengthOf .intermediateCarryValidation = 7094)
    (continuation : lengthOf .segmentContinuation = 38065)
    (carry : lengthOf .memoryCarryValidation = 7094)
    (stateOutput : lengthOf .stateOutput = 24497) :
    18026916 ≤ (ownerOrder.map lengthOf).sum := by
  simp only [ownerOrder, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil]
  omega

/-- The exact V2 recursive schema already requires more than the complete
24-variable row cube before any opaque paper-NIFS, accumulator, or control
rows are counted. -/
theorem Artifact.knownRows_lower_bound
    (artifact : Artifact ProductFullClaimDecoder.widths) :
    18026916 ≤ artifact.programRows.length := by
  have priorBinding :
      47188 ≤ (artifact.partRows .priorStateBinding).length := by
    change 47188 ≤
      (artifact.other.priorStateBinding ++
        MemoryCarryPublicRows.rows artifact.layouts.priorMemoryCarry ++
        PriorStateLinkRows.rows artifact.layouts.priorStateLink ++
        StateAuthorityBoundaryRows.rows
          artifact.layouts.priorStateBoundary).length
    rw [List.length_append, List.length_append, List.length_append,
      artifact.priorMemoryCarry_rows_length,
      artifact.priorStateLink_rows_length,
      artifact.priorStateBoundary_rows_length]
    omega
  have nifs : 11066930 ≤ (artifact.partRows .nifsVerifier).length := by
    change 11066930 ≤
      (ProductNifsRunningFieldRows.rows artifact.layouts.runningClaim ++
        artifact.other.nifsVerifier).length
    rw [List.length_append, artifact.runningClaim_rows_length]
    omega
  have fullClaim := artifact.fullClaim_rows_length
  rw [ProductFullClaimDecoder.widths_totalBits] at fullClaim
  have memoryClaim := artifact.memoryClaim_rows_length
  have memoryTransition := artifact.memoryTransition_rows_length
  have compactChain := artifact.compactChain_rows_length
  have bundle := artifact.bundleForwarding_rows_length
  rw [Nightstream.Protocol.NebulaV2.MemoryWireGeometry.mandatoryBundleBits_exact]
    at bundle
  have intermediate := artifact.intermediateMemoryCarry_rows_length
  have continuation := artifact.segmentContinuation_rows_length
  have carry := artifact.memoryCarry_rows_length
  have stateOutput := artifact.stateOutput_rows_length
  rw [artifact.programRows_length]
  exact knownOwnerLengthSum_lower_bound
    (fun owner => (artifact.partRows owner).length)
    priorBinding fullClaim nifs memoryClaim memoryTransition compactChain bundle
    intermediate continuation carry stateOutput

theorem knownRows_exceed_24_variable_cube :
    2 ^ 24 < 18026916 := by decide

theorem knownRows_fit_inside_25_variable_cube :
    18026916 < 2 ^ 25 := by decide

/-- A fitting generated artifact cannot use more rows above the proved
mandatory minimum than its independently selected row cube permits. -/
theorem Artifact.rows_above_known_minimum_bound
    (artifact : Artifact ProductFullClaimDecoder.widths)
    (fits : artifact.FitsGeneratedDomain) :
    artifact.programRows.length - 18026916 ≤
      2 ^ artifact.rowVariableCount - 18026916 := by
  unfold Artifact.FitsGeneratedDomain at fits
  omega

/-- The canonical owner ranges cover every program row exactly once, with no
gap or overlap between consecutive families. -/
theorem Artifact.rowRanges_exact_cover {widths : CompilerWidths}
    (artifact : Artifact widths) :
    Covers 0 artifact.programRows.length artifact.rowRanges := by
  rw [artifact.programRows_length]
  simpa [Artifact.rowRanges] using rangesFrom_cover artifact 0 ownerOrder

/-- Every nonconstant target has one ownership window, and no second distinct
window can own the same target. -/
theorem Artifact.target_has_unique_window {widths : CompilerWidths}
    (artifact : Artifact widths) {row : Row} {column : Nat}
    (rowMember : row ∈ artifact.programRows)
    (target : Mentions row.c column) (nonzero : column ≠ 0) :
    ∃ window ∈ artifact.columnWindows,
      window.owns column ∧
        ∀ other ∈ artifact.columnWindows,
          other.owns column → other = window := by
  have covered := artifact.targetColumnsCovered row (by
    simpa [Artifact.programRows, Artifact.partRows] using rowMember)
    column target
  rcases covered with zero | ⟨window, member, owns⟩
  · exact False.elim (nonzero zero)
  · refine ⟨window, member, owns, ?_⟩
    intro other otherMember otherOwns
    exact windowsOf_no_collision 0 artifact.columnWidths
      other otherMember window member column otherOwns owns

/-- The full-claim link call site is derived from the manifest. A caller does
not supply or assume its row-inclusion certificate. -/
def Artifact.fullClaimCallSite
    {widths : CompilerWidths} (artifact : Artifact widths)
    (assignment : Nat → Nat) (value : Value widths)
    (input : FixedBits.Word widths.totalBits)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FullClaimEnvelopeRows.Placed artifact.layouts.fullClaim
      assignment value input) :
    FullClaimEnvelopeRows.CallSite artifact.programRows assignment value input :=
  { layout := artifact.layouts.fullClaim
    rowsIncluded := artifact.owner_rows_included .fullClaimLink
    canonicalAssignment := canonical
    one := one
    placed := placed }

end Nightstream.Implementation.NebulaV2.RecursiveManifestSchema
