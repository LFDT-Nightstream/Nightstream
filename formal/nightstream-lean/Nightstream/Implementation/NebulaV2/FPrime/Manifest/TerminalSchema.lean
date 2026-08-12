import Nightstream.Implementation.NebulaV2.Commitment.Bundle.ForwardingRows
import Nightstream.Implementation.NebulaV2.Commitment.Bundle.FieldRows
import Nightstream.Implementation.NebulaV2.Commitment.Compact.CheckedStepChainRows
import Nightstream.Implementation.NebulaV2.Commitment.Compact.ChainHeaderRows
import Nightstream.Implementation.NebulaV2.FPrime.Claim.EnvelopeRows
import Nightstream.Implementation.NebulaV2.FPrime.Claim.NifsReceipt
import Nightstream.Implementation.NebulaV2.Memory.Carry.PublicRows
import Nightstream.Implementation.NebulaV2.Memory.Claim.Rows
import Nightstream.Implementation.NebulaV2.Memory.Product.BalanceRows
import Nightstream.Implementation.NebulaV2.Memory.Segment.SourceRows
import Nightstream.Implementation.NebulaV2.Memory.Transition.TransitionRows
import Nightstream.Implementation.NebulaV2.FPrime.State.PriorLinkRows
import Nightstream.Implementation.NebulaV2.NIFS.Running.RunningFieldRows
import Nightstream.Implementation.NebulaV2.FPrime.State.AuthorityBoundaryRows
import Nightstream.Implementation.NebulaV2.NIFS.Running.FullClaimDecoder
import Nightstream.Implementation.NebulaV2.Commitment.Terminal.BundleOpeningRows
import Nightstream.Implementation.NebulaV2.FPrime.Terminal.ClosedCarryRows
import Nightstream.Implementation.R1CS.Canonical.ColumnWindows
import Nightstream.Protocol.NebulaV2.Terminal

/-!
Contract: parametric generated-row manifest schema for the V2 terminal branch.

Assurance tier: implementation schema.

Owns the terminal-only row-family order, exact numeric row cover, exact
full-claim and memory blocks, exact transition to one validated intermediate
carry, the unconditional closed-phase row, all fourteen accumulator-output
bundle decoders, and the fourteen separately typed common-witness bundle
openings.

The terminal branch has no segment continuation, outgoing carry validator,
next-state hash, or next fresh claim. The NIFS verifier, accumulator fold,
terminal CCS relation, public-result check, and control lowering remain named
nonempty compiler boundaries. The typed opening rows are not silently counted
as numeric rows; a combined compiler must prove their shared-assignment
refinement.

Emits constraints: through `Artifact.programRows` and
`Artifact.typedOpeningRows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.TerminalManifestSchema

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.ColumnWindows
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Program
open Nightstream.Protocol.NebulaV2.Terminal
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- Canonical terminal-branch numeric row owners. -/
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
  | terminalClosed
  | accumulator
  | foldedBundleValidation
  | terminalRelation
  | publicResult
  | counterAndControl
deriving DecidableEq, Fintype, Repr

def ownerOrder : List Owner :=
  [.prelude, .priorStateBinding, .fullClaimLink, .nifsVerifier,
    .memoryClaimValidation, .memoryTransition, .compactChain,
    .bundleForwarding, .intermediateCarryValidation, .terminalClosed,
    .accumulator, .foldedBundleValidation, .terminalRelation, .publicResult,
    .counterAndControl]

theorem ownerOrder_nodup : ownerOrder.Nodup := by decide

theorem Owner.mem_order (owner : Owner) : owner ∈ ownerOrder := by
  cases owner <;> simp [ownerOrder]

/-- Numeric compiler families whose exact V2 row semantics are still release
obligations. Nonemptiness prevents a generated manifest from omitting one. -/
structure OpaqueRows where
  prelude : List Row
  priorStateBinding : List Row
  nifsVerifier : List Row
  accumulator : List Row
  terminalRelation : List Row
  publicResult : List Row
  counterAndControl : List Row
  preludeNonempty : prelude ≠ []
  priorStateBindingNonempty : priorStateBinding ≠ []
  nifsVerifierNonempty : nifsVerifier ≠ []
  accumulatorNonempty : accumulator ≠ []
  terminalRelationNonempty : terminalRelation ≠ []
  publicResultNonempty : publicResult ≠ []
  counterAndControlNonempty : counterAndControl ≠ []

/-- Exact local numeric and typed layouts selected by one terminal artifact. -/
structure Layouts
    (widths : CompilerWidths) (manifest : SeedSchedule.Manifest)
    (fullShape operationsShape snapshotShape : Shape) where
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
  terminalClosed : TerminalClosedCarryRows.Layout
  /-- Canonical field decoders for all final folded accumulator children.
  These bit windows are not the trailing fresh-claim bundle window. -/
  foldedBundleFields : FoldedChild → CommitmentBundleFieldRows.Layout
  /-- One complete four-component opening per post-PiDEC child. All openings
  live in one typed assignment selected by the combined terminal compiler. -/
  terminalOpening : FoldedChild →
    TerminalBundleOpeningRows.Layout manifest fullShape operationsShape
      snapshotShape

structure Layouts.Valid
    {widths : CompilerWidths} {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (layouts : Layouts widths manifest fullShape operationsShape
      snapshotShape) : Prop where
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
  compactChainBundleFromFullClaim :
    layouts.compactChain.bundleFields.publicBitStart =
      layouts.fullClaim.claimBitStart +
        Section.commitmentBundle.bitOffset widths
  bundleInputFromFullClaim :
    layouts.bundleForwarding.inputStart =
      layouts.fullClaim.claimBitStart +
        Section.commitmentBundle.bitOffset widths
  terminalClosedUsesIntermediateCarry :
    layouts.terminalClosed.carry = layouts.intermediateMemoryCarry
  terminalOpeningUsesFoldedBundle : ∀ child,
    (layouts.terminalOpening child).bundleFields =
      layouts.foldedBundleFields child

/-- All fourteen canonical numeric folded-bundle decoders, in child order. -/
def Layouts.foldedBundleRows
    {widths : CompilerWidths} {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (layouts : Layouts widths manifest fullShape operationsShape
      snapshotShape) : List Row :=
  (List.ofFn fun child : FoldedChild =>
    CommitmentBundleFieldRows.rows (layouts.foldedBundleFields child)).flatten

/-- All fourteen typed same-witness opening programs, in child order. -/
def Layouts.typedOpeningRows
    {widths : CompilerWidths} {manifest : SeedSchedule.Manifest}
    {fullShape operationsShape snapshotShape : Shape}
    (layouts : Layouts widths manifest fullShape operationsShape
      snapshotShape) :=
  (List.ofFn fun child : FoldedChild =>
    TerminalBundleOpeningRows.rows (layouts.terminalOpening child)).flatten

private theorem typedSatisfies_flatten_iff
    (pieces : List
      (List Nightstream.Implementation.Lowering.Goldilocks.OwnedRow))
    (assignment : Nightstream.Implementation.Lowering.Goldilocks.ColumnId →
      Nightstream.SuperNeo.Concrete.F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies pieces.flatten
        assignment ↔
      ∀ piece ∈ pieces,
        Nightstream.Implementation.Lowering.Goldilocks.Satisfies piece
          assignment := by
  induction pieces with
  | nil => simp
  | cons head tail inductionHypothesis =>
      rw [List.flatten_cons]
      rw [Nightstream.Implementation.Lowering.Goldilocks.satisfies_append_iff]
      rw [inductionHypothesis]
      simp only [List.mem_cons]
      constructor
      · rintro ⟨headHolds, tailHolds⟩ piece (rfl | member)
        · exact headHolds
        · exact tailHolds piece member
      · intro all
        exact ⟨all head (Or.inl rfl), fun piece member =>
          all piece (Or.inr member)⟩

/-- One terminal schema artifact. Numeric rows and typed opening rows stay
separate until the combined generated compiler supplies an exact refinement. -/
structure Artifact
    (widths : CompilerWidths)
    (fullShape operationsShape snapshotShape : Shape) where
  profile : Nightstream.Protocol.NebulaV2.Profile.Identity
  profileExact : profile = Nightstream.Protocol.NebulaV2.Profile.v2
  /-- Row variables of the SuperNeo relation that the finalizer opens. -/
  superNeoRowVariableCount : Nat
  /-- The opener uses the row exponent of the selected relation shape. It
  must not reuse the obsolete fixed-25 reference exponent. -/
  superNeoRowVariableCountExact :
    superNeoRowVariableCount = fullShape.rowVariables
  /-- Independently generated row variables of the finalizer circuit. This
  value is not implied by the SuperNeo relation shape. -/
  terminalRowVariableCount : Nat
  verifierKeyDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  relationManifestDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  terminalManifestDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  seedManifest : SeedSchedule.Manifest
  seedManifestProfile : seedManifest.profile = profile
  other : OpaqueRows
  layouts : Layouts widths seedManifest fullShape operationsShape snapshotShape
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
          | .terminalClosed => TerminalClosedCarryRows.rows layouts.terminalClosed
          | .accumulator => other.accumulator
          | .foldedBundleValidation =>
              layouts.foldedBundleRows
          | .terminalRelation => other.terminalRelation
          | .publicResult => other.publicResult
          | .counterAndControl => other.counterAndControl),
      ∀ column, Mentions row.c column →
        column = 0 ∨
          ∃ window ∈ windowsOf 0 columnWidths, window.owns column

structure Artifact.MatchesSelected
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
    (selected : FullClaimNifsReceipt.SelectedVerifier widths) : Prop where
  profile : artifact.profile = selected.profile
  verifierKeyDigest :
    artifact.verifierKeyDigest = selected.verifierKeyDigest
  relationManifestDigest :
    artifact.relationManifestDigest = selected.relationManifestDigest

/-- Authority-bearing terminal-key fields decoded from the verifier-owned
container. The digest values identify manifests; they do not replace manifest
recomputation or row validation. -/
structure TerminalKeyIdentity where
  profile : Nightstream.Protocol.NebulaV2.Profile.Identity
  verifierKeyDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  terminalManifestDigest : Nightstream.Protocol.NebulaV2.Digest.Value
  rowVariableCount : Nat

/-- The generated finalizer and the verifier-owned terminal key select the
same profile, aggregate key, terminal manifest, and independent row domain. -/
structure Artifact.MatchesTerminalKey
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
    (key : TerminalKeyIdentity) : Prop where
  profile : artifact.profile = key.profile
  verifierKeyDigest : artifact.verifierKeyDigest = key.verifierKeyDigest
  terminalManifestDigest :
    artifact.terminalManifestDigest = key.terminalManifestDigest
  rowVariableCount :
    artifact.terminalRowVariableCount = key.rowVariableCount

/-- The terminal opener uses the exact row exponent carried by the selected
SuperNeo relation shape. -/
theorem Artifact.opensSelectedRelationExponent
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    artifact.superNeoRowVariableCount = fullShape.rowVariables :=
  artifact.superNeoRowVariableCountExact

def Artifact.partRows
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    Owner → List Row
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
  | .terminalClosed => TerminalClosedCarryRows.rows artifact.layouts.terminalClosed
  | .accumulator => artifact.other.accumulator
  | .foldedBundleValidation =>
      artifact.layouts.foldedBundleRows
  | .terminalRelation => artifact.other.terminalRelation
  | .publicResult => artifact.other.publicResult
  | .counterAndControl => artifact.other.counterAndControl

def Artifact.programRows
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    List Row :=
  ownerOrder.flatMap artifact.partRows

theorem Artifact.programRows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
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
    (terminalClosed : lengthOf .terminalClosed = 1)
    (foldedBundle : lengthOf .foldedBundleValidation = 7239456) :
    25196717 ≤ (ownerOrder.map lengthOf).sum := by
  simp only [ownerOrder, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil]
  omega

def Artifact.typedOpeningRows
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :=
  artifact.layouts.typedOpeningRows

/-- The complete generated finalizer, including all typed same-witness
openings, fits its own padded row domain. A release artifact must provide this
certificate. The recursive SuperNeo row count does not provide it. -/
def Artifact.FitsGeneratedDomain
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    Prop :=
  artifact.programRows.length + artifact.typedOpeningRows.length ≤
    2 ^ artifact.terminalRowVariableCount

def Artifact.columnWindows
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    List Window :=
  windowsOf 0 artifact.columnWidths

theorem Artifact.owner_rows_in_program
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
    (owner : Owner) :
    ∀ row ∈ artifact.partRows owner, row ∈ artifact.programRows := by
  intro row member
  exact List.mem_flatMap.mpr ⟨owner, owner.mem_order, member⟩

theorem Artifact.owner_rows_included
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
    (owner : Owner) :
    rowsIncluded (artifact.partRows owner) artifact.programRows = true := by
  rw [rowsIncluded, List.all_eq_true]
  intro row member
  exact decide_eq_true (artifact.owner_rows_in_program owner row member)

theorem Artifact.owner_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) (owner : Owner) :
    Satisfies (artifact.partRows owner) assignment := by
  intro row member
  exact satisfies row
    (rowsIncluded_sound (artifact.owner_rows_included owner) row member)

theorem Artifact.fullClaim_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .fullClaimLink).length = widths.totalBits :=
  FullClaimEnvelopeRows.rows_length artifact.layouts.fullClaim

theorem Artifact.memoryClaim_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .memoryClaimValidation).length = 10244 :=
  MemoryClaimRows.rows_length_exact artifact.layouts.memoryClaim

theorem Artifact.runningClaim_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (ProductNifsRunningFieldRows.rows artifact.layouts.runningClaim).length =
      11066930 :=
  ProductNifsRunningFieldRows.rows_length_exact artifact.layouts.runningClaim

theorem Artifact.priorStateBoundary_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (StateAuthorityBoundaryRows.rows
      artifact.layouts.priorStateBoundary).length = 4 :=
  StateAuthorityBoundaryRows.rows_length_exact
    artifact.layouts.priorStateBoundary

/-- The complete strict bit-to-field bridge is a mandatory prefix of the
paper-NIFS verifier owner. -/
theorem Artifact.runningClaim_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (ProductNifsRunningFieldRows.rows artifact.layouts.runningClaim)
      assignment := by
  have nifs := artifact.owner_satisfied satisfies .nifsVerifier
  intro row member
  exact nifs row (List.mem_append_left _ member)

theorem Artifact.memoryTransition_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .memoryTransition).length = 26977 := by
  simp [Artifact.partRows, MemorySourceRows.checkedRows_length_exact,
    MemoryTransitionRows.rows_length_exact,
    MemoryProductBalanceRows.rows_length_exact]

theorem Artifact.compactChain_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .compactChain).length = 962271 := by
  simp [Artifact.partRows,
    CompactChainHeaderRows.rows_length_exact artifact.compactHeadersValid,
    CompactCheckedStepChainRows.rows_length_exact artifact.compactChainValid]

theorem Artifact.bundleForwarding_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .bundleForwarding).length =
      Nightstream.Protocol.NebulaV2.MemoryWireGeometry.mandatoryBundleBits :=
  BundleForwardingRows.rows_length artifact.layouts.bundleForwarding

theorem Artifact.intermediateCarry_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .intermediateCarryValidation).length = 7094 :=
  MemoryCarryPublicRows.rows_length_exact
    artifact.layouts.intermediateMemoryCarry

theorem Artifact.terminalClosed_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .terminalClosed).length = 1 :=
  TerminalClosedCarryRows.rows_length artifact.layouts.terminalClosed

theorem Artifact.foldedBundle_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    (artifact.partRows .foldedBundleValidation).length = 7239456 := by
  simp [Artifact.partRows, Layouts.foldedBundleRows, foldedChildCount,
    CommitmentBundleFieldRows.rows_length_exact]

theorem Artifact.typedOpening_rows_length
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape) :
    artifact.typedOpeningRows.length =
      14 * (2 * fullShape.carrierWidth + 3888) := by
  simp [Artifact.typedOpeningRows, Layouts.typedOpeningRows,
    foldedChildCount, TerminalBundleOpeningRows.rows_length]
  omega

/-- Numeric terminal rows known before the opaque terminal compiler families
already consume most of a 25-variable row cube. -/
theorem Artifact.knownNumericRows_lower_bound
    {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact ProductFullClaimDecoder.widths fullShape
      operationsShape snapshotShape) :
    25196717 ≤ artifact.programRows.length := by
  have priorBinding :
      47188 ≤ (artifact.partRows .priorStateBinding).length := by
    change 47188 ≤
      (artifact.other.priorStateBinding ++
        MemoryCarryPublicRows.rows artifact.layouts.priorMemoryCarry ++
        PriorStateLinkRows.rows artifact.layouts.priorStateLink ++
        StateAuthorityBoundaryRows.rows
          artifact.layouts.priorStateBoundary).length
    rw [List.length_append, List.length_append, List.length_append,
      MemoryCarryPublicRows.rows_length_exact,
      PriorStateLinkRows.rows_length_exact
        artifact.layoutsValid.priorStateLinkValid,
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
  have intermediate := artifact.intermediateCarry_rows_length
  have terminalClosed := artifact.terminalClosed_rows_length
  have foldedBundle := artifact.foldedBundle_rows_length
  rw [artifact.programRows_length]
  exact knownOwnerLengthSum_lower_bound
    (fun owner => (artifact.partRows owner).length)
    priorBinding fullClaim nifs memoryClaim memoryTransition compactChain bundle
    intermediate terminalClosed foldedBundle

/-- The full relation has at least the 540-coordinate public carrier, so the
fourteen typed same-witness opening programs add at least 69,552 rows. -/
theorem Artifact.typedOpeningRows_lower_bound
    {fullShape operationsShape snapshotShape : Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (artifact : Artifact ProductFullClaimDecoder.widths fullShape
      operationsShape snapshotShape) :
    69552 ≤ artifact.typedOpeningRows.length := by
  rw [artifact.typedOpening_rows_length]
  have publicWidth : fullShape.publicWidth = 540 := by
    simpa [MemoryBoundCcsPublic.coordinateCount] using contract.publicWidth
  have carrierLower : 540 ≤ fullShape.carrierWidth := by
    rw [← publicWidth]
    exact fullShape.publicFits
  omega

/-- Once the full carrier reaches 1,196,429 coordinates, the fourteen typed
same-witness openings alone exceed a 25-variable finalizer row cube. -/
theorem Artifact.typedOpeningRows_exceed_25_variable_cube
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
    (carrierLower : 1196429 ≤ fullShape.carrierWidth) :
    2 ^ 25 < artifact.typedOpeningRows.length := by
  rw [artifact.typedOpening_rows_length]
  have cube : 2 ^ 25 = 33554432 := by decide
  rw [cube]
  omega

/-- A 25-variable SuperNeo relation does not authorize a 25-variable
finalizer. Above the concrete carrier threshold, such a finalizer cannot hold
the mandatory opening rows, even before its numeric rows are added. -/
theorem Artifact.cannot_fit_generated_domain_at_25
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
    (carrierLower : 1196429 ≤ fullShape.carrierWidth)
    (terminalAt25 : artifact.terminalRowVariableCount = 25) :
    ¬ artifact.FitsGeneratedDomain := by
  intro fits
  have openingExceeds :=
    artifact.typedOpeningRows_exceed_25_variable_cube carrierLower
  unfold Artifact.FitsGeneratedDomain at fits
  rw [terminalAt25] at fits
  omega

/-- Any direct combined terminal compiler that emits both row programs has
this unconditional minimum before its opaque NIFS, accumulator, terminal,
public-result, and control families. -/
theorem Artifact.combinedKnownRows_lower_bound
    {fullShape operationsShape snapshotShape : Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (artifact : Artifact ProductFullClaimDecoder.widths fullShape
      operationsShape snapshotShape) :
    25266269 ≤
      artifact.programRows.length + artifact.typedOpeningRows.length := by
  have numeric := artifact.knownNumericRows_lower_bound
  have typed := artifact.typedOpeningRows_lower_bound contract
  omega

theorem combinedKnownRows_fit_inside_25_variable_cube :
    25266269 < 2 ^ 25 := by decide

theorem Artifact.memoryClaim_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (MemoryClaimRows.rows artifact.layouts.memoryClaim)
      assignment :=
  artifact.owner_satisfied satisfies .memoryClaimValidation

theorem Artifact.priorCarry_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (MemoryCarryPublicRows.rows artifact.layouts.priorMemoryCarry)
      assignment := by
  have prior := artifact.owner_satisfied satisfies .priorStateBinding
  intro row member
  exact prior row (by simp [Artifact.partRows, member])

theorem Artifact.priorStateLink_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (PriorStateLinkRows.rows artifact.layouts.priorStateLink)
      assignment := by
  have prior := artifact.owner_satisfied satisfies .priorStateBinding
  intro row member
  exact prior row (by simp [Artifact.partRows, member])

/-- Four generated wrapper rows link the prior public-state digest to the
typed state recomputed from the terminal trailing claim and prior carry. -/
theorem Artifact.priorStateBoundary_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (StateAuthorityBoundaryRows.rows artifact.layouts.priorStateBoundary)
      assignment := by
  have prior := artifact.owner_satisfied satisfies .priorStateBinding
  intro row member
  exact prior row (by simp [Artifact.partRows, member])

theorem Artifact.memoryCheckedStep_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (MemorySourceRows.checkedRows artifact.layouts.memorySource)
      assignment := by
  have transition := artifact.owner_satisfied satisfies .memoryTransition
  intro row member
  exact transition row (by simp [Artifact.partRows, member])

theorem Artifact.exactMemoryTransition_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (MemoryTransitionRows.rows artifact.layouts.memoryTransition)
      assignment := by
  have transition := artifact.owner_satisfied satisfies .memoryTransition
  intro row member
  exact transition row (by simp [Artifact.partRows, member])

theorem Artifact.memoryBalance_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (MemoryProductBalanceRows.rows artifact.layouts.memoryBalance)
      assignment := by
  have transition := artifact.owner_satisfied satisfies .memoryTransition
  intro row member
  exact transition row (by simp [Artifact.partRows, member])

theorem Artifact.intermediateCarry_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies
      (MemoryCarryPublicRows.rows artifact.layouts.intermediateMemoryCarry)
      assignment :=
  artifact.owner_satisfied satisfies .intermediateCarryValidation

theorem Artifact.terminalClosed_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    Satisfies (TerminalClosedCarryRows.rows artifact.layouts.terminalClosed)
      assignment :=
  artifact.owner_satisfied satisfies .terminalClosed

theorem Artifact.foldedBundle_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment : Nat → Nat}
    (satisfies : Satisfies artifact.programRows assignment) :
    ∀ child,
    Satisfies
      (CommitmentBundleFieldRows.rows
        (artifact.layouts.foldedBundleFields child)) assignment := by
  intro child row member
  have owner := artifact.owner_satisfied satisfies .foldedBundleValidation
  apply owner row
  change row ∈ artifact.layouts.foldedBundleRows
  apply List.mem_flatten.mpr
  exact ⟨CommitmentBundleFieldRows.rows
      (artifact.layouts.foldedBundleFields child), by simp, member⟩

/-- Satisfaction of the combined typed terminal opening program gives the
exact opening rows for every one of the fourteen folded children. -/
theorem Artifact.typedOpening_satisfied
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {assignment :
      Nightstream.Implementation.Lowering.Goldilocks.ColumnId →
        Nightstream.SuperNeo.Concrete.F}
    (satisfies :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        artifact.typedOpeningRows assignment) :
    ∀ child,
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (TerminalBundleOpeningRows.rows
          (artifact.layouts.terminalOpening child)) assignment := by
  intro child
  exact (typedSatisfies_flatten_iff
      (List.ofFn fun index : FoldedChild =>
        TerminalBundleOpeningRows.rows
          (artifact.layouts.terminalOpening index)) assignment).mp
    satisfies _ (by simp)

/-- Every final folded accumulator child has one bounded common-witness
opening when the combined compiler supplies one physical assignment and the
accumulator places all exact output bits in the declared child windows. The
four-component opening equation for each child is derived from both row
programs. -/
theorem Artifact.foldedBundlesCommonOpenings
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    {artifact : Artifact widths fullShape operationsShape snapshotShape}
    {numericAssignment : Nat → Nat}
    (typedAssignment :
      Nightstream.Implementation.Lowering.Goldilocks.ColumnId →
        Nightstream.SuperNeo.Concrete.F)
    (bundles : FoldedChild → CommitmentBundleCodec.Value)
    (numericCanonical : ∀ column, numericAssignment column < goldilocksP)
    (numericOne : numericAssignment 0 = 1)
    (bundleBits : ∀ child, CommitmentBundleFieldRows.BitsPlaced
      (artifact.layouts.foldedBundleFields child) numericAssignment
        (bundles child))
    (numericSatisfies : Satisfies artifact.programRows numericAssignment)
    (assignmentAgreement : TerminalBundleOpeningRows.Layout.NumericAgreement
      numericAssignment typedAssignment)
    (typedOne : ∀ child,
      typedAssignment (artifact.layouts.terminalOpening child).one = 1)
    (typedSatisfies :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        artifact.typedOpeningRows typedAssignment) :
    ∀ child,
      Nightstream.SuperNeo.Concrete.Phi81Relation.assignmentNormBounded 2
          ((artifact.layouts.terminalOpening child).fullAssignment
            typedAssignment) ∧
        TerminalBundleOpeningRows.exactBundle
            (artifact.layouts.terminalOpening child) typedAssignment =
          TerminalBundleOpeningRows.Layout.codecBundle (bundles child) := by
  intro child
  have openingBits : CommitmentBundleFieldRows.BitsPlaced
      (artifact.layouts.terminalOpening child).bundleFields numericAssignment
        (bundles child) := by
    rw [artifact.layoutsValid.terminalOpeningUsesFoldedBundle child]
    exact bundleBits child
  have openingNumericRows : Satisfies
      (CommitmentBundleFieldRows.rows
        (artifact.layouts.terminalOpening child).bundleFields)
      numericAssignment := by
    rw [artifact.layoutsValid.terminalOpeningUsesFoldedBundle child]
    exact artifact.foldedBundle_satisfied numericSatisfies child
  exact TerminalBundleOpeningRows.sound_opens_codec_bundle
    (artifact.layouts.terminalOpening child) numericAssignment typedAssignment
    (bundles child)
    numericCanonical numericOne openingBits openingNumericRows
    assignmentAgreement (typedOne child)
    (artifact.typedOpening_satisfied typedSatisfies child)

/-- The full-claim call site gets its row-inclusion certificate from the
terminal manifest. -/
def Artifact.fullClaimCallSite
    {widths : CompilerWidths} {fullShape operationsShape snapshotShape : Shape}
    (artifact : Artifact widths fullShape operationsShape snapshotShape)
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

end Nightstream.Implementation.NebulaV2.TerminalManifestSchema
