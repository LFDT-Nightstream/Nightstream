import Nightstream.Implementation.NebulaV2.TerminalProductRelationBridge

/-!
Contract: exact row-derived V2 terminal acceptance for the selected product
verifier.

Assurance tier: implementation-to-protocol refinement.

Owns the exact trailing full-claim receipt, its delayed transition to the
closed carry, the selected final-fold output, all fourteen fresh-stage product
CE relations, one common four-component opening per child, and the external
result predicate. The selected verifier output used by the final fold is the
same output decoded by the terminal bundle and relation rows.

Does not prove knowledge soundness of the selected NIFS verifier, generated
row refinement for its opaque verifier or accumulator row families, compact
terminal proof soundness, public-result row refinement, Rust, or deployed
parser conformance. These are separate release obligations.

Emits constraints: no additional rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductTerminalAcceptance

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Implementation.NebulaV2.ProductSelectedVerifier
open Nightstream.Implementation.NebulaV2.TerminalManifestNifsCall
open Nightstream.Implementation.NebulaV2.TerminalManifestSchema
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.Protocol.NebulaV2.Terminal
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

variable {widths : CompilerWidths}
variable {fullShape operationsShape snapshotShape : Shape}

abbrev ComponentCommitment :=
  Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Commitment.Value
    ProductCommitmentAlgebra.Rank

/-- The exact final-fold predicate represented by the selected verifier call.
The output equality prevents a proof accepted for one output from being paired
with a different terminal child family. -/
def FinalFoldRelation
    (selected : SelectedVerifier widths)
    (_running : Unit)
    (claim : Claim selected)
    (proof : PackedProof selected)
    (output : selected.Output) : Prop :=
  proof.2 = output ∧ VerifyClaim selected proof claim

/-- The final accumulator output's exact four-component product bundle for
one post-PiDEC child. -/
def bundleOf
    (profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape)
    (output : profile.selected.Output) (child : FoldedChild) :
    CommitmentBundle.Bundle ComponentCommitment :=
  (profile.children output child).commitment

/-- The exact fourteen-child terminal relation selected by one profile. -/
def terminalRelation
    (profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape)
    (output : profile.selected.Output)
    (assignments : FoldedChild → Assignment fullShape) : Prop :=
  ProductTerminalRelation.Holds profile.config (profile.children output)
    assignments

/-- The exact independent terminal acceptance type instantiated by the V2
product profile. -/
abbrev Acceptance
    (profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape)
    (Statement : Type)
    (resultCheck : Statement → ClosedCarry Digest.Value → Prop)
    (before : Carry Digest.Value (Challenges K) (State K))
    (statement : Statement) :=
  Accepted
    (VerifyClaim profile.selected)
    MemoryProductBalanceRows.ConcreteBalanced
    Unit (PackedProof profile.selected) profile.selected.Output
    (Assignment fullShape) ComponentCommitment Statement
    (FinalFoldRelation profile.selected)
    (bundleOf profile)
    (ProductCommitmentAlgebra.commit profile.config)
    (assignmentNormBounded 2)
    (terminalRelation profile)
    resultCheck () before statement

/-- Exact terminal rows construct the independent terminal acceptance object.
No execution witness, cryptographic bad event, or acceptance conclusion is an
assumption. The remaining premises are concrete row placement/refinement facts
and the verifier-owned external result predicate. -/
def acceptedOfRows
    {artifact : TerminalManifestSchema.Artifact widths fullShape
      operationsShape snapshotShape}
    {profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape}
    {numericAssignment : Nat → Nat}
    (typedAssignment :
      Nightstream.Implementation.Lowering.Goldilocks.ColumnId →
        Nightstream.SuperNeo.Concrete.F)
    (call : TerminalManifestNifsCall.Call artifact profile.selected
      numericAssignment)
    (carry : call.CarryBlocks)
    (configured : TerminalProductRelationBridge.Configured artifact profile)
    (numericSatisfies : Satisfies artifact.programRows numericAssignment)
    (bundleBits : ∀ child,
      CommitmentBundleFieldRows.BitsPlaced
        (artifact.layouts.foldedBundleFields child) numericAssignment
        (profile.bundles call.output child))
    (assignmentAgreement : TerminalBundleOpeningRows.Layout.NumericAgreement
      numericAssignment typedAssignment)
    (typedOne : ∀ child,
      typedAssignment (artifact.layouts.terminalOpening child).one = 1)
    (typedSatisfies :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        artifact.typedOpeningRows typedAssignment)
    (coreAccepted : ProductTerminalRelation.checkCore
      (profile.children call.output)
      (TerminalProductRelationBridge.assignments artifact.layouts
        typedAssignment) = true)
    {Statement : Type}
    (resultCheck : Statement → ClosedCarry Digest.Value → Prop)
    (statement : Statement)
    (resultAccepted : resultCheck statement
      (MemoryOpenSegmentSound.closedOfWire carry.intermediateValue)) :
    Acceptance profile Statement resultCheck
      (MemoryCarryParser.semanticCarry carry.priorValue
        (MemoryCarryParser.parse_value_canonical
          (carry.priorAccepted numericSatisfies)).stepIndex)
      statement := by
  let receipt := call.receiptOfRows numericSatisfies
  let childAssignments :=
    TerminalProductRelationBridge.assignments artifact.layouts typedAssignment
  have terminalHolds : ProductTerminalRelation.Holds profile.config
      (profile.children call.output) childAssignments :=
    TerminalProductRelationBridge.terminal_children_hold_of_rows
      typedAssignment call configured numericSatisfies bundleBits
      assignmentAgreement typedOne typedSatisfies coreAccepted
  refine
    { trailing := receipt.toVerified
      final := MemoryOpenSegmentSound.closedOfWire carry.intermediateValue
      consumesTrailing := ?_
      finalFold :=
        { proof := (call.proof, call.output)
          folded := call.output
          accepted := ?_ }
      assignments := childAssignments
      assignmentsBounded := ?_
      opensCompleteFoldedBundles := ?_
      terminalRelationAccepted := terminalHolds
      resultAccepted := resultAccepted }
  · simpa [receipt] using
      call.selectedTransitionToClosed carry numericSatisfies
  · exact ⟨rfl, (call.receiptOfRows numericSatisfies).accepted⟩
  · intro child
    have stageFresh := (terminalHolds child).1
    have normBound := (terminalHolds child).2.1.2.2
    simpa [stageFresh, productionGlobalParams] using normBound
  · intro child
    exact ProductTerminalRelation.commitment_of_holds profile.config
      (profile.children call.output) childAssignments terminalHolds child

/-- Row-derived terminal acceptance consumes the exact selected trailing
claim. This projects the authority fact needed by the lifetime theorem. -/
theorem consumes_exact_selected_trailing_claim
    {profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape}
    {Statement : Type}
    {resultCheck : Statement → ClosedCarry Digest.Value → Prop}
    {before : Carry Digest.Value (Challenges K) (State K)}
    {statement : Statement}
    (accepted : Acceptance profile Statement resultCheck before statement) :
    VerifyClaim profile.selected accepted.trailing.proof
        accepted.trailing.claim ∧
      Consumes MemoryProductBalanceRows.ConcreteBalanced before
        accepted.trailing.claim.memory (.closed accepted.final) :=
  Accepted.consumes_exact_verified_trailing_claim accepted

/-- Row-derived terminal acceptance uses the exact same fourteen assignment
values for product openings and the terminal CE relation. -/
theorem common_product_witnesses
    {profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape}
    {Statement : Type}
    {resultCheck : Statement → ClosedCarry Digest.Value → Prop}
    {before : Carry Digest.Value (Challenges K) (State K)}
    {statement : Statement}
    (accepted : Acceptance profile Statement resultCheck before statement) :
    ∃ assignments : FoldedChild → Assignment fullShape,
      (∀ child, assignmentNormBounded 2 (assignments child)) ∧
      (∀ child,
        ProductCommitmentAlgebra.commit profile.config (assignments child) =
          bundleOf profile accepted.finalFold.folded child) ∧
      terminalRelation profile accepted.finalFold.folded assignments :=
  Accepted.common_witness accepted

end Nightstream.Implementation.NebulaV2.ProductTerminalAcceptance
