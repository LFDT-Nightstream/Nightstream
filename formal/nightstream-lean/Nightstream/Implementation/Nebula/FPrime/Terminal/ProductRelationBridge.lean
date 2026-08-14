import Nightstream.Implementation.Nebula.NIFS.Core.SelectedVerifier
import Nightstream.Implementation.Nebula.FPrime.Manifest.TerminalNifsCall

/-!
Contract: row-to-terminal-CE bridge for all fourteen V2 accumulator children.

Assurance tier: implementation-to-semantic refinement.

Owns the exact assignment family read from the fourteen terminal opening
layouts, equality of every layout's verifier-selected product configuration,
the binding of accumulator output bundles to their canonical numeric fields,
and the theorem from numeric rows, typed opening rows, and the exact CE-core
checker to all fourteen complete product CE relations.

Does not own NIFS soundness, generated terminal-core rows, compact-proof
soundness, public-result checks, Rust, or the deployed verifier.

Emits constraints: no additional rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.TerminalProductRelationBridge

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.Nebula.ProductSelectedVerifier
open Nightstream.Implementation.Nebula.TerminalManifestNifsCall
open Nightstream.Implementation.Nebula.TerminalManifestSchema
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula.Terminal
open Nightstream.SuperNeo.Concrete.Phi81Relation

variable {widths : CompilerWidths}
variable {fullShape operationsShape snapshotShape : Shape}

/-- The exact fourteen assignments allocated by the terminal opening
program. -/
def assignments
    {manifest : SeedSchedule.Manifest}
    (layouts : TerminalManifestSchema.Layouts widths manifest fullShape
      operationsShape snapshotShape)
    (typedAssignment :
      Nightstream.Implementation.Lowering.Goldilocks.ColumnId →
        Nightstream.SuperNeo.Concrete.F) :
    ProductTerminalRelation.Assignments fullShape :=
  fun child =>
    (layouts.terminalOpening child).fullAssignment typedAssignment

/-- One selected terminal artifact must use the product profile's exact lane
layout and three exact verifier-selected keys for every child. -/
def Configured
    (artifact : TerminalManifestSchema.Artifact widths fullShape
      operationsShape snapshotShape)
    (profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape) : Prop :=
  ∀ child,
    TerminalProductCommitmentBridge.config
        (artifact.layouts.terminalOpening child) =
      profile.config

/-- Exact terminal rows plus the exact CE-core checker establish the complete
fresh-stage CE relation for every post-PiDEC child. -/
theorem terminal_children_hold_of_rows
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
    (configured : Configured artifact profile)
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
        (assignments artifact.layouts typedAssignment) = true) :
    ProductTerminalRelation.Holds profile.config
      (profile.children call.output)
      (assignments artifact.layouts typedAssignment) := by
  have common := artifact.foldedBundlesCommonOpenings typedAssignment
    (profile.bundles call.output) call.canonicalAssignment call.one bundleBits
    numericSatisfies assignmentAgreement typedOne typedSatisfies
  apply ProductTerminalRelation.holds_of_common_openings profile.config
    (profile.children call.output)
    (assignments artifact.layouts typedAssignment)
  · intro child
    exact (common child).1
  · intro child
    have layoutConfig := configured child
    calc
      ProductCommitmentAlgebra.commit profile.config
          (assignments artifact.layouts typedAssignment child) =
          ProductCommitmentAlgebra.commit profile.config
            ((artifact.layouts.terminalOpening child).fullAssignment
              typedAssignment) := rfl
      _ =
          ProductCommitmentAlgebra.commit
            (TerminalProductCommitmentBridge.config
              (artifact.layouts.terminalOpening child))
            ((artifact.layouts.terminalOpening child).fullAssignment
              typedAssignment) := by
        rw [layoutConfig]
      _ = TerminalBundleOpeningRows.exactBundle
            (artifact.layouts.terminalOpening child) typedAssignment :=
        TerminalProductCommitmentBridge.commit_eq_exactBundle
          (artifact.layouts.terminalOpening child) typedAssignment
      _ = TerminalBundleOpeningRows.Layout.codecBundle
            (profile.bundles call.output child) := (common child).2
      _ = (profile.children call.output child).commitment :=
        profile.bundleCommitmentExact call.output child
  · exact (ProductTerminalRelation.checkCore_eq_true_iff
      (profile.children call.output)
      (assignments artifact.layouts typedAssignment)).1 coreAccepted

/-- Honest terminal witnesses satisfy the exact value-level CE-core checker.
This is the local completeness direction and does not construct a compact
proof. -/
theorem terminal_core_check_complete
    {profile : ProductSelectedVerifier.Profile widths fullShape operationsShape
      snapshotShape}
    {output : profile.selected.Output}
    {assignments : ProductTerminalRelation.Assignments fullShape}
    (holds : ProductTerminalRelation.Holds profile.config
      (profile.children output) assignments) :
    ProductTerminalRelation.checkCore (profile.children output) assignments =
      true :=
  (ProductTerminalRelation.checkCore_eq_true_iff
    (profile.children output) assignments).2
      (ProductTerminalRelation.core_of_holds profile.config
        (profile.children output) assignments holds)

end Nightstream.Implementation.Nebula.TerminalProductRelationBridge
