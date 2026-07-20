import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessCommitment
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionBoundary
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionChecker
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal.TerminalCEBridge

/-!
Production delayed-`y_zcol` contract over full packed witness matrices.

Assurance tier: model-level until Rust verifier dataflow, generated
assignment-decoder rows, combined-NC rows, and commitment openings instantiate
the inputs below.

Owns: execution of the canonical production checker with
`Sources.Data.runningAssignments` definitionally decoded from complete
`CcsWitness.Z`/`CeWitness.Z` matrices; exact derivation of logical production
acceptance from that Boolean checker; adjacent-step delayed closure with
executable accumulator-state equality checks; and the terminal theorem over
explicit full raw openings.

Does not own: sparse rows that construct `SemanticInput` or `ChildOpenings`,
the Rust handoff of complete witness matrices, verifier-key continuity,
terminal commitment openings, Ajtai binding, y-ring authority, costs, or row
removal. No `CeClaim.y_zcol`, `SourceProjectionMatches`,
`UpstreamProducerColumnsBound`, `BindingsHoldFor`, or generic `outputUnbound`
premise occurs here.

Emits constraints: none; correspondence theorem only.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.full_witness.check` | execute combined NC over full packed `Z` witnesses | checked/model |
| `f_prime.pi_ccs_nc.delayed.full_witness.recursive` | successor acceptance closes the previous packed projection or named event | derived/security partition |
| `f_prime.pi_ccs_nc.delayed.full_witness.terminal` | final full-witness openings close the last pending projection | checked boundary |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open PackedWitness

universe uState uEncoding uDigest

variable
  {shape : SemanticShape}
  {State : Type uState}
  {Encoding : Type uEncoding}
  {Digest : Type uDigest}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Execute the claims-only production checker. Complete witness matrices do
not enter this predicate; they are used only by the extraction/refinement
theorem below. -/
def messageCheck
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize) : Bool :=
  CombinedNc.ProductionChecker.messageCheck context certificate

/-- The claims-only fixed-profile checker is exact to the public NIFS
predicate. -/
theorem messageCheck_eq_true_iff_accepted
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize) :
    messageCheck context certificate = true <->
      CombinedNc.ProductionNifs.MessageAccepted context.materialize
        certificate := by
  exact CombinedNc.ProductionChecker.messageCheck_eq_true_iff_accepted
    context certificate

/-- Execute the production NIFS checker with every running source decoded
directly from its full packed witness matrix. -/
def check
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize) : Bool :=
  CombinedNc.ProductionChecker.check context
    (decodedData template witnesses) certificate

/-- The executable checker is exact to production acceptance over the full
packed-witness source data. Raw-child authority is not a proposition supplied
by the caller: it is the definition of `decodedData`. -/
theorem check_eq_true_iff_accepted
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize) :
    check context template witnesses certificate = true <->
      CombinedNc.ProductionNifs.Accepted context.materialize
        (decodedData template witnesses) certificate := by
  exact CombinedNc.ProductionChecker.check_eq_true_iff_accepted context
    (decodedData template witnesses) certificate

/-- Execute final delayed closure directly over complete packed child
openings. Flat assignments are produced only by the proved Rust-order
`PackedWitness.unpack`; no caller can substitute a public `y_zcol` sidecar. -/
def terminalCheck
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize)
    (witnesses : Fin productionGlobalParams.k -> Matrix shape) : Bool :=
  CombinedNc.ProductionTerminal.check context.materialize certificate
    (fun child => unpack (witnesses child))

/-- The packed terminal checker is exact to the generic terminal relation over
the assignments decoded from those same complete matrices. -/
theorem terminalCheck_eq_true_iff_accepted
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize)
    (witnesses : Fin productionGlobalParams.k -> Matrix shape) :
    terminalCheck context certificate witnesses = true <->
      CombinedNc.ProductionTerminal.Accepted context.materialize certificate
        (fun child => unpack (witnesses child)) := by
  exact CombinedNc.ProductionTerminal.check_eq_true_iff context.materialize
    certificate (fun child => unpack (witnesses child))

/-- The actual terminal-CE authority path and the delayed projection check
compose into the packed terminal checker. Terminal CE supplies the fourteen
ordered raw openings; the projection is recomputed separately over those
same decoded witness matrices. -/
theorem terminalCheck_of_terminalCE_and_projection
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize)
    (witnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminalAccepted : Nightstream.Protocol.TerminalCE.Holds
      (CombinedNc.ProductionTerminal.TerminalCEBridge.semantics
        context.materialize)
      (CombinedNc.ProductionTerminal.TerminalCEBridge.terminalInstance
        context.materialize certificate
        (fun child => unpack (witnesses child))))
    (projectionAccepted :
      CombinedNc.ProductionTerminal.projectionCheck context.materialize
        certificate (fun child => unpack (witnesses child)) = true) :
    terminalCheck context certificate witnesses = true := by
  apply (terminalCheck_eq_true_iff_accepted context certificate witnesses).2
  exact
    CombinedNc.ProductionTerminal.TerminalCEBridge.accepted_of_terminalCE_and_projectionCheck
      context.materialize certificate (fun child => unpack (witnesses child))
      terminalAccepted projectionAccepted

/-- Rust-shaped terminal child-loop success and the independent projection
comparison imply the complete packed terminal checker. -/
theorem terminalCheck_of_rustVerifyPairs_and_projection
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (certificate : FixedActive.Certificate context.materialize)
    (witnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminalAccepted :
      CombinedNc.ProductionTerminal.TerminalCEBridge.rustVerifyPairs
        context.materialize certificate
        (fun child => unpack (witnesses child)) =
          (Except.ok () : Except
            Nightstream.Implementation.Rust.Terminal.Error Unit))
    (projectionAccepted :
      CombinedNc.ProductionTerminal.projectionCheck context.materialize
        certificate (fun child => unpack (witnesses child)) = true) :
    terminalCheck context certificate witnesses = true := by
  apply (terminalCheck_eq_true_iff_accepted context certificate witnesses).2
  exact
    CombinedNc.ProductionTerminal.TerminalCEBridge.accepted_of_rustVerifyPairs_and_projectionCheck
      context.materialize certificate (fun child => unpack (witnesses child))
      terminalAccepted projectionAccepted

/-- A successful public verifier execution refines to the post-extraction
full-witness checker or exposes exactly the packed output-opening failure.
The decoded data is built from complete matrices, never from public sidecar
values. -/
theorem messageCheck_implies_check_or_outputBindingFailure
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (accepted : messageCheck context certificate = true) :
    check context template witnesses certificate = true ∨
      CombinedNc.ProductionPiCcs.OutputBindingFailure context.materialize
        (decodedData template witnesses) certificate := by
  have claims : CombinedNc.ProductionNifs.MessageAccepted
      context.materialize certificate :=
    (messageCheck_eq_true_iff_accepted context certificate).mp accepted
  rcases
      CombinedNc.ProductionNifs.messageAccepted_implies_accepted_or_outputBindingFailure
        context.materialize (decodedData template witnesses) certificate
        claims with raw | failure
  · exact Or.inl
      ((check_eq_true_iff_accepted context template witnesses certificate).mpr
        raw)
  · exact Or.inr failure

/-- Two adjacent successful full-witness checks, together with executable
accumulator-state equalities, close the previous independent semantic fold or
expose only the named y-ring/algebraic/binding events. The one-fold delay is
explicit: successor acceptance closes `previousContext`, not the current
output. -/
theorem checkedPair_of_stateChecks_implies_previousSemanticFold_or_badEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousInput : SemanticInput previousContext.materialize
      (decodedData previousTemplate previousWitnesses))
    (previousChildren : ChildOpenings previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate)
    (previousChecked :
      check previousContext previousTemplate previousWitnesses
        previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextInput : SemanticInput nextContext.materialize
      (decodedData nextTemplate nextWitnesses))
    (nextChecked :
      check nextContext nextTemplate nextWitnesses nextCertificate = true)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key = previousContext.materialize.key) :
    SemanticFold.Holds previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate := by
  let nextParent :=
    nextContext.input.parent.materialize nextContext.input.system
  have previousAccepted : CombinedNc.ProductionNifs.Accepted
      previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate :=
    (check_eq_true_iff_accepted previousContext previousTemplate
      previousWitnesses previousCertificate).mp previousChecked
  have nextAccepted : CombinedNc.ProductionNifs.Accepted
      nextContext.materialize (decodedData nextTemplate nextWitnesses)
      nextCertificate :=
    (check_eq_true_iff_accepted nextContext nextTemplate nextWitnesses
      nextCertificate).mp nextChecked
  have previousBinds : StateBinds scheme stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate)) :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate))).mp previousStateChecked
  have nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.materialize.input.running nextContext.materialize.pending :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest nextParent nextContext.materialize.input.running
      nextContext.materialize.pending).mp nextStateChecked
  exact
    CombinedNc.ProductionBoundary.acceptedPair_implies_previousSemanticFold_or_badEvent
      noZeroDivisors scheme stateDigest previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate
      previousInput previousChildren previousAccepted nextContext.materialize
      (decodedData nextTemplate nextWitnesses) nextCertificate nextInput
      nextAccepted nextParent rfl sameKey previousBinds nextBinds

/-- Y-zcol-only backward step over complete packed running witnesses.  The
successor `nextPacked` hypothesis extracts its claims check to the raw
combined-NC predicate over `decodedData nextTemplate nextWitnesses`.  Both
state equalities are executable checks.  No semantic input, previous child
opening, or verifier-key-equality premise is needed to derive the predecessor
packed equation or the exact raw-parent-state failure partition. -/
theorem messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPacked_or_rawParentBadEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked : messageCheck previousContext previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked : messageCheck nextContext nextCertificate = true)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      nextContext.materialize.covers (decodedData nextTemplate nextWitnesses)
      (CombinedNc.ProductionPiCcs.ncPoint nextContext.materialize
        nextCertificate).block nextCertificate.piCcs.output)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true) :
    Terminal.PackedYZcolBoundAtBlock previousContext.materialize.covers
        (decodedData previousTemplate previousWitnesses)
        (DelayedProduction.outgoingPending previousContext.materialize
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      CombinedNc.ProductionBoundary.RawParentRecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate := by
  let nextParent :=
    nextContext.input.parent.materialize nextContext.input.system
  have previousClaims : CombinedNc.ProductionNifs.MessageAccepted
      previousContext.materialize previousCertificate :=
    (messageCheck_eq_true_iff_accepted previousContext previousCertificate).mp
      previousChecked
  have nextClaims : CombinedNc.ProductionNifs.MessageAccepted
      nextContext.materialize nextCertificate :=
    (messageCheck_eq_true_iff_accepted nextContext nextCertificate).mp
      nextChecked
  have previousBinds : StateBinds scheme stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate)) :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate))).mp previousStateChecked
  have nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.materialize.input.running nextContext.materialize.pending :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest nextParent nextContext.materialize.input.running
      nextContext.materialize.pending).mp nextStateChecked
  exact
    CombinedNc.ProductionBoundary.messageAcceptedPair_of_nextPacked_implies_previousPacked_or_rawParentBadEvent
      noZeroDivisors scheme stateDigest previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate
      previousClaims nextContext.materialize
      (decodedData nextTemplate nextWitnesses) nextCertificate nextClaims
      nextPacked nextParent rfl previousBinds nextBinds

/-- Exact full-witness extraction failures for the direct-parent edge. The
first is only the predecessor canonical-parent commitment/norm fact. The
second is only alignment of the successor's decoded raw running assignments
with their public commitments. Neither constructor contains public inputs,
evaluation sidecars, y-ring, or a packed `y_zcol` equation. -/
inductive ParentOpeningExternalBindingFailure
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape) : Prop where
  | previousParentBinding
      (failure : ¬ DelayedRawChildren.CanonicalParentBinding
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        previousCertificate) :
      ParentOpeningExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | nextRawCommitments
      (failure : ¬ DelayedRawChildren.RawRunningCommitmentsBound
        nextContext.materialize (decodedData nextTemplate nextWitnesses)) :
      ParentOpeningExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses

/-- The full-witness external binding branch contains exactly a predecessor
canonical-parent failure or one concrete successor matrix/commitment
mismatch. In particular, it cannot conceal a prover-carried `y_zcol`
sidecar, digest mismatch, or generic output-unbound proposition. -/
theorem parentOpeningExternalBindingFailure_iff
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape) :
    ParentOpeningExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses ↔
      (¬ DelayedRawChildren.CanonicalParentBinding
          previousContext.materialize
          (decodedData previousTemplate previousWitnesses)
          previousCertificate) ∨
        ∃ child,
          PackedWitnessCommitment.matrixCommit nextContext.materialize.key
              (nextWitnesses
                (nextContext.materialize.alignment.semanticRunningIndex child)) ≠
            (nextContext.materialize.input.running child).commitment := by
  constructor
  . intro failure
    cases failure with
    | previousParentBinding bad => exact Or.inl bad
    | nextRawCommitments bad =>
        exact Or.inr <|
          (PackedWitnessCommitment.rawRunningCommitmentsUnbound_iff_exists_matrixCommit_ne
            nextContext.materialize nextTemplate nextWitnesses).mp bad
  . rintro (bad | ⟨child, mismatch⟩)
    . exact .previousParentBinding bad
    . exact .nextRawCommitments <|
        (PackedWitnessCommitment.rawRunningCommitmentsUnbound_iff_exists_matrixCommit_ne
          nextContext.materialize nextTemplate nextWitnesses).mpr
            ⟨child, mismatch⟩

/-- Full-witness direct-parent edge.  Every delayed-projection premise is
derived from the two Boolean message/state checks and the positive successor
packed equation. The exact parent commitment/norm and successor raw-table
commitment predicates are exhaustively partitioned into the explicit external
binding type above; neither is assumed in the theorem conclusion. -/
theorem messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPacked_or_parentOpeningBadEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked : messageCheck previousContext previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked : messageCheck nextContext nextCertificate = true)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      nextContext.materialize.covers (decodedData nextTemplate nextWitnesses)
      (CombinedNc.ProductionPiCcs.ncPoint nextContext.materialize
        nextCertificate).block nextCertificate.piCcs.output)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key =
      previousContext.materialize.key) :
    Terminal.PackedYZcolBoundAtBlock previousContext.materialize.covers
        (decodedData previousTemplate previousWitnesses)
        (DelayedProduction.outgoingPending previousContext.materialize
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      CombinedNc.ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate ∨
      ParentOpeningExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses := by
  classical
  by_cases previousParentBound : DelayedRawChildren.CanonicalParentBinding
      previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate
  · by_cases nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
    · let nextParent :=
        nextContext.input.parent.materialize nextContext.input.system
      have previousClaims : CombinedNc.ProductionNifs.MessageAccepted
          previousContext.materialize previousCertificate :=
        (messageCheck_eq_true_iff_accepted previousContext
          previousCertificate).mp previousChecked
      have nextClaims : CombinedNc.ProductionNifs.MessageAccepted
          nextContext.materialize nextCertificate :=
        (messageCheck_eq_true_iff_accepted nextContext nextCertificate).mp
          nextChecked
      have previousBinds : StateBinds scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending previousContext.materialize
            previousCertificate)) :=
        (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
          stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending previousContext.materialize
            previousCertificate))).mp previousStateChecked
      have nextBinds : StateBinds scheme stateDigest nextParent
          nextContext.materialize.input.running
          nextContext.materialize.pending :=
        (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
          stateDigest nextParent nextContext.materialize.input.running
          nextContext.materialize.pending).mp nextStateChecked
      rcases
          CombinedNc.ProductionBoundary.messageAcceptedPair_of_nextPacked_of_parentOpening_implies_previousPacked_or_badEvent
            noZeroDivisors scheme stateDigest previousContext.materialize
            (decodedData previousTemplate previousWitnesses)
            previousCertificate previousParentBound previousClaims
            nextContext.materialize (decodedData nextTemplate nextWitnesses)
            nextCertificate nextCommitments nextClaims nextPacked nextParent rfl
            sameKey previousBinds nextBinds with packed | bad
      · exact Or.inl packed
      · exact Or.inr (Or.inl bad)
    · exact Or.inr (Or.inr (.nextRawCommitments nextCommitments))
  · exact Or.inr (Or.inr (.previousParentBinding previousParentBound))

/-- Raw-matrix form of the direct-parent edge.  Exact Ajtai opening equations
for the successor's fourteen packed witnesses derive the raw-child commitment
premise, so that branch cannot be supplied by a `y_zcol` sidecar or remain as
an unexplained assumption.  The predecessor canonical-parent opening remains
an explicit commitment/binding boundary. -/
theorem messageCheckedPair_of_nextPacked_of_openedPackedWitnesses_implies_previousPacked_or_parentBindingFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked : messageCheck previousContext previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked : messageCheck nextContext nextCertificate = true)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      nextContext.materialize.covers (decodedData nextTemplate nextWitnesses)
      (CombinedNc.ProductionPiCcs.ncPoint nextContext.materialize
        nextCertificate).block nextCertificate.piCcs.output)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key =
      previousContext.materialize.key)
    (nextOpened : forall child,
      PackedWitnessCommitment.matrixCommit nextContext.materialize.key
          (nextWitnesses
            (nextContext.materialize.alignment.semanticRunningIndex child)) =
        (nextContext.materialize.input.running child).commitment) :
    Terminal.PackedYZcolBoundAtBlock previousContext.materialize.covers
        (decodedData previousTemplate previousWitnesses)
        (DelayedProduction.outgoingPending previousContext.materialize
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      CombinedNc.ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate ∨
      ¬ DelayedRawChildren.CanonicalParentBinding previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        previousCertificate := by
  have nextBound : DelayedRawChildren.RawRunningCommitmentsBound
      nextContext.materialize (decodedData nextTemplate nextWitnesses) :=
    PackedWitnessCommitment.rawRunningCommitmentsBound_of_openedPackedWitnesses
      nextContext.materialize nextTemplate nextWitnesses nextOpened
  rcases
      messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPacked_or_parentOpeningBadEvent
        noZeroDivisors scheme stateDigest previousContext previousTemplate
        previousWitnesses previousCertificate previousChecked
        previousStateChecked nextContext nextTemplate nextWitnesses
        nextCertificate nextChecked nextPacked nextStateChecked sameKey with
    packed | bad | binding
  · exact Or.inl packed
  · exact Or.inr (Or.inl bad)
  · cases binding with
    | previousParentBinding failure => exact Or.inr (Or.inr failure)
    | nextRawCommitments failure => exact (failure nextBound).elim

/-- Strong backward claims-level recursive step over complete packed running
witnesses. The successful branch keeps the derived predecessor packed equation
beside its semantic fold so a finite trace can continue one edge backward. -/
theorem messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPackedAndSemanticFold_or_badEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousInput : SemanticInput previousContext.materialize
      (decodedData previousTemplate previousWitnesses))
    (previousChildren : ChildOpenings previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate)
    (previousChecked : messageCheck previousContext previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextInput : SemanticInput nextContext.materialize
      (decodedData nextTemplate nextWitnesses))
    (nextChecked : messageCheck nextContext nextCertificate = true)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      nextContext.materialize.covers (decodedData nextTemplate nextWitnesses)
      (CombinedNc.ProductionPiCcs.ncPoint nextContext.materialize
        nextCertificate).block nextCertificate.piCcs.output)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key = previousContext.materialize.key) :
    (Terminal.PackedYZcolBoundAtBlock previousContext.materialize.covers
          (decodedData previousTemplate previousWitnesses)
          (DelayedProduction.outgoingPending previousContext.materialize
            previousCertificate).oldBlock
          previousCertificate.piCcs.output ∧
        SemanticFold.Holds previousContext.materialize
          (decodedData previousTemplate previousWitnesses)
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate := by
  let nextParent :=
    nextContext.input.parent.materialize nextContext.input.system
  have previousClaims : CombinedNc.ProductionNifs.MessageAccepted
      previousContext.materialize previousCertificate :=
    (messageCheck_eq_true_iff_accepted previousContext previousCertificate).mp
      previousChecked
  have nextClaims : CombinedNc.ProductionNifs.MessageAccepted
      nextContext.materialize nextCertificate :=
    (messageCheck_eq_true_iff_accepted nextContext nextCertificate).mp
      nextChecked
  have previousBinds : StateBinds scheme stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate)) :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest
      (derive previousContext.materialize previousCertificate).piRlcOutput
      (outputChildren previousContext.materialize previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext.materialize
        previousCertificate))).mp previousStateChecked
  have nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.materialize.input.running nextContext.materialize.pending :=
    (CombinedNc.ProductionChecker.stateBindingCheck_eq_true_iff scheme
      stateDigest nextParent nextContext.materialize.input.running
      nextContext.materialize.pending).mp nextStateChecked
  exact
    CombinedNc.ProductionBoundary.messageAcceptedPair_of_nextPacked_implies_previousPackedAndSemanticFold_or_badEvent
      noZeroDivisors scheme stateDigest previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate
      previousInput previousChildren previousClaims nextContext.materialize
      (decodedData nextTemplate nextWitnesses) nextCertificate nextInput
      nextClaims nextPacked nextParent rfl sameKey previousBinds nextBinds

/-- Backward claims-level recursive step over complete packed running
witnesses. `nextPacked` is the induction hypothesis obtained from the terminal
check or a later accepted successor. Both public-message checks are then
refined without an output-binding failure result. -/
theorem messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousSemanticFold_or_badEvent
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousInput : SemanticInput previousContext.materialize
      (decodedData previousTemplate previousWitnesses))
    (previousChildren : ChildOpenings previousContext.materialize
      (decodedData previousTemplate previousWitnesses) previousCertificate)
    (previousChecked : messageCheck previousContext previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextInput : SemanticInput nextContext.materialize
      (decodedData nextTemplate nextWitnesses))
    (nextChecked : messageCheck nextContext nextCertificate = true)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      nextContext.materialize.covers (decodedData nextTemplate nextWitnesses)
      (CombinedNc.ProductionPiCcs.ncPoint nextContext.materialize
        nextCertificate).block nextCertificate.piCcs.output)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key = previousContext.materialize.key) :
    SemanticFold.Holds previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate := by
  rcases
      messageCheckedPair_of_nextPacked_of_stateChecks_implies_previousPackedAndSemanticFold_or_badEvent
        noZeroDivisors scheme stateDigest previousContext previousTemplate
        previousWitnesses previousCertificate previousInput previousChildren
        previousChecked previousStateChecked nextContext nextTemplate
        nextWitnesses nextCertificate nextInput nextChecked nextPacked
        nextStateChecked sameKey with success | yRing | bad
  · exact Or.inl success.2
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr bad)

/-- Packed terminal direct-parent contract.  The Boolean terminal checker
validates the fourteen ordered full-matrix child openings and recomputes the
delayed projection from those same assignments. Given the independently
checked canonical parent commitment/norm and strict Π_DEC tail, it yields the
packed output or only mixing/parent-binding events. -/
theorem terminalCheck_of_parentOpening_implies_packed_or_badEvent
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (parentBound : DelayedRawChildren.CanonicalParentBinding
      context.materialize (decodedData template witnesses) certificate)
    (piDecAccepted : PiDEC.Accepted (decAlgebra context.materialize.key)
      ((derive context.materialize certificate).piDecAttempt certificate))
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : terminalCheck context certificate terminalWitnesses = true) :
    Terminal.PackedYZcolBoundAtBlock context.materialize.covers
        (decodedData template witnesses)
        (derive context.materialize certificate).piCcs.ncPoint.block
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.materialize.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments (decodedData template witnesses)
          context.materialize.alignment)
        (DelayedProduction.outgoingPending context.materialize
          certificate).oldBlock
        (PackedYZcol.sourceClaims context.materialize certificate) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics context.materialize.key) productionGlobalParams
        (derive context.materialize certificate).piRlcOutput.commitment) := by
  exact
    CombinedNc.ProductionTerminal.accepted_of_parentOpening_implies_packedYZcolBound_or_badEvent
      context.materialize (decodedData template witnesses) certificate
      parentBound piDecAccepted (fun child => unpack (terminalWitnesses child))
      ((terminalCheck_eq_true_iff_accepted context certificate
        terminalWitnesses).mp terminal)

/-- A successful final full-witness check plus verifier-driven raw child
openings closes the last pending projection. No successor context is invented
at the terminal boundary. -/
theorem checkedTerminal_implies_semanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (input : SemanticInput context.materialize
      (decodedData template witnesses))
    (children : ChildOpenings context.materialize
      (decodedData template witnesses) certificate)
    (checked : check context template witnesses certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : terminalCheck context certificate terminalWitnesses = true) :
    SemanticFold.Holds context.materialize
        (decodedData template witnesses)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate := by
  exact
    CombinedNc.ProductionBoundary.acceptedTerminal_implies_previousSemanticFold_or_badEvent
      noZeroDivisors context.materialize (decodedData template witnesses)
      certificate input children
      ((check_eq_true_iff_accepted context template witnesses certificate).mp
        checked)
      (fun child => unpack (terminalWitnesses child)) terminal

/-- The claims-only production check can be closed directly by the complete
packed terminal witnesses.  Terminal authority is established first, so the
claims-to-raw refinement contributes no output-binding failure branch. -/
theorem messageCheckedTerminal_implies_semanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (input : SemanticInput context.materialize
      (decodedData template witnesses))
    (children : ChildOpenings context.materialize
      (decodedData template witnesses) certificate)
    (checked : messageCheck context certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : terminalCheck context certificate terminalWitnesses = true) :
    SemanticFold.Holds context.materialize
        (decodedData template witnesses)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate := by
  exact
    CombinedNc.ProductionBoundary.messageAcceptedTerminal_implies_previousSemanticFold_or_badEvent
      noZeroDivisors context.materialize (decodedData template witnesses)
      certificate input children
      ((messageCheck_eq_true_iff_accepted context certificate).mp checked)
      (fun child => unpack (terminalWitnesses child)) terminal

/-- Claims-level production acceptance composed with the concrete terminal-CE
opening path and the independent delayed projection check. The positive
projection is derived before claims extraction, so no output-binding failure
appears in the result. -/
theorem messageCheckedTerminal_of_terminalCE_and_projection_implies_semanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (input : SemanticInput context.materialize
      (decodedData template witnesses))
    (children : ChildOpenings context.materialize
      (decodedData template witnesses) certificate)
    (checked : messageCheck context certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminalAccepted : Nightstream.Protocol.TerminalCE.Holds
      (CombinedNc.ProductionTerminal.TerminalCEBridge.semantics
        context.materialize)
      (CombinedNc.ProductionTerminal.TerminalCEBridge.terminalInstance
        context.materialize certificate
        (fun child => unpack (terminalWitnesses child))))
    (projectionAccepted :
      CombinedNc.ProductionTerminal.projectionCheck context.materialize
        certificate (fun child => unpack (terminalWitnesses child)) = true) :
    SemanticFold.Holds context.materialize
        (decodedData template witnesses)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate := by
  exact messageCheckedTerminal_implies_semanticFold_or_badEvent
    noZeroDivisors context template witnesses certificate input children
    checked terminalWitnesses
    (terminalCheck_of_terminalCE_and_projection context certificate
      terminalWitnesses terminalAccepted projectionAccepted)

/-- Rust-shaped terminal child verification plus the independent delayed
projection check closes claims-level production acceptance without an
output-binding branch. -/
theorem messageCheckedTerminal_of_rustVerifyPairs_and_projection_implies_semanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (input : SemanticInput context.materialize
      (decodedData template witnesses))
    (children : ChildOpenings context.materialize
      (decodedData template witnesses) certificate)
    (checked : messageCheck context certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminalAccepted :
      CombinedNc.ProductionTerminal.TerminalCEBridge.rustVerifyPairs
        context.materialize certificate
        (fun child => unpack (terminalWitnesses child)) =
          (Except.ok () : Except
            Nightstream.Implementation.Rust.Terminal.Error Unit))
    (projectionAccepted :
      CombinedNc.ProductionTerminal.projectionCheck context.materialize
        certificate (fun child => unpack (terminalWitnesses child)) = true) :
    SemanticFold.Holds context.materialize
        (decodedData template witnesses)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate := by
  exact messageCheckedTerminal_implies_semanticFold_or_badEvent
    noZeroDivisors context template witnesses certificate input children
    checked terminalWitnesses
    (terminalCheck_of_rustVerifyPairs_and_projection context certificate
      terminalWitnesses terminalAccepted projectionAccepted)

/-- Exact physical bindings still outside the current generated-row bridge.
These are the only semantic premises partitioned by the headline recursive
theorem below. A running-source commitment/public-input mismatch, including a
full-witness Ajtai alignment failure, lands in `previousInput` or `nextInput`;
it cannot be hidden as a `y_zcol` output mismatch. -/
inductive ExternalBindingFailure
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape) : Prop where
  | previousInput
      (failure : ¬ SemanticInput previousContext.materialize
        (decodedData previousTemplate previousWitnesses)) :
      ExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | previousChildren
      (failure : ¬ ChildOpenings previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        previousCertificate) :
      ExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses
  | nextInput
      (failure : ¬ SemanticInput nextContext.materialize
        (decodedData nextTemplate nextWitnesses)) :
      ExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses

/-- Two adjacent full-witness executions plus the two executable state checks
yield the previous independent semantic fold, the specifically named y-ring
branch, an existing delayed algebraic/binding event, or one exact external
input/child-opening failure.

`ProductionNifs.Accepted`, state binding, raw-child authority, and all former
`y_zcol` producer/consumer premises are derived rather than assumed. Key
continuity remains a verifier-owned setup identity until the active setup
facade specializes both contexts to the same key. -/
theorem checkedPair_of_stateChecks_implies_previousSemanticFold_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked :
      check previousContext previousTemplate previousWitnesses
        previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked :
      check nextContext nextTemplate nextWitnesses nextCertificate = true)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key =
      previousContext.materialize.key) :
    SemanticFold.Holds previousContext.materialize
        (decodedData previousTemplate previousWitnesses)
        (derive previousContext.materialize previousCertificate).piRlcOutput
        (outputChildren previousContext.materialize previousCertificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate ∨
      ExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses := by
  classical
  by_cases previousInput : SemanticInput previousContext.materialize
      (decodedData previousTemplate previousWitnesses)
  · by_cases previousChildren : ChildOpenings previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
    · by_cases nextInput : SemanticInput nextContext.materialize
          (decodedData nextTemplate nextWitnesses)
      · rcases
          checkedPair_of_stateChecks_implies_previousSemanticFold_or_badEvent
            noZeroDivisors scheme stateDigest previousContext previousTemplate
            previousWitnesses previousCertificate previousInput
            previousChildren previousChecked previousStateChecked nextContext
            nextTemplate nextWitnesses nextCertificate nextInput nextChecked
            nextStateChecked sameKey with semantic | yRing | bad
        · exact Or.inl semantic
        · exact Or.inr (Or.inl yRing)
        · exact Or.inr (Or.inr (Or.inl bad))
      · exact Or.inr (Or.inr (Or.inr (.nextInput nextInput)))
    · exact Or.inr (Or.inr (Or.inr (.previousChildren previousChildren)))
  · exact Or.inr (Or.inr (Or.inr (.previousInput previousInput)))

/-- The recursive full-witness checker reaches the independent paper
statement, not merely an implementation-shaped acceptance predicate.  The
one-fold delay is still explicit: the successor check proves the paper
statement for `previousTemplate` and `previousWitnesses`. -/
theorem checkedPair_of_stateChecks_implies_previousPaper_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate :
      FixedActive.Certificate previousContext.materialize)
    (previousChecked :
      check previousContext previousTemplate previousWitnesses
        previousCertificate = true)
    (previousStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (derive previousContext.materialize previousCertificate).piRlcOutput
          (outputChildren previousContext.materialize previousCertificate)
          (some (DelayedProduction.outgoingPending
            previousContext.materialize previousCertificate)) = true)
    (nextContext : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate nextContext.materialize)
    (nextChecked :
      check nextContext nextTemplate nextWitnesses nextCertificate = true)
    (nextStateChecked :
      CombinedNc.ProductionChecker.stateBindingCheck scheme stateDigest
          (nextContext.input.parent.materialize nextContext.input.system)
          nextContext.materialize.input.running
          nextContext.materialize.pending = true)
    (sameKey : nextContext.materialize.key =
      previousContext.materialize.key) :
    Semantics.Paper.Holds
        (decodedData previousTemplate previousWitnesses) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate ∨
      CombinedNc.ProductionBoundary.RecursiveBadEvent scheme
        previousContext.materialize
        (decodedData previousTemplate previousWitnesses) previousCertificate
        nextContext.materialize (decodedData nextTemplate nextWitnesses)
        nextCertificate ∨
      ExternalBindingFailure previousContext previousTemplate
        previousWitnesses previousCertificate nextContext nextTemplate
        nextWitnesses := by
  rcases
      checkedPair_of_stateChecks_implies_previousSemanticFold_or_namedFailure
        noZeroDivisors scheme stateDigest previousContext previousTemplate
        previousWitnesses previousCertificate previousChecked
        previousStateChecked nextContext nextTemplate nextWitnesses
        nextCertificate nextChecked nextStateChecked sameKey with
    semantic | yRing | bad | binding
  · rcases semantic with ⟨_, realized⟩
    exact Or.inl realized.paper
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr binding))

/-- Exact physical facts still outside the terminal generated-row bridge.
The verifier-owned terminal opening is checked executably; only the semantic
input and child-opening refinements are partitioned here. -/
inductive TerminalExternalBindingFailure
    (context : FixedActive.Canonical.Context shape State
      publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize) : Prop where
  | input
      (failure : ¬ SemanticInput context.materialize
        (decodedData template witnesses)) :
      TerminalExternalBindingFailure context template witnesses certificate
  | children
      (failure : ¬ ChildOpenings context.materialize
        (decodedData template witnesses) certificate) :
      TerminalExternalBindingFailure context template witnesses certificate

/-- The terminal checker derives its own production acceptance and partitions
the two remaining physical semantic refinements.  In particular it does not
assume the desired packed projection or read a carried child `y_zcol`. -/
theorem checkedTerminal_implies_semanticFold_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (checked : check context template witnesses certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : terminalCheck context certificate terminalWitnesses = true) :
    SemanticFold.Holds context.materialize
        (decodedData template witnesses)
        (derive context.materialize certificate).piRlcOutput
        (outputChildren context.materialize certificate) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate ∨
      TerminalExternalBindingFailure context template witnesses certificate := by
  classical
  by_cases input : SemanticInput context.materialize
      (decodedData template witnesses)
  · by_cases children : ChildOpenings context.materialize
        (decodedData template witnesses) certificate
    · rcases checkedTerminal_implies_semanticFold_or_badEvent
          noZeroDivisors context template witnesses certificate input children
          checked terminalWitnesses terminal with semantic | yRing | bad
      · exact Or.inl semantic
      · exact Or.inr (Or.inl yRing)
      · exact Or.inr (Or.inr (Or.inl bad))
    · exact Or.inr (Or.inr (Or.inr (.children children)))
  · exact Or.inr (Or.inr (Or.inr (.input input)))

/-- Final full-witness checking reaches the independent paper statement or a
precisely named y-ring, algebraic, opening, or external binding failure. -/
theorem checkedTerminal_implies_paper_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (context : FixedActive.Canonical.Context shape State publicRingColumns
      publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate context.materialize)
    (checked : check context template witnesses certificate = true)
    (terminalWitnesses : Fin productionGlobalParams.k -> Matrix shape)
    (terminal : terminalCheck context certificate terminalWitnesses = true) :
    Semantics.Paper.Holds (decodedData template witnesses) ∨
      CombinedNc.ProductionPiCcs.YRingUnbound context.materialize
        (decodedData template witnesses) certificate ∨
      CombinedNc.ProductionBoundary.TerminalBadEvent context.materialize
        (decodedData template witnesses) certificate ∨
      TerminalExternalBindingFailure context template witnesses certificate := by
  rcases checkedTerminal_implies_semanticFold_or_namedFailure
      noZeroDivisors context template witnesses certificate checked
      terminalWitnesses terminal with semantic | yRing | bad | binding
  · rcases semantic with ⟨_, realized⟩
    exact Or.inl realized.paper
  · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl bad))
  · exact Or.inr (Or.inr (Or.inr binding))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessProduction
