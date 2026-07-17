import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.PublicInputBridge
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PiRlc.Reduction.Profiles
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.Certificate

/-!
Fixed-carrier diagnostic PiRLC profiles compose independently owned
challenge wiring, equation wiring, and reduction. They pass only the active-X
widths and parent-X column equality needed to cross the typed 270-coordinate
public-input boundary into strict PiDEC.

Assurance tier: artifact-checked. Generated profile facts are explicit theorem
premises or guarded `native_decide` results; no Rust-conformant or
security-reduced end-to-end claim is made.

Owns: profile-level equation refinement; the exact 29-public-trace
bad-root boundary; the terminal sampler-to-canonically-constructed challenge
handoff; and typed PiRLC-to-PiDEC public-input recomposition.

Does not own: the two delayed-NC identities, commitment or evaluation
composition, private CE membership, PiCCS output or transcript authority,
challenge membership, PiRLC/full NIFS/F-prime acceptance, bad-event
probability, Rust conformance, costs, or row removal.

Emits constraints: no.

Authority boundary: the parent is the structurally relabeled strict-PiDEC
claim, never a digest. Challenge columns are structural dataflow; membership
must be derived later by a source-bound NIFS sampler. Projection failure
remains exactly a bad root among the 29 public identities.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.verify.identities.public` | challenge wiring, equation-only carrier wiring, and Phi81 reduction form one equation refinement | derived | `recursiveEquationRefinement_or_badRoot`, `terminalEquationRefinement_or_badRoot` |
| `nifs.pi_rlc.challenge` | accepted terminal sampler rows construct the semantic challenge field and establish exact batch-column equality | computed/checked | `terminalSampler_refines_decodedBatchChallenges` |
| `nifs.pi_rlc.verify.identities.x` | typed Phi81 parent equals radix recomposition of strict-PiDEC children | derived | `TypedPublicInputComposition` |
| `nifs.pi_rlc.verify.identities.x` | recursive profile implies the typed equation or a public bad root | security boundary | `recursiveTypedComposition_or_badRoot` |
| `nifs.pi_rlc.verify.identities.x` | terminal profile implies the typed equation or a public bad root | security boundary | `terminalTypedComposition_or_badRoot` |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.ProjectionCheck
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection
open Nightstream.Implementation.R1CS.FPrimeFullHistoryPiDec
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.PublicInputBridge

set_option maxRecDepth 16384
set_option maxHeartbeats 1000000

/-! ## Profile refinement -/

/-- The independently owned recursive equation artifacts form one refinement,
or one of the same 29 public identities is a bad root. -/
theorem recursiveEquationRefinement_or_badRoot
    {assignment : Nat → Nat}
    (canonical :
      PiRlcChallenge.Transcript.ChunkOrder.CanonicalAssignment assignment)
    (constantOne : assignment 0 = 1)
    (projectionRows :
      Reduction.Profiles.PublicHolds RecursiveSamplerArtifact.tree assignment) :
    EquationRefinement assignment RecursiveCarrierArtifact.columns
        RecursiveSamplerArtifact.tree ∨
      BatchBadRoot K.ops (BatchIdentity
        RecursiveSamplerArtifact.tree.flatten assignment) := by
  rcases Reduction.Profiles.recursiveReduction_or_badRoot canonical constantOne
      projectionRows with reduction | badRoot
  · exact Or.inl {
      challengeWiring := RecursiveCarrierArtifact.challengeWiringArtifact
      wiring := RecursiveCarrierArtifact.equationWiringArtifact
      reduction := reduction
    }
  · exact Or.inr badRoot

/-- Terminal counterpart of `recursiveEquationRefinement_or_badRoot`. -/
theorem terminalEquationRefinement_or_badRoot
    {assignment : Nat → Nat}
    (canonical :
      PiRlcChallenge.Transcript.ChunkOrder.CanonicalAssignment assignment)
    (constantOne : assignment 0 = 1)
    (projectionRows :
      Reduction.Profiles.PublicHolds TerminalSamplerArtifact.tree assignment) :
    EquationRefinement assignment TerminalCarrierArtifact.columns
        TerminalSamplerArtifact.tree ∨
      BatchBadRoot K.ops (BatchIdentity
        TerminalSamplerArtifact.tree.flatten assignment) := by
  rcases Reduction.Profiles.terminalReduction_or_badRoot canonical constantOne
      projectionRows with reduction | badRoot
  · exact Or.inl {
      challengeWiring := TerminalCarrierArtifact.challengeWiringArtifact
      wiring := TerminalCarrierArtifact.equationWiringArtifact
      reduction := reduction
    }
  · exact Or.inr badRoot

/-! ## Context-bound terminal challenge authority -/

/-- Accepted terminal sampler rows construct the semantic challenge field,
establish the independent context-bound predicate, and identify every
challenge used by the fixed public PiRLC equations.

This theorem does not promote `Equations` to full PiRLC or NIFS acceptance;
the upstream message/post-NC bindings and all remaining source and transition
bindings stay explicit at their owners. -/
theorem terminalSampler_refines_decodedBatchChallenges
    (prime : EuclidPrime goldilocksP)
    {shape : PiCCS.SplitNc.SemanticShape}
    {domain : PiCCS.SplitNc.FlatNcDomain}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      ConcretePhi81.Context shape domain PiRlcChallenge.TranscriptMachine.State
        publicRingColumns publicFits verifierRows terminalArity}
    {certificate :
      ConcretePhi81.Certificate (domain := domain) (arity := terminalArity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    (profile : PiCcsOutputDigest.Projection.SplitNc.Profile shape)
    (contextBinding :
      PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.ContextBinding
        profile context)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (constantOne : assignment 0 = 1)
    (outputAccepted :
      FPrimeFullHistoryTerminalPiCcsOutputMessageHashes.Accepted assignment)
    (catchupAccepted :
      FPrimeFullHistoryTerminalPiCcsCatchup.Accepted assignment)
    (rlcAccepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (messageBound :
      PiCcsOutputDigest.SemanticHandoff.MessageBound
        profile certificate.piCcs.output assignment canonical)
    (postNcBoundary :
      PiCcsOutputDigest.SemanticHandoff.CatchupInputBound
        (PiCCS.SplitNc.Verifier.Protocol.derive context.feMachine
          context.ncMachine context.initialState certificate.piCcs).finalState
        assignment canonical) :
    ConcretePhi81.Sampler.CertificateAccepted context
        (PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.withDecodedChallenges
          assignment certificate) ∧
      forall index,
        (PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.withDecodedChallenges
          assignment certificate).piRlcChallenges index =
          ringOfList (values assignment
            (TerminalCarrierArtifact.columns.challenges index)) := by
  constructor
  · exact
      PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.accepted_refines_withDecodedChallenges
        prime certificate profile contextBinding canonical constantOne
        outputAccepted catchupAccepted rlcAccepted messageBound postNcBoundary
  · intro index
    exact
      PiRlcChallenge.Sampler.Refinement.Terminal.Certificate.withDecodedChallenges_challenge_eq_columns
        assignment certificate index

/-! ## Typed public-input composition -/

/-- Exact typed public-input equation shared by both fixed profiles. -/
def TypedPublicInputComposition
    {matrixCount : Nat}
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat → Nat)
    (columns : BatchColumns params arity matrixCount)
    (columnMap : List Nat) (dimensions : Dimensions) : Prop :=
  PublicInput.combinePublicInputs
      (fun index => ringOfList (values assignment (columns.challenges index)))
      (fun index =>
        decodeXRings (dimensions := dimensions)
          (decodeOpening assignment (columns.inputs index)).x) =
    PiDECAlgebra.PublicInput.recomposePublicInput fun index =>
      PiDecBridge.decodedPublicInput dimensions
        (Relabel.assignment columnMap assignment) (childLayout index)

/-- Recursive profile refinement and strict PiDEC imply the typed
parent/children equation, or the precisely scoped public bad root. -/
theorem recursiveTypedComposition_or_badRoot
    {assignment : Nat → Nat}
    (canonical :
      PiRlcChallenge.Transcript.ChunkOrder.CanonicalAssignment assignment)
    (constantOne : assignment 0 = 1)
    (projectionRows :
      Reduction.Profiles.PublicHolds RecursiveSamplerArtifact.tree assignment)
    (dimensions : Dimensions)
    (piDecAccepted : PiDecStrictCompiler.Accepted layout
      (Relabel.assignment recursiveColumnMap assignment)) :
    TypedPublicInputComposition assignment RecursiveCarrierArtifact.columns
        recursiveColumnMap dimensions ∨
      BatchBadRoot K.ops (BatchIdentity
        RecursiveSamplerArtifact.tree.flatten assignment) := by
  rcases recursiveEquationRefinement_or_badRoot canonical constantOne
      projectionRows with refinement | badRoot
  · apply Or.inl
    exact typedPiRlcPiDecPublicInputComposition_relabel
      assignment RecursiveCarrierArtifact.columns RecursiveSamplerArtifact.tree
      refinement
      (fun block => RecursiveCarrierArtifact.outputWidth (.x block))
      RecursiveCarrierArtifact.parentArtifact.x recursiveColumnMap rfl
      dimensions piDecAccepted
  · exact Or.inr badRoot

/-- Terminal counterpart of `recursiveTypedComposition_or_badRoot`. -/
theorem terminalTypedComposition_or_badRoot
    {assignment : Nat → Nat}
    (canonical :
      PiRlcChallenge.Transcript.ChunkOrder.CanonicalAssignment assignment)
    (constantOne : assignment 0 = 1)
    (projectionRows :
      Reduction.Profiles.PublicHolds TerminalSamplerArtifact.tree assignment)
    (dimensions : Dimensions)
    (piDecAccepted : PiDecStrictCompiler.Accepted layout
      (Relabel.assignment terminalColumnMap assignment)) :
    TypedPublicInputComposition assignment TerminalCarrierArtifact.columns
        terminalColumnMap dimensions ∨
      BatchBadRoot K.ops (BatchIdentity
        TerminalSamplerArtifact.tree.flatten assignment) := by
  rcases terminalEquationRefinement_or_badRoot canonical constantOne
      projectionRows with refinement | badRoot
  · apply Or.inl
    exact typedPiRlcPiDecPublicInputComposition_relabel
      assignment TerminalCarrierArtifact.columns TerminalSamplerArtifact.tree
      refinement
      (fun block => TerminalCarrierArtifact.outputWidth (.x block))
      TerminalCarrierArtifact.parentArtifact.x terminalColumnMap rfl
      dimensions piDecAccepted
  · exact Or.inr badRoot

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc.Profiles
