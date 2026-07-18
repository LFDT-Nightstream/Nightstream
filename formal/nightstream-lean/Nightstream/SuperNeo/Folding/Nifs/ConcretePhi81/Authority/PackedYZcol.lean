import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PiRlcParentOpening
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Transition
import Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

/-!
Packed `yZcol` authority at the concrete Phi81 NIFS boundary.

Protocol: SuperNeo NIFS.
Phases: canonical block×lane `Pi_CCS` output followed by `Pi_RLC`.
Constraint families: source/product indexing, sidecar aggregation, and one
parent-projection equality target; this file emits no rows.

Assurance tier: model-level.

Owns: exact transport between semantic source order and public-product order;
the typed delayed-projection premises needed to use the packed `Pi_RLC`
sidecar reduction; and replacement of the broad `yZcol`-unbound outcome by a
named `Pi_RLC` mixing collision, degree-53 projection bad root, or explicit
parent-projection scalar mismatch.

Does not own: parent/child opening extraction, the independently justified
equality that must establish the parent projection by direct recomputation or
a sound opening, transcript timing, collision probability, Poseidon2,
Rust/R1CS refinement, costs, or row removal.

Emits constraints: no.

Authority boundary: the source-side scalar equality does not promote a digest,
aggregate, or raw scalar to authority. The source aggregate, parent assignment,
and its packed projection are computed. The theorem returns
`ParentProjectionMismatch` unless a later refinement establishes the claimed
parent projection from authoritative data. One-point acceptance yields exact packed
equality only outside a named bad root; `Pi_DEC` is not part of this obligation.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.output.packed_y_zcol.index` | semantic and public-product source orders are exact inverses | derived | `sourceBound_iff_packedYZcolBound` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.aggregate` | compute the source sidecar fold directly from claims and challenges | computed | `DelayedPackedProjection.sourceAggregate` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_assignment` | compute the parent assignment directly from sources and challenges | computed | `canonicalParentAssignment` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_assignment.raw` | independent raw fold equals the canonical semantic parent | derived | `rawSourceFold_eq_canonicalParentAssignment` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_claim` | compute the parent packed projection and its exact source-derived aggregate | computed / derived | `canonicalParentClaim`, `canonicalParentClaim_eq_sourceAggregate` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.parent_scalar` | claimed scalar equals the canonical parent projection | semantic target | `ParentProjectionMatches`, `ParentProjectionMismatch` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.identity` | the claimed scalar equals the computed source projection | checked premise | `SourceProjectionMatches` |
| `nifs.pi_rlc.verify.authority.packed_y_zcol.soundness` | source binding, mixing collision, projection bad root, or parent mismatch | derived | `packedYZcolBound_or_mixingCollision_or_badRoot_or_parentProjectionMismatch` |
| `nifs.concrete.soundness.output_partition` | conditionally refine accepted NIFS semantics, or expose `yRing`, mixing, projection mismatch, or `Pi_CCS` failure | conditional derived result | `accepted_implies_refinement_or_yRingUnbound_or_mixingCollision_or_projectionBadRoot_or_parentProjectionMismatch_or_badEvent` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Read one semantic output message in exact public-product source order. -/
def claimsInProductOrder
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    (alignment : SourceAlignment shape params arity)
    (message : OutputMessage shape) : Fin arity.total -> RingK :=
  fun source => message.yZcol (alignment.semanticIndex source)

/-- Product-order source binding is exactly the canonical block×lane
`Pi_CCS` output-binding predicate. No source is omitted or duplicated. -/
theorem sourceBound_iff_packedYZcolBound
    {shape : SemanticShape}
    {params : GlobalParams}
    {arity : BatchArity params}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (alignment : SourceAlignment shape params arity)
    (block : CubePoint K domain.blockVariables)
    (message : OutputMessage shape) :
    PiRlcSidecar.SourceBound covers
        (InputAuthority.productAssignments data alignment) block
        (claimsInProductOrder alignment message) <->
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        covers data block message := by
  constructor
  · intro bound source lane
    have atProduct := congrFun (bound (alignment.productIndex source)) lane
    simpa [claimsInProductOrder, InputAuthority.productAssignments] using
      atProduct
  · intro bound source
    funext lane
    simpa [claimsInProductOrder, InputAuthority.productAssignments] using
      bound (alignment.semanticIndex source) lane

section

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- The exact packed sidecar values consumed by the active product-order
`Pi_RLC` fold. -/
def sourceClaims
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    Fin arity.total -> RingK :=
  claimsInProductOrder context.alignment certificate.piCcs.output

/-- Reuse the canonical semantic parent assignment. This is private semantic
dataflow and is not added to the public execution carrier. -/
abbrev canonicalParentAssignment
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) :=
  SemanticFold.combinedAssignment context data
    (CertificateRefinement.semanticWitness certificate)

/-- The independent width-only fold used by packed sidecar semantics is
exactly the canonical typed NIFS parent assignment. -/
theorem rawSourceFold_eq_canonicalParentAssignment
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    PiRLCFinite.Raw.combineAssignments certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment) =
      canonicalParentAssignment context data certificate := by
  simpa [canonicalParentAssignment, SemanticFold.combinedAssignment,
    SemanticFold.assignments, CertificateRefinement.semanticWitness,
    rlcAlgebra] using
    (PiRLCFinite.raw_combineAssignments_eq
      (shape := RelationShape shape publicRingColumns publicFits)
      certificate.piRlcChallenges
      (InputAuthority.productAssignments data context.alignment))

/-- Canonical packed projection of the already-computed `Pi_RLC` parent
assignment. No new parent sidecar is added to the public claim. -/
def canonicalParentClaim
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    RingK :=
  PackedBlockAction.packedYZcol context.covers
    (canonicalParentAssignment context data certificate)
    (derive context certificate).piCcs.ncPoint.block

/-- The canonical parent projection is exactly the `Pi_RLC` aggregate of the
source-derived packed projections. -/
theorem canonicalParentClaim_eq_sourceAggregate
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput) :
    canonicalParentClaim context data certificate =
      PiRLCFinite.combineEvaluation certificate.piRlcChallenges fun source =>
        PackedBlockAction.packedYZcol context.covers
          (InputAuthority.productAssignments data context.alignment source)
          (derive context certificate).piCcs.ncPoint.block := by
  unfold canonicalParentClaim
  rw [← rawSourceFold_eq_canonicalParentAssignment context data certificate]
  exact PackedBlockAction.Finite.packedYZcol_combine context.covers
    certificate.piRlcChallenges
    (InputAuthority.productAssignments data context.alignment)
    (derive context certificate).piCcs.ncPoint.block

/-- The claimed scalar equals the canonical parent projection. This is the
semantic correctness target for a future opening proof; it is not itself
commitment-binding evidence. -/
def ParentProjectionMatches
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput)
    (claimedParentProjection producerBeta : K) : Prop :=
  DelayedPackedProjection.PairRightScalarMatches
    (canonicalParentClaim context data certificate)
    claimedParentProjection producerBeta

/-- The claimed scalar does not equal the canonical parent projection. -/
def ParentProjectionMismatch
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (data : Data shape)
    (certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput)
    (claimedParentProjection producerBeta : K) : Prop :=
  ¬ ParentProjectionMatches context data certificate claimedParentProjection
      producerBeta

/-- Deterministic packed-output authority reduction for active NIFS indexing.
Every false source vector is retained as either the named `Pi_RLC` mixing
collision, the delayed projection's degree-53 bad root, or a claimed-parent
projection mismatch. -/
theorem packedYZcolBound_or_mixingCollision_or_badRoot_or_parentProjectionMismatch
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    {claimedParentProjection : K}
    {producerBeta : K}
    (sourceProjectionMatches :
      DelayedPackedProjection.SourceProjectionMatches
        certificate.piRlcChallenges (sourceClaims context certificate)
        claimedParentProjection producerBeta) :
    Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        context.covers data
        (derive context certificate).piCcs.ncPoint.block
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (derive context certificate).piCcs.ncPoint.block
        (sourceClaims context certificate) ∨
      ProjectionCheck.BadRoot DelayedPackedProjection.projectionOps
        (DelayedPackedProjection.pairIdentity
          (DelayedPackedProjection.sourceAggregate certificate.piRlcChallenges
            (sourceClaims context certificate))
          (canonicalParentClaim context data certificate) producerBeta) ∨
      ParentProjectionMismatch context data certificate claimedParentProjection
        producerBeta := by
  by_cases parentMatches : ParentProjectionMatches context data certificate
      claimedParentProjection producerBeta
  · have pairAccepted : DelayedPackedProjection.PairAccepted
        (DelayedPackedProjection.sourceAggregate certificate.piRlcChallenges
          (sourceClaims context certificate))
        (canonicalParentClaim context data certificate) producerBeta :=
    DelayedPackedProjection.pairAccepted_of_scalar_matches
        (DelayedPackedProjection.sourceAggregate certificate.piRlcChallenges
          (sourceClaims context certificate))
        (canonicalParentClaim context data certificate)
        claimedParentProjection producerBeta sourceProjectionMatches
        parentMatches
    rcases DelayedPackedProjection.pairAccepted_implies_exact_or_badRoot
        (DelayedPackedProjection.sourceAggregate certificate.piRlcChallenges
          (sourceClaims context certificate))
        (canonicalParentClaim context data certificate) producerBeta
        pairAccepted with aggregateEquality | badRoot
    · have canonicalAggregate : PiRlcSidecar.CanonicalAggregateEquality
          context.covers certificate.piRlcChallenges
          (InputAuthority.productAssignments data context.alignment)
          (derive context certificate).piCcs.ncPoint.block
          (sourceClaims context certificate) := by
        unfold PiRlcSidecar.CanonicalAggregateEquality
        exact aggregateEquality.trans
          (canonicalParentClaim_eq_sourceAggregate context data certificate)
      rcases
          (PiRlcSidecar.sourceBound_or_mixingCollision_iff_aggregateEquality
            context.covers certificate.piRlcChallenges
            (InputAuthority.productAssignments data context.alignment)
            (derive context certificate).piCcs.ncPoint.block
            (sourceClaims context certificate)).2 canonicalAggregate with
        sourceBound | collision
      · exact Or.inl <|
          (sourceBound_iff_packedYZcolBound context.covers data context.alignment
            (derive context certificate).piCcs.ncPoint.block
            certificate.piCcs.output).1 sourceBound
      · exact Or.inr (Or.inl collision)
    · exact Or.inr (Or.inr (Or.inl badRoot))
  · exact Or.inr (Or.inr (Or.inr parentMatches))

/-- Sharpen the physical NIFS soundness boundary when the source projection
check is available. The former undifferentiated output failure becomes either the
CE-owned `yRing` authority gap, one precise `Pi_RLC` mixing collision, or the
degree-53 delayed-projection bad root, while a false claimed parent projection
remains an explicit `ParentProjectionMismatch` outcome.

This theorem remains conditional on semantic input authority, canonical child
opening authority for the separate `Pi_DEC` transition, and the typed
source-projection check. It does not assume a parent opening: a scalar mismatch
is returned. This is not a production or
security-reduced result. -/
theorem accepted_implies_refinement_or_yRingUnbound_or_mixingCollision_or_projectionBadRoot_or_parentProjectionMismatch_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      Context shape State publicRingColumns publicFits verifierRows arity}
    {data : Data shape}
    {certificate :
      Certificate (arity := arity)
        publicRingColumns publicFits verifierRows context.piCcsInput}
    {claimedParentProjection : K}
    {producerBeta : K}
    (input : SemanticInput context data)
    (children : ChildOpenings context data certificate)
    (accepted : Accepted context certificate)
    (sourceProjectionMatches :
      DelayedPackedProjection.SourceProjectionMatches
        certificate.piRlcChallenges (sourceClaims context certificate)
        claimedParentProjection producerBeta) :
    CertificateRefinement context data certificate ∨
      ¬ PiRlcParentOpening.YRingBound context data certificate ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (derive context certificate).piCcs.ncPoint.block
        (sourceClaims context certificate) ∨
      ProjectionCheck.BadRoot DelayedPackedProjection.projectionOps
        (DelayedPackedProjection.pairIdentity
          (DelayedPackedProjection.sourceAggregate certificate.piRlcChallenges
            (sourceClaims context certificate))
          (canonicalParentClaim context data certificate) producerBeta) ∨
      ParentProjectionMismatch context data certificate claimedParentProjection
        producerBeta ∨
      PiCcsBadEvent context data certificate := by
  rcases
      packedYZcolBound_or_mixingCollision_or_badRoot_or_parentProjectionMismatch
        sourceProjectionMatches with
    packedBound | collision | badRoot | parentProjectionMismatch
  · by_cases yRing : PiRlcParentOpening.YRingBound context data certificate
    · have outputBound : OutputBound context data certificate := by
        exact ⟨yRing, packedBound⟩
      rcases accepted_implies_refinement_or_outputUnbound_or_badEvent
          noZeroDivisors input children accepted with
        refinement | unbound | bad
      · exact Or.inl refinement
      · exact False.elim (unbound outputBound)
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr bad))))
    · exact Or.inr (Or.inl yRing)
  · exact Or.inr (Or.inr (Or.inl collision))
  · exact Or.inr (Or.inr (Or.inr (Or.inl badRoot)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl parentProjectionMismatch))))

end

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol
