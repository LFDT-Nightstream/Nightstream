import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionNifs
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality

/-!
Terminal closure for the final delayed packed-`yZcol` output.

Assurance tier: model-level, pending concrete terminal-row refinement.

Owns: a terminal relation over the final authoritative child openings;
exact all-54-lane evaluation of their radix recomposition at the carried old
block point; reduction of alternate child openings to a fresh-bound
commitment collision; and promotion of the final delayed refinement to the
independent semantic fold.

Does not own: the concrete terminal witness decoder, exact terminal A/B/C
rows, Ajtai hardness, Poseidon2, Rust conformance, costs, or row removal.

Emits constraints: none.

Authority boundary: the terminal relation consumes complete private
assignments which genuinely open the actual final public children. It never
reads child `CeClaim.y_zcol` sidecars. The final projection is an exact vector
equation rather than a digest or a sampled equality, so terminal closure adds
no new projection-root event.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.production.terminal.children` | every terminal raw assignment opens the actual final child statement | checked/security boundary | `childrenCheck`, `Accepted.children` |
| `nifs.production.terminal.projection` | all 54 pending lanes equal the old-point projection of the radix-recomposed raw children | checked | `projectionCheck`, `Accepted.projection` |
| `nifs.production.terminal.binding` | terminal openings equal canonical previous splits or expose one child collision | derived/security boundary | `accepted_implies_previousSemanticFold_or_badEvent` |
| `nifs.production.terminal.semantic` | close the final delayed output and obtain the independent fold | derived | same theorem |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
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
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact authority facts checked for one verifier-owned raw child assignment.
This compact proposition avoids making the executable checker normalize the
entire generic `CE.Holds` relation merely to expose its four concrete fields. -/
structure ChildAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (assignment : Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits))
    (child : Fin productionGlobalParams.k) : Prop where
  commits :
    commit context.key assignment =
      (outputChildren context certificate child).commitment
  publicInput :
    Phi81Relation.projectPublicInput assignment =
      (outputChildren context certificate child).publicInput
  norm :
    Phi81Relation.assignmentNormBounded
      ((outputChildren context certificate child).stage.bound
        productionGlobalParams)
      assignment
  evaluations :
    Phi81Relation.evaluations
        (outputChildren context certificate child).constraintSystem
        assignment
        (outputChildren context certificate child).point =
      (outputChildren context certificate child).evaluations

/-- Final-decider relation. `rawChildren` are decoded witness assignments,
not public sidecar evaluations. -/
structure Accepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Prop where
  children : forall child,
    ChildAccepted context certificate (rawChildren child) child
  projection :
    (DelayedProduction.outgoingPending context certificate).parentYZcol =
      PackedBlockAction.packedYZcol context.covers
        (PiDEC.Raw.recomposeAssignment rawChildren)
        (DelayedProduction.outgoingPending context certificate).oldBlock

/-- Minimal terminal obligations used by the delayed-`y_zcol` authority
track.  The raw children are the same assignments opened against the ordered
PiDEC output commitments; no child evaluation sidecar occurs.  Unlike
`Accepted`, this contract deliberately excludes public-input, `y_ring`, and
ordinary CCS-evaluation obligations, which are owned by the independent paper
track. -/
structure ProjectionOpeningAccepted
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Prop where
  childCommitment : forall child,
    commit context.key (rawChildren child) =
      (outputChildren context certificate child).commitment
  childNorm : forall child,
    Phi81Relation.assignmentNormBounded
      ((outputChildren context certificate child).stage.bound
        productionGlobalParams)
      (rawChildren child)
  projection :
    (DelayedProduction.outgoingPending context certificate).parentYZcol =
      PackedBlockAction.packedYZcol context.covers
        (PiDEC.Raw.recomposeAssignment rawChildren)
        (DelayedProduction.outgoingPending context certificate).oldBlock

theorem Accepted.projectionOpeningAccepted
    {context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows}
    {certificate : FixedActive.Certificate context}
    {rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)}
    (accepted : Accepted context certificate rawChildren) :
    ProjectionOpeningAccepted context certificate rawChildren := by
  exact {
    childCommitment := fun child => (accepted.children child).commits
    childNorm := fun child => (accepted.children child).norm
    projection := accepted.projection
  }

/-- Executable complete-carrier norm check. -/
def assignmentNormCheck
    {relationShape : Phi81Relation.Shape}
    (bound : Nat)
    (assignment : Phi81Relation.Assignment relationShape) : Bool :=
  (List.finRange relationShape.carrierWidth).all fun column =>
    decide (centeredMagnitude (assignment column) < bound)

theorem assignmentNormCheck_eq_true_iff
    {relationShape : Phi81Relation.Shape}
    (bound : Nat)
    (assignment : Phi81Relation.Assignment relationShape) :
    assignmentNormCheck bound assignment = true <->
      Phi81Relation.assignmentNormBounded bound assignment := by
  constructor
  · intro checked column
    exact of_decide_eq_true
      ((List.all_eq_true.mp checked) column (by simp))
  · intro bounded
    apply List.all_eq_true.mpr
    intro column _member
    exact decide_eq_true (bounded column)

/-- Exact finite terminal opening check for one verifier-owned child. -/
def childCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (child : Fin productionGlobalParams.k) : Bool :=
  let statement := outputChildren context certificate child
  let assignment := rawChildren child
  commitmentEqual (commit context.key assignment) statement.commitment &&
    (publicInputEqual (Phi81Relation.projectPublicInput assignment)
        statement.publicInput &&
      (assignmentNormCheck (statement.stage.bound productionGlobalParams)
          assignment &&
        evaluationsEqual
          (Phi81Relation.evaluations statement.constraintSystem assignment
            statement.point)
          statement.evaluations))

theorem childCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (child : Fin productionGlobalParams.k) :
    childCheck context certificate rawChildren child = true <->
      ChildAccepted context certificate (rawChildren child) child := by
  simp only [childCheck, Bool.and_eq_true, commitmentEqual_eq_true_iff,
    publicInputEqual_eq_true_iff, assignmentNormCheck_eq_true_iff,
    evaluationsEqual_eq_true_iff]
  constructor
  · rintro ⟨commits, publicInput, norm, evaluations⟩
    exact ⟨commits, publicInput, norm, evaluations⟩
  · intro accepted
    exact ⟨accepted.commits, accepted.publicInput, accepted.norm,
      accepted.evaluations⟩

/-- Check all fourteen children in canonical index order. -/
def childrenCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Bool :=
  (List.finRange productionGlobalParams.k).all fun child =>
    childCheck context certificate rawChildren child

theorem childrenCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    childrenCheck context certificate rawChildren = true <->
      forall child,
        ChildAccepted context certificate (rawChildren child) child := by
  constructor
  · intro checked child
    exact (childCheck_eq_true_iff context certificate rawChildren child).mp
      ((List.all_eq_true.mp checked) child (List.mem_finRange child))
  · intro children
    apply List.all_eq_true.mpr
    intro child _member
    exact (childCheck_eq_true_iff context certificate rawChildren child).mpr
      (children child)

/-- Exact executable owner of the terminal delayed projection. This check is
separate from child CE opening authority because the native terminal verifier
already owns the latter, while production still must add this recomposition
comparison over those same raw opened assignments. -/
def projectionCheck
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Bool :=
  ringKEqual
    (DelayedProduction.outgoingPending context certificate).parentYZcol
    (PackedBlockAction.packedYZcol context.covers
      (PiDEC.Raw.recomposeAssignment rawChildren)
      (DelayedProduction.outgoingPending context certificate).oldBlock)

theorem projectionCheck_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    projectionCheck context certificate rawChildren = true <->
      (DelayedProduction.outgoingPending context certificate).parentYZcol =
        PackedBlockAction.packedYZcol context.covers
          (PiDEC.Raw.recomposeAssignment rawChildren)
          (DelayedProduction.outgoingPending context certificate).oldBlock := by
  exact ringKEqual_eq_true_iff _ _

/-- Canonical executable terminal check. Every child is checked against its
complete raw assignment, and the final delayed vector is recomputed from the
same ordered assignments. No child sidecar or digest supplies either result. -/
def check
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) : Bool :=
  childrenCheck context certificate rawChildren &&
    projectionCheck context certificate rawChildren

/-- The executable terminal accepts exactly the independently stated terminal
relation. In particular, callers no longer provide `Accepted` as evidence. -/
theorem check_eq_true_iff
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits)) :
    check context certificate rawChildren = true <->
      Accepted context certificate rawChildren := by
  simp only [check, Bool.and_eq_true, childrenCheck_eq_true_iff,
    projectionCheck_eq_true_iff]
  constructor
  · rintro ⟨children, projection⟩
    exact ⟨children, projection⟩
  · intro accepted
    exact ⟨accepted.children, accepted.projection⟩

/-- Success-path spelling used by active terminal composition. -/
theorem accepted_of_check
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : check context certificate rawChildren = true) :
    Accepted context certificate rawChildren :=
  (check_eq_true_iff context certificate rawChildren).mp accepted

/-- Composition seam for the actual terminal implementation: native full-CE
opening validation supplies `childrenCheck`, while the delayed terminal path
must execute `projectionCheck` over the identical raw child assignments. -/
theorem accepted_of_component_checks
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (certificate : FixedActive.Certificate context)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (children : childrenCheck context certificate rawChildren = true)
    (projection : projectionCheck context certificate rawChildren = true) :
    Accepted context certificate rawChildren := by
  exact {
    children := (childrenCheck_eq_true_iff context certificate rawChildren).mp
      children
    projection := (projectionCheck_eq_true_iff context certificate
      rawChildren).mp projection
  }

/-- The final raw openings bind the current packed output before any
claims-to-source extraction is attempted.  If the verifier-owned openings are
not the canonical child assignments, the two openings of one indexed child
exhibit the commitment-binding failure directly. -/
theorem accepted_implies_packedYZcolBound_or_badEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (children : ChildOpenings context data certificate)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : Accepted context certificate rawChildren) :
    Terminal.PackedYZcolBoundAtBlock context.covers data
        (derive context certificate).piCcs.ncPoint.block
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate) ∨
      (∃ child, Nonempty
        (Opening.BindingCollision (semantics context.key)
          productionGlobalParams.b
          (outputChildren context certificate child).commitment)) := by
  classical
  let canonical :=
    DelayedRawChildren.previousChildAssignment context data certificate
  by_cases childrenEq : rawChildren = canonical
  · have recomposes :
        PiDEC.Raw.recomposeAssignment rawChildren =
          PackedYZcol.canonicalParentAssignment context data certificate := by
      rw [childrenEq]
      simpa [canonical, PackedYZcol.canonicalParentAssignment,
        SemanticFold.combinedAssignment, SemanticFold.assignments,
        CertificateRefinement.semanticWitness] using
        (DelayedRawChildren.previousChildAssignments_recompose context data
          certificate)
    rcases DelayedProduction.packedBound_or_mixingCollision_of_rawRecomposition
        context data certificate rawChildren recomposes accepted.projection with
      packed | mixingCollision
    · exact Or.inl (by simpa using packed)
    · exact Or.inr (Or.inl mixingCollision)
  · apply Or.inr
    apply Or.inr
    have differs : ∃ child, rawChildren child ≠ canonical child := by
      exact Classical.byContradiction fun noDifference => childrenEq (by
        funext child
        exact Classical.byContradiction fun different =>
          noDifference ⟨child, different⟩)
    rcases differs with ⟨child, different⟩
    have rawAccepted := accepted.children child
    have canonicalHolds :
        CE.Holds (semantics context.key) productionGlobalParams
          (outputChildren context certificate child) (canonical child) := by
      simpa [canonical, DelayedRawChildren.previousChildAssignment] using
        children child
    exact ⟨child, ⟨{
      leftOpening := rawChildren child
      rightOpening := canonical child
      leftCommits := rawAccepted.commits
      rightCommits := canonicalHolds.1.1
      leftNorm := by
        simpa [production_norm_stages.1] using rawAccepted.norm
      rightNorm := by
        simpa [production_norm_stages.1] using canonicalHolds.1.2.2
      different := different
    }⟩⟩

/-- The minimal raw-opening/projection obligations close the direct-parent
edge.  Only ordered child commitments/norms and the raw terminal projection
are consumed; public-input, `y_ring`, and carried child `y_zcol` values are
irrelevant to this authority track. -/
theorem projectionOpeningAccepted_of_parentOpening_implies_packedYZcolBound_or_badEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (parentBound : DelayedRawChildren.CanonicalParentBinding context data
      certificate)
    (piDecAccepted : PiDEC.Accepted (decAlgebra context.key)
      ((derive context certificate).piDecAttempt certificate))
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : ProjectionOpeningAccepted context certificate rawChildren) :
    Terminal.PackedYZcolBoundAtBlock context.covers data
        (derive context certificate).piCcs.ncPoint.block
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics context.key) productionGlobalParams
        (derive context certificate).piRlcOutput.commitment) := by
  have childCommitments : forall child,
      (semantics context.key).commit (rawChildren child) =
        (((derive context certificate).piDecAttempt certificate).children
          child).commitment := by
    intro child
    simpa [outputChildren] using accepted.childCommitment child
  have childNorms : forall child,
      (semantics context.key).normBounded productionGlobalParams.b
        (rawChildren child) := by
    intro child
    simpa [outputChildren, production_norm_stages.1] using
      accepted.childNorm child
  rcases
      DelayedRawChildren.rawChildren_recompose_eq_canonicalParent_or_bindingCollision
        context data certificate piDecAccepted parentBound rawChildren
        childCommitments childNorms with
    recomposesCanonical | bindingCollision
  ·
    rcases DelayedProduction.packedBound_or_mixingCollision_of_rawRecomposition
        context data certificate rawChildren recomposesCanonical
        accepted.projection with packed | mixing
    · exact Or.inl (by simpa using packed)
    · exact Or.inr (Or.inl mixing)
  · exact Or.inr (Or.inr bindingCollision)

/-- Terminal direct-parent closure without a separate canonical child-opening
family.  The terminal checker validates every ordered raw child assignment
against the actual public Π_DEC child. Strict Π_DEC, the canonical parent
commitment/norm, and only the raw child commitments/norms therefore bind the
radix recomposition to the source/challenge parent assignment, or expose one
standard parent-opening collision.

No child `y_zcol` sidecar or desired packed equation occurs in the premises. -/
theorem accepted_of_parentOpening_implies_packedYZcolBound_or_badEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (parentBound : DelayedRawChildren.CanonicalParentBinding context data
      certificate)
    (piDecAccepted : PiDEC.Accepted (decAlgebra context.key)
      ((derive context certificate).piDecAttempt certificate))
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : Accepted context certificate rawChildren) :
    Terminal.PackedYZcolBoundAtBlock context.covers data
        (derive context certificate).piCcs.ncPoint.block
        certificate.piCcs.output ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics context.key) productionGlobalParams
        (derive context certificate).piRlcOutput.commitment) := by
  exact
    projectionOpeningAccepted_of_parentOpening_implies_packedYZcolBound_or_badEvent
      context data certificate parentBound piDecAccepted rawChildren
      accepted.projectionOpeningAccepted

/-- Exact terminal closure yields the previous semantic fold, a `Pi_RLC`
source-mixing collision, or one indexed terminal child commitment collision. -/
theorem accepted_implies_previousSemanticFold_or_badEvent
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (refinement : CombinedNc.ProductionNifs.DelayedRefinement context data
      certificate)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (accepted : Accepted context certificate rawChildren) :
    SemanticFold.Holds context data
        (derive context certificate).piRlcOutput
        (outputChildren context certificate) ∨
      PiRlcSidecar.MixingCollision context.covers
        certificate.piRlcChallenges
        (InputAuthority.productAssignments data context.alignment)
        (DelayedProduction.outgoingPending context certificate).oldBlock
        (PackedYZcol.sourceClaims context certificate) ∨
      (∃ child, Nonempty
        (Opening.BindingCollision (semantics context.key)
          productionGlobalParams.b
          (outputChildren context certificate child).commitment)) := by
  rcases accepted_implies_packedYZcolBound_or_badEvent context data certificate
      refinement.children rawChildren accepted with packed | mixing | binding
  · exact Or.inl (refinement.toSemanticFold packed)
  · exact Or.inr (Or.inl mixing)
  · exact Or.inr (Or.inr binding)

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionTerminal
