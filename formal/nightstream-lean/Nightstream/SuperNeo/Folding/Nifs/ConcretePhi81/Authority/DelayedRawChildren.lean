import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.PackedYZcol
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/-!
Cross-step raw-child authority for delayed packed `yZcol` checking.

Protocol: active concrete Phi81 NIFS.
Phase: next-step running inputs back to the previous `Pi_DEC` children.
Constraint family: commitment-opening continuity; this file emits no rows.

Assurance tier: model-level.

Owns: the exact production-order raw running assignments read from
`Sources.Data.runningAssignments`; genuine CE openings for those assignments;
the direct parent-recomposition-or-binding-collision reduction from strict
`Pi_DEC`; and the stronger assignment-equality-or-fresh-binding-collision
reduction when the same public child statement also has a previous canonical
opening.

Does not own: equality of consecutive public child statements, derivation of
paper truth or input binding from physical acceptance, the combined NC
SumCheck, delayed transcript challenges, packed projection algebra, Ajtai
hardness, Rust/R1CS refinement, costs, or row removal.

Emits constraints: none.

Authority boundary: the child assignment is always read from the next
independent source table. No `yZcol` field, output sidecar, digest, or
caller-provided projection is consulted. A public statement can identify that
assignment with a previous opening only outside the explicitly returned
fresh-bound commitment collision.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.delayed.raw_children.index` | map each fixed child to the exact active running-source coordinate | direct dataflow | `rawRunningAssignment` |
| `nifs.pi_ccs.nc.delayed.raw_children.opening` | the raw running assignment opens its public CE statement | derived from independent paper/input authority | `rawRunningAssignment_holds` |
| `nifs.pi_ccs.nc.delayed.raw_children.parent` | accepted strict `Pi_DEC` binds the raw-child recomposition to the previous combined opening | derived/security boundary | `rawRunningAssignments_recompose_eq_parent_or_bindingCollision` |
| `nifs.pi_ccs.nc.delayed.raw_children.binding` | two openings of the same fresh child agree or expose a binding collision | security boundary | `openings_eq_or_freshBindingCollision` |
| `nifs.pi_ccs.nc.delayed.raw_children.family` | recover every previous child assignment or one indexed collision | derived | `rawRunningAssignments_eq_or_freshBindingCollision` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState uPreviousState

section GenericOpening

variable
  {shape : Phi81Relation.Shape}
  {Assignment Commitment : Type}
  {semantics : RelationSemantics
    (Phi81Relation.Structure shape) Assignment
    (Phi81Relation.PublicInput shape) (Phi81Relation.Point shape)
    Phi81Relation.Evaluation Commitment}

/-- Two genuine openings of one exact fresh CE statement are equal, or they
are an ordinary `b`-bounded commitment-opening collision. Public-input and
evaluation agreement are intentionally not used as commitment authority. -/
theorem openings_eq_or_freshBindingCollision
    (statement : Phi81Relation.CEStatement shape Commitment)
    (left right : Assignment)
    (fresh : statement.stage = .fresh)
    (leftHolds : CE.Holds semantics productionGlobalParams statement left)
    (rightHolds : CE.Holds semantics productionGlobalParams statement right) :
    left = right ∨
      Nonempty (Opening.BindingCollision semantics productionGlobalParams.b
        statement.commitment) := by
  by_cases equal : left = right
  · exact Or.inl equal
  · apply Or.inr
    exact ⟨{
      leftOpening := left
      rightOpening := right
      leftCommits := leftHolds.1.1
      rightCommits := rightHolds.1.1
      leftNorm := by
        simpa [fresh, production_norm_stages.1] using leftHolds.1.2.2
      rightNorm := by
        simpa [fresh, production_norm_stages.1] using rightHolds.1.2.2
      different := equal
    }⟩

end GenericOpening

section FixedActive

variable
  {shape : SemanticShape}
  {State : Type uState}
  {PreviousState : Type uPreviousState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact raw assignment for one active running child. The two casts are
owned by `SourceAlignment` and the fixed-active arity; no list position or
prover sidecar participates. -/
def rawRunningAssignment
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (child : Fin productionGlobalParams.k) :
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) :=
  data.runningAssignments
    (context.alignment.semanticRunningIndex child)

/-- The complete ordered raw child family consumed by the production-native
delayed projection. -/
def rawRunningAssignments
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape) :
    Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits) :=
  fun child => rawRunningAssignment context data child

@[simp] theorem rawRunningAssignments_apply
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (child : Fin productionGlobalParams.k) :
    rawRunningAssignments context data child =
      data.runningAssignments
        (context.alignment.semanticRunningIndex child) := by
  rfl

/-- Exact parent facts consumed by the delayed `y_zcol` binding reduction.
This deliberately excludes public-input and evaluation-array equality. -/
def CanonicalParentBinding
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop :=
  (semantics context.key).commit
        (PackedYZcol.canonicalParentAssignment context data certificate) =
      (derive context certificate).piRlcOutput.commitment ∧
    (semantics context.key).normBounded productionGlobalParams.bigB
      (PackedYZcol.canonicalParentAssignment context data certificate)

/-- Commitment-only binding of the successor's exact raw running table to
the public running statements. No point, public-input, or evaluation sidecar
is part of this predicate. -/
def RawRunningCommitmentsBound
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape) : Prop :=
  ∀ child,
    (semantics context.key).commit
        (rawRunningAssignment context data child) =
      (context.input.running child).commitment

/-- Full semantic input authority implies the narrower raw-commitment fact,
but the delayed binding theorem below consumes only the latter. -/
theorem rawRunningCommitmentsBound_of_semanticInput
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (input : SemanticInput context data) :
    RawRunningCommitmentsBound context data := by
  intro child
  simpa [RawRunningCommitmentsBound, semantics, productSemantics,
    rawRunningAssignment] using (input.sources.running child).commitment

/-- NC truth alone supplies the fresh norm for one exact raw assignment. -/
theorem rawRunningAssignment_norm_of_ncTruth
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (ncTruth : Semantics.Nc.Truth data)
    (child : Fin productionGlobalParams.k) :
    (semantics context.key).normBounded productionGlobalParams.b
      (rawRunningAssignment context data child) := by
  change ∀ column,
    centeredMagnitude (rawRunningAssignment context data child column) <
      productionGlobalParams.b
  intro column
  change centeredMagnitude
      (data.runningAssignments
        (context.alignment.semanticRunningIndex child) column) <
    productionGlobalParams.b
  rw [show productionGlobalParams.b = 2 by rfl]
  rw [← congrFun (data.assignment_runningIndex
    (context.alignment.semanticRunningIndex child)) column]
  exact ncTruth
    (Data.runningIndex
      (context.alignment.semanticRunningIndex child)) column

/-- A genuine canonical parent opening implies exactly the commitment and
big-B norm facts used by the delayed binding reduction. -/
theorem canonicalParentBinding_of_ceHolds
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context)
    (holds : CE.Holds (semantics context.key) productionGlobalParams
      (derive context certificate).piRlcOutput
      (PackedYZcol.canonicalParentAssignment context data certificate)) :
    CanonicalParentBinding context data certificate := by
  constructor
  · exact holds.1.1
  · simpa [CanonicalParentBinding, PackedYZcol.canonicalParentAssignment,
      SemanticFold.combinedAssignment, SemanticFold.assignments,
      CertificateRefinement.semanticWitness, production_norm_stages.2]
      using holds.1.2.2

/-- Independent paper truth and the exact public/source input bridge make one
raw next-step running assignment a genuine opening of the actual public child
statement. -/
theorem rawRunningAssignment_holds
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : SemanticInput context data)
    (child : Fin productionGlobalParams.k) :
    CE.Holds (semantics context.key) productionGlobalParams
      (context.input.running child)
      (rawRunningAssignment context data child) := by
  simpa [rawRunningAssignment, semantics] using
    (InputAuthority.runningSource_holds publicRingColumns publicFits
      (commit context.key) data context.alignment context.input
      production_norm_stages.1 paper child (input.sources.running child))

/-- The actual public running child is visibly at the strict fresh stage.
This is derived from the source-input bridge, not from a carried stage label. -/
theorem runningChild_stage
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (input : SemanticInput context data)
    (child : Fin productionGlobalParams.k) :
    (context.input.running child).stage = .fresh :=
  (input.sources.running child).stage

/-- The current raw running assignment opens the actual child commitment and
has the exact fresh-stage norm once the independently checked NC relation is
known.  Unlike `rawRunningAssignment_holds`, this fact needs neither CCS
truth nor the carried evaluation array, so it can be used after the combined
NC check without assuming the paper conclusion being proved. -/
theorem rawRunningAssignment_commitment_and_norm
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (input : SemanticInput context data)
    (ncTruth : Semantics.Nc.Truth data)
    (child : Fin productionGlobalParams.k) :
    (semantics context.key).commit
          (rawRunningAssignment context data child) =
        (context.input.running child).commitment ∧
      (semantics context.key).normBounded productionGlobalParams.b
        (rawRunningAssignment context data child) := by
  constructor
  · exact rawRunningCommitmentsBound_of_semanticInput context data input child
  · exact rawRunningAssignment_norm_of_ncTruth context data ncTruth child

/-- If every actual next running statement also has a candidate previous
opening, the raw assignment family is exactly that candidate family, or one
precisely indexed child commitment has two distinct fresh-bound openings.

The premise is deliberately about the same `context.input.running child`.
The outer prior-link refinement must first derive that this statement is the
corresponding previous output child; a digest equality is insufficient. -/
theorem rawRunningAssignments_eq_or_freshBindingCollision
    (context : FixedActive.Context shape State publicRingColumns publicFits
      verifierRows)
    (data : Data shape)
    (input : SemanticInput context data)
    (ncTruth : Semantics.Nc.Truth data)
    (candidate : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (candidateHolds : forall child,
      CE.Holds (semantics context.key) productionGlobalParams
        (context.input.running child) (candidate child)) :
    rawRunningAssignments context data = candidate ∨
      ∃ child, Nonempty
        (Opening.BindingCollision (semantics context.key)
          productionGlobalParams.b
          (context.input.running child).commitment) := by
  classical
  by_cases familyEqual : rawRunningAssignments context data = candidate
  · exact Or.inl familyEqual
  · apply Or.inr
    have differs : ∃ child,
        rawRunningAssignments context data child ≠ candidate child := by
      exact Classical.byContradiction fun noDifference => familyEqual (by
        funext child
        exact Classical.byContradiction fun different =>
          noDifference ⟨child, different⟩)
    rcases differs with ⟨child, different⟩
    have rawAuthority := rawRunningAssignment_commitment_and_norm
      context data input ncTruth child
    exact ⟨child, ⟨{
      leftOpening := rawRunningAssignments context data child
      rightOpening := candidate child
      leftCommits := by
        simpa [rawRunningAssignments] using rawAuthority.1
      rightCommits := (candidateHolds child).1.1
      leftNorm := by
        simpa [rawRunningAssignments] using rawAuthority.2
      rightNorm := by
        simpa [runningChild_stage context data input child,
          production_norm_stages.1] using (candidateHolds child).1.2.2
      different := different
    }⟩⟩

/-! ## Consecutive-production specialization -/

/-- Canonical private assignment of one child emitted by the previous active
fold. This is computed from the previous source assignments and exact
`Pi_RLC` challenges; no child-side projection is part of the definition. -/
def previousChildAssignment
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (child : Fin productionGlobalParams.k) :
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) :=
  (decAlgebra previousContext.key).splitAssignment
    (PiRLC.combinedWitness (rlcAlgebra previousContext.key)
      previousCertificate.piRlcChallenges
      (InputAuthority.productAssignments previousData
        previousContext.alignment)) child

/-- Recombining all previous canonical child assignments recovers the exact
previous `Pi_RLC` private parent assignment. This is the production radix
theorem, not a commitment or public-field recomposition assumption. -/
theorem previousChildAssignments_recompose
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext) :
    Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
        (previousChildAssignment previousContext previousData
          previousCertificate) =
      PiRLC.combinedWitness (rlcAlgebra previousContext.key)
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment) := by
  rw [Phi81Relation.EvaluationHomomorphism.PiDEC.raw_recomposeAssignment_eq]
  exact (decAlgebra previousContext.key).split_recompose
    (PiRLC.combinedWitness (rlcAlgebra previousContext.key)
      previousCertificate.piRlcChallenges
      (InputAuthority.productAssignments previousData
        previousContext.alignment))

/-- Exact commitment-and-norm reduction shared by recursive and terminal
delayed projection.  These are the only parent/child facts consumed by the
strict `Pi_DEC` binding argument; public inputs and evaluation sidecars are
deliberately absent. -/
theorem rawChildren_recompose_eq_canonicalParent_or_bindingCollision
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (piDecAccepted : PiDEC.Accepted (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate))
    (parentBound : CanonicalParentBinding previousContext previousData
      previousCertificate)
    (rawChildren : Fin productionGlobalParams.k ->
      Phi81Relation.Assignment
        (RelationShape shape publicRingColumns publicFits))
    (childCommitments : forall child,
      (semantics previousContext.key).commit (rawChildren child) =
        (((derive previousContext previousCertificate).piDecAttempt
          previousCertificate).children child).commitment)
    (childNorms : forall child,
      (semantics previousContext.key).normBounded productionGlobalParams.b
        (rawChildren child)) :
    Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
        rawChildren =
        PackedYZcol.canonicalParentAssignment previousContext previousData
          previousCertificate \/
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics previousContext.key) productionGlobalParams
        (derive previousContext previousCertificate).piRlcOutput.commitment) := by
  let attempt :=
    (derive previousContext previousCertificate).piDecAttempt
      previousCertificate
  have commitmentsAgree :
      (fun child => (semantics previousContext.key).commit
        (rawChildren child)) =
        (fun child => (attempt.children child).commitment) := by
    funext child
    simpa [attempt] using childCommitments child
  have recomposedCommits :
      (semantics previousContext.key).commit
          ((decAlgebra previousContext.key).recomposeAssignment rawChildren) =
        attempt.parent.commitment := by
    exact ((decAlgebra previousContext.key).commit_hom rawChildren).trans
      ((congrArg (decAlgebra previousContext.key).recomposeCommitment
        commitmentsAgree).trans piDecAccepted.commitmentEquation.symm)
  have recomposedNorm :
      (semantics previousContext.key).normBounded productionGlobalParams.bigB
        ((decAlgebra previousContext.key).recomposeAssignment rawChildren) :=
    (decAlgebra previousContext.key).recompose_norm rawChildren childNorms
  let parentAssignment :=
    PackedYZcol.canonicalParentAssignment previousContext previousData
      previousCertificate
  by_cases same :
      parentAssignment =
        (decAlgebra previousContext.key).recomposeAssignment rawChildren
  · apply Or.inl
    rw [Phi81Relation.EvaluationHomomorphism.PiDEC.raw_recomposeAssignment_eq]
    exact same.symm
  · exact Or.inr ⟨{
      parentOpening := parentAssignment
      recomposedOpening :=
        (decAlgebra previousContext.key).recomposeAssignment rawChildren
      parentCommits := by
        simpa [parentAssignment, CanonicalParentBinding, attempt] using
          parentBound.1
      recomposedCommits := recomposedCommits
      parentNorm := by
        simpa [parentAssignment, CanonicalParentBinding] using parentBound.2
      recomposedNorm := recomposedNorm
      different := same
    }⟩

/-- Strict public `Pi_DEC`, one valid previous combined-parent opening, and
valid openings of the continued child statements by the *next raw running
assignments* bind their radix recomposition directly to the previous private
parent.  This is the exact authority needed by delayed `yZcol`: it does not
require the raw children to equal a canonical digit vector individually and
does not consume a child sidecar.

If the assignments differ, the already-derived commitment and norm facts are
returned as the standard `B`-bounded parent-opening collision. -/
theorem rawRunningAssignments_recompose_eq_parent_or_bindingCollision
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (nextData : Data shape)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (piDecAccepted : PiDEC.Accepted (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate))
    (parentValid : CE.Holds (semantics previousContext.key)
      productionGlobalParams
      (derive previousContext previousCertificate).piRlcOutput
      (PiRLC.combinedWitness (rlcAlgebra previousContext.key)
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment)))
    (rawChildrenValid : forall child,
      CE.Holds (semantics nextContext.key) productionGlobalParams
        (nextContext.input.running child)
        (rawRunningAssignment nextContext nextData child)) :
    Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
        (rawRunningAssignments nextContext nextData) =
      PiRLC.combinedWitness (rlcAlgebra previousContext.key)
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment) \/
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics previousContext.key) productionGlobalParams
        (derive previousContext previousCertificate).piRlcOutput.commitment) := by
  have rawChildrenValidForPrevious : forall child,
      CE.Holds (semantics previousContext.key) productionGlobalParams
        (((derive previousContext previousCertificate).piDecAttempt
          previousCertificate).children child)
        (rawRunningAssignments nextContext nextData child) := by
    intro child
    have statementEq := congrFun childrenContinue child
    have valid := rawChildrenValid child
    rw [statementEq] at valid
    simpa [sameKey, rawRunningAssignments, outputChildren] using valid
  rcases PiDEC.accepted_parent_eq_recompose_or_bindingCollision
      (semantics previousContext.key) productionGlobalParams
      (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate)
      (PiRLC.combinedWitness (rlcAlgebra previousContext.key)
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment))
      (rawRunningAssignments nextContext nextData) piDecAccepted parentValid
      rawChildrenValidForPrevious with equal | collision
  · apply Or.inl
    rw [Phi81Relation.EvaluationHomomorphism.PiDEC.raw_recomposeAssignment_eq]
    exact equal.symm
  · exact Or.inr collision

/-- The direct-parent binding argument needs only the authority actually
available from an accepted raw NC table: each next running assignment opens
the corresponding commitment and satisfies the fresh norm bound.  Public
input and evaluation sidecars are irrelevant to this commitment-binding
dichotomy.

The NC truth premise is subsequently derived from the accepted combined-NC
certificate.  `input` binds the complete decoded assignment table to the
public running commitments; no carried child evaluation is consulted. -/
theorem rawRunningAssignments_recompose_eq_parent_or_bindingCollision_of_ncTruth
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (nextData : Data shape)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (piDecAccepted : PiDEC.Accepted (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate))
    (parentBound : CanonicalParentBinding previousContext previousData
      previousCertificate)
    (nextCommitments : RawRunningCommitmentsBound nextContext nextData)
    (nextNcTruth : Semantics.Nc.Truth nextData) :
    Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
        (rawRunningAssignments nextContext nextData) =
      PiRLC.combinedWitness (rlcAlgebra previousContext.key)
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics previousContext.key) productionGlobalParams
        (derive previousContext previousCertificate).piRlcOutput.commitment) := by
  let rawChildren := rawRunningAssignments nextContext nextData
  have childCommitments : forall child,
      (semantics previousContext.key).commit (rawChildren child) =
        (((derive previousContext previousCertificate).piDecAttempt
          previousCertificate).children child).commitment := by
    intro child
    have authority := nextCommitments child
    have statementEq := congrFun childrenContinue child
    rw [statementEq] at authority
    simpa [rawChildren, rawRunningAssignments, outputChildren, sameKey] using
      authority
  have childNorms : ∀ child,
      (semantics previousContext.key).normBounded productionGlobalParams.b
        (rawChildren child) := by
    intro child
    simpa [rawChildren, rawRunningAssignments, sameKey] using
      rawRunningAssignment_norm_of_ncTruth nextContext nextData nextNcTruth child
  simpa [rawChildren, PackedYZcol.canonicalParentAssignment,
    SemanticFold.combinedAssignment, SemanticFold.assignments,
    CertificateRefinement.semanticWitness] using
    rawChildren_recompose_eq_canonicalParent_or_bindingCollision
      previousContext previousData previousCertificate piDecAccepted
      parentBound rawChildren childCommitments childNorms

/-- The prior fold's child-opening theorem, exact public child continuity,
and verifier-key continuity turn every previous canonical child assignment
into a valid opening of the corresponding next public running statement.

`childrenContinue` is equality of the complete typed child family. A digest
may establish it only through a separate binding-failure dichotomy. -/
theorem previousChildAssignment_holds_for_next
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (previousOpenings :
      ChildOpenings previousContext previousData previousCertificate)
    (child : Fin productionGlobalParams.k) :
    CE.Holds (semantics nextContext.key) productionGlobalParams
      (nextContext.input.running child)
      (previousChildAssignment previousContext previousData
        previousCertificate child) := by
  have statementEq := congrFun childrenContinue child
  rw [statementEq]
  unfold previousChildAssignment
  simpa [sameKey] using previousOpenings child

/-- Concrete cross-step raw-child authority. Actual next-step raw running
assignments equal the exact radix splits of the previous computed `Pi_RLC`
parent, or one actual next child commitment exposes two distinct fresh-bound
openings.

All premises are direct production boundaries: input authority plus checked
NC truth for the next raw table, exact verifier-key continuity, ordered child
continuity, and previous child openings. No `CeClaim.y_zcol`-like sidecar is
accepted. -/
theorem rawRunningAssignments_eq_previousChildren_or_freshBindingCollision
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (nextData : Data shape)
    (nextInput : SemanticInput nextContext nextData)
    (nextNcTruth : Semantics.Nc.Truth nextData)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (previousOpenings :
      ChildOpenings previousContext previousData previousCertificate) :
    rawRunningAssignments nextContext nextData =
        previousChildAssignment previousContext previousData
          previousCertificate ∨
      ∃ child, Nonempty
        (Opening.BindingCollision (semantics nextContext.key)
          productionGlobalParams.b
          (nextContext.input.running child).commitment) := by
  exact rawRunningAssignments_eq_or_freshBindingCollision
    nextContext nextData nextInput nextNcTruth
    (previousChildAssignment previousContext previousData previousCertificate)
    (previousChildAssignment_holds_for_next previousContext previousData
      previousCertificate nextContext sameKey childrenContinue
      previousOpenings)

/-- Exact raw-family recovery immediately yields the private parent
recomposition equality consumed by the packed delayed-projection theorem. -/
theorem rawRunningAssignments_recompose_eq_previousParent
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (nextData : Data shape)
    (childrenEqual :
      rawRunningAssignments nextContext nextData =
        previousChildAssignment previousContext previousData
          previousCertificate) :
    Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
        (rawRunningAssignments nextContext nextData) =
      PiRLC.combinedWitness (rlcAlgebra previousContext.key)
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment) := by
  rw [childrenEqual]
  exact previousChildAssignments_recompose previousContext previousData
    previousCertificate

end FixedActive

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedRawChildren
