import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Types

/-!
Checked incoming accumulator authority for the concrete Phi81 NIFS verifier.

Protocol: production SuperNeo NIFS, including its zero-running bootstrap.
Phase: validate the incoming running accumulator before `Pi_CCS`.
Constraint family: strict `Pi_DEC` recomposition of the carried parent and
running children; this file emits no rows.

Owns: the exact mode split; absence of parent authority in bootstrap mode;
exact presence and transcript binding of the parent in active mode; the strict
`Pi_DEC` attempt over that parent and the public running children; and the
derivation that all running children share one evaluation point.

Does not own: parent hashing, Poseidon2, `Pi_CCS` transcript replay, semantic
openings, the outgoing `Pi_RLC` parent, Rust/R1CS refinement, physical row
counts, or row removal.

Emits constraints: no.

Authority boundary: a parent digest is never authority. Bootstrap acceptance
requires the complete parent carrier to be absent. Active acceptance requires
the complete parent statement bound by `Context.runningParent` and a strict
`Pi_DEC.Accepted` proof against the exact public running children. The shared
running point is a theorem of that check, not an additional protocol
obligation.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.concrete.running_authority.mode` | accept exactly zero-running bootstrap or full active mode | checked | `Accepted.bootstrap`, `Accepted.active` |
| `nifs.concrete.running_authority.bootstrap_parent_absence` | zero-running bootstrap carries no parent authority | checked | `Accepted.bootstrap`, `Accepted.parentAbsent_of_bootstrap` |
| `nifs.concrete.running_authority.parent_presence` | active transcript-bound carrier is exactly `some parent` | checked | `Bound.parentBound` |
| `nifs.concrete.running_authority.parent_dec` | parent stage, child stages, structure, point, commitment, public input, and evaluations satisfy strict `Pi_DEC` | checked | `Bound.piDec` |
| `nifs.concrete.running_authority.shared_point` | every pair of running children has the same evaluation point | derived | `Accepted.children_sharePoint` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

variable
  {shape : SemanticShape}
  {State : Type uState}
  {publicRingColumns verifierRows : Nat}
  {publicFits :
    ringDegree * publicRingColumns <= shape.carrierWidth}
  {arity : BatchArity productionGlobalParams}

/-- Reindex one fixed child into an active arity's running-input index. -/
def activeIndex
    (active : arity.mode = .active)
    (child : Fin productionGlobalParams.k) :
    Fin (arity.mode.count productionGlobalParams) :=
  Fin.cast (by rw [active]; rfl) child

/-- Reindex the fixed `k` child family into an active arity's running input. -/
def children
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity)
    (active : arity.mode = .active) :
    Fin productionGlobalParams.k →
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows) :=
  fun child =>
    context.input.running (activeIndex active child)

@[simp] theorem children_apply
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (active : arity.mode = .active)
    (child : Fin productionGlobalParams.k) :
    children context active child =
      context.input.running (activeIndex active child) := by
  rfl

/-- Exact strict `Pi_DEC` view checked before the incoming parent may
authorize transcript compression. -/
def attempt
    (context :
      Context shape State publicRingColumns publicFits verifierRows
        arity)
    (active : arity.mode = .active)
    (parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) :
    PiDEC.Attempt
      (Phi81Relation.Structure
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.PublicInput
        (RelationShape shape publicRingColumns publicFits))
      (Phi81Relation.Point
        (RelationShape shape publicRingColumns publicFits))
      Phi81Relation.Evaluation (CommitmentValue verifierRows)
      productionGlobalParams := {
  parent := parent
  children := children context active
}

/-- Typed evidence for the complete active incoming-accumulator check. -/
structure Bound
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity) where
  active : arity.mode = .active
  parent :
    Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows)
  parentBound : context.runningParent = some parent
  piDec :
    PiDEC.Accepted (decAlgebra context.key)
      (attempt context active parent)

/-- Exact incoming-authority split used by both native and circuit verifiers.

The constructors are mathematical cases, not Rust branch labels. Bootstrap
has no running inputs and therefore must not carry a parent. Active mode has
the full `k`-child product and must validate its complete parent through the
strict `Pi_DEC` equations retained in `Bound`. -/
inductive Accepted
    (context :
      Context shape State publicRingColumns publicFits verifierRows arity) :
    Prop where
  | bootstrap
      (mode : arity.mode = .bootstrap)
      (parentAbsent : context.runningParent = none) :
      Accepted context
  | active (bound : Bound context) : Accepted context

namespace Bound

/-- Every running child uses the checked parent's structure. -/
theorem childStructure_eq_parent
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (bound : Bound context)
    (child : Fin productionGlobalParams.k) :
    (children context bound.active child).constraintSystem =
      bound.parent.constraintSystem :=
  bound.piDec.sameStructure child

/-- Every running child uses the checked parent's evaluation point. -/
theorem childPoint_eq_parent
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (bound : Bound context)
    (child : Fin productionGlobalParams.k) :
    (children context bound.active child).point =
      bound.parent.point :=
  bound.piDec.samePoint child

/-- The strict `Pi_DEC` point equation applies directly to each public running
input, after eliminating the active-mode index cast. -/
theorem inputPoint_eq_parent
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (bound : Bound context)
    (child : Fin (arity.mode.count productionGlobalParams)) :
    (context.input.running child).point = bound.parent.point := by
  let fixedChild : Fin productionGlobalParams.k :=
    Fin.cast (by rw [bound.active]; rfl) child
  have same := bound.childPoint_eq_parent fixedChild
  simpa [children, activeIndex, fixedChild] using same

/-- Pairwise shared-`r` equality is already implied by strict incoming
`Pi_DEC`; it is not an independent semantic obligation. -/
theorem children_sharePoint
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (bound : Bound context)
    (left right : Fin (arity.mode.count productionGlobalParams)) :
    (context.input.running left).point =
      (context.input.running right).point := by
  exact
    (bound.inputPoint_eq_parent left).trans
      (bound.inputPoint_eq_parent right).symm

end Bound

namespace Accepted

/-- Bootstrap acceptance exposes the exact absence of a parent carrier. -/
theorem parentAbsent_of_bootstrap
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (accepted : Accepted context)
    (mode : arity.mode = .bootstrap) :
    context.runningParent = none := by
  cases accepted with
  | bootstrap _ parentAbsent => exact parentAbsent
  | active bound =>
      have impossible : RunningMode.active = RunningMode.bootstrap :=
        bound.active.symm.trans mode
      cases impossible

/-- In a verifier-owned bootstrap profile, parent absence is the complete
incoming-authority obligation. -/
theorem iff_parentAbsent_of_bootstrap
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (mode : arity.mode = .bootstrap) :
    Accepted context ↔ context.runningParent = none := by
  constructor
  · intro accepted
    exact accepted.parentAbsent_of_bootstrap mode
  · intro parentAbsent
    exact .bootstrap mode parentAbsent

/-- In a verifier-owned active profile, acceptance is exactly existence of
the complete strict parent proof. -/
theorem iff_nonemptyBound_of_active
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (mode : arity.mode = .active) :
    Accepted context ↔ Nonempty (Bound context) := by
  constructor
  · intro accepted
    cases accepted with
    | bootstrap bootstrapMode _ =>
        have impossible : RunningMode.active = RunningMode.bootstrap :=
          mode.symm.trans bootstrapMode
        cases impossible
    | active bound => exact ⟨bound⟩
  · rintro ⟨bound⟩
    exact .active bound

/-- A zero-running bootstrap cannot accept a supplied parent authority. -/
theorem rejects_parent_in_bootstrap
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (mode : arity.mode = .bootstrap)
    (parent :
      Phi81Relation.CEStatement
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows))
    (parentPresent : context.runningParent = some parent) :
    ¬ Accepted context := by
  intro accepted
  have parentAbsent := accepted.parentAbsent_of_bootstrap mode
  rw [parentPresent] at parentAbsent
  cases parentAbsent

/-- Physical running-authority acceptance alone derives pairwise shared-`r`
for the exact public running family. -/
theorem children_sharePoint
    {context :
      Context shape State publicRingColumns publicFits verifierRows
        arity}
    (accepted : Accepted context)
    (left right : Fin (arity.mode.count productionGlobalParams)) :
    (context.input.running left).point =
      (context.input.running right).point := by
  cases accepted with
  | bootstrap mode _ =>
      have impossible := left.isLt
      simp [mode, RunningMode.count] at impossible
  | active bound => exact bound.children_sharePoint left right

end Accepted

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.RunningAuthority
