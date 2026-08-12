import Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra
import Nightstream.Protocol.NebulaV2.Terminal

/-!
Contract: exact terminal CE relation for the V2 product commitment.

Assurance tier: concrete semantic relation.

Owns the fourteen post-PiDEC CE children, their exact fresh-stage public and
evaluation obligations, the one assignment per child used by those
obligations, and the theorem that combines this core with the product-opening
and strict-bound facts checked by the terminal opening rows.

Does not own the final NIFS accumulator, generated terminal rows, proof-system
soundness, commitment binding, Rust, or the deployed verifier.

Emits constraints: no. `checkCore` is the exact value-level checker whose
generated-row refinement is a separate implementation obligation.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductTerminalRelation

open Nightstream.Protocol.NebulaV2.Terminal
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation

variable {fullShape operationsShape snapshotShape : Shape}

abbrev Assignment (shape : Shape) := Phi81Relation.Assignment shape
abbrev BundleValue := ProductCommitmentAlgebra.BundleValue
abbrev ChildClaim (shape : Shape) :=
  CE.Instance (Structure shape) (PublicInput shape) (Point shape)
    Evaluation BundleValue
abbrev Children (shape : Shape) := FoldedChild → ChildClaim shape
abbrev Assignments (shape : Shape) := FoldedChild → Assignment shape

/-- The part of terminal CE membership that is not already checked by the
product-opening rows. The exact same `assignments child` value is used for the
public projection and every matrix evaluation. -/
def CoreHolds (children : Children fullShape)
    (assignments : Assignments fullShape) : Prop :=
  ∀ child,
    (children child).stage = NormStage.fresh ∧
    projectPublicInput (assignments child) = (children child).publicInput ∧
    evaluations (children child).constraintSystem (assignments child)
        (children child).point =
      (children child).evaluations

/-- Exact complete terminal relation for all fourteen accumulator children. -/
def Holds
    (config : ProductCommitmentAlgebra.Config fullShape operationsShape
    snapshotShape)
    (children : Children fullShape)
    (assignments : Assignments fullShape) : Prop :=
  ∀ child,
    (children child).stage = NormStage.fresh ∧
      CE.Holds (ProductCommitmentAlgebra.semantics config)
        productionGlobalParams (children child) (assignments child)

/-- Exact value-level terminal-core checker. It does not accept a supplied
validity bit. -/
noncomputable def checkCore (children : Children fullShape)
    (assignments : Assignments fullShape) : Bool := by
  classical
  exact decide (CoreHolds children assignments)

@[simp] theorem checkCore_eq_true_iff (children : Children fullShape)
    (assignments : Assignments fullShape) :
    checkCore children assignments = true ↔ CoreHolds children assignments := by
  classical
  simp [checkCore]

/-- Product opening and norm rows plus the exact CE core establish full CE
membership for every child. No part of CE membership is a premise with the
same name as the conclusion. -/
theorem holds_of_common_openings
    (config : ProductCommitmentAlgebra.Config fullShape operationsShape
      snapshotShape)
    (children : Children fullShape) (assignments : Assignments fullShape)
    (bounded : ∀ child,
      assignmentNormBounded 2 (assignments child))
    (opens : ∀ child,
      ProductCommitmentAlgebra.commit config (assignments child) =
        (children child).commitment)
    (core : CoreHolds children assignments) :
    Holds config children assignments := by
  intro child
  rcases core child with ⟨stageFresh, publicExact, evaluationsExact⟩
  refine ⟨stageFresh, ?_⟩
  apply (ceMembership_iff (ProductCommitmentAlgebra.commit config)
    productionGlobalParams (children child) (assignments child)).2
  refine ⟨opens child, publicExact, ?_, evaluationsExact⟩
  simpa [stageFresh] using bounded child

/-- Full CE membership projects back to the exact core. This is the
completeness direction for the value-level terminal-core checker. -/
theorem core_of_holds
    (config : ProductCommitmentAlgebra.Config fullShape operationsShape
      snapshotShape)
    (children : Children fullShape) (assignments : Assignments fullShape)
    (holds : Holds config children assignments) :
    CoreHolds children assignments := by
  intro child
  have member :=
    (ceMembership_iff (ProductCommitmentAlgebra.commit config)
      productionGlobalParams (children child) (assignments child)).1
      (holds child).2
  exact ⟨(holds child).1, member.2.1, member.2.2.2⟩

/-- Honest fresh CE children satisfy the complete product terminal relation
when their assignments meet the selected strict bound. -/
theorem canonical_children_hold
    (config : ProductCommitmentAlgebra.Config fullShape operationsShape
      snapshotShape)
    (systems : FoldedChild → Structure fullShape)
    (points : FoldedChild → Point fullShape)
    (assignments : Assignments fullShape)
    (bounded : ∀ child, assignmentNormBounded 2 (assignments child)) :
    Holds config
      (fun child => canonicalCEStatement
        (ProductCommitmentAlgebra.commit config) (systems child)
        NormStage.fresh
        (points child) (assignments child))
      assignments := by
  intro child
  exact ⟨rfl, canonicalCE_holds (ProductCommitmentAlgebra.commit config)
    productionGlobalParams (systems child) NormStage.fresh (points child)
    (assignments child) (bounded child)⟩

/-- Each complete terminal CE opening includes the exact product commitment
equation. -/
theorem commitment_of_holds
    (config : ProductCommitmentAlgebra.Config fullShape operationsShape
      snapshotShape)
    (children : Children fullShape) (assignments : Assignments fullShape)
    (holds : Holds config children assignments) (child : FoldedChild) :
    ProductCommitmentAlgebra.commit config (assignments child) =
      (children child).commitment :=
  (holds child).2.1.1

/-! ## Countermodel for an omitted terminal stage check -/

/-- A family of canonical combined-stage CE statements can satisfy ordinary
`CE.Holds` for every child. It still cannot satisfy the V2 terminal relation,
because terminal PiDEC children must be at the strict fresh stage. This shows
that checking only `CE.Holds` does not recover the required terminal bound. -/
theorem combined_children_can_satisfy_ce_but_not_terminal
    (config : ProductCommitmentAlgebra.Config fullShape operationsShape
      snapshotShape)
    (systems : FoldedChild → Structure fullShape)
    (points : FoldedChild → Point fullShape)
    (assignments : Assignments fullShape)
    (bounded : ∀ child,
      assignmentNormBounded
        (NormStage.combined.bound productionGlobalParams)
        (assignments child)) :
    (∀ child,
      CE.Holds (ProductCommitmentAlgebra.semantics config)
        productionGlobalParams
        (canonicalCEStatement (ProductCommitmentAlgebra.commit config)
          (systems child) NormStage.combined (points child)
          (assignments child))
        (assignments child)) ∧
      ¬ Holds config
        (fun child => canonicalCEStatement
          (ProductCommitmentAlgebra.commit config) (systems child)
          NormStage.combined (points child) (assignments child))
        assignments := by
  constructor
  · intro child
    exact canonicalCE_holds (ProductCommitmentAlgebra.commit config)
      productionGlobalParams (systems child) NormStage.combined
      (points child) (assignments child) (bounded child)
  · intro terminal
    have stage := (terminal ⟨0, by decide⟩).1
    cases stage

end Nightstream.Implementation.NebulaV2.ProductTerminalRelation
