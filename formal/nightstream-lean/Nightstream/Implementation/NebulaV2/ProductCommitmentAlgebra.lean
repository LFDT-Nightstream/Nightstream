import Nightstream.Implementation.NebulaV2.AlignedLaneAction
import Nightstream.Protocol.NebulaV2.MemoryWireGeometry
import Nightstream.Protocol.NebulaV2.Terminal
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Algebra
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.PaperVerifier
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiRLCAlgebra.Algebra
import Nightstream.SuperNeo.Folding.Nifs.PaperProfile

/-!
Contract: exact four-component product-commitment algebra for the V2
SuperNeo profile.

Assurance tier: concrete semantic algebra.

Owns the mandatory four-component commitment type, verifier-key-selected
full/operations/shared-snapshot maps, exact whole-ring projections from one
complete assignment, componentwise PiRLC combination, componentwise PiDEC
recomposition, both homomorphism proofs, and the complete paper-profile
instantiation.

Does not own PiCCS/PiRLC/PiDEC generated rows, transcript derivation, Ajtai or
Module-SIS binding, seeded-key refinement, Rust, the terminal backend, or the
deployed verifier.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CommitmentBundle
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding

abbrev Rank := MemoryWireGeometry.commitmentRank
abbrev BundleValue := Bundle (PiRLCAlgebra.Commitment.Value Rank)

/-- Exact verifier-owned V2 commitment configuration. Initial and final
snapshots share the same key by construction. -/
structure Config
    (fullShape operationsShape snapshotShape : Phi81Relation.Shape) where
  lanes : LaneLayout.Layout fullShape.carrierWidth
    operationsShape.carrierWidth snapshotShape.carrierWidth
  fullKey : PiRLCAlgebra.Commitment.Key fullShape Rank
  operationsKey : PiRLCAlgebra.Commitment.Key operationsShape Rank
  snapshotKey : PiRLCAlgebra.Commitment.Key snapshotShape Rank

namespace Config

variable {fullShape operationsShape snapshotShape : Phi81Relation.Shape}

def operationsSlice
    (config : Config fullShape operationsShape snapshotShape) :
    AlignedLaneAction.Slice fullShape operationsShape :=
  AlignedLaneAction.operationsSlice config.lanes

def initialSnapshotSlice
    (config : Config fullShape operationsShape snapshotShape) :
    AlignedLaneAction.Slice fullShape snapshotShape :=
  AlignedLaneAction.initialSnapshotSlice config.lanes

def finalSnapshotSlice
    (config : Config fullShape operationsShape snapshotShape) :
    AlignedLaneAction.Slice fullShape snapshotShape :=
  AlignedLaneAction.finalSnapshotSlice config.lanes

end Config

/-- One authority-bearing four-component map from one complete full
assignment. -/
def commit
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape)
    (assignment : Assignment fullShape) : BundleValue
  | .full => PiRLCAlgebra.Commitment.commit config.fullKey assignment
  | .operations =>
      PiRLCAlgebra.Commitment.commit config.operationsKey
        (config.operationsSlice.project assignment)
  | .initialSnapshot =>
      PiRLCAlgebra.Commitment.commit config.snapshotKey
        (config.initialSnapshotSlice.project assignment)
  | .finalSnapshot =>
      PiRLCAlgebra.Commitment.commit config.snapshotKey
        (config.finalSnapshotSlice.project assignment)

/-- Public PiRLC combination. Every component uses the identical challenge
vector and source order. -/
def combineBundles {count : Nat}
    (challenges : Fin count → RingF)
    (bundles : Fin count → BundleValue) : BundleValue :=
  fun component =>
    PiRLCAlgebra.Commitment.combineCommitments challenges
      (fun source => bundles source component)

/-- The complete bundle map commutes with the exact PiRLC assignment fold. -/
theorem commit_combine
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape)
    {count : Nat} (challenges : Fin count → RingF)
    (assignments : Fin count → Assignment fullShape) :
    commit config (PiRLCFinite.combineAssignments challenges assignments) =
      combineBundles challenges (fun source => commit config (assignments source)) := by
  funext component
  cases component with
  | full =>
      exact PiRLCAlgebra.Commitment.commit_combine config.fullKey
        challenges assignments
  | operations =>
      unfold commit combineBundles
      rw [config.operationsSlice.project_combineAssignments]
      exact PiRLCAlgebra.Commitment.commit_combine config.operationsKey
        challenges (fun source =>
          config.operationsSlice.project (assignments source))
  | initialSnapshot =>
      unfold commit combineBundles
      rw [config.initialSnapshotSlice.project_combineAssignments]
      exact PiRLCAlgebra.Commitment.commit_combine config.snapshotKey
        challenges (fun source =>
          config.initialSnapshotSlice.project (assignments source))
  | finalSnapshot =>
      unfold commit combineBundles
      rw [config.finalSnapshotSlice.project_combineAssignments]
      exact PiRLCAlgebra.Commitment.commit_combine config.snapshotKey
        challenges (fun source =>
          config.finalSnapshotSlice.project (assignments source))

/-- Public PiDEC recomposition. Every component uses the identical fourteen
children and radix powers. -/
def recomposeBundles
    (children : PiDECAlgebra.Radix.ChildIndex → BundleValue) : BundleValue :=
  fun component =>
    PiDECAlgebra.Commitment.recomposeCommitment
      (fun child => children child component)

/-- The complete bundle map commutes with the exact PiDEC assignment
recomposition. -/
theorem commit_recompose
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape)
    (assignments : PiDECAlgebra.Radix.ChildIndex → Assignment fullShape) :
    commit config (PiDECAlgebra.Radix.recomposeAssignment assignments) =
      recomposeBundles (fun child => commit config (assignments child)) := by
  funext component
  cases component with
  | full =>
      exact PiDECAlgebra.Commitment.commit_recompose config.fullKey assignments
  | operations =>
      unfold commit recomposeBundles
      rw [config.operationsSlice.project_recomposeAssignment]
      exact PiDECAlgebra.Commitment.commit_recompose config.operationsKey
        (fun child => config.operationsSlice.project (assignments child))
  | initialSnapshot =>
      unfold commit recomposeBundles
      rw [config.initialSnapshotSlice.project_recomposeAssignment]
      exact PiDECAlgebra.Commitment.commit_recompose config.snapshotKey
        (fun child => config.initialSnapshotSlice.project (assignments child))
  | finalSnapshot =>
      unfold commit recomposeBundles
      rw [config.finalSnapshotSlice.project_recomposeAssignment]
      exact PiDECAlgebra.Commitment.commit_recompose config.snapshotKey
        (fun child => config.finalSnapshotSlice.project (assignments child))

/-- The exact full Phi81 relation with the product bundle as its commitment
coordinate. -/
def semantics
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape) :=
  relationSemantics (commit config)

/-- Complete concrete PiRLC algebra with the mandatory bundle as the sole
commitment type. -/
def piRlcAlgebra
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape) :
    PiRLC.Algebra
      (Structure fullShape) (Assignment fullShape) (PublicInput fullShape)
      (Point fullShape) Evaluation BundleValue RingF (semantics config)
      productionGlobalParams where
  challengeValid := PiRLCAlgebra.Challenge.challengeValid
  combineAssignment := PiRLCFinite.combineAssignments
  combineCommitment := combineBundles
  combinePublicInput := PiRLCAlgebra.PublicInput.combinePublicInputs
  combineEvaluations := PiRLCFinite.combineEvaluations (shape := fullShape)
  commit_hom := by
    intro count challenges assignments
    exact commit_combine config challenges assignments
  publicInput_hom := by
    intro count challenges assignments
    exact PiRLCAlgebra.PublicInput.relation_publicInput_hom
      (commit config) challenges assignments
  evaluations_hom := by
    intro count system point challenges assignments
    exact PiRLCFinite.relation_evaluations_hom
      (commit config) system point challenges assignments
  norm_growth := by
    intro count totalBound challenges assignments challengesValid assignmentsFresh
    exact PiRLCAlgebra.Norm.relation_norm_growth
      (commit config) totalBound challenges assignments challengesValid
        assignmentsFresh

/-- Complete concrete PiDEC algebra with component-complete public
recomposition. -/
def piDecAlgebra
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape) :
    PiDEC.Algebra
      (Structure fullShape) (Assignment fullShape) (PublicInput fullShape)
      (Point fullShape) Evaluation BundleValue (semantics config)
      productionGlobalParams where
  splitAssignment := PiDECAlgebra.Radix.splitAssignment
  recomposeAssignment := PiDECAlgebra.Radix.recomposeAssignment
  recomposeCommitment := recomposeBundles
  recomposePublicInput := PiDECAlgebra.PublicInput.recomposePublicInput
  recomposeEvaluations := EvaluationHomomorphism.PiDEC.recomposeEvaluations
  split_recompose := PiDECAlgebra.Radix.split_recompose
  split_norm := PiDECAlgebra.Radix.split_norm
  recompose_norm := PiDECAlgebra.Radix.recompose_norm
  commit_hom := commit_recompose config
  publicInput_hom :=
    PiDECAlgebra.PublicInput.relation_publicInput_hom (commit config)
  evaluations_hom :=
    EvaluationHomomorphism.PiDEC.relation_evaluations_hom (commit config)

/-- Verifier-computed public-input split for the product algebra. -/
def publicInputSplit
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape) :
    PiDEC.PaperVerifier.PublicInputSplit (piDecAlgebra config) where
  split := PiDECAlgebra.PublicInput.splitPublicInput
  recompose_split := PiDECAlgebra.PublicInput.splitPublicInput_recompose
  split_project := PiDECAlgebra.PublicInput.splitPublicInput_project

/-- Exact evaluation arity of the product-commitment relation. -/
def evaluationArity
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape) :
    PiDEC.PaperVerifier.EvaluationArity (semantics config) where
  count := fun _ => fullShape.matrixCount
  evaluations_size := Phi81Relation.evaluations_size

/-- Complete paper-profile instantiation. The product bundle is not a
sidecar: it is the CCS/CE commitment coordinate through all three reductions. -/
def paperProfile
    {fullShape operationsShape snapshotShape : Phi81Relation.Shape}
    (config : Config fullShape operationsShape snapshotShape) :
    Nifs.PaperProfile.Profile
      (Structure fullShape) (Assignment fullShape) (PublicInput fullShape)
      (Point fullShape) Evaluation BundleValue RingF where
  semantics := semantics config
  rlcAlgebra := piRlcAlgebra config
  decAlgebra := piDecAlgebra config
  decPublicInputSplit := publicInputSplit config
  decEvaluationArity := evaluationArity config

/-- The V2 terminal child family has exactly the selected SuperNeo PiDEC
arity. -/
theorem terminal_children_match_superNeo :
    Terminal.foldedChildCount = productionGlobalParams.k := by
  decide

end Nightstream.Implementation.NebulaV2.ProductCommitmentAlgebra
