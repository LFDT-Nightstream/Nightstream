import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.Authority
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc

/-!
Authoritative source construction for the opening-derived fixed-active NIFS.

Assurance tier: model-level carrier refinement.

Owns: the sole matrices, constraint polynomial, fresh assignment, canonical
opening, and source-product alignment; deterministic construction of the
Split-NC source data; computation of both public input surfaces from that
data; and derivation of computed-child source validity from independent
Split-NC paper truth.

Does not own: proof of the fresh CCS and norm obligations, transcript replay,
physical row refinement, commitment binding, an opening serializer, costs, or
row removal.

Emits constraints: no.

Authority boundary: callers supply no running assignment, carried evaluation
claim, source commitment, source public input, or source evaluation array.
The fourteen running assignments are the canonical split of the one opening
in semantic running order. Every carried claim and both public input surfaces
are computed from the same source data.
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.SourceInput

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open PaperLinearAlgebra

universe uState

/-- Complete authoritative input needed to reconstruct the fixed-active
Split-NC source family. -/
structure Carrier
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth) where
  matrices : Fin shape.matrixCount ->
    BooleanMatrix F shape.rowVariables shape.logicalWidth
  constraintPolynomial :
    CCSResidualTable.ConstraintPolynomial F shape.matrixCount
  freshAssignment : Assignment F shape.logicalWidth
  opening : OpeningPayload shape publicRingColumns publicFits
  alignment : SourceAlignment shape productionGlobalParams arity

namespace Carrier

/-- The sole relation structure, computed before source statements are
materialized. -/
def system
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits) :
    Structure (RelationShape shape publicRingColumns publicFits) where
  matrices := carrier.matrices
  constraintPolynomial := carrier.constraintPolynomial

/-- The verifier-owned child digit of the canonical opening assignment. -/
def childAssignment
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (child : Fin productionGlobalParams.k) :
    Phi81Relation.Assignment
      (RelationShape shape publicRingColumns publicFits) :=
  fun column => PiDECAlgebra.Radix.splitScalar
    (carrier.opening.assignment column) child

/-- The explicitly typed child assignment is the canonical radix split. -/
@[simp] theorem childAssignment_eq_splitAssignment
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (child : Fin productionGlobalParams.k) :
    carrier.childAssignment child =
      PiDECAlgebra.Radix.splitAssignment carrier.opening.assignment child := by
  rfl

/-- One semantic running assignment is the opening digit at the inverse
product index selected by the partition alignment. -/
def runningAssignment
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (running : Fin shape.runningCount) : SourceAssignment shape :=
  let child : PiDECAlgebra.Radix.ChildIndex :=
    show Fin productionGlobalParams.k from
      carrier.alignment.productRunningIndex running
  fun column => PiDECAlgebra.Radix.splitScalar
    (carrier.opening.assignment column) child

@[simp] theorem runningAssignment_semanticRunningIndex
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (running : Fin (arity.mode.count productionGlobalParams)) :
    carrier.runningAssignment
        (carrier.alignment.semanticRunningIndex running) =
      PiDECAlgebra.Radix.splitAssignment carrier.opening.assignment
        running := by
  funext column
  simp [runningAssignment, PiDECAlgebra.Radix.splitAssignment]

/-- The prior evaluation claim is the relation evaluation of the same
matrix, running assignment, and opening point. -/
def claimedCoefficient
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (coordinate : CarriedCoordinate shape.paperShape) : K :=
  Phi81Relation.matrixEvaluation carrier.system
    (carrier.runningAssignment coordinate.running) carrier.opening.point
    coordinate.matrix coordinate.coefficient

/-- The sole independent Split-NC source family. No claimed coefficient or
running assignment remains caller supplied. -/
def data
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits) : Data shape where
  matrices := carrier.matrices
  constraintPolynomial := carrier.constraintPolynomial
  freshAssignments := fun _ => carrier.freshAssignment
  runningAssignments := carrier.runningAssignment
  priorPoint := carrier.opening.point
  claimedCoefficient := carrier.claimedCoefficient

@[simp] theorem data_runningAssignments
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (running : Fin shape.runningCount) :
    carrier.data.runningAssignments running =
      PiDECAlgebra.Radix.splitAssignment carrier.opening.assignment
        (carrier.alignment.productRunningIndex running) := by
  rfl

@[simp] theorem data_priorPoint
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits) :
    carrier.data.priorPoint = carrier.opening.point := by
  rfl

@[simp] theorem system_eq_ofSourceData
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits) :
    carrier.system =
      Structure.ofSourceData publicRingColumns publicFits carrier.data := by
  rfl

/-- The one completed fresh assignment used by both source data and its
public statement. -/
def freshSourceAssignment
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits) :
    SourceAssignment shape :=
  Phi81CarrierLayout.extendAssignment 0 carrier.freshAssignment

/-- Canonical fresh public payload computed from the sole fresh assignment. -/
def freshPayload
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits) :
    Canonical.FreshPayload shape publicRingColumns publicFits verifierRows where
  commitment := ConcretePhi81.commit key carrier.freshSourceAssignment
  publicInput := sourcePublicInput publicRingColumns publicFits
    carrier.freshSourceAssignment

/-- Opening-derived input whose fresh payload and relation are computed from
the same authoritative source carrier. -/
def input
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits) :
    CanonicalOpening.Input shape publicRingColumns publicFits verifierRows where
  system := carrier.system
  fresh := carrier.freshPayload key
  opening := carrier.opening

@[simp] theorem data_freshAssignment
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (fresh : Fin shape.freshCount) :
    carrier.data.freshAssignment fresh = carrier.freshSourceAssignment := by
  rfl

@[simp] theorem priorEvaluations_eq
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (running : Fin shape.runningCount) :
    InputAuthority.priorEvaluations carrier.data running =
      Phi81Relation.evaluations carrier.system
        (carrier.runningAssignment running) carrier.opening.point := by
  rfl

/-- Exact public source product computed from the authoritative carrier. -/
def sourceProduct
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits) :
    SourceProduct shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams arity where
  fresh := fun _ => {
    constraintSystem := carrier.system
    commitment := ConcretePhi81.commit key carrier.freshSourceAssignment
    publicInput := sourcePublicInput publicRingColumns publicFits
      carrier.freshSourceAssignment
    stage := .fresh
  }
  running := fun running =>
    let semantic := carrier.alignment.semanticRunningIndex running
    {
      constraintSystem := carrier.system
      commitment := ConcretePhi81.commit key
        (carrier.runningAssignment semantic)
      publicInput := sourcePublicInput publicRingColumns publicFits
        (carrier.runningAssignment semantic)
      point := carrier.opening.point
      evaluations := InputAuthority.priorEvaluations carrier.data semantic
      stage := .fresh
    }

private theorem sourceProduct_eq_of_fields
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (left right : SourceProduct shape publicRingColumns publicFits
      (CommitmentValue verifierRows) productionGlobalParams arity)
    (fresh : left.fresh = right.fresh)
    (running : left.running = right.running) :
    left = right := by
  cases left with
  | mk leftFresh leftRunning =>
    cases right with
    | mk rightFresh rightRunning =>
      cases fresh
      cases running
      rfl

/-- The existing two-stage canonical input materializer computes exactly the
direct source product above. This is the sole place that unfolds both layers. -/
@[simp] theorem input_materialize_eq_sourceProduct
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits) :
    ((carrier.input key).materialize key).materialize =
      carrier.sourceProduct key := by
  have freshEq :
      ((carrier.input key).materialize key).materialize.fresh =
        (carrier.sourceProduct key).fresh := by
    funext fresh
    rfl
  have runningEq :
      ((carrier.input key).materialize key).materialize.running =
        (carrier.sourceProduct key).running := by
    funext running
    change carrier.opening.children key carrier.system running = {
      constraintSystem := carrier.system
      commitment := ConcretePhi81.commit key
        (carrier.runningAssignment
          (carrier.alignment.semanticRunningIndex running))
      publicInput := sourcePublicInput publicRingColumns publicFits
        (carrier.runningAssignment
          (carrier.alignment.semanticRunningIndex running))
      point := carrier.opening.point
      evaluations := InputAuthority.priorEvaluations carrier.data
        (carrier.alignment.semanticRunningIndex running)
      stage := .fresh
    }
    rw [priorEvaluations_eq, runningAssignment_semanticRunningIndex]
    rfl
  exact sourceProduct_eq_of_fields
    ((carrier.input key).materialize key).materialize
    (carrier.sourceProduct key) freshEq runningEq

/-- Each opening-derived child is the corresponding running statement in the
direct authoritative source product. -/
@[simp] theorem opening_children_eq_sourceProduct_running
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits)
    (running : Fin (arity.mode.count productionGlobalParams)) :
    carrier.opening.children key carrier.system running =
      (carrier.sourceProduct key).running running := by
  exact congrArg (fun input => input.running running)
    (carrier.input_materialize_eq_sourceProduct key)

/-- The public Split-NC polynomial input is computed from the same data. -/
def piCcsInput
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits) :
    PiCCS.SplitNc.Verifier.PublicInput shape :=
  PiCCS.SplitNc.Verifier.PublicInput.ofSources carrier.data

/-- The one public fresh source is computed from its authoritative completed
assignment and sole relation structure. -/
theorem freshSourceBound
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits)
    (fresh : Fin arity.freshCount) :
    @InputAuthority.FreshSourceBound shape productionGlobalParams arity
      (CommitmentValue verifierRows) publicRingColumns publicFits
      (ConcretePhi81.commit key) carrier.data carrier.alignment
      (carrier.sourceProduct key) fresh := by
  exact {
    constraintSystem := by
      change carrier.system =
        Structure.ofSourceData publicRingColumns publicFits carrier.data
      exact carrier.system_eq_ofSourceData
    commitment := by
      change ConcretePhi81.commit key
          (carrier.data.freshAssignment
            (carrier.alignment.semanticFreshIndex fresh)) =
        ConcretePhi81.commit key carrier.freshSourceAssignment
      rw [data_freshAssignment]
    publicInput := by
      change sourcePublicInput publicRingColumns publicFits
          (carrier.data.freshAssignment
            (carrier.alignment.semanticFreshIndex fresh)) =
        sourcePublicInput publicRingColumns publicFits
          carrier.freshSourceAssignment
      rw [data_freshAssignment]
    stage := by
      change NormStage.fresh = NormStage.fresh
      rfl
  }

/-- One public running source is computed from the aligned canonical opening
digit and its computed prior evaluation array. -/
theorem runningSourceBound
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits)
    (running : Fin (arity.mode.count productionGlobalParams)) :
    @InputAuthority.RunningSourceBound shape productionGlobalParams arity
      (CommitmentValue verifierRows) publicRingColumns publicFits
      (ConcretePhi81.commit key) carrier.data carrier.alignment
      (carrier.sourceProduct key) running := by
  exact {
    constraintSystem := by
      change carrier.system =
        Structure.ofSourceData publicRingColumns publicFits carrier.data
      exact carrier.system_eq_ofSourceData
    commitment := rfl
    publicInput := rfl
    point := rfl
    evaluations := rfl
    stage := rfl
  }

/-- Every public source field is definitionally computed from the aligned
authoritative assignment. -/
theorem boundToSources
    {shape : SemanticShape}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (carrier : Carrier shape publicRingColumns publicFits) :
    @InputAuthority.BoundToSources shape productionGlobalParams arity
      (CommitmentValue verifierRows) publicRingColumns publicFits
      (ConcretePhi81.commit key) carrier.data carrier.alignment
      (carrier.sourceProduct key) where
  fresh := carrier.freshSourceBound key
  running := carrier.runningSourceBound key

/-- Install both computed public input surfaces into an otherwise unchanged
opening-derived context. -/
def install
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows) :
    CanonicalOpening.Context shape State publicRingColumns publicFits
      verifierRows := {
  context with
    alignment := carrier.alignment
    input := carrier.input context.key
    piCcsInput := carrier.piCcsInput
}

/-- Both public input surfaces are authoritative by construction. -/
theorem semanticInput
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows) :
    SemanticFold.Input (carrier.install context).full carrier.data := by
  exact {
    publicInput := rfl
    sources := by
      change @InputAuthority.BoundToSources shape productionGlobalParams arity
        (CommitmentValue verifierRows) publicRingColumns publicFits
        (ConcretePhi81.commit context.key) carrier.data carrier.alignment
        (((carrier.input context.key).materialize context.key).materialize)
      rw [input_materialize_eq_sourceProduct]
      exact carrier.boundToSources context.key
  }

/-- The public source-product half of `semanticInput`, exposed for callers
that compose the polynomial-input bridge separately. -/
theorem inputBound
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows) :
    SemanticFold.InputBound (carrier.install context).full carrier.data :=
  (carrier.semanticInput context).sources

/-- The independent NC fact used below, kept opaque so public theorem
declarations do not repeatedly normalize the computed source data. -/
def NcTruth
    {shape : SemanticShape}
    {publicRingColumns : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits) : Prop :=
  Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.Truth carrier.data

/-- Child validity for the installed computed input, kept opaque at the
public theorem boundary. -/
def ComputedChildSourcesValid
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows) : Prop :=
  ∀ child : Fin productionGlobalParams.k,
    CE.Holds (ConcretePhi81.semantics context.key)
      productionGlobalParams
      (carrier.opening.children context.key carrier.system child)
      (carrier.childAssignment child)

private theorem childHolds_of_ncTruth
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows)
    (truth : carrier.NcTruth)
    (child : Fin productionGlobalParams.k) :
    CE.Holds (ConcretePhi81.semantics context.key)
      productionGlobalParams
      (carrier.opening.children context.key carrier.system child)
      (carrier.childAssignment child) := by
  let running := carrier.alignment.semanticRunningIndex child
  change Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.Truth
    carrier.data at truth
  have normAt (column : Fin shape.carrierWidth) :
      centeredMagnitude (carrier.runningAssignment running column) <
        NormStage.bound productionGlobalParams .fresh := by
    rw [production_norm_stages.1]
    simpa only [
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.assignment_runningIndex]
      using truth
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources.Data.runningIndex
          running) column
  have childEq :
      carrier.runningAssignment running =
        carrier.childAssignment child := by
    rw [carrier.childAssignment_eq_splitAssignment]
    simpa only [running] using
      carrier.runningAssignment_semanticRunningIndex child
  rw [opening_children_eq_sourceProduct_running, ← childEq]
  refine ⟨⟨rfl, rfl, ?_⟩, carrier.opening.point.dimension, ?_⟩
  · intro column
    exact normAt column
  · exact (carrier.priorEvaluations_eq running).symm

/-- Independent NC truth supplies the only non-computed fact needed for the
canonical running children: their strict fresh norm. -/
theorem childSourcesValid_of_ncTruth
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows)
    (truth : carrier.NcTruth) :
    carrier.ComputedChildSourcesValid context := by
  change ∀ child,
    CE.Holds (ConcretePhi81.semantics context.key)
      productionGlobalParams
      (carrier.opening.children context.key carrier.system child)
      (carrier.childAssignment child)
  intro child
  exact carrier.childHolds_of_ncTruth context truth child

/-- NC truth discharges the existing running-authority interface through the
computed canonical child family. -/
theorem runningAuthority_of_ncTruth
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows)
    (truth : carrier.NcTruth) :
    RunningAuthority.Accepted (carrier.install context).full := by
  apply CanonicalOpening.runningAuthority
  intro child
  let holds := carrier.childSourcesValid_of_ncTruth context truth child
  exact Eq.mp
    (congrArg
      (fun assignment =>
        CE.Holds (ConcretePhi81.semantics context.key)
          productionGlobalParams
          (carrier.opening.children context.key carrier.system child)
          assignment)
      (carrier.childAssignment_eq_splitAssignment child))
    holds

/-- Full independent Split-NC paper truth in particular contains the NC norm
truth needed by every computed canonical child. -/
theorem childSourcesValid_of_paper
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (carrier : Carrier shape publicRingColumns publicFits)
    (context : CanonicalOpening.Context shape State publicRingColumns
      publicFits verifierRows)
    (paper :
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Paper.Holds
        carrier.data) :
    carrier.ComputedChildSourcesValid context := by
  apply carrier.childSourcesValid_of_ncTruth context
  change Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.Truth
    carrier.data
  intro source column
  exact paper.2.1 source column

end Carrier

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.SourceInput
