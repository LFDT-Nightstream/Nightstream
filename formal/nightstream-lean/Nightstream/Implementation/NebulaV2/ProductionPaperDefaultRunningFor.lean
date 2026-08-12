import Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor

/-!
Contract: verifier-owned universal default running product for the generated
Nebula V2 SuperNeo relation.

The value is the unique all-zero public running carrier. Every one of its
fourteen claims opens with the all-zero full assignment for every selected
product-commitment key and every generated relation structure. This is the
concrete `u_perp` required by HyperNova Construction 2.

Does not own generated base rows, base-branch selection, recursive NIFS rows,
Rust, or cryptographic binding.

Assurance tier: concrete semantic algebra.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 30000

namespace Nightstream.Implementation.NebulaV2.ProductionPaperDefaultRunningFor

open Nightstream.Implementation.NebulaV2
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev FullShape
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.FullShape rowVariables logicalWidth publicFits

abbrev Assignment
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  ProductPaperAlgebraFor.Assignment rowVariables logicalWidth publicFits

/-- Canonical opening for every coordinate of the default product. -/
def zeroAssignment
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    Assignment rowVariables logicalWidth publicFits :=
  BaseLinear.assignmentZero

/-- Canonical cube point for the selected generated-relation exponent. -/
def zeroPoint (rowVariables : Nat) : ProductPaperAlgebraFor.Point rowVariables where
  coordinates := List.replicate rowVariables K.zero
  dimension := by simp

/-- Canonical public zero for all four commitment components. -/
def zeroBundle : ProductPaperAlgebraFor.Commitment :=
  fun _ => PiRLCAlgebra.Commitment.commitmentZero

/-- One deterministic value, fixed by the profile and relation exponent. -/
def value
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :
    ProductNifsCodec.RunningFor rowVariables
      (FullShape rowVariables logicalWidth publicFits) where
  point := zeroPoint rowVariables
  commitments := fun _ => zeroBundle
  publicInputs := fun _ => PiRLCAlgebra.PublicInput.publicZero
  evaluations := fun _ => ProductPaperAlgebraFor.evaluationZero rowVariables

/-- The four-component map sends the complete zero assignment to the public
zero bundle. The lane components use projections of the same zero witness. -/
theorem commit_zero
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape) :
    ProductCommitmentAlgebra.commit config
        (zeroAssignment rowVariables logicalWidth publicFits) =
      zeroBundle := by
  funext component
  cases component with
  | full => exact PiRLCAlgebra.Commitment.commit_zero config.fullKey
  | operations =>
      exact PiRLCAlgebra.Commitment.commit_zero config.operationsKey
  | initialSnapshot =>
      exact PiRLCAlgebra.Commitment.commit_zero config.snapshotKey
  | finalSnapshot =>
      exact PiRLCAlgebra.Commitment.commit_zero config.snapshotKey

/-- The concrete CE statement represented by one default running slot. -/
def slotStatement
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (slot : Fin (ProductNifsCodec.shapeFor rowVariables).runningCount) :
    CE.Instance (ProductPaperAlgebraFor.Structure rowVariables logicalWidth)
      (ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits)
      (ProductPaperAlgebraFor.Point rowVariables)
      (ProductPaperAlgebraFor.Evaluation rowVariables)
      ProductPaperAlgebraFor.Commitment where
  constraintSystem := ProductPaperAlgebraFor.matrixSource system
  commitment := (value rowVariables logicalWidth publicFits).commitments slot
  publicInput := (value rowVariables logicalWidth publicFits).publicInputs slot
  point := (value rowVariables logicalWidth publicFits).point
  evaluations := #[(value rowVariables logicalWidth publicFits).evaluations slot]
  stage := .fresh

/-- Every default slot is a valid fresh-stage CE claim with the same
all-zero full witness. This is the substantive universal-default property. -/
theorem slot_holds
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth
      publicFits operationsShape snapshotShape)
    (system : Phi81Relation.Structure
      (FullShape rowVariables logicalWidth publicFits))
    (slot : Fin (ProductNifsCodec.shapeFor rowVariables).runningCount) :
    CE.Holds (ProductPaperAlgebraFor.semantics config)
      productionGlobalParams (slotStatement system slot)
      (zeroAssignment rowVariables logicalWidth publicFits) := by
  refine ⟨?_, True.intro, ?_⟩
  · refine ⟨commit_zero config, ?_, ?_⟩
    · exact PiRLCAlgebra.PublicInput.projectPublicInput_zero
    · intro column
      exact Nat.zero_lt_succ 1
  · change
      #[ProductPaperAlgebraFor.evaluationFamily
          (ProductPaperAlgebraFor.matrixSource system)
          (zeroAssignment rowVariables logicalWidth publicFits)
          (zeroPoint rowVariables)] =
        #[ProductPaperAlgebraFor.evaluationZero rowVariables]
    apply congrArg (fun evaluation => #[evaluation])
    funext matrix
    exact BaseLinear.matrixEvaluation_zero
      (ProductPaperAlgebraFor.canonicalStructure
        (ProductPaperAlgebraFor.matrixSource system))
      (zeroPoint rowVariables) matrix

end Nightstream.Implementation.NebulaV2.ProductionPaperDefaultRunningFor
