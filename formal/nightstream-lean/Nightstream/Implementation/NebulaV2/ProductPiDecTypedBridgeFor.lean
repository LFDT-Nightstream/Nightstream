import Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge
import Nightstream.Implementation.NebulaV2.ProductPaperAlgebraFor

/-!
Contract: exponent-indexed typed refinement of the PiDEC coordinate rows.

The physical PiDEC relation always recomposes four commitment components and
fourteen quadratic-extension evaluation families. This geometry is independent
of the augmented-relation exponent. This module gives those wires the exact
exponent-indexed paper types and derives the independent PiDEC `Accepted`
predicate from row satisfaction.

Placement binds only decoded row values to one verifier-computed attempt. It
does not assume a recomposition equation or verifier acceptance.

Does not own PiCCS, PiRLC, generated placement, terminal verification,
cryptographic security, or implementation refinement.

Assurance tier: exponent-indexed row-to-paper refinement.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridgeFor

open Nightstream.Implementation.NebulaV2.ProductPiDecRows
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev decodeBundle := ProductPiDecTypedBridge.decodeBundle

/-- Decode one complete evaluation family at the selected exponent. -/
def decodeEvaluation
    (rowVariables : Nat) (layout : EvaluationLayout)
    (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductPaperAlgebraFor.Evaluation rowVariables :=
  ProductPiDecTypedBridge.decodeEvaluation layout assignment canonical

/-- Satisfied PiDEC rows derive exact exponent-indexed commitment and
evaluation recomposition. -/
theorem typedEquations_of_rows
    {rowVariables : Nat} {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ProductPiDecRows.rows layout) assignment) :
    (decodeBundle layout.parentBundle assignment canonical =
        ProductCommitmentAlgebra.recomposeBundles fun child =>
          decodeBundle (layout.childBundle child) assignment canonical) /\
      (decodeEvaluation rowVariables layout.parentEvaluation assignment
          canonical =
        ProductPaperAlgebraFor.recomposeEvaluationFamily fun child =>
          decodeEvaluation rowVariables (layout.childEvaluation child)
            assignment canonical) := by
  exact ProductPiDecTypedBridge.typedEquations_of_rows canonical one satisfies

/-- Operational PiDEC attempt at the exact relation exponent. -/
abbrev ExactAttempt
    (rowVariables logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  PiDEC.PaperVerifier.Attempt
    (ProductPaperAlgebraFor.Structure rowVariables logicalWidth)
    (ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits)
    (ProductPaperAlgebraFor.Point rowVariables)
    (ProductPaperAlgebraFor.Evaluation rowVariables)
    ProductPaperAlgebraFor.Commitment productionGlobalParams

/-- Column placement for one verifier-computed exponent-indexed attempt. -/
structure Placement
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (attempt : ExactAttempt rowVariables logicalWidth publicFits) : Prop where
  parentBundle :
    attempt.parent.commitment =
      decodeBundle layout.parentBundle assignment canonical
  childBundle : forall child,
    (attempt.messages child).commitment =
      decodeBundle (layout.childBundle child) assignment canonical
  parentEvaluation :
    attempt.parent.evaluations =
      #[decodeEvaluation rowVariables layout.parentEvaluation assignment
        canonical]
  childEvaluation : forall child,
    (attempt.messages child).evaluations =
      #[decodeEvaluation rowVariables (layout.childEvaluation child) assignment
        canonical]

/-- Exact rows imply independent paper PiDEC acceptance at one exponent. -/
theorem paperAccepted_of_rows_for_attempt
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : ProductPaperAlgebraFor.Config rowVariables logicalWidth publicFits
      operationsShape snapshotShape)
    (attempt : ExactAttempt rowVariables logicalWidth publicFits)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ProductPiDecRows.rows layout) assignment)
    (placement : Placement layout assignment canonical attempt)
    (parentCombined : attempt.parent.stage = .combined)
    (parentEvaluationCount :
      (ProductPaperAlgebraFor.evaluationArity config).count
        attempt.parent.constraintSystem = 1) :
    PiDEC.PaperVerifier.Accepted (ProductPaperAlgebraFor.piDecAlgebra config)
      (ProductPaperAlgebraFor.evaluationArity config) attempt := by
  have equations := typedEquations_of_rows
    (rowVariables := rowVariables) canonical one satisfies
  constructor
  · exact parentCombined
  · rw [placement.parentEvaluation, parentEvaluationCount]
    rfl
  · intro child
    rw [placement.childEvaluation, parentEvaluationCount]
    rfl
  · change
      attempt.parent.commitment =
        ProductCommitmentAlgebra.recomposeBundles fun child =>
          (attempt.messages child).commitment
    have childBundles :
        (fun child => (attempt.messages child).commitment) =
          fun child =>
            decodeBundle (layout.childBundle child) assignment canonical := by
      funext child
      exact placement.childBundle child
    calc
      attempt.parent.commitment =
          decodeBundle layout.parentBundle assignment canonical :=
        placement.parentBundle
      _ = ProductCommitmentAlgebra.recomposeBundles (fun child =>
            decodeBundle (layout.childBundle child) assignment canonical) :=
        equations.1
      _ = ProductCommitmentAlgebra.recomposeBundles (fun child =>
            (attempt.messages child).commitment) :=
        (congrArg ProductCommitmentAlgebra.recomposeBundles childBundles).symm
  · change
      attempt.parent.evaluations =
        ProductPaperAlgebraFor.recomposeEvaluations fun child =>
          (attempt.messages child).evaluations
    rw [placement.parentEvaluation]
    have childArrays :
        (fun child => (attempt.messages child).evaluations) =
          fun child =>
            #[decodeEvaluation rowVariables (layout.childEvaluation child)
              assignment canonical] := by
      funext child
      exact placement.childEvaluation child
    calc
      #[decodeEvaluation rowVariables layout.parentEvaluation assignment
          canonical] =
          #[ProductPaperAlgebraFor.recomposeEvaluationFamily (fun child =>
            decodeEvaluation rowVariables (layout.childEvaluation child)
              assignment canonical)] :=
        congrArg (fun value => #[value]) equations.2
      _ = ProductPaperAlgebraFor.recomposeEvaluations (fun child =>
            #[decodeEvaluation rowVariables (layout.childEvaluation child)
              assignment canonical]) := by
        rfl
      _ = ProductPaperAlgebraFor.recomposeEvaluations (fun child =>
            (attempt.messages child).evaluations) :=
        (congrArg ProductPaperAlgebraFor.recomposeEvaluations childArrays).symm

end Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridgeFor
