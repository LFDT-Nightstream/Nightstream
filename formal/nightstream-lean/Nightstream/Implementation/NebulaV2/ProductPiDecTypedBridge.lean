import Nightstream.Implementation.NebulaV2.ProductConcreteNifs
import Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination

/-!
Contract: typed refinement of the exact V2 PiDEC coordinate rows.

Owns decoding of the parent and fourteen child wire families into the exact
four-component commitment and packed quadratic-extension evaluation types;
derivation of both typed PiDEC recomposition equations from row satisfaction;
and the final bridge to the paper verifier's independent `Accepted` predicate.

The placement interface states only equality between decoded wire values and
the exact fields of the verifier-computed attempt. It does not assume either
PiDEC recomposition equation or verifier acceptance.

Does not own column placement in a full recursive artifact, PiCCS, PiRLC,
transcript rows, Rust refinement, or cryptographic soundness.

Emits constraints: no; it proves the meaning of `ProductPiDecRows.rows`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge

open Nightstream.Implementation.NebulaV2.ProductPiDecRows
open Nightstream.Implementation.NebulaV2.ProductPiDecLinearCombination
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Decode one complete four-component public commitment. -/
def decodeBundle
    (layout : BundleLayout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductCommitmentAlgebra.BundleValue :=
  fun component row lane =>
    fieldAt assignment canonical (layout.column component row lane)

/-- Decode one complete 14-by-54 quadratic-extension evaluation family. -/
def decodeEvaluation
    (layout : EvaluationLayout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductPaperAlgebra.Evaluation :=
  fun matrix coefficient =>
    ⟨fieldAt assignment canonical
        (layout.column matrix coefficient ⟨0, by decide⟩),
      fieldAt assignment canonical
        (layout.column matrix coefficient ⟨1, by decide⟩)⟩

/-- The commitment coordinates imply the exact four-component typed paper
equation. -/
theorem commitmentEquation_of_accepted
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (accepted : ProductPiDecRows.Accepted layout assignment) :
    decodeBundle layout.parentBundle assignment canonical =
      ProductCommitmentAlgebra.recomposeBundles fun child =>
        decodeBundle (layout.childBundle child) assignment canonical := by
  funext component row lane
  rw [recomposeBundles_coordinate]
  apply recomposes_field canonical
  simpa [ProductPiDecRows.radixPowers] using
    accepted.commitment component row lane

/-- The evaluation coordinates imply the exact packed quadratic-extension
paper equation. -/
theorem evaluationEquation_of_accepted
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (accepted : ProductPiDecRows.Accepted layout assignment) :
    decodeEvaluation layout.parentEvaluation assignment canonical =
      ProductPaperAlgebra.recomposeEvaluationFamily fun child =>
        decodeEvaluation (layout.childEvaluation child) assignment canonical := by
  funext matrix coefficient
  change K.mk _ _ = K.mk _ _
  congr 1
  · change
      fieldAt assignment canonical
          (layout.parentEvaluation.column matrix coefficient ⟨0, by decide⟩) =
        (BaseLinear.combineEvaluations PiDEC.radixWeight
          (fun child =>
            decodeEvaluation (layout.childEvaluation child)
              assignment canonical matrix) coefficient).c0
    rw [combineEvaluations_c0]
    apply recomposes_field canonical
    simpa [ProductPiDecRows.radixPowers] using
      accepted.evaluation matrix coefficient ⟨0, by decide⟩
  · change
      fieldAt assignment canonical
          (layout.parentEvaluation.column matrix coefficient ⟨1, by decide⟩) =
        (BaseLinear.combineEvaluations PiDEC.radixWeight
          (fun child =>
            decodeEvaluation (layout.childEvaluation child)
              assignment canonical matrix) coefficient).c1
    rw [combineEvaluations_c1]
    apply recomposes_field canonical
    simpa [ProductPiDecRows.radixPowers] using
      accepted.evaluation matrix coefficient ⟨1, by decide⟩

/-- Direct row-level form of both typed equations. -/
theorem typedEquations_of_rows
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ProductPiDecRows.rows layout) assignment) :
    (decodeBundle layout.parentBundle assignment canonical =
        ProductCommitmentAlgebra.recomposeBundles fun child =>
          decodeBundle (layout.childBundle child) assignment canonical) /\
      (decodeEvaluation layout.parentEvaluation assignment canonical =
        ProductPaperAlgebra.recomposeEvaluationFamily fun child =>
          decodeEvaluation (layout.childEvaluation child) assignment canonical) := by
  have accepted := ProductPiDecRows.rows_sound canonical one satisfies
  exact
    ⟨commitmentEquation_of_accepted canonical accepted,
      evaluationEquation_of_accepted canonical accepted⟩

/-! ## Exact paper-verifier placement and acceptance -/

/-- Exact operational PiDEC attempt type selected by the V2 key. -/
abbrev ExactAttempt
    (logicalWidth : Nat)
    (publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth) :=
  PiDEC.PaperVerifier.Attempt
    (ProductPaperAlgebra.Structure logicalWidth)
    (ProductPaperAlgebra.PublicInput logicalWidth publicFits)
    ProductPaperAlgebra.Point
    ProductPaperAlgebra.Evaluation
    ProductPaperAlgebra.Commitment
    productionGlobalParams

/-- Column placement for one verifier-computed PiDEC attempt.

This interface binds decoded row values to the attempt fields. It does not
assume a recomposition equation, a combined parent stage, or verifier
acceptance. A complete recursive artifact must derive this placement from
its parser and column map. -/
structure Placement
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP)
    (attempt : ExactAttempt logicalWidth publicFits) : Prop where
  parentBundle :
    attempt.parent.commitment =
      decodeBundle layout.parentBundle assignment canonical
  childBundle : forall child,
    (attempt.messages child).commitment =
      decodeBundle (layout.childBundle child) assignment canonical
  parentEvaluation :
    attempt.parent.evaluations =
      #[decodeEvaluation layout.parentEvaluation assignment canonical]
  childEvaluation : forall child,
    (attempt.messages child).evaluations =
      #[decodeEvaluation (layout.childEvaluation child) assignment canonical]

/-- Exact PiDEC rows imply acceptance for any verifier-computed attempt over
the fixed product algebra. Transcript-profile selection is outside this
theorem. -/
theorem paperAccepted_of_rows_for_attempt
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (attempt : ExactAttempt logicalWidth publicFits)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ProductPiDecRows.rows layout) assignment)
    (placement : Placement layout assignment canonical attempt)
    (parentCombined : attempt.parent.stage = .combined)
    (parentEvaluationCount :
      (ProductPaperAlgebra.evaluationArity config).count
        attempt.parent.constraintSystem = 1) :
    PiDEC.PaperVerifier.Accepted (ProductPaperAlgebra.piDecAlgebra config)
      (ProductPaperAlgebra.evaluationArity config) attempt := by
  have equations := typedEquations_of_rows canonical one satisfies
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
        ProductPaperAlgebra.recomposeEvaluations fun child =>
          (attempt.messages child).evaluations
    rw [placement.parentEvaluation]
    have childArrays :
        (fun child => (attempt.messages child).evaluations) =
          fun child =>
            #[decodeEvaluation (layout.childEvaluation child)
                assignment canonical] := by
      funext child
      exact placement.childEvaluation child
    calc
      #[decodeEvaluation layout.parentEvaluation assignment canonical] =
          #[ProductPaperAlgebra.recomposeEvaluationFamily (fun child =>
            decodeEvaluation (layout.childEvaluation child)
              assignment canonical)] :=
        congrArg (fun value => #[value]) equations.2
      _ = ProductPaperAlgebra.recomposeEvaluations (fun child =>
            #[decodeEvaluation (layout.childEvaluation child)
                assignment canonical]) := by
        rfl
      _ = ProductPaperAlgebra.recomposeEvaluations (fun child =>
            (attempt.messages child).evaluations) :=
        (congrArg ProductPaperAlgebra.recomposeEvaluations childArrays).symm

/-- The exact 5,400 coordinate rows imply the independent paper PiDEC
acceptance predicate for the verifier-computed V2 attempt.

The only artifact-specific premise is `Placement`, which states where the
already computed attempt fields occur. In particular, it does not contain
either recomposition equation or an acceptance Boolean. -/
theorem paperAccepted_of_rows
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof :
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Proof
        K ProductPaperAlgebra.Commitment ProductNifsCodec.shape 9)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ProductPiDecRows.rows layout) assignment)
    (placement : Placement layout assignment canonical
      ((ProductConcreteNifs.key statementId config artifact).piDecAttempt
        running fresh proof)) :
    PiDEC.PaperVerifier.Accepted
      (ProductConcreteNifs.key statementId config artifact).piDecAlgebra
      (ProductConcreteNifs.key statementId config artifact).piDecEvaluationArity
      ((ProductConcreteNifs.key statementId config artifact).piDecAttempt
        running fresh proof) := by
  change PiDEC.PaperVerifier.Accepted
    (ProductPaperAlgebra.piDecAlgebra config)
    (ProductPaperAlgebra.evaluationArity config)
    ((ProductConcreteNifs.key statementId config artifact).piDecAttempt
      running fresh proof)
  exact paperAccepted_of_rows_for_attempt config
    ((ProductConcreteNifs.key statementId config artifact).piDecAttempt
      running fresh proof) canonical one satisfies placement rfl rfl

/-- The same exact rows make the concrete paper verifier's PiDEC Boolean
true. No caller supplies an acceptance Boolean. -/
theorem piDecCheck_true_of_rows
    {logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {operationsShape snapshotShape : Phi81Relation.Shape}
    (statementId : ProductConcreteNifs.StatementId)
    (config : ProductPaperAlgebra.Config logicalWidth publicFits
      operationsShape snapshotShape)
    (artifact : ProductConcreteNifs.RelationArtifact logicalWidth publicFits)
    (running : ProductNifsCodec.Running
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (fresh : ProductNifsCodec.Fresh
      (ProductPaperAlgebra.FullShape logicalWidth publicFits))
    (proof :
      Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.Proof
        K ProductPaperAlgebra.Commitment ProductNifsCodec.shape 9)
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (ProductPiDecRows.rows layout) assignment)
    (placement : Placement layout assignment canonical
      ((ProductConcreteNifs.key statementId config artifact).piDecAttempt
        running fresh proof)) :
    Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piDecCheck
        (ProductConcreteNifs.key statementId config artifact)
        running fresh proof = true := by
  apply
    (Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.piDecCheck_eq_true_iff
      (ProductConcreteNifs.key statementId config artifact)
      running fresh proof).2
  exact paperAccepted_of_rows statementId config artifact running fresh proof
    canonical one satisfies placement

end Nightstream.Implementation.NebulaV2.ProductPiDecTypedBridge
