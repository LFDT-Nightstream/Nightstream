import Nightstream.Implementation.NebulaV2.NIFS.PiRLC.AlgebraSound
import Nightstream.Implementation.NebulaV2.NIFS.Core.PaperAlgebraFor

/-!
Contract: exponent-indexed typed meaning of the complete PiRLC algebra rows.

The physical PiRLC algebra has 110 ring families for every relation exponent:
72 commitment rings, ten public rings, and 28 evaluation-limb rings. The row
exponent changes the SumCheck point width, but it does not change these ring
families. This module gives their decoded values the exact exponent-indexed
paper types and transfers the existing row theorem to those types.

It does not own transcript sampling, PiCCS, PiDEC, generated placement,
cryptographic security, or implementation refinement.

Assurance tier: exponent-indexed row-to-paper refinement.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSoundFor

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraRows
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

abbrev decodeChallenges := ProductPiRlcAlgebraSound.decodeChallenges
abbrev decodeInputBundles := ProductPiRlcAlgebraSound.decodeInputBundles
abbrev decodeOutputBundle := ProductPiRlcAlgebraSound.decodeOutputBundle
abbrev decodeInputPublicRings :=
  ProductPiRlcAlgebraSound.decodeInputPublicRings
abbrev decodeOutputPublicRings :=
  ProductPiRlcAlgebraSound.decodeOutputPublicRings
abbrev decodeInputEvaluationLimb :=
  ProductPiRlcAlgebraSound.decodeInputEvaluationLimb
abbrev decodeOutputEvaluationLimb :=
  ProductPiRlcAlgebraSound.decodeOutputEvaluationLimb

/-- Reconstruct the exact 540-field public input at any row exponent. -/
def decodeInputPublic
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Source -> ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth
      publicFits :=
  ProductPiRlcAlgebraSound.decodeInputPublic
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    layout assignment canonical

/-- Reconstruct the combined 540-field public input at any row exponent. -/
def decodeOutputPublic
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductPaperAlgebraFor.PublicInput rowVariables logicalWidth publicFits :=
  ProductPiRlcAlgebraSound.decodeOutputPublic
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    layout assignment canonical

/-- Decode all source evaluation families at the selected exponent. -/
def decodeInputEvaluations
    (rowVariables : Nat) (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    Source -> ProductPaperAlgebraFor.Evaluation rowVariables :=
  ProductPiRlcAlgebraSound.decodeInputEvaluations layout assignment canonical

/-- Decode the combined evaluation family at the selected exponent. -/
def decodeOutputEvaluation
    (rowVariables : Nat) (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    ProductPaperAlgebraFor.Evaluation rowVariables :=
  ProductPiRlcAlgebraSound.decodeOutputEvaluation layout assignment canonical

/-- The physical public decoder is definitionally the exponent-indexed public
decoder because every profile has exactly ten public rings. -/
theorem decodeInputPublic_eq_reference
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    decodeInputPublic (rowVariables := rowVariables)
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        layout assignment canonical =
      ProductPiRlcAlgebraSound.decodeInputPublic
        (logicalWidth := logicalWidth) (publicFits := publicFits)
        layout assignment canonical := by
  rfl

/-- The physical evaluation decoder is definitionally exponent-independent
because the profile fixes fourteen matrices and two extension limbs. -/
theorem decodeInputEvaluations_eq_reference
    (rowVariables : Nat) (layout : Layout) (assignment : Nat -> Nat)
    (canonical : forall column, assignment column < goldilocksP) :
    decodeInputEvaluations rowVariables layout assignment canonical =
      ProductPiRlcAlgebraSound.decodeInputEvaluations layout assignment
        canonical := by
  rfl

/-- Satisfied rows derive all three exact exponent-indexed PiRLC parent
equations. -/
theorem typedEquations_of_rows
    {rowVariables logicalWidth : Nat}
    {publicFits : 540 <= Phi81CarrierLayout.carrierWidth logicalWidth}
    {layout : Layout} {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : forall source lane,
      assignment (layout.challengeSymbol source lane) < 5)
    (satisfied : Satisfies (ProductPiRlcAlgebraRows.rows layout) assignment) :
    (decodeOutputBundle layout assignment canonical =
        ProductCommitmentAlgebra.combineBundles
          (decodeChallenges layout assignment range)
          (decodeInputBundles layout assignment canonical)) /\
      (decodeOutputPublic (rowVariables := rowVariables)
          (logicalWidth := logicalWidth) (publicFits := publicFits)
          layout assignment canonical =
        Phi81Relation.PiRLCAlgebra.PublicInput.combinePublicInputs
          (decodeChallenges layout assignment range)
          (decodeInputPublic (rowVariables := rowVariables)
            (logicalWidth := logicalWidth) (publicFits := publicFits)
            layout assignment canonical)) /\
      (decodeOutputEvaluation rowVariables layout assignment canonical =
        ProductPaperAlgebraFor.combineEvaluationFamily
          (decodeChallenges layout assignment range)
          (decodeInputEvaluations rowVariables layout assignment canonical)) := by
  exact ProductPiRlcAlgebraSound.typedEquations_of_rows
    (logicalWidth := logicalWidth) (publicFits := publicFits)
    canonical one range satisfied

end Nightstream.Implementation.NebulaV2.ProductPiRlcAlgebraSoundFor
