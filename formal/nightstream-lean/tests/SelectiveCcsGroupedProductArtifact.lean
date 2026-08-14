import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FinalRowsReconstructSource

namespace Tests.SelectiveCcsGroupedProductArtifact

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Boolean
open Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowSourceBridge
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowsReconstructSource
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureSourceReconstruction
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowSemantics
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowRefinement

example : decodedSteps.length = 6 :=
  decodedSteps_length

example : decodedRows.length = 6 :=
  decodedRows_length

example (index : Fin 6) :
    (decodedStep index).emittedRow = (decodedRow index).emittedRow.val :=
  (generated_steps_and_rows_join index).1

example (index : Fin 6) :
    PortImagesMatch finalColumnCount_positive
      (decodedFixedRow index).ports (decodedStep index)
      decodedSourceSlots decodedSourceDefinitions decodedDerivedSlots
      sourceFuel :=
  generated_step_images_match index

example (index : Fin 6) (assignment : Fin finalColumnCount → F) :
    action ((decodedFixedRow index).ports Role.c.index) assignment =
      Form.evaluate
        (expectedCForm finalColumnCount_positive decodedSourceSlots
          decodedSourceDefinitions decodedDerivedSlots sourceFuel
          (decodedStep index))
        assignment :=
  generated_c_action_eq_source_image index assignment

example (index : Fin 6) (factor : Fin 5)
    (assignment : Fin finalColumnCount → F) :
    action
          ((decodedFixedRow index).ports (factorRoles factor).1.index)
          assignment =
        Form.evaluate
          (factorFormAt finalColumnCount_positive decodedSourceSlots
            decodedSourceDefinitions sourceFuel factor.val true
            (decodedStep index)) assignment ∧
      action
          ((decodedFixedRow index).ports (factorRoles factor).2.index)
          assignment =
        Form.evaluate
          (factorFormAt finalColumnCount_positive decodedSourceSlots
            decodedSourceDefinitions sourceFuel factor.val false
            (decodedStep index)) assignment :=
  generated_factor_actions_eq_source_images index factor assignment

example : decodedSourceDefinitions.length = 1 :=
  decodedSourceDefinitions_length

example :
    (decodedSourceDefinitions.get ⟨0, by
      rw [decodedSourceDefinitions_length]
      decide⟩).target.val = 15 :=
  generated_affine_definition_exact.1

example (assignment : Fin finalColumnCount → F) :
    action
          ((decodedFixedRow 2).ports (factorRoles 0).1.index)
          assignment =
        generatedAffineFactor.coefficient *
          sourceLinearValue finalColumnCount_positive decodedSourceSlots
            decodedSourceDefinitions sourceFuel generatedAffineFactor.left
            assignment :=
  (generated_affine_factor_actions_eq_source_values assignment).1

example
    (index : Fin 6)
    (assignment : Fin (decodedRow index).columns → F)
    (selectorOne : assignment (selectorColumn index) = 1) :
    residual (decodedRow index) assignment = 0 ↔
      action ((decodedRow index).port Role.c.index) assignment =
        Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.GroupedProduct.fiveProductSum
          (action ((decodedRow index).port Role.bit.index) assignment)
          (action ((decodedRow index).port Role.a.index) assignment)
          (action ((decodedRow index).port Role.b.index) assignment)
          (action ((decodedRow index).port Role.sboxInput.index) assignment)
          (action ((decodedRow index).port Role.centeredUnit.index) assignment)
          (action ((decodedRow index).port Role.canonicalDigit.index) assignment)
          (action ((decodedRow index).port Role.canonicalBorrow.index) assignment)
          (action ((decodedRow index).port Role.canonicalNextBorrow.index) assignment)
          (action ((decodedRow index).port Role.canonicalBoundDigit.index) assignment)
          (action ((decodedRow index).port Role.evalTailRight.index) assignment) :=
  generated_row_zero_iff_fiveProduct index assignment selectorOne

example :
    Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.PolynomialCertificate.identityPolynomial
        (decodedStep 0) (decodedStep 1) = qCertificatePolynomial :=
  q_polynomial_exact

example :
    Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.PolynomialCertificate.identityPolynomial
        (decodedStep 2) (decodedStep 3) = pCertificatePolynomial :=
  p_polynomial_exact

example :
    Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.PolynomialCertificate.identityPolynomial
        (decodedStep 4) (decodedStep 5) = rCertificatePolynomial :=
  r_polynomial_exact

example (assignment : Fin sourceColumnCount → F)
    (holds : RowsHold decodedSourceRows assignment 1) :
    ∀ index : Fin 6,
      StepHolds (decodedStep index) assignment 1
        (fixtureDerivedValues assignment) :=
  sourceRows_imply_all_steps_hold assignment holds

example (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1)
    (selectors : SelectorsOne assignment)
    (holds : ActiveRowsHold assignment) :
    ∃ sourceAssignment : Nat → F,
      boundary sourceAssignment =
          boundary (retainedSource (fixtureSourceAssignment assignment)) ∧
        ∀ row ∈ rawSourceRows, Holds 1 sourceAssignment row :=
  active_final_rows_have_source_witness assignment constantOne selectors holds

end Tests.SelectiveCcsGroupedProductArtifact
