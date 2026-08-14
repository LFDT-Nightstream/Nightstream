import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FinalRowSourceBridge
import Nightstream.Implementation.R1CS.Correspondence.SelectiveCcs.Rewrite.Artifact.FixtureSourceReconstruction

/-!
Contract: end-to-end same-assignment reconstruction for the exact generated
grouped-product fixture.

Assurance tier: artifact-checked same-assignment refinement.

Owns: transport from the six active final rows to the three compact semantic
equations, and canonical reconstruction of all thirty source-only temporary
values needed by the exact thirty-three source rows.

Does not own: low-norm validity, selector authority outside the supplied
hypotheses, production-family coverage, or permission to remove a production
row or coordinate.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 10000
set_option maxHeartbeats 1000000

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowsReconstructSource

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveGroupedProductRewriteFixture
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.Decoder
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowSourceBridge
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureRefinement
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FixtureSourceReconstruction
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceImage
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRelation
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.SourceRowSemantics
open Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.KaratsubaDotProduct

local instance : Std.Associative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_assoc⟩

local instance : Std.Commutative (fun (left right : F) => left + right) :=
  ⟨Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws.add_comm⟩

private theorem mappedValue (value : F) :
    (ZMod.finEquiv goldilocksModulus) value =
      (value.val : ZMod goldilocksModulus) := by
  symm
  exact ZMod.natCast_zmod_val ((ZMod.finEquiv goldilocksModulus) value)

private theorem zmodMinusSeven :
    (18446744069414584314 : ZMod goldilocksModulus) = -7 := by
  decide

private theorem zmodMinusSix :
    (18446744069414584315 : ZMod goldilocksModulus) = -6 := by
  decide

/-- Total view of a bounded source assignment. Out-of-range columns fail
closed to zero. -/
def retainedSource (assignment : Fin sourceColumnCount → F) (column : Nat) : F :=
  if inRange : column < sourceColumnCount then
    assignment ⟨column, inRange⟩
  else
    0

/-- The six exact decoded recurrences imply the three emitted Karatsuba
equations on the same source assignment. -/
theorem decoded_steps_imply_emitted
    (assignment : Fin sourceColumnCount → F) (derived : Nat → F)
    (steps : ∀ index : Fin 6,
      StepHolds (decodedStep index) assignment 1 derived) :
    EmittedHolds (boundary (retainedSource assignment)) := by
  have step00 := steps 0
  have step01 := steps 1
  have step02 := steps 2
  have step03 := steps 3
  have step04 := steps 4
  have step05 := steps 5
  have stepFacts := And.intro step00
    (And.intro step01
      (And.intro step02
        (And.intro step03 (And.intro step04 step05))))
  dsimp [decodedStep, decodedSteps, decodeStep, decodeRange, decodeOutput,
    decodeFactor, decodeSourceLinearCombination, decodeSourceTerm,
    SourceTermsValid,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
    sourceRowCount, sourceColumnCount, rawSteps, rawStep00, rawStep01,
    rawStep02, rawStep03, rawStep04, rawStep05] at stepFacts
  simp [decodeStep, decodeRange, decodeOutput, decodeFactor,
    decodeSourceLinearCombination, decodeSourceTerm, SourceTermsValid,
    Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Decoder.decodeField,
    goldilocksModulus,
    StepHolds, outputValue, previousValue, linearValue,
    directSourceLinearValue, factorSum, factorValue] at stepFacts
  rcases stepFacts with ⟨step00, step01, step02, step03, step04, step05⟩
  unfold EmittedHolds boundary retainedSource
  simp [sourceColumnCount, lowLeftColumn, highLeftColumn, lowRightColumn,
    highRightColumn, sumSix, highHighValue, lowLowValue, crossValue]
  rw [step00] at step01
  rw [step02] at step03
  rw [step04] at step05
  constructor
  · apply (ZMod.finEquiv goldilocksModulus).injective
    have mapped := congrArg (ZMod.finEquiv goldilocksModulus) step01
    simp only [map_add, map_mul, mappedValue] at mapped ⊢
    norm_num [goldilocksModulus] at mapped ⊢
    convert mapped using 1 <;> try ring
  constructor
  · apply (ZMod.finEquiv goldilocksModulus).injective
    have mapped := congrArg (ZMod.finEquiv goldilocksModulus) step03
    simp only [map_add, map_sub, map_mul, mappedValue] at mapped ⊢
    norm_num [goldilocksModulus] at mapped ⊢
    rw [zmodMinusSeven] at mapped
    convert mapped using 1 <;> try ring
  · apply (ZMod.finEquiv goldilocksModulus).injective
    have mapped := congrArg (ZMod.finEquiv goldilocksModulus) step05
    simp only [map_add, map_sub, map_mul, mappedValue] at mapped ⊢
    norm_num [goldilocksModulus] at mapped ⊢
    rw [zmodMinusSix] at mapped
    convert mapped using 1 <;> try ring

/-- The six decoded recurrences imply the compact semantic relation. -/
theorem decoded_steps_imply_reduced
    (assignment : Fin sourceColumnCount → F) (derived : Nat → F)
    (steps : ∀ index : Fin 6,
      StepHolds (decodedStep index) assignment 1 derived) :
    ReducedHolds (boundary (retainedSource assignment)) :=
  (emitted_iff_reduced _).mp
    (decoded_steps_imply_emitted assignment derived steps)

/-- The canonical reconstruction changes no retained input or output in the
compact boundary. -/
theorem boundary_reconstructed (retained : Nat → F) :
    boundary (reconstructedAssignment retained) = boundary retained := by
  unfold boundary
  congr <;> funext index <;> fin_cases index <;> rfl

/-- Six active generated final rows imply the three compact semantic
equations on the source values extracted from the same final assignment. -/
theorem active_final_rows_imply_reduced
    (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1)
    (selectors : SelectorsOne assignment)
    (holds : ActiveRowsHold assignment) :
    ReducedHolds
      (boundary (retainedSource (fixtureSourceAssignment assignment))) :=
  decoded_steps_imply_reduced
    (fixtureSourceAssignment assignment)
    (fixtureDerivedAssignment assignment)
    (stepHolds_of_activeRows assignment constantOne selectors holds)

/-- Six active generated final rows reconstruct one explicit assignment that
satisfies every exact source row. -/
theorem active_final_rows_reconstruct_source
    (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1)
    (selectors : SelectorsOne assignment)
    (holds : ActiveRowsHold assignment) :
    ∀ row ∈ rawSourceRows,
      Holds 1
        (reconstructedAssignment
          (retainedSource (fixtureSourceAssignment assignment))) row :=
  generated_source_rows_hold _
    (active_final_rows_imply_reduced assignment constantOne selectors holds)

/-- End-to-end same-boundary existential refinement for the exact generated
fixture. The witness is the canonical reconstruction of only the thirty
source-only temporary values. -/
theorem active_final_rows_have_source_witness
    (assignment : Fin finalColumnCount → F)
    (constantOne : assignment ⟨0, finalColumnCount_positive⟩ = 1)
    (selectors : SelectorsOne assignment)
    (holds : ActiveRowsHold assignment) :
    ∃ sourceAssignment : Nat → F,
      boundary sourceAssignment =
          boundary (retainedSource (fixtureSourceAssignment assignment)) ∧
        ∀ row ∈ rawSourceRows, Holds 1 sourceAssignment row := by
  let retained := retainedSource (fixtureSourceAssignment assignment)
  exact ⟨reconstructedAssignment retained,
    boundary_reconstructed retained,
    active_final_rows_reconstruct_source assignment constantOne selectors
      holds⟩

end Nightstream.Implementation.R1CS.SelectiveCcs.Rewrite.Artifact.FinalRowsReconstructSource
