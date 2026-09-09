import NightstreamFPrime.Lifecycle.Nifs.InteractiveOutput
import NightstreamFPrime.Lifecycle.Nifs.InteractiveAgreement
import NightstreamFPrime.Lifecycle.Nifs.SupportedContinuation

/-!
Selected interactive extraction from the actual checked prefix and its
reachable continuations. The ring algebra is fixed by the production key;
low-norm invertibility and correctness of costed primitives remain explicit.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.SupportedExtraction

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw

variable {Context State Tape : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (contexts : PMF Context)
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape relation ajtai running fresh contexts
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck))
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))
  (checkCorrect : ∀ context probe, publicCheck context probe = true ↔
    probe.FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) 9)
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
  (sourceCorrect : ∀ context, CheckedWitnessExtraction.Correct (width := 9)
    (sourceProgram context) (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context)))

include checkCorrect lowNorm correct bounded sourceCorrect in
/-- Every probability term uses the same reachable continuation. The right
side counts the source values that the checked projection actually returns.
There is no finite-work requirement on an impossible private state. -/
theorem returned_source_success_bound :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        publicCheck continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (PaperCompositionProbability.disagreementProbability contexts
        (InteractiveComposition.firstPhase originalFirstPhase publicCheck)
        (none : InteractiveComposition.Endpoint relation ajtai)
        (InteractiveComposition.suffixLaw relation ajtai running fresh continuation)
        (InteractiveComposition.consume relation ajtai program)
        (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
        (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) +
          IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program sourceProgram contexts := by
  dsimp only
  rw [InteractiveOutput.returnedSourceProbability_eq relation ajtai running fresh
    originalFirstPhase publicCheck _ program sourceProgram sourceCorrect contexts]
  exact InteractiveComposition.source_success_bound relation ajtai running fresh
    originalFirstPhase publicCheck _ program checkCorrect
    (PaperExtractionAlgebra.extractionAlgebra ajtai)
    (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm)
    correct bounds bounded contexts

include checkCorrect lowNorm correct bounded sourceCorrect in
/-- The selected interactive loss is the weak retry loss plus the strong
test loss and the measured binding event of two actual executions. A bound
on that event from public-seed MSIS hardness remains a separate contract. -/
theorem returned_source_bound_with_binding :
    let continuation := SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider
    StrongProbability.clockMean contexts
      (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
        publicCheck continuation) - InteractiveComposition.weakLoss relation ajtai -
      Real.sqrt (InteractiveAgreement.bindingProbability relation ajtai running fresh
        originalFirstPhase publicCheck continuation program contexts +
        IndependentExecution.testError productionShape 9) ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        publicCheck continuation program sourceProgram contexts := by
  dsimp only
  have source := returned_source_success_bound relation ajtai running fresh contexts
    originalFirstPhase publicCheck abortTape provider program sourceProgram checkCorrect lowNorm
    correct bounds bounded sourceCorrect
  have agreement := InteractiveAgreement.disagreement_le_bindingProbability relation ajtai running fresh
    originalFirstPhase publicCheck
    (SupportedContinuation.extension relation ajtai running fresh contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck) abortTape provider)
    program (Phi81Relation.PiRLCAlgebra.ForkStrongSet.strongSetUnits lowNorm) correct contexts
  exact (sub_le_sub_left (Real.sqrt_le_sqrt
    (_root_.add_le_add agreement (le_refl (IndependentExecution.testError productionShape 9)))) _).trans source

end NightstreamFPrime.Lifecycle.Nifs.SupportedExtraction
