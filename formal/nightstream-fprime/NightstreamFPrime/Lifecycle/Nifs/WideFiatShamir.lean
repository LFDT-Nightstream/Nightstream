import NightstreamFPrime.Lifecycle.Nifs.FiatShamirTransfer
import NightstreamFPrime.Lifecycle.PiRLC.Wide.Key
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.TranscriptHistory

/-! Security consumers for the total wide-sampler key. The real event uses
this key's verifier and openings for its exact sixteen returned children.
The approved classical FS transfer remains an explicit assumption; the
interactive extraction, same-key MSIS and test losses are unchanged.

The separate adaptive block comparison preserves raw replies and repeated
queries. Its q counts all block calls. Q in the FS model counts permutation
queries, including replay; neither number is inferred from the other.
A general g need not preserve additive error, so q * distance is not moved
through g. The concrete-to-ideal model error remains a separate hypothesis. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.WideFiatShamir

open scoped BigOperators
attribute [local instance] Classical.propDecidable

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev RealOutput (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :=
  FiatShamirTransfer.RealOutput relation

variable
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- The actual NIFS verifier accepts, and the supplied witnesses open its
exact sixteen returned children. The same proof supplies the PiDEC attempt. -/
def RealSuccess
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    Option (RealOutput relation) → Prop
  | none => False
  | some output =>
      let key := PiRLC.Wide.Key.key relation ajtai
      ∃ result attempt,
        PaperNonInteractive.verify key running fresh output.proof = some result ∧
        key.piDecAttempt running fresh output.proof = some attempt ∧
        ∀ child, CE.Holds key.piRlcSemantics key.params
          (PiDEC.OutputWitnessConsumer.runningStatement key result child) (output.children child)

/-- The real event's witnesses are for the exact verifier-computed PiDEC
children, in their original order, with no assumed output correspondence. -/
theorem realSuccess_implies_exact_children
    (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
    (output : RealOutput relation) (success : RealSuccess relation ajtai running fresh (some output)) :
    let key := PiRLC.Wide.Key.key relation ajtai
    ∃ result attempt,
      PaperNonInteractive.verify key running fresh output.proof = some result ∧
      key.piDecAttempt running fresh output.proof = some attempt ∧
      ∀ child, CE.Holds key.piRlcSemantics key.params
        (PiDEC.PaperVerifier.children key.piDecPublicInputSplit attempt child)
        (output.children (Fin.cast key.outputCount_eq child)) := by
  dsimp only
  rcases success with ⟨result, attempt, accepted, attemptEq, valid⟩
  refine ⟨result, attempt, accepted, attemptEq, ?_⟩
  intro child
  rw [← PiDEC.OutputWitnessConsumer.runningStatement_eq_child
    (PiRLC.Wide.Key.key relation ajtai) running fresh output.proof result attempt attemptEq accepted child]
  exact valid (Fin.cast (PiRLC.Wide.Key.key relation ajtai).outputCount_eq child)

variable {Context : Type*}
  (running : Context → Lifecycle.Running
    (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh
    (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- Success mass under the supplied classical adversary output law.
There is no caller-supplied scalar standing in for verifier success. -/
noncomputable def realSuccessProbability
    (law : PMF (Context × Option (RealOutput relation))) : ℝ :=
  ∑' outcome, if RealSuccess relation ajtai (running outcome.1) (fresh outcome.1) outcome.2
    then (law outcome).toReal else 0

/-- The translated experiment keeps the same context and hence the same
running/fresh public input law. The relation and Ajtai key are shared parameters. -/
noncomputable def contextLaw (law : PMF (Context × Option (RealOutput relation))) : PMF Context :=
  law.map Prod.fst

variable {State Tape : Type*}
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (law : PMF (Context × Option (RealOutput relation)))
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape relation ajtai running fresh (contextLaw relation law)
    (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running)))

/-- Exactly the success mean consumed by SupportedExtraction, on the
supported R/D continuation of the same checked causal prefix. -/
noncomputable def originalSuccessProbability : ℝ :=
  StrongProbability.clockMean (contextLaw relation law)
    (InteractiveComposition.originalSuccess relation ajtai running fresh originalFirstPhase
      (SupportedExtraction.publicCheck running)
      (SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
        (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
        abortTape provider))

/-- The existing parametric classical transfer boundary, for the wide verifier. Its sole field
transfers success to the typed interactive experiment. No source conclusion,
local correctness, replay/query theorem, or time bound is assumed here.
The admitted adversaries, history depth and total-query interpretation are
specified in FIAT_SHAMIR_MODEL.md. The protocol change was approved on
2026-09-24. No numerical model, g or deltaFS is supplied here. -/
structure FiatShamirModel (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat) : Prop where
  successTransfer :
    g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q ≤
      originalSuccessProbability relation ajtai running fresh law originalFirstPhase abortTape provider

variable
  (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
  (model : FiatShamirModel relation ajtai running fresh law originalFirstPhase abortTape provider g deltaFS Q)
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
  (sourceCorrect : ∀ context, CheckedWitnessExtraction.Correct (width := 9)
    (sourceProgram context) (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context)))

include model lowNorm correct bounded sourceCorrect in
/-- Transfer the v1.2 additive bound without changing the approved FS
model. The MSIS term is the actual stopped reduction under the same context
law; efficient translation and query applicability remain external. -/
theorem returned_source_bound_with_adaptive_msis :
    let continuation := SupportedContinuation.extension relation ajtai running fresh (contextLaw relation law)
      (InteractiveComposition.firstPhase originalFirstPhase (SupportedExtraction.publicCheck running))
      abortTape provider
    g Q (realSuccessProbability relation ajtai running fresh law) - deltaFS Q -
      InteractiveComposition.weakLoss relation ajtai - IndependentExecution.testError productionShape 9 -
      AdaptiveBindingProbability.successProbability relation ajtai program running fresh
        originalFirstPhase (SupportedExtraction.publicCheck running) continuation sourceProgram
        (contextLaw relation law) * PaperProfile.arity.total ≤
      InteractiveOutput.returnedSourceProbability relation ajtai running fresh originalFirstPhase
        (SupportedExtraction.publicCheck running) continuation program sourceProgram (contextLaw relation law) := by
  dsimp only
  have transfer := model.successTransfer
  unfold originalSuccessProbability at transfer
  have extracted := SupportedExtraction.returned_source_bound_with_adaptive_msis relation ajtai running fresh
    (contextLaw relation law) originalFirstPhase abortTape provider program sourceProgram lowNorm
    correct bounds bounded sourceCorrect
  exact (sub_le_sub_right (sub_le_sub_right (sub_le_sub_right transfer _) _) _).trans extracted

end NightstreamFPrime.Lifecycle.Nifs.WideFiatShamir
