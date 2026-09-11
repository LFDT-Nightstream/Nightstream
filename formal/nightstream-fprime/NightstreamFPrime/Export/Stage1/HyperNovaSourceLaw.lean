import NightstreamFPrime.Spec.Folding.Nifs.SequentialOutputLaw
import NightstreamFPrime.Export.Stage1.PiCCSStoredSourceProbability
import NightstreamFPrime.Lifecycle.Nifs.SupportedExtraction

/-!
Owns the selected checked source-result PMF used by a reverse HyperNova
step. It maps the existing sequential NIFS output through the same stored
checker, preserving its context and every abort.

The continuation and primitive program are existing experiment parameters.
The selected NifsClosure consumer supplies its concrete provider extension
and primitive constructor. The equalities here need no correctness, work,
Fiat--Shamir, or event-equality premise.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaSourceLaw

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction
open NightstreamFPrime.Lifecycle
open PiRLC.PaperForkExtractionWork (Primitives)
open PiRLC.CoordinateForkLaw (Challenge)
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiCCSStoredWitnessCheck (carrier commit statement)
open PiDECInputCheck (relation)

private theorem map_eq_on_support {Sample Left : Type*}
    (distribution : PMF Sample) (first second : Sample → Left)
    (same : ∀ sample ∈ distribution.support, first sample = second sample) :
    distribution.map first = distribution.map second := by
  classical
  apply PMF.ext
  intro output
  simp only [PMF.map_apply]
  apply tsum_congr
  intro sample
  by_cases zero : distribution sample = 0
  · simp only [zero, ite_self]
  · rw [same sample ((distribution.mem_support_iff sample).mpr zero)]

private theorem reattach_context {Context Value : Type*}
    (distribution : PMF (Context × Value)) (context : Context)
    (marginal : distribution.map Prod.fst = PMF.pure context) :
    (distribution.map Prod.snd).map (fun value => (context, value)) = distribution := by
  rw [PMF.map_comp]
  calc
    distribution.map ((fun value => (context, value)) ∘ Prod.snd) =
        distribution.map id := by
      apply map_eq_on_support
      intro sample supported
      have observed : sample.1 ∈ (distribution.map Prod.fst).support :=
        (PMF.mem_support_map_iff Prod.fst distribution sample.1).mpr
          ⟨sample, supported, rfl⟩
      rw [marginal] at observed
      have selected := (PMF.mem_support_pure_iff context sample.1).mp observed
      exact Prod.ext selected.symm rfl
    _ = distribution := PMF.map_id distribution

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  (contexts : PMF Context)
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      Lifecycle.Nifs.WeakExtraction.Continuation Tape relation productionAjtaiKey
        (PiCCSInputCheck.running (inputs context)) (PiCCSInputCheck.fresh (inputs context))
        coins output)
  (primitives : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := PiDECInputCheck.logicalWidth)
      (publicFits := PiDECInputCheck.publicFits)))

/-- Run the actual checked prefix and its exact receipt's suffix law, then
apply the selected stored source return. Contexts survive all abort paths. -/
noncomputable def law :
    PMF (Context × Option (WitnessProjection.SourceWitness productionShape carrier)) :=
  (SequentialOutputLaw.law contexts
    (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => PiCCSInputCheck.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.suffixLaw relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) continuation)
    (Lifecycle.Nifs.InteractiveComposition.consume relation productionAjtaiKey primitives)).map
      fun sample =>
        (sample.1, PiCCSStoredWitnessCheck.finishValue (inputs sample.1)
          (PiCCSStoredSourceProbability.storeOutcome sample.2))

/-- The source-event mass is exactly the sequential event on the right side
of NifsClosure's probability bound. No distribution or event correspondence
is assumed, and the original captured suffix state remains in that law. -/
theorem source_event_mass_eq :
    ((law inputs contexts originalFirstPhase continuation primitives).toOuterMeasure
      { sample | CheckedWitnessExtraction.SourceReturned commit productionGlobalParams
        (statement (inputs sample.1)) sample.2 }).toReal =
      PaperCompositionProbability.eventProbability contexts
        (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
          (Lifecycle.Nifs.SupportedExtraction.publicCheck
            (fun context => PiCCSInputCheck.running (inputs context))))
        (Lifecycle.Nifs.InteractiveComposition.suffixLaw relation productionAjtaiKey
          (fun context => PiCCSInputCheck.running (inputs context))
          (fun context => PiCCSInputCheck.fresh (inputs context)) continuation)
        (Lifecycle.Nifs.InteractiveComposition.consume relation productionAjtaiKey primitives)
        (fun context outcome => CheckedWitnessExtraction.SourceReturned commit productionGlobalParams
          (statement (inputs context))
          (PiCCSStoredWitnessCheck.finishValue (inputs context)
            (PiCCSStoredSourceProbability.storeOutcome outcome))) := by
  rw [law, PMF.toOuterMeasure_map_apply]
  exact SequentialOutputLaw.eventProbability_eq contexts
    (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => PiCCSInputCheck.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.suffixLaw relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) continuation)
    (Lifecycle.Nifs.InteractiveComposition.consume relation productionAjtaiKey primitives)
    (fun context outcome => CheckedWitnessExtraction.SourceReturned commit productionGlobalParams
      (statement (inputs context))
      (PiCCSStoredWitnessCheck.finishValue (inputs context)
        (PiCCSStoredSourceProbability.storeOutcome outcome)))

/-- Mapping to the checked source result leaves the exact supplied context
law unchanged, including contexts whose prefix or suffix aborts. -/
theorem context_marginal :
    (law inputs contexts originalFirstPhase continuation primitives).map Prod.fst = contexts := by
  rw [law, PMF.map_comp]
  exact SequentialOutputLaw.context_marginal contexts
    (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => PiCCSInputCheck.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.suffixLaw relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) continuation)
    (Lifecycle.Nifs.InteractiveComposition.consume relation productionAjtaiKey primitives)

/-- Two continuations that agree on actual supported checked receipts give
the same selected source-result law, including unsuccessful source returns.
This is the interface used to compare a fixed operational continuation with
its proof-only supported extension under each visited-context law. -/
theorem law_eq_of_continuation_eq_on_return
    (other : ∀ context (coins : PublicCoins K productionShape)
      (output : FullOutputCoordinates.FullOutput K productionShape), State →
        Lifecycle.Nifs.WeakExtraction.Continuation Tape relation productionAjtaiKey
          (PiCCSInputCheck.running (inputs context)) (PiCCSInputCheck.fresh (inputs context))
          coins output)
    (same : ∀ context ∈ contexts.support,
      ∀ alpha gamma point (receipt : Probe K productionShape × State),
        InteractivePrefix.run
          (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
            (Lifecycle.Nifs.SupportedExtraction.publicCheck
              (fun context => PiCCSInputCheck.running (inputs context))) context)
          alpha gamma point = some receipt →
        continuation context receipt.1.coins receipt.1.response.fullOutput receipt.2 =
          other context receipt.1.coins receipt.1.response.fullOutput receipt.2) :
    law inputs contexts originalFirstPhase continuation primitives =
      law inputs contexts originalFirstPhase other primitives := by
  unfold law
  apply congrArg (fun distribution : PMF (Context ×
    CheckedWitnessExtraction.Outcome productionShape carrier) => distribution.map _)
  apply SequentialOutputLaw.law_eq_of_suffix_eq_on_return
  intro context supported alpha gamma point receipt returned
  unfold Lifecycle.Nifs.InteractiveComposition.suffixLaw
  rw [same context supported alpha gamma point receipt returned]

/-- The same experiment at one supplied context. Its continuation and
primitive program are unchanged; only the context PMF is a point mass. -/
noncomputable def atContext (context : Context) :
    PMF (Option (WitnessProjection.SourceWitness productionShape carrier)) :=
  (law inputs (PMF.pure context) originalFirstPhase continuation primitives).map Prod.snd

/-- Drawing the context and then its exact source result reconstructs the
joint source law. No conditioning, support test, or new continuation is used.
The returned context is retained even when the source result is `none`. -/
theorem law_eq_bind_atContext :
    law inputs contexts originalFirstPhase continuation primitives =
      contexts.bind (fun context =>
        (atContext inputs originalFirstPhase continuation primitives context).map
          (fun result => (context, result))) := by
  have splitContexts : law inputs contexts originalFirstPhase continuation primitives =
      contexts.bind (fun context =>
        law inputs (PMF.pure context) originalFirstPhase continuation primitives) := by
    simp only [law, SequentialOutputLaw.law, PMF.pure_bind, PMF.map_bind]
  rw [splitContexts]
  apply congrArg (PMF.bind contexts)
  funext context
  exact (reattach_context
    (law inputs (PMF.pure context) originalFirstPhase continuation primitives) context
    (context_marginal inputs (PMF.pure context) originalFirstPhase continuation primitives)).symm

end NightstreamFPrime.Export.Stage1.HyperNovaSourceLaw
