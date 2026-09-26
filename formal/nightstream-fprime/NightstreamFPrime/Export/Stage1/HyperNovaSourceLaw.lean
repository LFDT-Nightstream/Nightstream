import NightstreamFPrime.Export.Stage1.SecurityInstance
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
open PiCCSStoredWitnessCheck (carrier commit statement)

variable (inst : SecurityInstance)

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
  [Fintype (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  (contexts : PMF Context)
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      Lifecycle.Nifs.WeakExtraction.Continuation Tape inst.relation inst.ajtai
        (inst.running (inputs context)) (inst.fresh (inputs context))
        coins output)
  (primitives : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := inst.logicalWidth)
      (publicFits := inst.publicFits)))

/-- Run the actual checked prefix and its exact receipt's suffix law, then
apply the selected stored source return. Contexts survive all abort paths. -/
noncomputable def law :
    PMF (Context × Option (WitnessProjection.SourceWitness productionShape (carrier inst))) :=
  (SequentialOutputLaw.law contexts
    (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => inst.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.suffixLaw inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context)) continuation)
    (Lifecycle.Nifs.InteractiveComposition.consume inst.relation inst.ajtai primitives)).map
      fun sample =>
        (sample.1, (PiCCSStoredWitnessCheck.finishValue inst) (inputs sample.1)
          (PiCCSStoredSourceProbability.storeOutcome inst sample.2))

/-- The source-event mass is exactly the sequential event on the right side
of NifsClosure's probability bound. No distribution or event correspondence
is assumed, and the original captured suffix state remains in that law. -/
theorem source_event_mass_eq :
    ((law inst inputs contexts originalFirstPhase continuation primitives).toOuterMeasure
      { sample | CheckedWitnessExtraction.SourceReturned (commit inst) productionGlobalParams
        (statement inst (inputs sample.1)) sample.2 }).toReal =
      PaperCompositionProbability.eventProbability contexts
        (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
          (Lifecycle.Nifs.SupportedExtraction.publicCheck
            (fun context => inst.running (inputs context))))
        (Lifecycle.Nifs.InteractiveComposition.suffixLaw inst.relation inst.ajtai
          (fun context => inst.running (inputs context))
          (fun context => inst.fresh (inputs context)) continuation)
        (Lifecycle.Nifs.InteractiveComposition.consume inst.relation inst.ajtai primitives)
        (fun context outcome => CheckedWitnessExtraction.SourceReturned (commit inst) productionGlobalParams
          (statement inst (inputs context))
          (PiCCSStoredWitnessCheck.finishValue inst (inputs context)
            (PiCCSStoredSourceProbability.storeOutcome inst outcome))) := by
  rw [law, PMF.toOuterMeasure_map_apply]
  exact SequentialOutputLaw.eventProbability_eq contexts
    (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => inst.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.suffixLaw inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context)) continuation)
    (Lifecycle.Nifs.InteractiveComposition.consume inst.relation inst.ajtai primitives)
    (fun context outcome => CheckedWitnessExtraction.SourceReturned (commit inst) productionGlobalParams
      (statement inst (inputs context))
      (PiCCSStoredWitnessCheck.finishValue inst (inputs context)
        (PiCCSStoredSourceProbability.storeOutcome inst outcome)))

/-- Mapping to the checked source result leaves the exact supplied context
law unchanged, including contexts whose prefix or suffix aborts. -/
theorem context_marginal :
    (law inst inputs contexts originalFirstPhase continuation primitives).map Prod.fst = contexts := by
  rw [law, PMF.map_comp]
  exact SequentialOutputLaw.context_marginal contexts
    (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => inst.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.suffixLaw inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context)) continuation)
    (Lifecycle.Nifs.InteractiveComposition.consume inst.relation inst.ajtai primitives)

/-- Two continuations that agree on actual supported checked receipts give
the same selected source-result law, including unsuccessful source returns.
This is the interface used to compare a fixed operational continuation with
its proof-only supported extension under each visited-context law. -/
theorem law_eq_of_continuation_eq_on_return
    (other : ∀ context (coins : PublicCoins K productionShape)
      (output : FullOutputCoordinates.FullOutput K productionShape), State →
        Lifecycle.Nifs.WeakExtraction.Continuation Tape inst.relation inst.ajtai
          (inst.running (inputs context)) (inst.fresh (inputs context))
          coins output)
    (same : ∀ context ∈ contexts.support,
      ∀ alpha gamma point (receipt : Probe K productionShape × State),
        InteractivePrefix.run
          (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
            (Lifecycle.Nifs.SupportedExtraction.publicCheck
              (fun context => inst.running (inputs context))) context)
          alpha gamma point = some receipt →
        continuation context receipt.1.coins receipt.1.response.fullOutput receipt.2 =
          other context receipt.1.coins receipt.1.response.fullOutput receipt.2) :
    (law inst) inputs contexts originalFirstPhase continuation primitives =
      (law inst) inputs contexts originalFirstPhase other primitives := by
  unfold law
  apply congrArg (fun distribution : PMF (Context ×
    CheckedWitnessExtraction.Outcome productionShape (carrier inst)) => distribution.map _)
  apply SequentialOutputLaw.law_eq_of_suffix_eq_on_return
  intro context supported alpha gamma point receipt returned
  unfold Lifecycle.Nifs.InteractiveComposition.suffixLaw
  rw [same context supported alpha gamma point receipt returned]

/-- The same experiment at one supplied context. Its continuation and
primitive program are unchanged; only the context PMF is a point mass. -/
noncomputable def atContext (context : Context) :
    PMF (Option (WitnessProjection.SourceWitness productionShape (carrier inst))) :=
  (law inst inputs (PMF.pure context) originalFirstPhase continuation primitives).map Prod.snd

/-- Drawing the context and then its exact source result reconstructs the
joint source law. No conditioning, support test, or new continuation is used.
The returned context is retained even when the source result is `none`. -/
theorem law_eq_bind_atContext :
    (law inst) inputs contexts originalFirstPhase continuation primitives =
      contexts.bind (fun context =>
        (atContext inst inputs originalFirstPhase continuation primitives context).map
          (fun result => (context, result))) := by
  have splitContexts : law inst inputs contexts originalFirstPhase continuation primitives =
      contexts.bind (fun context =>
        (law inst) inputs (PMF.pure context) originalFirstPhase continuation primitives) := by
    simp only [law, SequentialOutputLaw.law, PMF.pure_bind, PMF.map_bind]
  rw [splitContexts]
  apply congrArg (PMF.bind contexts)
  funext context
  exact (reattach_context
    (law inst inputs (PMF.pure context) originalFirstPhase continuation primitives) context
    (context_marginal inst inputs (PMF.pure context) originalFirstPhase continuation primitives)).symm

end NightstreamFPrime.Export.Stage1.HyperNovaSourceLaw
