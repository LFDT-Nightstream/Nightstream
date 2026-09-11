import NightstreamFPrime.Export.Stage1.HyperNovaVisitedLaw
import NightstreamFPrime.Export.Stage1.HyperNovaSourceLaw
import NightstreamFPrime.Export.Stage1.HyperNovaRealInput

/-!
The unconditional guarded NIFS experiment at an actual visited history law.
The analytical mark masks this experiment's output, while the operational
history still uses its original source kernel on every active visit. Both
source laws retain the same continuation, receipt and primitive program.

These are value and probability-law identities. No Fiat--Shamir model,
accepted-source premise, efficient translation or work bound is supplied.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaGuardedSourceLaw

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction (Probe PublicCoins OutputWitness)
open NightstreamFPrime.Lifecycle
open PiRLC.PaperForkExtractionWork (Primitives)
open PiRLC.CoordinateForkLaw (Challenge)
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiDECInputCheck (relation)
open HyperNovaHistory (Statement Payload SourceResult)
open HyperNovaVisitedLaw (Visit goodActive)

attribute [local instance] Classical.propDecidable

private def inactiveInput : PiCCSInputCheck.Input where
  commitment := Vector.replicate _ 0
  publicInput := Vector.replicate _ 0
  rounds := Vector.replicate _ (Vector.replicate _ K.zero)
  evalK := Vector.replicate _ (Vector.replicate _ K.zero)
  evalA := Vector.replicate _ (Vector.replicate _ (Vector.replicate _ K.zero))
  running := {
    point := Vector.replicate _ K.zero
    commitments := Vector.replicate _ (Vector.replicate _ 0)
    publicInputs := Vector.replicate _ (Vector.replicate _ 0)
    evalK := Vector.replicate _ (Vector.replicate _ K.zero)
    evalA := Vector.replicate _ (Vector.replicate _ (Vector.replicate _ K.zero)) }

/-- The exact decoded source input on every recursive payload. The typed
zero value only totalizes inactive branches, which the guarded prefix aborts
before any input check or continuation. It is not an accepted base proof. -/
def inputs (visit : Visit) : PiCCSInputCheck.Input :=
  match visit.1 with
  | some (_, .recursive payload) => HyperNovaHistory.sourceInput payload
  | _ => inactiveInput

/-- The existing recursive terminal pair with the analytical mark set true.
This fixes the source kernel independently of a later visited-context law. -/
def recursiveVisit (statement : Statement) (payload : Payload) : Visit :=
  (some (statement, .recursive payload), true)

private theorem good_recursive (visit : Visit) (good : goodActive visit) :
    ∃ statement payload, visit = recursiveVisit statement payload := by
  rcases visit with ⟨current, mark⟩
  rcases good with ⟨marked, active, _safe⟩
  cases current with
  | none => exact False.elim active
  | some input =>
      rcases input with ⟨statement, proof⟩
      cases proof with
      | bottom => exact False.elim active
      | recursive payload =>
          exact ⟨statement, payload, Prod.ext rfl marked⟩

private def abortPrefix {State : Type*} : InteractivePrefix.Prover State productionShape 9 where
  rounds := fun _ _ _ => none
  output := fun _ _ _ => none

private theorem abortPrefix_run {State : Type*}
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    InteractivePrefix.run (abortPrefix (State := State)) alpha gamma point = none := by
  unfold InteractivePrefix.run
  split <;> rfl

/-- The actual causal prefix on the marked active branch, and an explicit
abort otherwise. The original history transition does not use this mask. -/
noncomputable def guardedPrefix {State : Type*}
    (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
    (visit : Visit) : InteractivePrefix.Prover State productionShape 9 :=
  if goodActive visit then originalFirstPhase visit else abortPrefix

/-- Inactive branches return no receipt before the public checker or suffix
can be called. This holds for every checker, without evaluating its input. -/
theorem guardedPrefix_abort {State : Type*}
    (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
    (visit : Visit) (inactive : ¬ goodActive visit)
    (check : Probe K productionShape → Bool)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    InteractivePrefix.run (InteractivePrefix.checked (guardedPrefix originalFirstPhase visit) check)
      alpha gamma point = none := by
  rw [InteractivePrefix.run_checked]
  simp only [guardedPrefix, if_neg inactive, abortPrefix_run, Option.filter_none]

/-- The actual local proof and current child witnesses on the good active
branch. Every other visit remains present with an absent real output. -/
noncomputable def realOutput (visit : Visit) :
    Option (Lifecycle.Nifs.FiatShamirTransfer.RealOutput relation) :=
  if goodActive visit then
    match visit.1 with
    | some (_, .recursive payload) => some (HyperNovaRealInput.output payload)
    | _ => none
  else none

/-- No real output is presented outside the same source-experiment guard. -/
theorem realOutput_off (visit : Visit) (inactive : ¬ goodActive visit) :
    realOutput visit = none := by
  simp only [realOutput, if_neg inactive]

/-- The normalized real experiment keeps the entire supplied visited law. -/
noncomputable def realLaw (contexts : PMF Visit) :
    PMF (Visit × Option (Lifecycle.Nifs.FiatShamirTransfer.RealOutput relation)) :=
  contexts.map (fun visit => (visit, realOutput visit))

/-- The guarded real experiment retains stopped, rejected and abort contexts
with their original mass; it performs no conditioning. -/
theorem realLaw_context_marginal (contexts : PMF Visit) :
    Lifecycle.Nifs.FiatShamirTransfer.contextLaw relation (realLaw contexts) = contexts := by
  rw [Lifecycle.Nifs.FiatShamirTransfer.contextLaw, realLaw, PMF.map_comp]
  exact PMF.map_id contexts

/-- The existing sequential law at one context draws the same public coins,
checked receipt and selected suffix, preserving every abort. -/
theorem sequential_atContext
    {Context State Endpoint : Type*} {shape : Shape} {columns width : Nat}
    (context : Context)
    (firstPhase : Context → InteractivePrefix.Prover State shape width)
    (suffixLaw : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → PMF Endpoint)
    (consume : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → Endpoint → Option (OutputWitness shape columns)) :
    SequentialOutputLaw.law (PMF.pure context) firstPhase suffixLaw consume =
      (VerifierCoinLaw.law shape).bind fun request =>
        let coins := VerifierCoinSpace.coins request
        match InteractivePrefix.run (firstPhase context) coins.alpha coins.gamma coins.roundPoint with
        | none => PMF.pure (context, none)
        | some receipt =>
            (suffixLaw context receipt.1.coins receipt.1.response.fullOutput receipt.2).map
              fun endpoint => (context,
                (consume context receipt.1.coins receipt.1.response.fullOutput receipt.2 endpoint).map
                  fun witness => (receipt.1, witness)) := by
  simp only [SequentialOutputLaw.law, PMF.pure_bind]
  rfl

private theorem sequential_atContext_congr
    {Context State Endpoint : Type*} {shape : Shape} {columns width : Nat}
    (context : Context)
    (first second : Context → InteractivePrefix.Prover State shape width)
    (suffixLaw : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → PMF Endpoint)
    (consume : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → Endpoint → Option (OutputWitness shape columns))
    (same : first context = second context) :
    SequentialOutputLaw.law (PMF.pure context) first suffixLaw consume =
      SequentialOutputLaw.law (PMF.pure context) second suffixLaw consume := by
  rw [sequential_atContext, sequential_atContext, same]

private theorem sequential_atContext_abort
    {Context State Endpoint : Type*} {shape : Shape} {columns width : Nat}
    (context : Context)
    (firstPhase : Context → InteractivePrefix.Prover State shape width)
    (suffixLaw : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → PMF Endpoint)
    (consume : Context → PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → Endpoint → Option (OutputWitness shape columns))
    (aborted : ∀ alpha gamma point, InteractivePrefix.run (firstPhase context) alpha gamma point = none) :
    SequentialOutputLaw.law (PMF.pure context) firstPhase suffixLaw consume = PMF.pure (context, none) := by
  rw [sequential_atContext]
  simp only [aborted, PMF.bind_const]

variable {State Tape : Type*}
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
  (continuation : ∀ visit (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      Lifecycle.Nifs.WeakExtraction.Continuation Tape relation productionAjtaiKey
        (PiCCSInputCheck.running (inputs visit)) (PiCCSInputCheck.fresh (inputs visit)) coins output)
  (primitives : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := PiDECInputCheck.logicalWidth)
      (publicFits := PiDECInputCheck.publicFits)))

/-- The operational history kernel at its actual statement and payload.
Its prefix, continuation and primitive values do not depend on the current
history mark or on which visited-context distribution will be observed. -/
noncomputable def source (statement : Statement) (payload : Payload) : PMF SourceResult :=
  HyperNovaSourceLaw.atContext inputs originalFirstPhase continuation primitives
    (recursiveVisit statement payload)

/-- The selected stored source law with only the causal prefix guarded.
The continuation is the literal one used by the operational source kernel. -/
noncomputable def law (contexts : PMF Visit) : PMF (Visit × SourceResult) :=
  HyperNovaSourceLaw.law inputs contexts (guardedPrefix originalFirstPhase) continuation primitives

/-- An inactive guarded call returns the absent source result with mass one.
The public checker and continuation cannot affect this branch. -/
theorem guarded_atContext_off (visit : Visit) (inactive : ¬ goodActive visit) :
    HyperNovaSourceLaw.atContext inputs (guardedPrefix originalFirstPhase)
      continuation primitives visit = PMF.pure none := by
  unfold HyperNovaSourceLaw.atContext HyperNovaSourceLaw.law
  rw [sequential_atContext_abort visit _ _ _ (fun alpha gamma point =>
    guardedPrefix_abort originalFirstPhase visit inactive _ alpha gamma point)]
  simp only [PMF.pure_map]
  rfl

/-- On the active marked branch, both experiments use the identical prefix,
receipt, continuation and returned source value. No equality is assumed. -/
theorem guarded_atContext_on (visit : Visit) (good : goodActive visit) :
    HyperNovaSourceLaw.atContext inputs (guardedPrefix originalFirstPhase)
      continuation primitives visit =
      HyperNovaSourceLaw.atContext inputs originalFirstPhase continuation primitives visit := by
  have same := sequential_atContext_congr visit
    (Lifecycle.Nifs.InteractiveComposition.firstPhase (guardedPrefix originalFirstPhase)
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => PiCCSInputCheck.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.firstPhase originalFirstPhase
      (Lifecycle.Nifs.SupportedExtraction.publicCheck
        (fun context => PiCCSInputCheck.running (inputs context))))
    (Lifecycle.Nifs.InteractiveComposition.suffixLaw relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) continuation)
    (Lifecycle.Nifs.InteractiveComposition.consume relation productionAjtaiKey primitives)
    (by simp only [Lifecycle.Nifs.InteractiveComposition.firstPhase, guardedPrefix, if_pos good])
  exact congrArg
    (fun distribution : PMF (Visit × CheckedWitnessExtraction.Outcome productionShape
        PiCCSStoredWitnessCheck.carrier) =>
      (distribution.map (fun sample =>
        (sample.1, PiCCSStoredWitnessCheck.finishValue (inputs sample.1)
          (PiCCSStoredSourceProbability.storeOutcome sample.2)))).map Prod.snd) same

/-- This unconditional selected NIFS law is exactly the guarded draw of the
operational history kernel, retaining every context and abort. -/
theorem law_eq_guardedDraw (contexts : PMF Visit) :
    law originalFirstPhase continuation primitives contexts =
      contexts.bind (HyperNovaVisitedLaw.guardedDraw (source originalFirstPhase continuation primitives)) := by
  rw [law, HyperNovaSourceLaw.law_eq_bind_atContext]
  apply congrArg (PMF.bind contexts)
  funext visit
  by_cases good : goodActive visit
  · rw [guarded_atContext_on originalFirstPhase continuation primitives visit good]
    rcases good_recursive visit good with ⟨statement, payload, rfl⟩
    rw [HyperNovaVisitedLaw.guardedDraw, if_pos good, HyperNovaVisitedLaw.draw, if_pos good.2.1]
    rfl
  · rw [guarded_atContext_off originalFirstPhase continuation primitives visit good, PMF.pure_map]
    simp only [HyperNovaVisitedLaw.guardedDraw, if_neg good]

private theorem event_ne_top {Sample : Type*} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

private theorem partition_mass {Sample : Type*} (distribution : PMF Sample)
    (active success : Sample → Prop)
    (contained : ∀ sample ∈ distribution.support, success sample → active sample) :
    (distribution.toOuterMeasure {sample | active sample ∧ ¬ success sample}).toReal =
      (distribution.toOuterMeasure {sample | active sample}).toReal -
        (distribution.toOuterMeasure {sample | success sample}).toReal := by
  have partition : distribution.toOuterMeasure {sample | active sample} =
      distribution.toOuterMeasure {sample | active sample ∧ ¬ success sample} +
        distribution.toOuterMeasure {sample | success sample} := by
    simp only [PMF.toOuterMeasure_apply, ← ENNReal.tsum_add]
    apply tsum_congr
    intro sample
    by_cases zero : distribution sample = 0
    · simp [Set.indicator, zero]
    · have supported := (distribution.mem_support_iff sample).mpr zero
      by_cases hs : success sample
      · simp [Set.indicator, hs, contained sample supported hs]
      · by_cases ha : active sample <;> simp [Set.indicator, hs, ha]
  have realPartition := congrArg ENNReal.toReal partition
  rw [ENNReal.toReal_add (event_ne_top distribution _) (event_ne_top distribution _)] at realPartition
  linarith

/-- The first source-failure mass is exactly good-active mass minus the
actual selected SourceReturned mass. This partitions the normalized law;
it assumes neither source validity nor conditional Fiat--Shamir security. -/
theorem first_source_failure_mass_eq (contexts : PMF Visit) :
    ((law originalFirstPhase continuation primitives contexts).toOuterMeasure
      {sample | goodActive sample.1 ∧
        ¬ CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
          (PiCCSStoredWitnessCheck.statement (inputs sample.1)) sample.2}).toReal =
      (contexts.toOuterMeasure {visit | goodActive visit}).toReal -
        ((law originalFirstPhase continuation primitives contexts).toOuterMeasure
          {sample | CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit
            productionGlobalParams (PiCCSStoredWitnessCheck.statement (inputs sample.1)) sample.2}).toReal := by
  have contained : ∀ sample ∈ (law originalFirstPhase continuation primitives contexts).support,
      CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
        (PiCCSStoredWitnessCheck.statement (inputs sample.1)) sample.2 → goodActive sample.1 := by
    intro sample supported returned
    rw [law_eq_guardedDraw originalFirstPhase continuation primitives contexts] at supported
    rcases (PMF.mem_support_bind_iff _ _ _).mp supported with ⟨visit, _visited, drawn⟩
    by_cases good : goodActive visit
    · rw [HyperNovaVisitedLaw.guardedDraw, if_pos good] at drawn
      rcases (PMF.mem_support_map_iff _ _ _).mp drawn with ⟨result, _produced, same⟩
      rw [← same]
      exact good
    · rw [HyperNovaVisitedLaw.guardedDraw, if_neg good] at drawn
      have same := (PMF.mem_support_pure_iff _ _).mp drawn
      subst sample
      rcases returned with ⟨values, impossible, _⟩
      cases impossible
  have partition := partition_mass (law originalFirstPhase continuation primitives contexts)
    (fun sample => goodActive sample.1)
    (fun sample => CheckedWitnessExtraction.SourceReturned PiCCSStoredWitnessCheck.commit
      productionGlobalParams (PiCCSStoredWitnessCheck.statement (inputs sample.1)) sample.2) contained
  have marginal : (law originalFirstPhase continuation primitives contexts).map Prod.fst = contexts :=
    HyperNovaSourceLaw.context_marginal inputs contexts (guardedPrefix originalFirstPhase)
      continuation primitives
  have activeMass : (law originalFirstPhase continuation primitives contexts).toOuterMeasure
      {sample | goodActive sample.1} = contexts.toOuterMeasure {visit | goodActive visit} := by
    calc
      _ = ((law originalFirstPhase continuation primitives contexts).map Prod.fst).toOuterMeasure
          {visit | goodActive visit} :=
        (PMF.toOuterMeasure_map_apply Prod.fst
          (law originalFirstPhase continuation primitives contexts) {visit | goodActive visit}).symm
      _ = _ := congrArg
        (fun distribution : PMF Visit => distribution.toOuterMeasure {visit | goodActive visit}) marginal
  rw [activeMass] at partition
  exact partition

end NightstreamFPrime.Export.Stage1.HyperNovaGuardedSourceLaw
