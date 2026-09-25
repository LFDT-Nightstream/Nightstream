import NightstreamFPrime.Lifecycle.Nifs.AdaptiveBinding
import NightstreamFPrime.Lifecycle.Nifs.BindingReduction
import NightstreamFPrime.Lifecycle.Nifs.BindingProbability
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.AcceptedRetry

/-!
The finite-prefix adaptive MSIS reduction. It checks the first actual NIFS
observation, retains it on acceptance, and searches fresh calls for the next
accepted observation. Only those two endpoints reach the existing integer
vector reduction. The supplied prefix describes oracle responses, not a
runtime table to construct; the driver ignores its unused suffix.

Each packet owns prefix and weak-query work through its retained endpoint.
`AdaptiveBinding.check` adds terminal extraction, decoding and source checks.
Four outer transitions cover the initial branch, first-response retention,
retry-result dispatch and return. Initial rejection uses only branch and
return. The retry loop and binding postprocessing retain their own clocks.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingRun

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork PiRLC.CoordinateForkLaw

variable {State : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : CheckedWitnessExtraction.Program productionShape (FullShape logicalWidth publicFits))

/-- Compute the trial integer vector from the first two accepted calls.
All consumed call/check clocks remain attached, including rejected retries. -/
def run
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total) : Result (Option (List Int)) :=
  let initial := AdaptiveBinding.check relation ajtai program sourceProgram first.value
  if initial.value then
    let searched := AcceptedRetry.search (AdaptiveBinding.check relation ajtai program sourceProgram) following
    match searched.value.1 with
    | none => ⟨none, first.work + initial.work + searched.work + 4⟩
    | some second =>
        let emitted := BindingReduction.runPair program sourceProgram.access
          ⟨first.value, first.work + initial.work⟩ ⟨second, searched.work⟩ coordinate
        ⟨emitted.value, emitted.work + 4⟩
  else ⟨none, first.work + initial.work + 2⟩

/-- Project the existing retry clock and its actual postprocessing work.
This is a view of `search` and `runPair`, not a separate execution model. -/
def retryWork
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total) : Nat :=
  let searched := AcceptedRetry.search (AdaptiveBinding.check relation ajtai program sourceProgram) following
  searched.work + match searched.value.1 with
    | none => 0
    | some second =>
        (BindingReduction.runPair program sourceProgram.access ⟨first, 0⟩ ⟨second, 0⟩ coordinate).work

/-- The exhausted prefix contributes the existing search return transition. -/
theorem retryWork_nil
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (coordinate : Fin PaperProfile.arity.total) :
    retryWork relation ajtai program sourceProgram first [] coordinate = 1 := rfl

/-- Exact retry recurrence: every entered call and check is charged, and
only its acceptance branch performs the integer-vector computation. -/
theorem retryWork_cons
    (first : PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)
    (packet : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total) :
    retryWork relation ajtai program sourceProgram first (packet :: following) coordinate =
      packet.work + (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).work + 3 +
        (if (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).value then
          (BindingReduction.runPair program sourceProgram.access
            ⟨first, 0⟩ ⟨packet.value, 0⟩ coordinate).work
        else retryWork relation ajtai program sourceProgram first following coordinate) := by
  dsimp only [retryWork]
  cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).value with
  | false =>
      simp only [AcceptedRetry.search, accepted, Bool.false_eq_true, ↓reduceIte]
      omega
  | true => simp only [AcceptedRetry.search, accepted, ↓reduceIte]

/-- The actual returned clock has the exact branch structure used by the
mean recurrence. No source cost is replaced by an unconditional upper bound
in this equality. -/
theorem run_work_eq
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total) :
    (run relation ajtai program sourceProgram first following coordinate).work =
      first.work + (AdaptiveBinding.check relation ajtai program sourceProgram first.value).work +
        (if (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value then
          retryWork relation ajtai program sourceProgram first.value following coordinate + 4
        else 2) := by
  dsimp only [run, retryWork]
  cases accepted : (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value with
  | false => simp only [Bool.false_eq_true, ↓reduceIte]
  | true =>
      simp only [↓reduceIte]
      cases selected : (AcceptedRetry.search
        (AdaptiveBinding.check relation ajtai program sourceProgram) following).value.1 with
      | none => simp only [Nat.add_zero, Nat.add_assoc]
      | some second =>
          dsimp only
          rw [BindingReduction.runPair_work_eq]
          simp only [Nat.add_assoc]

/-- A rejected first call makes no retry or binding call. Its actual prefix,
query and checker costs remain, together with branch and return. -/
theorem run_rejected
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total)
    (rejected : (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value = false) :
    run relation ajtai program sourceProgram first following coordinate =
      ⟨none, first.work + (AdaptiveBinding.check relation ajtai program sourceProgram first.value).work + 2⟩ := by
  simp only [run, rejected, Bool.false_eq_true, ↓reduceIte]

/-- The first accepted retry is the second endpoint used by the computed
binding reduction. Every earlier rejection contributes its complete call,
checker and loop cost; the unused suffix contributes none. -/
theorem run_firstHit
    (first last : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (before after : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total)
    (firstAccepted : (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value = true)
    (rejected : ∀ packet ∈ before,
      (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).value = false)
    (lastAccepted : (AdaptiveBinding.check relation ajtai program sourceProgram last.value).value = true) :
    run relation ajtai program sourceProgram first (before ++ last :: after) coordinate =
      let retryWork := (before.map fun packet => packet.work +
        (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).work + 3).sum +
        last.work + (AdaptiveBinding.check relation ajtai program sourceProgram last.value).work + 3
      let emitted := BindingReduction.runPair program sourceProgram.access
        ⟨first.value, first.work + (AdaptiveBinding.check relation ajtai program sourceProgram first.value).work⟩
        ⟨last.value, retryWork⟩ coordinate
      ⟨emitted.value, emitted.work + 4⟩ := by
  simp only [run, firstAccepted, ↓reduceIte,
    AcceptedRetry.search_firstHit _ before last after rejected lastAccepted]

/-- Exhausting a rejected finite prefix emits no vector. All supplied calls
were consumed, so their complete costs remain in the returned clock. -/
theorem run_exhausted
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total)
    (firstAccepted : (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value = true)
    (rejected : ∀ packet ∈ following,
      (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).value = false) :
    run relation ajtai program sourceProgram first following coordinate =
      ⟨none, first.work + (AdaptiveBinding.check relation ajtai program sourceProgram first.value).work +
        ((following.map fun packet => packet.work +
          (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).work + 3).sum + 1) + 4⟩ := by
  simp only [run, firstAccepted, ↓reduceIte, AcceptedRetry.search_exhausted _ following rejected]

/-- The full driver charges retry work only after first-call acceptance.
The additional bound is the existing actual integer-vector postprocessing
bound and the four outer control transitions. -/
theorem run_work_le
    (bounds : PrimitiveBounds)
    (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra ajtai).ring program bounds)
    (accessBound : Nat)
    (accessBounded : CostedWitnessProjection.Bounded sourceProgram.access accessBound)
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (coordinate : Fin PaperProfile.arity.total) :
    (run relation ajtai program sourceProgram first following coordinate).work ≤
      first.work + (AdaptiveBinding.check relation ajtai program sourceProgram first.value).work +
      (if (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value then
        (AcceptedRetry.search (AdaptiveBinding.check relation ajtai program sourceProgram) following).work
      else 0) + BindingOutput.crossWork bounds +
        (FullShape logicalWidth publicFits).carrierWidth * (accessBound + 6) + 16 := by
  dsimp only [run]
  split
  · split
    · dsimp only
      omega
    · rename_i second _
      have emitted := BindingReduction.runPair_work_le ajtai program bounds bounded sourceProgram.access
        accessBound accessBounded
        ⟨first.value, first.work + (AdaptiveBinding.check relation ajtai program sourceProgram first.value).work⟩
        ⟨second, (AcceptedRetry.search (AdaptiveBinding.check relation ajtai program sourceProgram) following).work⟩
        coordinate
      dsimp only at emitted ⊢
      omega
  · dsimp only
    omega

attribute [local instance] Classical.propDecidable

/-- Actual emitted-vector success for a fixed oracle prefix and one uniform
trial coordinate. The sum describes the experiment; the driver executes
only the supplied coordinate. -/
noncomputable def coordinateSuccess
    (first : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (following : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))) : ℝ :=
  (∑ coordinate, if BindingReduction.Succeeds ajtai
    (run relation ajtai program sourceProgram first following coordinate).value then (1 : ℝ) else 0) /
      (PaperProfile.arity.total : ℝ)

/-- On the first-hit branch, the actual driver's success is the existing
computed binding success on exactly the two selected observations. Clocks
remain in `run_firstHit`; this equation erases them only from the event. -/
theorem coordinateSuccess_firstHit
    (first last : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (before after : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (firstAccepted : (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value = true)
    (rejected : ∀ packet ∈ before,
      (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).value = false)
    (lastAccepted : (AdaptiveBinding.check relation ajtai program sourceProgram last.value).value = true) :
    coordinateSuccess relation ajtai program sourceProgram first (before ++ last :: after) =
      BindingProbability.observationSuccess ajtai program sourceProgram.access relation first.value last.value := by
  unfold coordinateSuccess
  simp only [run_firstHit relation ajtai program sourceProgram first last before after _
    firstAccepted rejected lastAccepted]
  exact (BindingProbability.observationSuccess_eq_runPair ajtai program sourceProgram.access relation
    ⟨first.value, first.work + (AdaptiveBinding.check relation ajtai program sourceProgram first.value).work⟩
    ⟨last.value, (before.map fun packet => packet.work +
      (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).work + 3).sum +
      last.work + (AdaptiveBinding.check relation ajtai program sourceProgram last.value).work + 3⟩).symm

variable {Context Tape : Type*}
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)
  (strongSet : StrongSetUnits (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)
  (accessCorrect : CostedWitnessProjection.Correct sourceProgram.access)

include strongSet correct accessCorrect in
/-- The binding event on the selected first-hit pair is bounded by the
actual emitted-vector success with the existing source-count loss. All
endpoint support is supplied by the actual suffix laws; no independent-pair
premise or existential choice of the output vector is used. The fixed-seed
hardness advantage stays in `PUBLIC_SEED_MSIS_ASSUMPTION.md`. -/
theorem selected_binding_le_success
    (context : Context)
    (first last : Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape))
    (before after : List (Result (PaperCompositionAgreement.Observation State
      (InteractiveComposition.Endpoint relation ajtai) productionShape)))
    (firstAccepted : (AdaptiveBinding.check relation ajtai program sourceProgram first.value).value = true)
    (rejected : ∀ packet ∈ before,
      (AdaptiveBinding.check relation ajtai program sourceProgram packet.value).value = false)
    (lastAccepted : (AdaptiveBinding.check relation ajtai program sourceProgram last.value).value = true)
    (firstSupported : PaperCompositionAgreement.Supported
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context) first.value)
    (lastSupported : PaperCompositionAgreement.Supported
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context) last.value) :
    (if InteractiveAgreement.BindingEvent relation ajtai running fresh program context first.value last.value
      then (1 : ℝ) else 0) ≤
      coordinateSuccess relation ajtai program sourceProgram first (before ++ last :: after) *
        PaperProfile.arity.total := by
  rw [coordinateSuccess_firstHit relation ajtai program sourceProgram first last before after
    firstAccepted rejected lastAccepted]
  exact BindingProbability.supported_binding_le_success ajtai program sourceProgram.access relation running fresh
    continuation strongSet correct accessCorrect context first.value last.value firstSupported lastSupported

end NightstreamFPrime.Lifecycle.Nifs.AdaptiveBindingRun
