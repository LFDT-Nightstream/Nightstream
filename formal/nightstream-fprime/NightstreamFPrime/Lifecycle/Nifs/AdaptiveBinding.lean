import NightstreamFPrime.Lifecycle.Nifs.InteractiveOutput
import NightstreamFPrime.Lifecycle.Nifs.InteractiveWork
import NightstreamFPrime.Spec.Folding.Nifs.SequentialObservationLaw

/-!
The adaptive binding reduction checks each retained NIFS observation with
the existing weak terminal program, decoder and checked source finish.
Endpoint evidence stays available for the later computed binding vector.
Query work remains with the same charged weak oracle; the checker below
includes terminal extraction, source checking and projection, even on abort.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.AdaptiveBinding

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
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
  (sourceProgram : CheckedWitnessExtraction.Program productionShape (FullShape logicalWidth publicFits))

/-- Executable specialization by the source count. `decode_eq_spec` proves
the same values and work as the decoder whose argument is the full key. -/
private def decode : Option (List
    (PaperLinearAlgebra.Assignment F (Phi81CarrierLayout.carrierWidth logicalWidth))) →
    Result (Option (OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)))
  | none => ⟨none, 1⟩
  | some values =>
      ⟨if length : values.length = productionShape.sourceCount then
        some ⟨fun source => values[source.val]'(by rw [length]; exact source.isLt)⟩
      else none, (values.length + 2) + 1⟩

omit [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
private theorem decode_eq_spec (values : Option (List
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))) :
    decode (logicalWidth := logicalWidth) values =
      PaperWeakOutput.decode (ProductionKey.key relation ajtai) values := by
  cases values with
  | none => rfl
  | some values =>
      have countEq := (ProductionKey.key relation ajtai).total_eq_sourceCount
      dsimp only [decode, PaperWeakOutput.decode, PaperWeakOutput.decodeResult,
        PaperWeakOutput.witnessOfList, PaperStrongInterface.outputWitnessOfAssignments]
      split_ifs <;> first | rfl | omega

/-- Complete the existing source call and test whether it returned a checked
source. The final presence test and return add one declared transition.
This does not test source membership or witness agreement. -/
def check : PaperCompositionAgreement.Observation State
    (InteractiveComposition.Endpoint relation ajtai) productionShape → Result Bool
  | none => ⟨false, 1⟩
  | some (receipt, endpoint) =>
      let terminal := PaperWeakLaw.terminalResult (count := PaperProfile.arity.total) program endpoint
      let finished := PaperCompositionWork.finish sourceProgram
        (decode (logicalWidth := logicalWidth)) receipt.1 terminal.value
      ⟨finished.value.isSome, terminal.work + finished.work + 1⟩

omit [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
/-- The executable checker runs exactly the existing terminal and source
finish, with their returned clocks and one final presence-test transition. -/
theorem check_some_eq (receipt : Probe K productionShape × State)
    (endpoint : InteractiveComposition.Endpoint relation ajtai) :
    check relation ajtai program sourceProgram (some (receipt, endpoint)) =
      let terminal := PaperWeakLaw.terminalResult program endpoint
      let finished := PaperCompositionWork.finish sourceProgram
        (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) receipt.1 terminal.value
      ⟨finished.value.isSome, terminal.work + finished.work + 1⟩ := by
  have decoder : (decode (logicalWidth := logicalWidth)) =
      PaperWeakOutput.decode (ProductionKey.key relation ajtai) :=
    funext (decode_eq_spec relation ajtai)
  simp only [check, decoder]
  rfl

variable
  (running : Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (sourceCorrect : CheckedWitnessExtraction.Correct (width := 9) sourceProgram
    (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement running fresh))

omit [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
include sourceCorrect in
private theorem finish_isSome_iff
    (outcome : CheckedWitnessExtraction.Outcome productionShape (FullShape logicalWidth publicFits)) :
    (CheckedWitnessExtraction.finish sourceProgram outcome).value.isSome = true ↔
      StrongProbability.RelaxedSuccess (width := 9) (PaperAlgebra.openingMaps ajtai)
        productionGlobalParams ((ProductionKey.key relation ajtai).statement running fresh) outcome := by
  constructor
  · intro present
    cases returned : (CheckedWitnessExtraction.finish sourceProgram outcome).value with
    | none => simp only [returned, Option.isSome_none, Bool.false_eq_true] at present
    | some values =>
        obtain ⟨probe, witness, issued, checked, _⟩ :=
          (CheckedWitnessExtraction.finish_return_iff sourceProgram outcome values).mp returned
        have accepted := (sourceCorrect.check probe witness).mp checked
        exact ⟨probe, witness, issued, accepted.1, accepted.2⟩
  · rintro ⟨probe, witness, issued, accepted, ambient⟩
    have returned := (CheckedWitnessExtraction.finish_return_iff sourceProgram outcome
      (CostedWitnessProjection.project sourceProgram.access witness).value).mpr
      ⟨probe, witness, issued, (sourceCorrect.check probe witness).mpr ⟨accepted, ambient⟩, rfl⟩
    rw [returned]
    rfl

omit [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
include sourceCorrect in
/-- Retry acceptance is exactly the actual relaxed-success event on this
retained observation. It uses the selected checker and decoded witness,
including rejected prefixes, malformed returns and ambient-check failures. -/
theorem check_correct (observation : PaperCompositionAgreement.Observation State
    (InteractiveComposition.Endpoint relation ajtai) productionShape) :
    (check relation ajtai program sourceProgram observation).value = true ↔
      StrongProbability.RelaxedSuccess (width := 9) (PaperAlgebra.openingMaps ajtai)
        productionGlobalParams ((ProductionKey.key relation ajtai).statement running fresh)
        (PaperCompositionAgreement.outputOf
          (fun _ _ _ endpoint => WeakExtraction.consume relation ajtai program endpoint) observation) := by
  cases observation with
  | none => simp [check, PaperCompositionAgreement.outputOf, StrongProbability.RelaxedSuccess]
  | some observation =>
      rcases observation with ⟨receipt, endpoint⟩
      rw [check_some_eq]
      exact finish_isSome_iff relation ajtai sourceProgram running fresh sourceCorrect _

/-- The same charged weak query law and the executable observation checker
give the existing complete suffix clock plus the final presence test. Every
rejected query and failed terminal/ambient check keeps its original cost. -/
theorem checked_suffix_work_eq
    (law : PiRLC.CoordinateChargedOracle.Law (Fin PaperProfile.arity.total)
      (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))
    (parentChecker : Response
      (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))
      RingF productionGlobalParams PaperProfile.arity → PiRLC.CoordinateCheckedCalls.CheckResult)
    (receipt : Probe K productionShape × State) :
    let algebra := (ProductionKey.key relation ajtai).piRlcAlgebra
    let typed := PiRLC.CoordinateExtraction.typedChecker algebra parentChecker
    let charged := PiRLC.CoordinateCheckedCalls.withChecker law typed
    PiRLC.CoordinateRetryWork.expectedQueryWork charged (PiRLC.CoordinateCheckedCalls.check typed) +
      (∑ endpoint, (PaperCompositionWork.endpointLaw algebra law parentChecker endpoint).toReal *
        ((check relation ajtai program sourceProgram (some (receipt, endpoint))).work : ℝ)) =
      PiRLC.CoordinateExtraction.expectedTotalWork algebra law parentChecker program +
        PaperCompositionWork.finishMean algebra law parentChecker program sourceProgram
          (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) receipt.1 + 1 := by
  dsimp only
  have weights {Sample : Type} [Fintype Sample] (distribution : PMF Sample) :
      (∑ sample, (distribution sample).toReal) = 1 := by
    rw [← ENNReal.toReal_sum (fun sample _ => distribution.apply_ne_top sample)]
    have total : (∑ sample, distribution sample) = 1 := by
      simpa only [tsum_fintype] using distribution.tsum_coe
    rw [total, ENNReal.toReal_one]
  simp only [check_some_eq, Nat.cast_add, Nat.cast_one, mul_add,
    Finset.sum_add_distrib, mul_one]
  rw [weights]
  unfold PiRLC.CoordinateExtraction.expectedTotalWork PaperCompositionWork.finishMean
  dsimp only [PaperWeakLaw.terminalValue, PaperCompositionWork.endpointLaw]
  rw [← PaperWeakLaw.terminal_work_mean]
  ring_nf
  rfl

variable {Context Tape : Type*}
  (runningOf : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (freshOf : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (runningOf context) (freshOf context) coins output)
  (call : Context → CubePoint K productionShape.cubeVariables → K →
    CubePoint K productionShape.cubeVariables → Result (Option (Probe K productionShape × State)))
  (sourcePrograms : Context → CheckedWitnessExtraction.Program productionShape (FullShape logicalWidth publicFits))

/-- Mean clock of the complete call checked by the adaptive reduction. A
prefix abort still executes the presence test. A receipt selects its own
charged query law and the checker above. Prefix dispatch adds one transition;
the final presence test has its separate transition in `check`. -/
noncomputable def callClock (context : Context)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) : ℝ :=
  let issued := call context alpha gamma point
  (issued.work : ℝ) + (match issued.value with
  | none => (check relation ajtai program (sourcePrograms context)
      (none : PaperCompositionAgreement.Observation State
        (InteractiveComposition.Endpoint relation ajtai) productionShape)).work
  | some receipt =>
      let algebra := (ProductionKey.key relation ajtai).piRlcAlgebra
      let law := InteractiveWork.law relation ajtai runningOf freshOf continuation context receipt
      let parentChecker := InteractiveWork.parentChecker relation ajtai runningOf freshOf continuation context receipt
      let typed := PiRLC.CoordinateExtraction.typedChecker algebra parentChecker
      let charged := PiRLC.CoordinateCheckedCalls.withChecker law typed
      PiRLC.CoordinateRetryWork.expectedQueryWork charged (PiRLC.CoordinateCheckedCalls.check typed) +
        ∑ endpoint, (PaperCompositionWork.endpointLaw algebra law parentChecker endpoint).toReal *
          ((check relation ajtai program (sourcePrograms context) (some (receipt, endpoint))).work : ℝ)) + 1

/-- The complete checked call has the existing selected NIFS clock plus
one presence-test transition. The same receipt determines values and costs;
all prefix and suffix failure branches remain included. -/
theorem callClock_eq (context : Context)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    callClock relation ajtai program runningOf freshOf continuation call sourcePrograms context alpha gamma point =
      InteractiveWork.totalClock relation ajtai runningOf freshOf continuation call program sourcePrograms
        context alpha gamma point + 1 := by
  dsimp only [callClock, InteractiveWork.totalClock, PaperCompositionWork.totalClock]
  cases returned : (call context alpha gamma point).value with
  | none => simp only [check, Nat.cast_one]; ring
  | some receipt =>
      have suffix := checked_suffix_work_eq relation ajtai program (sourcePrograms context)
        (InteractiveWork.law relation ajtai runningOf freshOf continuation context receipt)
        (InteractiveWork.parentChecker relation ajtai runningOf freshOf continuation context receipt) receipt
      have lifted := congrArg (fun work : ℝ => ((call context alpha gamma point).work : ℝ) + work + 1) suffix
      simpa only [_root_.add_assoc] using lifted

end NightstreamFPrime.Lifecycle.Nifs.AdaptiveBinding
