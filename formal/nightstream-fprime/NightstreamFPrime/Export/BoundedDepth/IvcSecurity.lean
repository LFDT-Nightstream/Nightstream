import NightstreamFPrime.Export.BoundedDepth.IvcWalk

/-! HyperNova Lemma 17 for the bounded-depth scheme: the IVC false-acceptance bound. An IVC adversary outputs a
terminal statement and a proof. It wins when the terminal relation accepts and no `iteration`
steps of the application lead from `z0` to `zi` (`FalseAcceptance`). The reverse extractor walks
back from the terminal one step per stage (`IvcWalk`), with the extractor that HyperNova errata
Assumption 1 gives for each stage's fold attempt. Then

  Pr[false acceptance] ≤ Σ j < depth, (Pr[collision at visit j] + error(stage j)),

where `depth` bounds the adversary's iteration. The collision at visit `j` is a Poseidon2
collision on two states that stage `j` computes from its tape: the next state of the step that
the visit's fresh witness decodes to, and the visit's terminal state
(`TerminalBridge.Collision`). Each fold attempt is linked to the prior state of its step
(`Lifecycle.BoundedDepth.PriorLink`), so its running claim is fixed by the fresh instance that the
fold transcript absorbs, up to a collision on a pair of prior states (`priorLink_binds`).
Assumption 1 is a premise; it is needed only at bounds up to `B + cap · T · (b − 1)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.IvcSecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Export.BoundedDepth.TerminalBridge
open NightstreamFPrime.Export.BoundedDepth.IvcWalk

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

abbrev Output (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) :=
  TerminalStatement AppState ×
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  {step : AppState → AppWitness → AppState}
  {cubeFits : Phi81CarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : Exports relation ajtai circuitKeys step cubeFits)

/-- The terminal accepts, and no `iteration` valid steps of the application lead from `z0` to
`zi`. -/
def FalseAcceptance (control : AppState → AppWitness → Fin 2) (cap : Nat)
    (output : Output logicalWidth publicFits) : Prop :=
  Terminal.HoldsFor cap ajtai circuitKeys step control output.1 output.2 ∧
    ¬ Reaches exports.program output.1

/-- The witnesses that the reverse extractor returns for a visit and its answers: one for each
step of the walk's trail, first step first. -/
def extracted (visit : Visit logicalWidth publicFits)
    (answers : List (Answer logicalWidth publicFits)) : List AppWitness :=
  witnessesOf (trail ajtai circuitKeys exports visit answers)

/-- The terminal accepts the visit, and the extracted witnesses are not `iteration` valid steps
of the application that lead from `z0` to `zi` (HyperNova Definition 11 (ii)). -/
def ExtractionFailure (control : AppState → AppWitness → Fin 2) (cap : Nat)
    (visit : Visit logicalWidth publicFits) (answers : List (Answer logicalWidth publicFits)) :
    Prop :=
  Terminal.HoldsFor cap ajtai circuitKeys step control visit.statement visit.proof ∧
    ¬ ReachedBy exports.program visit.statement (extracted ajtai circuitKeys exports visit answers)

/-- The walk's start: the IVC adversary's output with every slot at bound `B`. -/
def walkAdversary (adversary : IvcAdversary (Output logicalWidth publicFits)) :
    IvcAdversary (Visit logicalWidth publicFits) :=
  ⟨adversary.Tape, adversary.tape, fun tape => initial (adversary.output tape)⟩

variable {Admitted : NifsAdversary (WalkAttempt logicalWidth publicFits) → Prop}
  {StageAdmitted : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits) → Prop}
  {Efficient : (adversary : NifsAdversary (WalkAttempt logicalWidth publicFits)) →
    NifsExtractor (Answer logicalWidth publicFits) adversary → Prop}
  {error : NifsAdversary (WalkAttempt logicalWidth publicFits) → ℝ}
  (assumption : Assumption1 (WalkSucceeds ajtai) (WalkExtracted ajtai) Admitted Efficient error)
  (closed : Closed (attemptOf ajtai circuitKeys exports) (back ajtai circuitKeys exports) Admitted
    StageAdmitted Efficient)
  (adversary : IvcAdversary (Output logicalWidth publicFits))
  (admitted : StageAdmitted (Stage.start (walkAdversary adversary)))

local notation "stages" => reverseStages assumption closed (walkAdversary adversary) admitted

/-- HyperNova Lemma 17 for the bounded-depth scheme, on the extractor's output: the probability that the terminal
accepts the adversary's output and the witnesses that the reverse extractor returns do not lead
from `z0` to `zi`. The law is the one of the last stage: the adversary's tape and the coins of
each stage's extractor. -/
theorem extraction_bound (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (cap depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    ((stages depth).1.tape.toOuterMeasure
        {tape | ExtractionFailure ajtai circuitKeys exports control cap
          ((stages depth).1.input tape) ((stages depth).1.results tape)}).toReal ≤
      ∑ j : Fin depth,
        (((stages j.val).1.tape.toOuterMeasure
            {tape | WalkCollision ajtai circuitKeys exports
              (visitAfter (back ajtai circuitKeys exports) ((stages j.val).1.input tape)
                ((stages j.val).1.results tape))}).toReal +
          error ((stages j.val).1.nifs (attemptOf ajtai circuitKeys exports)
            (back ajtai circuitKeys exports))) := by
  refine bad_le assumption closed (walkAdversary adversary) admitted depth
    (ExtractionFailure ajtai circuitKeys exports control cap)
    (WalkCollision ajtai circuitKeys exports) ?_
  intro origin supported results length failure
  obtain ⟨accepted, unreached⟩ := failure
  obtain ⟨j, inside, found⟩ := covered ajtai circuitKeys exports keysShaped control cap
    (adversary.output origin) results (length ▸ depthBound origin supported) accepted
    fun leads => unreached (reachedBy_of_leads _ _ _ leads
      (trail_lengths ajtai circuitKeys exports _ _))
  exact ⟨⟨j, length ▸ inside⟩, found⟩

/-- HyperNova Lemma 17 for the bounded-depth scheme. A false acceptance is an extraction failure, because no
list of witnesses reaches its statement. -/
theorem probability_bound (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (cap depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure
        {tape | FalseAcceptance ajtai circuitKeys exports control cap
          (adversary.output tape)}).toReal ≤
      ∑ j : Fin depth,
        (((stages j.val).1.tape.toOuterMeasure
            {tape | WalkCollision ajtai circuitKeys exports
              (visitAfter (back ajtai circuitKeys exports) ((stages j.val).1.input tape)
                ((stages j.val).1.results tape))}).toReal +
          error ((stages j.val).1.nifs (attemptOf ajtai circuitKeys exports)
            (back ajtai circuitKeys exports))) := by
  let event : Set (Visit logicalWidth publicFits) :=
    {visit | FalseAcceptance ajtai circuitKeys exports control cap
      (visit.statement, visit.proof)}
  have moved := start_event assumption closed (walkAdversary adversary) admitted depth event
  change adversary.tape.toOuterMeasure
      {tape | FalseAcceptance ajtai circuitKeys exports control cap
        (adversary.output tape)} = _ at moved
  rw [moved]
  refine (ENNReal.toReal_mono (event_ne_top _ _) (MeasureTheory.measure_mono ?_)).trans
    (extraction_bound ajtai circuitKeys exports assumption closed adversary admitted keysShaped
      control cap depth depthBound)
  intro tape falseAcceptance
  exact ⟨falseAcceptance.1, fun reached => falseAcceptance.2 ⟨_, reached⟩⟩

end NightstreamFPrime.Export.BoundedDepth.IvcSecurity
