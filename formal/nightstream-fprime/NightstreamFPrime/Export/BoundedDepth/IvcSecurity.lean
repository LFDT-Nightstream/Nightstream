import NightstreamFPrime.Export.BoundedDepth.IvcWalk

/-! HyperNova Lemma 18 for the bounded-depth scheme: the IVC false-acceptance bound. An IVC adversary outputs a
terminal statement and a proof. It wins when the terminal relation accepts and no `iteration`
steps of the application lead from `z0` to `zi` (`FalseAcceptance`). The reverse extractor walks
back from the terminal one step per stage (`IvcWalk`), with the extractor that HyperNova errata
Assumption 1 gives for each stage's fold attempt. Then

  Pr[false acceptance] ≤ walkBound depth = Σ j < depth, (finderChance(stage j) + error(stage j)),

where `depth` bounds the adversary's iteration. `finderChance` is the success chance of an
explicit Poseidon2 collision finder (`finder`): it runs stage `j` on its tape, decodes the
visit, and outputs the pair of `IvcWalk.visitCollision`. Each success is a Poseidon2 collision
(`PoseidonCollision`). A collision at visit `j` (`TerminalBridge.Collision`, on two states that
stage `j` computes from its tape: the next state of the step that the visit's fresh witness
decodes to, and the visit's terminal state) is a success (`IvcWalk.visitCollision_isSome`).
Each fold attempt is linked to the prior state of its step
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
    RingCarrierLayout.carrierWidth logicalWidth}

abbrev Output (logicalWidth : Nat)
    (publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
      RingCarrierLayout.carrierWidth logicalWidth) :=
  TerminalStatement AppState ×
    Terminal.ProofEnvelope (logicalWidth := logicalWidth) (publicFits := publicFits)

variable {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
  {step : AppState → AppWitness → AppState}
  {cubeFits : RingCarrierLayout.carrierWidth logicalWidth ≤ 2 ^ cubeVariables}
  (exports : Exports relation ajtai circuitKeys step cubeFits)

/-- The terminal accepts, and no `iteration` valid steps of the application lead from `z0` to
`zi`. -/
def FalseAcceptance (control : AppState → AppWitness → Fin 2)
    (output : Output logicalWidth publicFits) : Prop :=
  Terminal.HoldsFor ajtai circuitKeys step control output.1 output.2 ∧
    ¬ Reaches exports.program output.1

/-- The witnesses that the reverse extractor returns for a visit and its answers: one for each
step of the walk's trail, first step first. -/
def extracted (visit : Visit logicalWidth publicFits)
    (answers : List (Answer logicalWidth publicFits)) : List AppWitness :=
  witnessesOf (trail ajtai circuitKeys exports visit answers)

/-- The terminal accepts the visit, and the extracted witnesses are not `iteration` valid steps
of the application that lead from `z0` to `zi` (HyperNova Definition 11 (ii)). -/
def ExtractionFailure (control : AppState → AppWitness → Fin 2)
    (visit : Visit logicalWidth publicFits) (answers : List (Answer logicalWidth publicFits)) :
    Prop :=
  Terminal.HoldsFor ajtai circuitKeys step control visit.statement visit.proof ∧
    ¬ ReachedBy exports.program visit.statement (extracted ajtai circuitKeys exports visit answers)

/-- The walk's start: the IVC adversary's output with every slot at bound `B`. -/
def walkAdversary (adversary : IvcAdversary (Output logicalWidth publicFits)) :
    IvcAdversary (Visit logicalWidth publicFits) :=
  ⟨adversary.Tape, adversary.tape, fun tape => initial (adversary.output tape)⟩

/-- The Poseidon2 collision finder of a reverse stage: it runs `stage` on `tape`, decodes the
visit after the stage's answers (`visitAfter`), and outputs `visitCollision` of that visit. Each
output is a Poseidon2 collision: two different inputs of one length with one `Poseidon2.hash`
(`PoseidonCollision`). -/
noncomputable def finder
    (stage : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits))
    (tape : stage.Tape) : Option PoseidonCollision :=
  visitCollision ajtai circuitKeys exports
    (visitAfter (back ajtai circuitKeys exports) (stage.input tape) (stage.results tape))

/-- The success chance of `finder` at `stage`: the probability over the stage's tape that the
finder, the stage followed by `visitCollision`, gives an output. Each output is a Poseidon2
collision (`PoseidonCollision`). It is the collision term of the IVC bounds. -/
noncomputable def finderChance
    (stage : Stage (Visit logicalWidth publicFits) (Answer logicalWidth publicFits)) : ℝ :=
  (stage.tape.toOuterMeasure {tape | (finder ajtai circuitKeys exports stage tape).isSome}).toReal

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

/-- The bound of the walk for the IVC adversary `adversary`: the sum, over the first `depth` stages
of the reverse extractor that starts at `walkAdversary adversary`, of the success chance of the
stage's Poseidon2 collision finder (`finderChance`) and the stage's Assumption 1 `error`. -/
noncomputable def walkBound (depth : Nat) : ℝ :=
  ∑ j : Fin depth,
    (finderChance ajtai circuitKeys exports (stages j.val).1 +
      error ((stages j.val).1.nifs (attemptOf ajtai circuitKeys exports)
        (back ajtai circuitKeys exports)))

/-- HyperNova Lemma 18 for the bounded-depth scheme, on the extractor's output: the probability that the terminal
accepts the adversary's output and the witnesses that the reverse extractor returns do not lead
from `z0` to `zi` is at most `walkBound`. The law is the one of the last stage: the adversary's
tape and the coins of each stage's extractor. -/
theorem extraction_bound (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    ((stages depth).1.tape.toOuterMeasure
        {tape | ExtractionFailure ajtai circuitKeys exports control
          ((stages depth).1.input tape) ((stages depth).1.results tape)}).toReal ≤
      walkBound ajtai circuitKeys exports assumption closed adversary admitted depth := by
  refine bad_le assumption closed (walkAdversary adversary) admitted depth
    (ExtractionFailure ajtai circuitKeys exports control)
    (fun visit => (visitCollision ajtai circuitKeys exports visit).isSome) ?_
  intro origin supported results length failure
  obtain ⟨accepted, unreached⟩ := failure
  obtain ⟨j, inside, found⟩ := covered ajtai circuitKeys exports keysShaped control
    (adversary.output origin) results (length ▸ depthBound origin supported) accepted
    fun leads => unreached (reachedBy_of_leads _ _ _ leads
      (trail_lengths ajtai circuitKeys exports _ _))
  exact ⟨⟨j, length ▸ inside⟩, found.imp_left (visitCollision_isSome ajtai circuitKeys exports)⟩

/-- HyperNova Lemma 18 for the bounded-depth scheme: the probability of a false acceptance is at
most `walkBound`. A false acceptance is an extraction failure, because no list of witnesses
reaches its statement. -/
theorem probability_bound (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (control : AppState → AppWitness → Fin 2) (depth : Nat)
    (depthBound : ∀ tape ∈ adversary.tape.support, (adversary.output tape).1.iteration ≤ depth) :
    (adversary.tape.toOuterMeasure
        {tape | FalseAcceptance ajtai circuitKeys exports control
          (adversary.output tape)}).toReal ≤
      walkBound ajtai circuitKeys exports assumption closed adversary admitted depth := by
  let event : Set (Visit logicalWidth publicFits) :=
    {visit | FalseAcceptance ajtai circuitKeys exports control
      (visit.statement, visit.proof)}
  have moved := start_event assumption closed (walkAdversary adversary) admitted depth event
  change adversary.tape.toOuterMeasure
      {tape | FalseAcceptance ajtai circuitKeys exports control
        (adversary.output tape)} = _ at moved
  rw [moved]
  refine (ENNReal.toReal_mono (event_ne_top _ _) (MeasureTheory.measure_mono ?_)).trans
    (extraction_bound ajtai circuitKeys exports assumption closed adversary admitted keysShaped
      control depth depthBound)
  intro tape falseAcceptance
  exact ⟨falseAcceptance.1, fun reached => falseAcceptance.2 ⟨_, reached⟩⟩

end NightstreamFPrime.Export.BoundedDepth.IvcSecurity
