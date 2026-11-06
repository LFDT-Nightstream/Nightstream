import NightstreamFPrime.Export.OptionB.IvcWalk

/-! HyperNova Lemma 17 for option B: the IVC false-acceptance bound. An IVC adversary outputs a
terminal statement and a proof. It wins when the terminal relation accepts and no `iteration`
steps of the application lead from `z0` to `zi` (`FalseAcceptance`). The reverse extractor walks
back from the terminal one step per stage (`IvcWalk`), with the extractor that HyperNova errata
Assumption 1 gives for each stage's fold attempt. Then

  Pr[false acceptance] ≤ Σ j < depth, (Pr[collision at visit j] + error(stage j)),

where `depth` bounds the adversary's iteration. The collision at visit `j` is a Poseidon2
collision on two states that stage `j` computes from its tape: the next state of the step that
the visit's fresh witness decodes to, and the visit's terminal state
(`TerminalBridge.Collision`). Each fold attempt is linked to the prior state of its step
(`Lifecycle.OptionB.PriorLink`), so its running claim is fixed by the fresh instance that the
fold transcript absorbs, up to a collision on a pair of prior states (`priorLink_binds`).
Assumption 1 is a premise; it is needed only at bounds up to `B + cap · T · (b − 1)`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.OptionB.IvcSecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.ReverseStages
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Export.OptionB.TerminalBridge
open NightstreamFPrime.Export.OptionB.IvcWalk

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

/-- HyperNova Lemma 17 for option B. -/
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
  refine bad_le assumption closed (walkAdversary adversary) admitted depth
    (fun input _ => input ∈ event) (WalkCollision ajtai circuitKeys exports) ?_
  intro origin supported results length falseAcceptance
  obtain ⟨accepted, unreached⟩ := falseAcceptance
  obtain ⟨j, inside, found⟩ := covered ajtai circuitKeys exports keysShaped control cap
    (adversary.output origin) results (length ▸ depthBound origin supported) accepted
    fun leads => unreached (reaches_of_leads _ _ _ leads
      (trail_lengths ajtai circuitKeys exports _ _))
  exact ⟨⟨j, length ▸ inside⟩, found⟩

end NightstreamFPrime.Export.OptionB.IvcSecurity
