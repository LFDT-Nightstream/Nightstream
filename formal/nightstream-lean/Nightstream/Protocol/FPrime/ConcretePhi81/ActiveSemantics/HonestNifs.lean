import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

/-!
Honest fixed-active NIFS premises shared by outer completeness and necessity.

Owns: the independent paper/source premises for one verifier-selected slot,
the exhaustive success-or-sampler-shortfall outcome, and a stronger
successful-sampler compatibility surface for callers that already have a
bounded replay witness.

Does not own: outer iteration, prior-slot, prior-link, structure, or dispatch
checks; executable checking; bad-event exclusion; Rust; R1CS; costs;
necessity; or row removal.

Emits constraints: no.

Authority boundary: `SemanticPremises` carries only independent source truth
and incoming-parent authority. It carries no challenge vector or sampler
success assumption. `Premises` adds the stronger bounded-sampler hypothesis
needed by compatibility callers. Neither structure carries an outer F-prime
equation or accepts a completed result as caller authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.active.honest_nifs.paper` | the independent source family satisfies the paper obligations | semantic premise | `SemanticPremises.paper` |
| `fprime.active.honest_nifs.input` | public and private source views describe that same family | semantic premise | `SemanticPremises.semanticInput` |
| `fprime.active.honest_nifs.running` | the incoming cached parent recomposes to the running children | checked premise | `SemanticPremises.running` |
| `fprime.active.honest_nifs.outcome` | honest sources yield a transition or one exact bounded-sampler shortfall | exhaustive model outcome | `SemanticPremises.exists_resultTransition_or_samplerShortfall` |
| `fprime.active.honest_nifs.sampler` | one fixed challenge vector works for every accepted PiCCS certificate | strong compatibility premise | `Premises.samplerAvailable` |
| `fprime.active.honest_nifs.result` | honest construction yields one independent semantic fold result | derived | `Premises.exists_resultTransition` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics

universe uOuterKey uAppState uWitness uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {domain : FlatNcDomain}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Independent honest NIFS inputs at one selected slot. This surface contains
no sampler-success assumption and no outer F-prime equation. -/
structure SemanticPremises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount) where
  data : Data shape
  paper : Semantics.Paper.Holds data
  semanticInput :
    ConcretePhi81.SemanticInput (contextAt setup input selected) data
  running :
    ConcretePhi81.RunningAuthority.Accepted
      (contextAt setup input selected)

/-- Strong successful-sampler premises retained for compatibility with callers
that already provide one challenge vector valid after every accepted PiCCS
prefix. New semantic completeness arguments should prefer `SemanticPremises`
and handle the explicit shortfall branch. -/
structure Premises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount) where
  data : Data shape
  paper : Semantics.Paper.Holds data
  semanticInput :
    ConcretePhi81.SemanticInput (contextAt setup input selected) data
  running :
    ConcretePhi81.RunningAuthority.Accepted
      (contextAt setup input selected)
  challenges : Fin FixedActive.arity.total -> RingF
  samplerAvailable :
    forall piCcsCertificate :
        Protocol.Certificate
          (contextAt setup input selected).piCcsInput domain,
      Protocol.Accepted
          (contextAt setup input selected).feMachine
          (contextAt setup input selected).ncMachine
          (contextAt setup input selected).initialState
          (contextAt setup input selected).profile
          (contextAt setup input selected).piCcsInput
          (contextAt setup input selected).feCoins
          (contextAt setup input selected).ncCoins
          piCcsCertificate ->
        ConcretePhi81.Sampler.Bound
          (contextAt setup input selected).piRlcMachine
          ((contextAt setup input selected).piCcsOutputHandoff
            (Protocol.derive
              (contextAt setup input selected).feMachine
              (contextAt setup input selected).ncMachine
              (contextAt setup input selected).initialState
              piCcsCertificate).finalState
            piCcsCertificate.output)
          challenges

namespace SemanticPremises

/-- Independent honest NIFS completeness has exactly two model-level outcomes:
one accepted certificate with its canonical semantic fold result, or one
finite coordinate at which the production-sized rejection sampler shortfalls.
No fixed challenge vector is assumed across unrelated PiCCS certificates. -/
theorem exists_resultTransition_or_samplerShortfall
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : SemanticPremises setup input selected) :
    (exists certificate : FixedActive.Certificate (contextAt setup input selected),
      ConcretePhi81.Accepted (contextAt setup input selected) certificate /\
        FixedActive.ResultTransition
          (contextAt setup input selected)
          (FixedActive.resultOf (contextAt setup input selected) certificate)) \/
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      ConcretePhi81.complete_or_samplerShortfall
        (contextAt setup input selected)
        premises.data premises.paper premises.semanticInput premises.running with
    completed | shortfall
  · rcases completed with
      ⟨_challenges, certificate, accepted, holds, _childrenValid⟩
    exact Or.inl ⟨certificate, accepted,
      ⟨premises.data, certificate, rfl, holds⟩⟩
  · exact Or.inr shortfall

end SemanticPremises

namespace Premises

/-- Forget the strong sampler-success hypothesis and retain only independent
semantic authority. -/
def toSemanticPremises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : Premises setup input selected) :
    SemanticPremises setup input selected where
  data := premises.data
  paper := premises.paper
  semanticInput := premises.semanticInput
  running := premises.running

/-- Honest NIFS completeness constructs one raw certificate whose canonical
result satisfies the independent fixed-active semantic transition. -/
theorem exists_resultTransition
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : Premises setup input selected) :
    exists certificate : FixedActive.Certificate (contextAt setup input selected),
      ConcretePhi81.Accepted (contextAt setup input selected) certificate /\
        FixedActive.ResultTransition
          (contextAt setup input selected)
          (FixedActive.resultOf (contextAt setup input selected) certificate) := by
  rcases
      ConcretePhi81.complete_of_paperObligations
        (contextAt setup input selected)
        premises.data premises.paper premises.semanticInput premises.running
        premises.challenges premises.samplerAvailable with
    ⟨certificate, accepted, holds, _childrenValid⟩
  exact ⟨certificate, accepted,
    ⟨premises.data, certificate, rfl, holds⟩⟩

end Premises

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
