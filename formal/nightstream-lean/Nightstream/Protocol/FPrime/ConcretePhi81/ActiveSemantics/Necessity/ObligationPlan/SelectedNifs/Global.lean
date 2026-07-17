import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs

/-!
Global-language lift of the selected-NIFS removal witness.

Owns: embedding the actual-type selected-NIFS mutation into the one global
fixed-active verifier language, both from an existing realization and from
independent honest-NIFS premises with explicit sampler shortfall.

Does not own: a closed fixed-active fixture, construction of honest premises,
any other obligation family, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: the local witness keeps its exact outer input when it is
embedded as `ObligationPlan.Global.Case`. The global theorem therefore changes
only the carrier of the counterexample; it does not weaken, reinterpret, or
manufacture any semantic premise.

| Stage path | Mathematical obligation | Assurance status | Lean owner |
|---|---|---|---|
| `fprime.active.necessity.selected_nifs.global.realization` | embed the local parent-stage mutation into the global case language | exact model lift | `necessary_of_realization` |
| `fprime.active.necessity.selected_nifs.global.outcome` | independent honest NIFS premises yield a global removal witness or one exact sampler shortfall | exhaustive model outcome | `necessary_or_samplerShortfall_of_semanticPremises` |
| `fprime.active.necessity.selected_nifs.global.honest` | successful-sampler premises plus the other five equations yield a global removal witness | compatibility necessity | `necessary_of_honestNifs` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {domain : FlatNcDomain}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- The local actual-type mutation is a witness in the common global verifier
language without any additional semantic assumption. -/
theorem necessary_of_realization
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (realization :
      SelectedNifs.Realization setup machine functionIndex input) :
    CheckPlan.NecessaryForSoundness
      (ObligationPlan.Global.semantics setup machine functionIndex)
      (ObligationPlan.Global.target setup machine functionIndex)
      checks .selectedNifs :=
  ObligationPlan.Global.lift_local_necessary
    setup machine functionIndex input .selectedNifs realization.necessary

/-- Independent honest NIFS premises produce the global selected-NIFS
removal witness, unless the fixed bounded sampler exposes one exact shortfall
coordinate. -/
theorem necessary_or_samplerShortfall_of_semanticPremises
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : HonestNifs.SemanticPremises setup input selected)
    (iterationPositive : 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicInput :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
        (ObligationPlan.Global.semantics setup machine functionIndex)
        (ObligationPlan.Global.target setup machine functionIndex)
        checks .selectedNifs ∨
      ConcretePhi81.HonestSamplerShortfall
        (contextAt setup input selected) premises.data := by
  rcases
      SelectedNifs.necessary_or_samplerShortfall_of_semanticPremises
        setup machine functionIndex input selected premises iterationPositive
          priorSlot priorPublicInput expectedStructure dispatch with
    localNecessary | shortfall
  · exact Or.inl <|
      ObligationPlan.Global.lift_local_necessary
        setup machine functionIndex input .selectedNifs localNecessary
  · exact Or.inr shortfall

/-- Strong successful-sampler premises plus the five retained equations give
the same selected-NIFS removal witness in the global verifier language.

This remains conditional because the repository has no closed actual
fixed-active fixture supplying these premises. -/
theorem necessary_of_honestNifs
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows slotCount)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount)
    (functionIndex : Fin slotCount)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (selected : Fin slotCount)
    (premises : HonestNifs.Premises setup input selected)
    (iterationPositive : 0 < input.iteration)
    (priorSlot : input.priorPc = selected.val + 1)
    (priorPublicInput :
      input.fresh.publicInput =
        machine.encodeInstance
          (machine.hash (Paper.priorHashPreimage input.toPaper)))
    (expectedStructure :
      input.fresh.constraintSystem =
        setup.expectedStructure input.verifierKey selected)
    (dispatch :
      machine.control input.zi input.witness =
        Paper.ProgramCounter.ofIndex functionIndex) :
    CheckPlan.NecessaryForSoundness
      (ObligationPlan.Global.semantics setup machine functionIndex)
      (ObligationPlan.Global.target setup machine functionIndex)
      checks .selectedNifs :=
  ObligationPlan.Global.lift_local_necessary
    setup machine functionIndex input .selectedNifs
      (SelectedNifs.necessary_of_honestNifs
        setup machine functionIndex input selected premises iterationPositive
          priorSlot priorPublicInput expectedStructure dispatch)

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global
