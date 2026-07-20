import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.SemanticBoundary

/-!
Compile-time surface regression for the fixed-active outer evaluator.

| Stage path | Property under test |
|---|---|
| `fprime.active.checkers` | nontrivial equalities have named exact Boolean owners |
| `fprime.active.physical` | successful execution has one complete physical characterization |
| `fprime.active.semantic` | soundness exposes named failures and closes only under explicit semantic/security premises |
| `fprime.active.completeness` | honest paper/source premises construct one accepted canonical output or exact sampler shortfall |
-/

open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
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

#check Certificate
#check Checkers
#check Checkers.priorLinkCheck_eq_true_iff
#check Checkers.freshStructureCheck_eq_true_iff
#check Checkers.dispatchCheck_eq_true_iff
#check OuterChecks
#check outerCheck
#check outerCheck_eq_true_iff
#check PhysicalChecks
#check run
#check run_eq_some_iff_physicalChecks
#check run_sound_or_outputUnbound_or_piCcsBadEvent
#check YRingUnbound
#check run_sound_or_yRingUnbound_or_piCcsBadEvent_of_packedYZcolBound
#check SoundnessClosure
#check run_sound_of_closure
#check exists_run_and_holds_or_samplerShortfall
#check run_complete_of_outer_and_honestNifs
#check exists_run_and_holds_of_outer_and_honestNifs

/-! This exact-result regression forbids a generic `outputUnbound` alternative
from reappearing after the packed `y_zcol` equation has been supplied. -/

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows slotCount : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

example
    {setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows slotCount}
    {machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows slotCount}
    {functionIndex : Fin slotCount}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (checkers : Checkers setup machine functionIndex)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows slotCount)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        slotCount)
    (data : Data shape)
    (semanticInput :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.SemanticInput
        (contextAt setup input certificate.selected) data)
    (childOpenings :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.ChildOpenings
        (contextAt setup input certificate.selected) data certificate.nifs)
    (packed :
      Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
        (contextAt setup input certificate.selected).covers data
        (derive (contextAt setup input certificate.selected)
          certificate.nifs).piCcs.ncPoint.block certificate.nifs.piCcs.output)
    (executed : run checkers input certificate = some output) :
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Holds
        setup machine functionIndex input output \/
      YRingUnbound certificate data \/
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsBadEvent
        (contextAt setup input certificate.selected) data certificate.nifs := by
  exact run_sound_or_yRingUnbound_or_piCcsBadEvent_of_packedYZcolBound
    noZeroDivisors checkers input certificate output data semanticInput
    childOpenings packed executed

end
