import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Evaluator
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.PaperConstruction2
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Checker
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.CanonicalOpening.SourceInput
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.CertificateRefinement
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.PaperProfile.PhysicalTransition

/-!
Paper-exact executable boundary for the fixed-one recursive F-prime verifier.

Assurance tier: model-level.

Owns: the fixed-one outer checks, the complete raw NIFS checker, the five
operational paper `Pi_DEC` output equations, canonical output construction,
and the exact Construction-2 refinement or a precise named binding/algebraic
failure.

Does not own: Fiat--Shamir/Poseidon2 instantiation, probability bounds,
source extraction, commitment binding, Rust, R1CS, generated rows, or costs.

Authority boundary: this checker strengthens the existing physical checker
with the paper output equations. It does not require child openings or a
deterministic private split. Semantic source binding is supplied by the typed
`SourceAuthority` receipt and therefore cannot reappear as a generic failure
branch. `SourceAuthority.ofCanonicalOpening` constructs that receipt from the
existing opening-derived carrier once the selected verifier context is
identified exactly.

Emits constraints: no.

| Boundary | Exact equation or ownership rule | Lean owner |
|---|---|---|
| acceptance | physical fixed-one checks and all paper output equations | `Accepted`, `check_eq_true_iff_accepted` |
| execution | successful checking returns the verifier-computed canonical output | `run`, `run_eq_some_iff` |
| source authority | one fixed source family binds both public input surfaces | `SourceAuthority`, `SourceAuthority.ofCanonicalOpening` |
| soundness | successful execution gives Construction 2 or one exact named failure | `run_refinesConstruction2_or_namedFailure` |
| completeness | honest paper inputs execute or expose a bounded-sampler shortfall | `exists_run_and_construction2_or_samplerShortfall` |
| child authority | the theorem retains the actual public child vector | `run_refinesConstruction2_or_namedFailure` |
-/

set_option autoImplicit false

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary

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

namespace PaperOutputChecker

export
  Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Checker
  (paperOutputCheck paperOutputCheck_eq_true_iff)

end PaperOutputChecker

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- The verifier-owned selected NIFS context used by the paper boundary. -/
abbrev SelectedContext
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :=
  (nifsContext setup input).materialize

/-- Exact semantic source authority consumed by the paper boundary.

The generic fixed-one public carrier deliberately contains no private opening,
so Boolean acceptance alone cannot manufacture this receipt. Production must
construct it from authoritative opening data or reduce failure of that
construction to its concrete commitment-binding contract. -/
structure SourceAuthority
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) : Type where
  data : Data shape
  semanticInput : SemanticInput (SelectedContext setup input) data

namespace SourceAuthority

/-- The canonical opening-derived carrier supplies source authority after one
exact context identity. No digest, existential source search, or verifier
acceptance bit is used. -/
def ofCanonicalOpening
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (carrier :
      FixedActive.CanonicalOpening.SourceInput.Carrier shape
        publicRingColumns publicFits)
    (context :
      FixedActive.CanonicalOpening.Context shape TranscriptState
        publicRingColumns publicFits verifierRows)
    (contextExact :
      SelectedContext setup input =
        (carrier.install context).full) :
    SourceAuthority setup input where
  data := carrier.data
  semanticInput := by
    rw [contextExact]
    exact carrier.semanticInput context

end SourceAuthority

/-- Complete positive meaning of the strengthened paper checker. -/
structure Accepted
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) : Prop where
  physical :
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Accepted
      machine setup input certificate
  paperOutput :
    FixedActive.PaperProfile.PhysicalOutput.PaperOutputEquations
      (SelectedContext setup input) certificate

/-- Execute the existing fixed-one verifier and additionally enforce every
paper `Pi_DEC` output equation on the actual public child family. -/
def check
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) : Bool :=
  Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.check
      machine setup input certificate &&
    PaperOutputChecker.paperOutputCheck (SelectedContext setup input) certificate

/-- The strengthened Boolean checker is exact to its two independently named
parts. -/
theorem check_eq_true_iff_accepted
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) :
    check machine setup input certificate = true <->
      Accepted machine setup input certificate := by
  rw [check, Bool.and_eq_true,
    Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.check_eq_true_iff_accepted,
    PaperOutputChecker.paperOutputCheck_eq_true_iff]
  exact ⟨fun parts => ⟨parts.1, parts.2⟩,
    fun accepted => ⟨accepted.physical, accepted.paperOutput⟩⟩

/-- Fail-closed paper evaluator. The result is the existing canonical outer
output, computed from the verifier-derived NIFS result. -/
def run
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) :
    Option
      (Output Digest AppState shape publicRingColumns publicFits verifierRows
        1) :=
  if check machine setup input certificate then
    some
      (ActiveSemantics.outputOf machine (input.toActive setup)
        ActiveSemantics.FixedOneCanonical.selected
        (FixedActive.resultOf (SelectedContext setup input) certificate))
  else
    none

/-- Exact successful-execution characterization. -/
theorem run_eq_some_iff
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        1) :
    run machine setup input certificate = some output <->
      Accepted machine setup input certificate /\
        output =
          ActiveSemantics.outputOf machine (input.toActive setup)
            ActiveSemantics.FixedOneCanonical.selected
            (FixedActive.resultOf (SelectedContext setup input) certificate) := by
  cases checked : check machine setup input certificate with
  | false =>
      constructor
      · intro executed
        simp [run, checked] at executed
      · rintro ⟨accepted, _⟩
        have trueCheck : check machine setup input certificate = true :=
          (check_eq_true_iff_accepted machine setup input certificate).2
            accepted
        rw [checked] at trueCheck
        contradiction
  | true =>
      have accepted : Accepted machine setup input certificate :=
        (check_eq_true_iff_accepted machine setup input certificate).1 checked
      constructor
      · intro executed
        have outputEq :
            output =
              ActiveSemantics.outputOf machine (input.toActive setup)
                ActiveSemantics.FixedOneCanonical.selected
                (FixedActive.resultOf (SelectedContext setup input)
                  certificate) := by
          simpa [run, checked] using executed.symm
        exact ⟨accepted, outputEq⟩
      · rintro ⟨_, outputEq⟩
        simpa [run, checked] using outputEq.symm

/-- Exact failure families left after the executable paper equations and
typed source authority are enforced. There is no generic source-binding,
refinement, or output-unbound constructor. -/
inductive NamedFailure
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) : Prop where
  | yRingBinding
      (data : Data shape)
      (bound : SemanticInput (SelectedContext setup input) data)
      (failure :
        ¬ certificate.piCcs.output.yRing =
          Polynomial.Fe.sourceYRingAt data
            (derive (SelectedContext setup input) certificate).piCcs.fePoint.row) :
      NamedFailure setup input certificate
  | packedYZcolBinding
      (data : Data shape)
      (bound : SemanticInput (SelectedContext setup input) data)
      (failure :
        ¬ Polynomial.Nc.BlockLane.Terminal.PackedYZcolBoundAtBlock
          (SelectedContext setup input).covers data
          (derive (SelectedContext setup input) certificate).piCcs.ncPoint.block
          certificate.piCcs.output) :
      NamedFailure setup input certificate
  | piCcsAlgebraic
      (data : Data shape)
      (bound : SemanticInput (SelectedContext setup input) data)
      (failure : PiCcsBadEvent (SelectedContext setup input) data certificate) :
      NamedFailure setup input certificate

/-- A successful source-authoritative paper-exact fixed-one execution is
precisely HyperNova Construction 2's recursive transition over the
paper-selected SuperNeo NIFS, or one exact output/algebraic failure occurs.

The target uses the actual public child vector. No child opening or
deterministic private split occurs in the theorem. -/
theorem run_refinesConstruction2_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input)
    (sourceAuthority : SourceAuthority setup input)
    (output :
      Output Digest AppState shape publicRingColumns publicFits verifierRows
        1)
    (executed : run machine setup input certificate = some output) :
    Paper.Construction2.RecursiveHolds
        (PaperSelectedNifsSemantics.family
          (ActiveSemantics.Construction2.selectedNifsSetup setup))
        machine functionIndex (input.toActive setup).toPaper output.toPaper ∨
      NamedFailure setup input certificate := by
  rcases
      (run_eq_some_iff machine setup input certificate output).1 executed with
    ⟨accepted, outputEq⟩
  let data := sourceAuthority.data
  have semanticInput :
      SemanticInput (SelectedContext setup input) data :=
    sourceAuthority.semanticInput
  rcases
      FixedActive.PaperProfile.PhysicalTransition.accepted_implies_transition_or_outputUnbound_or_badEvent
        noZeroDivisors (SelectedContext setup input) data certificate
        semanticInput accepted.paperOutput.canonicalPublicInput
        accepted.paperOutput.parentEvaluationSize
        accepted.paperOutput.childEvaluationSize
        accepted.physical.nifs with
    paperTransition | outputUnbound | badEvent
  · let activeCertificate :
        ActiveEvaluator.Certificate setup (input.toActive setup) := {
      selected := ActiveSemantics.FixedOneCanonical.selected
      nifs := certificate
    }
    have activePhysical :
        ActiveEvaluator.PhysicalChecks setup machine functionIndex
          (input.toActive setup) activeCertificate
          (FixedActive.resultOf (SelectedContext setup input) certificate)
          output := by
      exact {
        outer := {
          iterationPositive := accepted.physical.outer.iterationPositive
          priorSlot :=
            ActiveSemantics.FixedOneCanonical.Input.priorSlot_derived
              setup (input.toSemantic setup)
              ActiveSemantics.FixedOneCanonical.selected
          priorPublicInput := accepted.physical.outer.priorPublicInput
          expectedStructure :=
            ActiveSemantics.FixedOneCanonical.Input.expectedStructure_derived
              setup (input.toSemantic setup)
              ActiveSemantics.FixedOneCanonical.selected
          dispatch :=
            ActiveSemantics.FixedOneCanonical.dispatch_derived machine
              functionIndex (input.toSemantic setup)
        }
        nifsAccepted := accepted.physical.nifs
        resultExact := rfl
        outputExact := outputEq
      }
    exact Or.inl
      (ActiveEvaluator.physicalChecks_refineConstruction2_of_paperTransition
        (input.toActive setup) activeCertificate
        (FixedActive.resultOf (SelectedContext setup input) certificate) output
        (by
          simpa [activeCertificate, SelectedContext, nifsContext_materialize]
            using paperTransition)
        activePhysical)
  · by_cases yRing :
        certificate.piCcs.output.yRing =
          Polynomial.Fe.sourceYRingAt data
            (derive (SelectedContext setup input) certificate).piCcs.fePoint.row
    · apply Or.inr
      apply NamedFailure.packedYZcolBinding data semanticInput
      intro packed
      exact outputUnbound ⟨yRing, packed⟩
    · exact Or.inr
        (NamedFailure.yRingBinding data semanticInput yRing)
  · exact Or.inr
      (NamedFailure.piCcsAlgebraic data semanticInput badEvent)

/-- Honest paper sources either construct one accepted paper-exact fixed-one
execution satisfying HyperNova Construction 2, or expose one exact coordinate
where the bounded production sampler shortfalls. The successful branch keeps
the actual public child vector and derives the paper output equations from the
existing certificate refinement. -/
theorem exists_run_and_construction2_or_samplerShortfall
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (outer :
      Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.OuterChecks
        machine setup input)
    (honest : ActiveSemantics.HonestNifs.SemanticPremises setup
      (input.toActive setup) ActiveSemantics.FixedOneCanonical.selected) :
    (∃ certificate : Certificate setup input,
      ∃ output :
          Output Digest AppState shape publicRingColumns publicFits
            verifierRows 1,
        run machine setup input certificate = some output /\
          Paper.Construction2.RecursiveHolds
            (PaperSelectedNifsSemantics.family
              (ActiveSemantics.Construction2.selectedNifsSetup setup))
            machine functionIndex (input.toActive setup).toPaper
              output.toPaper) \/
      ConcretePhi81.HonestSamplerShortfall
        (SelectedContext setup input) honest.data := by
  have semanticInput :
      ConcretePhi81.SemanticInput (SelectedContext setup input) honest.data := by
    simpa [SelectedContext, nifsContext_materialize] using honest.semanticInput
  have running :
      ConcretePhi81.RunningAuthority.Accepted (SelectedContext setup input) := by
    simpa [SelectedContext, nifsContext_materialize] using honest.running
  rcases
      ConcretePhi81.complete_or_samplerShortfall
        (SelectedContext setup input) honest.data honest.paper semanticInput
        running with
    completed | shortfall
  · rcases completed with
      ⟨_challenges, certificate, nifsAccepted, refinement, _childrenValid⟩
    have paperOutput :
        FixedActive.PaperProfile.PhysicalOutput.PaperOutputEquations
          (SelectedContext setup input) certificate :=
      FixedActive.PaperProfile.CertificateRefinement.paperOutputEquations
        (SelectedContext setup input) honest.data certificate nifsAccepted
        refinement
    have accepted : Accepted machine setup input certificate := {
      physical := {
        outer := outer
        nifs := nifsAccepted
      }
      paperOutput := paperOutput
    }
    have checked : check machine setup input certificate = true :=
      (check_eq_true_iff_accepted machine setup input certificate).2 accepted
    let output :=
      ActiveSemantics.outputOf machine (input.toActive setup)
        ActiveSemantics.FixedOneCanonical.selected
        (FixedActive.resultOf (SelectedContext setup input) certificate)
    have executed : run machine setup input certificate = some output := by
      simp [run, checked, output]
    have resultTransition :
        FixedActive.ResultTransition (SelectedContext setup input)
          (FixedActive.resultOf (SelectedContext setup input) certificate) :=
      ConcretePhi81.Result.resultOf_refines refinement
    rcases
        (FixedActive.resultTransition_iff_exists_paperDecomposition
          (SelectedContext setup input)
          (FixedActive.resultOf (SelectedContext setup input) certificate)).1
          resultTransition with
      ⟨_data, _witness, decomposition⟩
    let activeCertificate :
        ActiveEvaluator.Certificate setup (input.toActive setup) := {
      selected := ActiveSemantics.FixedOneCanonical.selected
      nifs := certificate
    }
    have activePhysical :
        ActiveEvaluator.PhysicalChecks setup machine functionIndex
          (input.toActive setup) activeCertificate
          (FixedActive.resultOf (SelectedContext setup input) certificate)
          output := by
      exact {
        outer := {
          iterationPositive := outer.iterationPositive
          priorSlot :=
            ActiveSemantics.FixedOneCanonical.Input.priorSlot_derived
              setup (input.toSemantic setup)
              ActiveSemantics.FixedOneCanonical.selected
          priorPublicInput := outer.priorPublicInput
          expectedStructure :=
            ActiveSemantics.FixedOneCanonical.Input.expectedStructure_derived
              setup (input.toSemantic setup)
              ActiveSemantics.FixedOneCanonical.selected
          dispatch :=
            ActiveSemantics.FixedOneCanonical.dispatch_derived machine
              functionIndex (input.toSemantic setup)
        }
        nifsAccepted := nifsAccepted
        resultExact := rfl
        outputExact := rfl
      }
    have recursive :
        Paper.Construction2.RecursiveHolds
          (PaperSelectedNifsSemantics.family
            (ActiveSemantics.Construction2.selectedNifsSetup setup))
          machine functionIndex (input.toActive setup).toPaper output.toPaper :=
      ActiveEvaluator.physicalChecks_refineConstruction2_of_paperTransition
        (input.toActive setup) activeCertificate
        (FixedActive.resultOf (SelectedContext setup input) certificate) output
        (by
          simpa [activeCertificate, SelectedContext, nifsContext_materialize]
            using
              (show FixedActive.PaperProfile.Transition
                  (FixedActive.paperProfileOf (SelectedContext setup input))
                  (SelectedContext setup input).input
                  (FixedActive.resultOf
                    (SelectedContext setup input) certificate).children from
                ⟨_data, _witness, decomposition.paper⟩))
        activePhysical
    exact Or.inl ⟨certificate, output, executed, recursive⟩
  · exact Or.inr shortfall

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.PaperBoundary
