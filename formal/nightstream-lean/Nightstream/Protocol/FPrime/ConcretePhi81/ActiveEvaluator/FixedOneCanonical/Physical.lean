import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical.Context
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.Checker

/-!
Physical checks for the payload-minimal fixed-one F-prime verifier.

Owns: the raw NIFS certificate, the two retained outer equations, their
Boolean checker, and exact equivalence to complete physical acceptance.

Does not own: output construction, semantic source truth, bad-event bounds,
Rust/R1CS refinement, physical rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: selection, prior counter, structure, stages, parent
presence, and dispatch are computed by `Context`. The prior link compares the
complete finite public input, and NIFS acceptance delegates to the exact raw
message checker.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.iteration` | recursive step has positive iteration | checked | `OuterChecks.iterationPositive`, `outerCheck` |
| `fprime.fixed_one.prior_link` | fresh public input equals the verifier-computed prior hash image | checked | `OuterChecks.priorPublicInput`, `outerCheck` |
| `fprime.fixed_one.nifs` | canonical raw NIFS certificate physically accepts | checked | `FixedActive.Canonical.Checker` |
| `fprime.fixed_one.physical_exact` | Boolean acceptance iff the retained physical obligations hold | exact model theorem | `check_eq_true_iff_accepted` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.CarrierEquality

universe uOuterKey uAppState uWitness uDigest uTranscriptState

section

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {domain : FlatNcDomain}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Raw fixed-one F-prime certificate: exactly the raw certificate of the
canonically reconstructed NIFS context. -/
abbrev Certificate
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :=
  FixedActive.Certificate (nifsContext setup input).materialize

/-- The two outer obligations not encoded by the canonical carrier or the
selected NIFS verifier. -/
structure OuterChecks
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) : Prop where
  iterationPositive : 0 < input.iteration
  priorPublicInput :
    input.fresh.publicInput =
      machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage (input.toActive setup).toPaper))

/-- Execute exactly the retained outer equation families. -/
def outerCheck
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) : Bool :=
  decide (0 < input.iteration) &&
    publicInputEqual input.fresh.publicInput
      (machine.encodeInstance
        (machine.hash (Paper.priorHashPreimage (input.toActive setup).toPaper)))

/-- The executable outer checker is exact to the two retained equations. -/
theorem outerCheck_eq_true_iff
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    outerCheck machine setup input = true <->
      OuterChecks machine setup input := by
  simp only [outerCheck, Bool.and_eq_true, decide_eq_true_eq,
    publicInputEqual_eq_true_iff]
  constructor
  · rintro ⟨iteration, prior⟩
    exact ⟨iteration, prior⟩
  · intro outer
    exact ⟨outer.iterationPositive, outer.priorPublicInput⟩

/-- Complete physical acceptance before output construction. -/
structure Accepted
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) : Prop where
  outer : OuterChecks machine setup input
  nifs : ConcretePhi81.Accepted
    (nifsContext setup input).materialize certificate

/-- Boolean decision procedure for the complete payload-minimal physical
F-prime verifier. -/
def check
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) : Bool :=
  outerCheck machine setup input &&
    FixedActive.Canonical.Checker.check (nifsContext setup input) certificate

/-- Boolean physical acceptance contains exactly the two outer families and
the independently exact raw NIFS checker. -/
theorem check_eq_true_iff_accepted
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape domain
        publicRingColumns publicFits verifierRows 1)
    (input :
      Input OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows)
    (certificate : Certificate setup input) :
    check machine setup input certificate = true <->
      Accepted machine setup input certificate := by
  rw [check, Bool.and_eq_true, outerCheck_eq_true_iff,
    FixedActive.Canonical.Checker.check_eq_true_iff_accepted]
  exact ⟨fun parts => ⟨parts.1, parts.2⟩,
    fun accepted => ⟨accepted.outer, accepted.nifs⟩⟩

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.FixedOneCanonical
