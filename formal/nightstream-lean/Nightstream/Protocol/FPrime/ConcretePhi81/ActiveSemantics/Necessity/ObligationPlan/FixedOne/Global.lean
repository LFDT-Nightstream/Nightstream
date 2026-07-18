import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne

/-!
Global canonical fixed-one obligation plan.

Owns: one verifier-language case containing a canonical fixed-one input and
one selected NIFS result; interpretation of the three retained families over
that case; exactness across all canonical inputs under one setup and machine;
and lifting of a pointwise removal witness into that common language.

Does not own: a concrete setup, machine, honest NIFS result, removal witness,
inclusion-minimality fixture, production decoding, Rust, R1CS, costs, or row
removal.

Emits constraints: no.

Authority boundary: different removal witnesses may use different canonical
outer inputs, but every case is interpreted under the same verifier-owned
setup and machine. The target is the independent
`FixedOneCanonical.Obligations`, not an executable acceptance predicate.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.fixed_one.canonical.global.case` | one canonical input plus one selected semantic result | typed verifier language | `Case` |
| `fprime.fixed_one.canonical.global.exact` | three retained leaves equal the independent canonical obligations for every case | exact model theorem | `exact` |
| `fprime.fixed_one.canonical.global.lift` | a pointwise removal witness embeds into the common input language | exact model theorem | `lift_local_necessary` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne.Global

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uOuterKey uAppState uWitness uDigest uTranscriptState

/-- One complete canonical verifier-language case. -/
structure Case
    (OuterKey : Type uOuterKey)
    (AppState : Type uAppState)
    (Witness : Type uWitness)
    (shape : SemanticShape)
    (publicRingColumns : Nat)
    (publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth)
    (verifierRows : Nat) where
  input :
    FixedOneCanonical.Input OuterKey AppState Witness shape
      publicRingColumns publicFits verifierRows
  selectedNext :
    Slot shape publicRingColumns publicFits verifierRows

section

variable {OuterKey : Type uOuterKey}
variable {Digest : Type uDigest}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {TranscriptState : Type uTranscriptState}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Interpret a retained family over the complete canonical case. -/
def semantics
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1) :
    Family ->
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows -> Prop :=
  fun family case =>
    Canonical.semantics setup machine functionIndex case.input family
      case.selectedNext

/-- Independent canonical obligations over the complete case. -/
def target
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) : Prop :=
  FixedOneCanonical.Obligations setup machine case.input case.selectedNext

/-- The global three-family plan accepts exactly the independent canonical
obligations for every canonical outer input and selected result. -/
theorem accepts_iff_obligations
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows) :
    CheckPlan.Accepts
        (semantics setup machine functionIndex) Canonical.checks case <->
      target setup machine case := by
  exact Canonical.accepts_iff_obligations setup machine functionIndex
    case.input case.selectedNext

/-- Exactness of the retained plan across the complete canonical input
language. -/
theorem exact
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1) :
    CheckPlan.Exact
      (semantics setup machine functionIndex)
      (target setup machine) Canonical.checks := by
  intro case
  exact accepts_iff_obligations setup machine functionIndex case

/-- Embed a pointwise canonical removal witness into the common global
verifier language. -/
theorem lift_local_necessary
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (functionIndex : Fin 1)
    (input :
      FixedOneCanonical.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (family : Family)
    (necessary :
      CheckPlan.NecessaryForSoundness
        (Canonical.semantics setup machine functionIndex input)
        (FixedOneCanonical.Obligations setup machine input)
        Canonical.checks family) :
    CheckPlan.NecessaryForSoundness
      (semantics setup machine functionIndex)
      (target setup machine) Canonical.checks family := by
  rcases necessary with ⟨selectedNext, weakened, rejected⟩
  let case :
      Case OuterKey AppState Witness shape publicRingColumns publicFits
        verifierRows := {
    input := input
    selectedNext := selectedNext
  }
  refine ⟨case, ?_, ?_⟩
  · simpa [semantics, case] using weakened
  · simpa [target, case] using rejected

end

end Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.FixedOne.Global
