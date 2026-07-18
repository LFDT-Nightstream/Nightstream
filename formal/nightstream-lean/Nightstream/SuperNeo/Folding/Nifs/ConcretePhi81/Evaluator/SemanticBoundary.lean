import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator

/-!
Conditional semantic closure for the fixed-active ConcretePhi81 evaluator.

Protocol: SuperNeo NIFS.
Phase: physically accepted certificate → independent semantic result.
Constraint family: semantic/security boundary only; this file emits no rows.

Owns: the exact additional premises that close the existing
soundness-or-output-unbound-or-bad-event theorem for one certificate, and an
exhaustive classification of every unresolved semantic/security failure.

Does not own: derivation of semantic source authority from public verifier
inputs, proof that output claims are source-bound, extraction/binding of
private child openings, a probability bound for the named `Pi_CCS` bad event,
probability or binding bounds for the named failures, unconditional verifier
exactness, Rust, R1CS, rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `SoundnessClosure` is a model-level/security-premise
package. None of its fields is inferred from a digest or from physical
acceptance. In particular, `outputBound` is semantic source authority and
`childOpenings` is extraction/binding authority; `noPiCcsBadEvent` is an
explicit security premise. This module must not be used to claim that the
current executable verifier decides `ResultTransition` unconditionally.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.soundness.input` | public/source carriers have one independent semantic interpretation | semantic premise | `SoundnessClosure.semanticInput` |
| `nifs.fixed_active.soundness.output` | the raw `Pi_CCS` output is bound to those same sources | semantic premise | `SoundnessClosure.outputBound` |
| `nifs.fixed_active.soundness.children` | every public Π_DEC child has its canonical split private opening | extraction/binding premise | `SoundnessClosure.childOpenings` |
| `nifs.fixed_active.soundness.bad_event` | the named FE/NC mixing failure did not occur | security premise | `SoundnessClosure.noPiCcsBadEvent` |
| `nifs.fixed_active.soundness.closed` | successful execution yields the independent result transition | conditional theorem | `run_sound_of_closure` |
| `nifs.fixed_active.soundness.partition` | every successful execution refines semantics or names the missing binding/security event | exhaustive theorem | `run_sound_or_securityFailure` |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

universe uState

/-- Per-certificate premises required to turn the existing soundness
disjunction into an independent semantic transition.

This is intentionally not a field of `Checker`: a Boolean verifier cannot
manufacture source openings, output authority, or exclusion of a
Fiat--Shamir bad event. -/
structure SoundnessClosure
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  semanticInput : ConcretePhi81.SemanticInput context data
  outputBound : ConcretePhi81.OutputBound context data certificate
  childOpenings : ConcretePhi81.ChildOpenings context data certificate
  noPiCcsBadEvent :
    ¬ ConcretePhi81.PiCcsBadEvent context data certificate

/-- Exhaustive unresolved outcomes for one physically successful execution.

These constructors are named proof obligations, not claims that the events
are possible or likely. A security reduction must eliminate or bound each
constructor without assuming the semantic conclusion. -/
inductive SecurityFailure
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows)
    (certificate : FixedActive.Certificate context) : Prop where
  | sourceUnbound
      (unbound : forall data : Data shape,
        ¬ ConcretePhi81.SemanticInput context data)
  | childOpeningsUnbound
      (data : Data shape)
      (input : ConcretePhi81.SemanticInput context data)
      (unbound : ¬ ConcretePhi81.ChildOpenings context data certificate)
  | outputUnbound
      (data : Data shape)
      (input : ConcretePhi81.SemanticInput context data)
      (children : ConcretePhi81.ChildOpenings context data certificate)
      (unbound : ¬ ConcretePhi81.OutputBound context data certificate)
  | piCcsBadEvent
      (data : Data shape)
      (input : ConcretePhi81.SemanticInput context data)
      (children : ConcretePhi81.ChildOpenings context data certificate)
      (bad : ConcretePhi81.PiCcsBadEvent context data certificate)

/-- Successful physical execution closes to the independent semantic result
when source interpretation, output authority, and bad-event exclusion are
supplied explicitly. -/
theorem run_sound_of_closure
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows}
    {data : Data shape}
    {checker : Checker context}
    {certificate : FixedActive.Certificate context}
    {result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows}
    (closure : SoundnessClosure context data certificate)
    (executed : run checker certificate = some result) :
    FixedActive.ResultTransition context result := by
  rcases run_sound noZeroDivisors closure.semanticInput closure.childOpenings
      executed with
    transition | outputUnbound | badEvent
  · exact transition
  · exact False.elim (outputUnbound closure.outputBound)
  · exact False.elim (closure.noPiCcsBadEvent badEvent)

/-- Every successful physical run either reaches the independent semantic
transition or exposes exactly one still-unproved binding/security family. -/
theorem run_sound_or_securityFailure
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows}
    {checker : Checker context}
    {certificate : FixedActive.Certificate context}
    {result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows}
    (executed : run checker certificate = some result) :
    FixedActive.ResultTransition context result ∨
      SecurityFailure context certificate := by
  classical
  by_cases source :
      ∃ data : Data shape, ConcretePhi81.SemanticInput context data
  · rcases source with ⟨data, input⟩
    by_cases children : ConcretePhi81.ChildOpenings context data certificate
    · rcases run_sound noZeroDivisors input children executed with
        transition | outputUnbound | badEvent
      · exact Or.inl transition
      · exact Or.inr (.outputUnbound data input children outputUnbound)
      · exact Or.inr (.piCcsBadEvent data input children badEvent)
    · exact Or.inr (.childOpeningsUnbound data input children)
  · apply Or.inr
    apply SecurityFailure.sourceUnbound
    intro data input
    exact source ⟨data, input⟩

/-- Unpacked form of `run_sound_of_closure`, useful at a caller that already
owns the three premises separately. -/
theorem run_sound_of_outputBound_noBadEvent
    {shape : SemanticShape}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      FixedActive.Context shape State publicRingColumns publicFits
        verifierRows}
    {data : Data shape}
    {checker : Checker context}
    {certificate : FixedActive.Certificate context}
    {result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows}
    (semanticInput : ConcretePhi81.SemanticInput context data)
    (outputBound : ConcretePhi81.OutputBound context data certificate)
    (childOpenings : ConcretePhi81.ChildOpenings context data certificate)
    (noPiCcsBadEvent :
      ¬ ConcretePhi81.PiCcsBadEvent context data certificate)
    (executed : run checker certificate = some result) :
    FixedActive.ResultTransition context result :=
  run_sound_of_closure noZeroDivisors
    {
      semanticInput := semanticInput
      outputBound := outputBound
      childOpenings := childOpenings
      noPiCcsBadEvent := noPiCcsBadEvent
    }
    executed

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator
