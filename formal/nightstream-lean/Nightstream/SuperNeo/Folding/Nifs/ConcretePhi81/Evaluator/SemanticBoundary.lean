import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Evaluator

/-!
Conditional semantic closure for the fixed-active ConcretePhi81 evaluator.

Protocol: SuperNeo NIFS.
Phase: physically accepted certificate → independent semantic result.
Constraint family: semantic/security boundary only; this file emits no rows.

Owns: the exact additional premises that close the existing
soundness-or-output-unbound-or-bad-event theorem for one certificate.

Does not own: derivation of semantic source authority from public verifier
inputs, proof that output claims are source-bound, a probability bound for
the named `Pi_CCS` bad event, unconditional verifier exactness, Rust, R1CS,
rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `SoundnessClosure` is a model-level/security-premise
package. None of its fields is inferred from a digest or from physical
acceptance. In particular, `outputBound` is semantic source authority and
`noPiCcsBadEvent` is an explicit security premise. This module must not be
used to claim that the current executable verifier decides
`ResultTransition` unconditionally.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.soundness.input` | public/source carriers have one independent semantic interpretation | semantic premise | `SoundnessClosure.semanticInput` |
| `nifs.fixed_active.soundness.output` | the raw `Pi_CCS` output is bound to those same sources | semantic premise | `SoundnessClosure.outputBound` |
| `nifs.fixed_active.soundness.bad_event` | the named FE/NC mixing failure did not occur | security premise | `SoundnessClosure.noPiCcsBadEvent` |
| `nifs.fixed_active.soundness.closed` | successful execution yields the independent result transition | conditional theorem | `run_sound_of_closure` |
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
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows)
    (data : Data shape)
    (certificate : FixedActive.Certificate context) : Prop where
  semanticInput : ConcretePhi81.SemanticInput context data
  outputBound : ConcretePhi81.OutputBound context data certificate
  noPiCcsBadEvent :
    ¬ ConcretePhi81.PiCcsBadEvent context data certificate

/-- Successful physical execution closes to the independent semantic result
when source interpretation, output authority, and bad-event exclusion are
supplied explicitly. -/
theorem run_sound_of_closure
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows}
    {data : Data shape}
    {checker : Checker context}
    {certificate : FixedActive.Certificate context}
    {result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows}
    (closure : SoundnessClosure context data certificate)
    (executed : run checker certificate = some result) :
    FixedActive.ResultTransition context result := by
  rcases run_sound noZeroDivisors closure.semanticInput executed with
    transition | outputUnbound | badEvent
  · exact transition
  · exact False.elim (outputUnbound closure.outputBound)
  · exact False.elim (closure.noPiCcsBadEvent badEvent)

/-- Unpacked form of `run_sound_of_closure`, useful at a caller that already
owns the three premises separately. -/
theorem run_sound_of_outputBound_noBadEvent
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows}
    {data : Data shape}
    {checker : Checker context}
    {certificate : FixedActive.Certificate context}
    {result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows}
    (semanticInput : ConcretePhi81.SemanticInput context data)
    (outputBound : ConcretePhi81.OutputBound context data certificate)
    (noPiCcsBadEvent :
      ¬ ConcretePhi81.PiCcsBadEvent context data certificate)
    (executed : run checker certificate = some result) :
    FixedActive.ResultTransition context result :=
  run_sound_of_closure noZeroDivisors
    {
      semanticInput := semanticInput
      outputBound := outputBound
      noPiCcsBadEvent := noPiCcsBadEvent
    }
    executed

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator
