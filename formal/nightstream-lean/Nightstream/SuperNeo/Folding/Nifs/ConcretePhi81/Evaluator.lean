import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive

/-!
Canonical fixed-active evaluator boundary for the concrete Phi81 NIFS.

Protocol: SuperNeo NIFS.
Phase: raw fixed-active certificate → computed parent-and-children result.
Constraint family: executable acceptance boundary only; this file emits no
rows.

Owns: the small exactness contract that a concrete Boolean checker must
discharge; fail-closed `Option` execution; exact acceptance/result
characterization; semantic soundness with explicit output-unbound and
Split-NC bad-event outcomes; and honest completeness.

Does not own: the implementation of the Boolean checker, Poseidon2, Rust,
R1CS, rows, costs, necessity, or row removal.

Emits constraints: no.

Authority boundary: `Checker` is a refinement interface, not a semantic
oracle. Its required `exact` theorem targets the independently defined
`ConcretePhi81.Accepted`. `run` computes the result with `resultOf`; no caller
may supply either the outgoing parent or the returned children separately.
No production checker is considered verified until it instantiates this
interface from raw messages and proves `exact`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.fixed_active.checker` | Boolean acceptance equals physical ConcretePhi81 acceptance | refinement contract | `Checker.exact` |
| `nifs.fixed_active.run` | reject on failed checking; otherwise compute one shared result | executable/computed | `run` |
| `nifs.fixed_active.run.exact` | successful execution iff physical acceptance and exact result equality | theorem | `run_eq_some_iff_accepted` |
| `nifs.fixed_active.run.sound` | semantic transition or explicit output/FE/NC failure | theorem | `run_sound` |
| `nifs.fixed_active.run.complete` | honest paper obligations plus bounded sampler success construct an accepted computed result | compatibility theorem | `run_complete` |
| `nifs.fixed_active.run.complete.outcome` | honest paper obligations construct a result or expose one exact sampler shortfall | exhaustive model theorem | `run_complete_or_samplerShortfall` |
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

/-- Refinement contract for a concrete raw-message checker.

This interface is intentionally small: a backend supplies only a Boolean
decision procedure and proves that it is exactly the physical verifier
predicate already defined below the F-prime layer. -/
structure Checker
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows) where
  check : FixedActive.Certificate context -> Bool
  exact : ∀ certificate,
    check certificate = true ↔ ConcretePhi81.Accepted context certificate

/-- Fail-closed evaluator. Successful execution returns only the canonical
result computed from the checked raw certificate. -/
def run
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows}
    (checker : Checker context)
    (certificate : FixedActive.Certificate context) :
    Option (FixedActive.FoldResult shape publicRingColumns publicFits
      verifierRows) :=
  if checker.check certificate then
    some (FixedActive.resultOf context certificate)
  else
    none

/-- Exact characterization of one evaluator result. The result cannot differ
from the value computed by the accepted certificate. -/
theorem run_eq_some_iff_accepted
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows}
    (checker : Checker context)
    (certificate : FixedActive.Certificate context)
    (result :
      FixedActive.FoldResult shape publicRingColumns publicFits verifierRows) :
    run checker certificate = some result ↔
      ConcretePhi81.Accepted context certificate ∧
        FixedActive.resultOf context certificate = result := by
  cases checked : checker.check certificate with
  | false =>
      have notAccepted : ¬ ConcretePhi81.Accepted context certificate := by
        intro accepted
        have trueCheck := (checker.exact certificate).2 accepted
        simp [checked] at trueCheck
      simp [run, checked, notAccepted]
  | true =>
      have accepted : ConcretePhi81.Accepted context certificate :=
        (checker.exact certificate).1 checked
      simp [run, checked, accepted]

/-- A physically accepted and semantically interpreted certificate yields the
independent result transition, or exposes exactly the existing semantic
failure outcomes. -/
theorem run_sound
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
    (input : ConcretePhi81.SemanticInput context data)
    (executed : run checker certificate = some result) :
    FixedActive.ResultTransition context result ∨
      ¬ ConcretePhi81.OutputBound context data certificate ∨
      ConcretePhi81.PiCcsBadEvent context data certificate := by
  rcases (run_eq_some_iff_accepted checker certificate result).1 executed with
    ⟨accepted, resultEq⟩
  rcases ConcretePhi81.accepted_implies_holds_or_outputUnbound_or_badEvent
      noZeroDivisors input accepted with
    holds | outputUnbound | bad
  · exact Or.inl ⟨data, certificate, resultEq.symm, holds⟩
  · exact Or.inr (Or.inl outputUnbound)
  · exact Or.inr (Or.inr bad)

/-- Package already established physical and semantic acceptance as one
successful canonical evaluation. -/
theorem run_of_accepted_holds
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    {context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows}
    {data : Data shape}
    (checker : Checker context)
    (certificate : FixedActive.Certificate context)
    (accepted : ConcretePhi81.Accepted context certificate)
    (holds : ConcretePhi81.Holds context data certificate) :
    run checker certificate =
        some (FixedActive.resultOf context certificate) ∧
      FixedActive.ResultTransition context
        (FixedActive.resultOf context certificate) := by
  constructor
  · have checked : checker.check certificate = true :=
      (checker.exact certificate).2 accepted
    simp [run, checked]
  · exact ⟨data, certificate, rfl, holds⟩

/-- Honest completeness for the fixed active evaluator. This is the existing
ConcretePhi81 honest construction specialized to exactly fifteen sources and
then passed through a checker whose exactness has been proved. -/
theorem run_complete
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows)
    (checker : Checker context)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : ConcretePhi81.SemanticInput context data)
    (running : ConcretePhi81.RunningAuthority.Accepted context)
    (challenges : Fin FixedActive.arity.total -> RingF)
    (samplerAvailable :
      ∀ piCcsCertificate :
          Protocol.Certificate context.piCcsInput domain,
        Protocol.Accepted context.feMachine context.ncMachine
            context.initialState context.profile
            context.piCcsInput context.feCoins context.ncCoins
            piCcsCertificate →
          ConcretePhi81.Sampler.Bound context.piRlcMachine
            (context.piCcsOutputHandoff
              (Protocol.derive context.feMachine context.ncMachine
                context.initialState piCcsCertificate).finalState
              piCcsCertificate.output)
            challenges) :
    ∃ certificate : FixedActive.Certificate context,
      ∃ result :
          FixedActive.FoldResult shape publicRingColumns publicFits
            verifierRows,
        run checker certificate = some result ∧
          FixedActive.ResultTransition context result := by
  rcases ConcretePhi81.complete_of_paperObligations
      context data paper input running challenges samplerAvailable with
    ⟨certificate, accepted, holds, _childrenValid⟩
  have evaluated :=
    run_of_accepted_holds checker certificate accepted holds
  exact ⟨certificate, FixedActive.resultOf context certificate,
    evaluated.1, evaluated.2⟩

/-- Honest completeness without assuming bounded-sampler success. The exact
checker either executes the canonical certificate and result, or the theorem
returns one coordinate whose fixed rejection-sampling prefix shortfalls. -/
theorem run_complete_or_samplerShortfall
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type uState}
    {publicRingColumns verifierRows : Nat}
    {publicFits :
      ringDegree * publicRingColumns <= shape.carrierWidth}
    (context :
      FixedActive.Context shape domain State publicRingColumns publicFits
        verifierRows)
    (checker : Checker context)
    (data : Data shape)
    (paper : Semantics.Paper.Holds data)
    (input : ConcretePhi81.SemanticInput context data)
    (running : ConcretePhi81.RunningAuthority.Accepted context) :
    (∃ certificate : FixedActive.Certificate context,
      ∃ result :
          FixedActive.FoldResult shape publicRingColumns publicFits
            verifierRows,
        run checker certificate = some result ∧
          FixedActive.ResultTransition context result) ∨
      ConcretePhi81.HonestSamplerShortfall context data := by
  rcases ConcretePhi81.complete_or_samplerShortfall
      context data paper input running with completed | shortfall
  · rcases completed with
      ⟨_challenges, certificate, accepted, holds, _childrenValid⟩
    have evaluated :=
      run_of_accepted_holds checker certificate accepted holds
    exact Or.inl ⟨certificate, FixedActive.resultOf context certificate,
      evaluated.1, evaluated.2⟩
  · exact Or.inr shortfall

end Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Evaluator
