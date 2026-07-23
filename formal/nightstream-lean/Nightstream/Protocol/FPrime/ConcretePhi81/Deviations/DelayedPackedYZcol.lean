import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.ChallengeAuthority
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Checker
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Trace

/-!
Public protocol facade for the one-fold delayed packed-`yZcol` deviation.

Assurance tier: model-proved production deviation.

Owns: the executable step checker, transcript timing contract, explicit
base/recursive/terminal trace carriers, and the exact trace-level success or
named-failure theorem.

Does not own: Rust/R1CS refinement, generated rows, primitive security, costs,
or row-removal authority. These exports do not modify the paper-authoritative
SuperNeo or HyperNova relations.

Emits constraints: no.

Authority boundary: this facade exports only the protocol-owned delayed
route. Authority remains with opening-derived carriers, raw assignments,
recomputed state bindings, and the exact named-failure partitions below.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.facade.transcript` | expose statement-bound delayed challenge timing | checked | `ChallengeAuthority` |
| `fprime.delayed.facade.checker` | expose executable active/base step checks and their exact propositions | checked | `Checker` |
| `fprime.delayed.facade.edge` | expose adjacent-step continuity and predecessor packed-output closure | derived/security partition | `Continuity`, `Edge` |
| `fprime.delayed.facade.terminal` | expose final raw-opening closure | derived/security partition | `Terminal` |
| `fprime.delayed.facade.trace` | expose finite base/recursive/terminal closure and its exact trace failure theorem | derived/security partition | `Trace` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol

export ChallengeAuthority
  (Holds holds)

export Checker.Binding
  (stateBindingCheck stateBindingCheck_eq_true_iff)

export Checker
  (piCcsMessageCheck piCcsMessageCheck_eq_true_iff_accepted
    canonicalPublicInputCheck canonicalPublicInputCheck_eq_true_iff
    parentEvaluationSizeCheck parentEvaluationSizeCheck_eq_true_iff
    childEvaluationSizeCheck childEvaluationSizeCheck_eq_true_iff
    paperOutputCheck paperOutputCheck_eq_true_iff
    canonicalParentOpeningCheck canonicalParentOpeningCheck_eq_true_iff
    check check_eq_true_iff_accepted
    BaseAccepted baseCheck baseCheck_eq_true_iff_accepted)

export PaperStep
  (PaperStepAccepted PaperRefinement
    accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent)

export Continuity
  (of_piDec_and_stateBindings)

export Terminal
  (ProjectionOpeningAccepted
    projectionOpeningAccepted_implies_packedYZcolBound_or_badEvent)

export Edge
  (acceptedPair_of_nextPacked_implies_previousClosed_or_failure)

export Trace
  (SharedContext CheckedStep BaseStep OutputClosed TerminalClosure Tail
    TerminalFailure EdgeFailure AllOutputsClosed Failure ClosedTrace
    singleton_implies_closed_or_terminalFailure
    baseSingleton_implies_noPendingAndClosed_or_terminalFailure
    tail_implies_allClosed_or_failure
    closedTrace_implies_baseAndAllClosed_or_failure)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol
