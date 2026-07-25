import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Trace

/-!
Explicit ownership map for FPR-DEV-DELAYED-PACKED-YZCOL.

Assurance tier: model-level registered-deviation refinement.

Owns: base, recursive-edge, and terminal ownership of the delayed packed
value, uniqueness of terminal discharge, and reduction of a closed typed trace
to paper transitions or existing named failures.

Does not own: construction of the underlying trace receipts, commitment
binding, Fiat--Shamir, concrete hashing, Rust/R1CS, costs, or rows.

Emits constraints: none.

Iteration `i` produces `DelayedProduction.outgoingPending` after its accepted
PiCCS/PiRLC output.  The recursive edge from `i` to `i+1` binds that exact
value into the predecessor state receipt and binds the successor's complete
`pending` field into the same receipt.  Thus `i+1` is the sole recursive
carrier/consumer, modulo the named accumulator-binding event already exposed
by `Trace.EdgeFailure`.  The last iteration is not followed by another edge:
its ordered fourteen raw child openings discharge its outgoing pending value.

A `Trace.Tail` is nonempty and structurally contains exactly one `terminal`
constructor.  Therefore a recursive trace cannot omit terminal discharge and
cannot discharge the terminal value twice.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.base` | own no predecessor pending value at the base | derived | `Trace.base_owns_no_predecessor` |
| `fprime.delayed.edge` | bind predecessor production and successor carriage/consumption in one receipt | checked | `Trace.edge_owns_production_and_consumption` |
| `fprime.delayed.terminal` | discharge the final pending value from ordered raw child openings | checked | `Trace.terminal_owns_discharge` |
| `fprime.delayed.unique` | contain exactly one terminal discharge | derived | `Trace.terminalCount_eq_one` |
| `fprime.delayed.soundness` | reduce the closed lifecycle to paper transitions or named failures | derived | `Trace.closedTrace_reduces_to_paper_transitions_or_named_failure` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Lifecycle

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

universe uState uEncoding uDigest

variable {shape : SemanticShape}
variable {State : Type uState}
variable {Encoding : Type uEncoding}
variable {Digest : Type uDigest}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

namespace Trace

/-- The base iteration owns no predecessor-carried delayed value. -/
theorem base_owns_no_predecessor
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    (base : DelayedPackedYZcol.Trace.BaseStep (State := State) key) :
    base.toCheckedStep.full.pending = none :=
  base.pending_eq_none

/-- A recursive edge records both lifecycle halves in one typed receipt:
production by the predecessor and carriage/consumption by the successor. -/
theorem edge_owns_production_and_consumption
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {previous next : DelayedPackedYZcol.Trace.CheckedStep (State := State) key}
    (edge : DelayedPackedYZcol.Trace.Edge scheme previous next) :
    StateBinds scheme edge.stateDigest
        (derive previous.full previous.certificate).piRlcOutput
        (outputChildren previous.full previous.certificate)
        (some (DelayedProduction.outgoingPending previous.full
          previous.certificate)) ∧
      StateBinds scheme edge.stateDigest
        (next.carrier.opening.parent next.context.key next.carrier.system)
        next.full.input.running next.full.pending :=
  ⟨edge.previousBinds, edge.nextBinds⟩

/-- The last iteration's ordered raw-child opening is the terminal discharge
of that iteration's exact outgoing delayed value. -/
theorem terminal_owns_discharge
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {last : DelayedPackedYZcol.Trace.CheckedStep (State := State) key}
    (closure : DelayedPackedYZcol.Trace.TerminalClosure last) :
    DelayedPackedYZcol.Terminal.ProjectionOpeningAccepted last.full
      last.certificate closure.rawChildren :=
  closure.accepted

/-- Count terminal-discharge constructors in a well-typed nonempty tail. -/
def terminalCount
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {head : DelayedPackedYZcol.Trace.CheckedStep (State := State) key} :
    DelayedPackedYZcol.Trace.Tail scheme head -> Nat
  | .terminal _ _ => 1
  | .prepend _ _ _ tail => terminalCount tail

/-- Every complete tail contains exactly one terminal discharge.  This rules
out both a delayed recursive trace with no terminal discharge and a terminal
value consumed twice. -/
@[simp] theorem terminalCount_eq_one
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {key : VerifierKey shape publicRingColumns publicFits verifierRows}
    {head : DelayedPackedYZcol.Trace.CheckedStep (State := State) key}
    (tail : DelayedPackedYZcol.Trace.Tail scheme head) :
    terminalCount tail = 1 := by
  induction tail with
  | terminal => rfl
  | prepend previous next edge tail ih =>
      exact ih

/-- Full lifecycle reduction: base ownership plus closure of every production
output, or one precisely located recursive/terminal event. -/
theorem closedTrace_reduces_to_paper_transitions_or_named_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (key : VerifierKey shape publicRingColumns publicFits verifierRows)
    (trace : DelayedPackedYZcol.Trace.ClosedTrace (State := State) scheme key) :
    (trace.base.toCheckedStep.full.pending = none ∧
      DelayedPackedYZcol.Trace.AllOutputsClosed trace.tail) ∨
      DelayedPackedYZcol.Trace.Failure trace.tail :=
  DelayedPackedYZcol.Trace.closedTrace_implies_baseAndAllClosed_or_failure
    noZeroDivisors scheme key trace

end Trace

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Lifecycle
