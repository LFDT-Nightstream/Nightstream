import Nightstream.Implementation.R1CS.Canonical.KTraceDecoder

/-!
Contract: the selected Lean-owned projection program for a list of minimal
canonical coefficient traces.

Owns one emitted program: the quotient-identity rows from
`KQuotientIdentity`, instantiated directly from each trace's authoritative
coefficient columns.  Satisfaction of those rows constructs
`ProjectionCheck.Accepted`; the quotient equation is never a premise.

Does not own NIFS call framing, application codecs, Fiat--Shamir, or a
probability bound for `BatchBadRoot`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KTraceProgram

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KProjectionTrace
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.ProjectionCheck

/-- Decode the trace's extension-field challenge columns into the carrier used
by the canonical Horner program. -/
def decodePoint (columns : KColumns) : Carried :=
  ⟨[(columns.c0, 1)], [(columns.c1, 1)]⟩

theorem carriedValue_decodePoint (assignment : Nat → Nat)
    (columns : KColumns) :
    carriedValue assignment (decodePoint columns)
      = KBridge.toPair (columns.value assignment) := by
  rcases columns with ⟨c0, c1⟩
  simp [decodePoint, carriedValue, lcEval, KBridge.toPair, KColumns.value,
    baseAt, residue]

/-- The complete coefficient inputs of one frozen trace. -/
def decodedPairs (trace : KProjectionTrace.Trace) :
    List (List Carried × List Carried) :=
  trace.pairs.map fun pair =>
    (KTraceDecoder.decodeVector pair.rho,
      KTraceDecoder.decodeVector pair.input)

/-- The sole selected row program for one projection identity. -/
def traceRows (base : Nat) (trace : KProjectionTrace.Trace) : List Row :=
  KQuotientIdentity.identityRows
    (decodePoint trace.beta)
    base
    (decodedPairs trace)
    (KTraceDecoder.decodeVector trace.output)
    (KTraceDecoder.decodeVector trace.quotient)
    KTraceDecoder.decodeModulus

/-- The number of auxiliary columns allocated by one trace. -/
def traceAuxWidth (trace : KProjectionTrace.Trace) : Nat :=
  (KQuotientIdentity.identityCost trace.pairs.length).auxiliaryColumns

/-- A batch layout contains only static shape and sharing facts.  It carries no
equation, acceptance result, or prover conclusion. -/
structure BatchLayout where
  traces : List KProjectionTrace.Trace
  sharedBeta : KColumns
  betaShared : ∀ trace ∈ traces, trace.beta = sharedBeta
  valid : ∀ trace ∈ traces, trace.Valid

/-- The exact recurring-row subtotal of a selected projection batch.  This is
only the public PiRLC quotient program; it is not the cost of `nifsVerify`. -/
def BatchLayout.rowCount (layout : BatchLayout) : Nat :=
  (layout.traces.map fun trace =>
    (KQuotientIdentity.identityCost trace.pairs.length).recurringRows).sum

/-- Concatenate trace programs while assigning each trace its own contiguous
auxiliary block. -/
def rowsFrom (sharedBeta : KColumns) :
    Nat → List KProjectionTrace.Trace → List Row
  | _, [] => []
  | base, trace :: traces =>
      KQuotientIdentity.identityRows
          (decodePoint sharedBeta)
          base
          (decodedPairs trace)
          (KTraceDecoder.decodeVector trace.output)
          (KTraceDecoder.decodeVector trace.quotient)
          KTraceDecoder.decodeModulus
        ++
      rowsFrom sharedBeta (base + traceAuxWidth trace) traces

def rows (base : Nat) (layout : BatchLayout) : List Row :=
  rowsFrom layout.sharedBeta base layout.traces

private theorem decodedPairs_sized
    (trace : KProjectionTrace.Trace) (valid : trace.Valid) :
    ∀ pair ∈ decodedPairs trace,
      pair.1.length = 54 ∧ pair.2.length = 54 := by
  intro decoded member
  rcases List.mem_map.mp member with ⟨pair, pairMember, rfl⟩
  simpa only [KTraceDecoder.decodeVector_length] using
    valid.2.1 pair pairMember

/-- One valid trace emits exactly its derived `identityCost` row receipt. -/
theorem traceRows_length
    (base : Nat) (trace : KProjectionTrace.Trace) (valid : trace.Valid) :
    (traceRows base trace).length =
      (KQuotientIdentity.identityCost
        trace.pairs.length).recurringRows := by
  simpa only [traceRows, decodedPairs, List.length_map] using
    KQuotientIdentity.identityCost_rows
      (decodePoint trace.beta) base (decodedPairs trace)
      (KTraceDecoder.decodeVector trace.output)
      (KTraceDecoder.decodeVector trace.quotient)
      KTraceDecoder.decodeModulus
      (decodedPairs_sized trace valid)
      (KTraceDecoder.decoded_output_sized trace valid)
      (KTraceDecoder.decoded_quotient_sized trace valid)
      KTraceDecoder.decodeModulus_length

private theorem rowsFrom_length
    (sharedBeta : KColumns) :
    ∀ (traces : List KProjectionTrace.Trace) (base : Nat),
      (∀ trace ∈ traces, trace.Valid) →
      (rowsFrom sharedBeta base traces).length =
        (traces.map fun trace =>
          (KQuotientIdentity.identityCost
            trace.pairs.length).recurringRows).sum
  | [], _, _ => rfl
  | trace :: traces, base, valid => by
      rw [rowsFrom, List.length_append,
        KQuotientIdentity.identityCost_rows
          (decodePoint sharedBeta) base (decodedPairs trace)
          (KTraceDecoder.decodeVector trace.output)
          (KTraceDecoder.decodeVector trace.quotient)
          KTraceDecoder.decodeModulus
          (decodedPairs_sized trace (valid trace List.mem_cons_self))
          (KTraceDecoder.decoded_output_sized trace
            (valid trace List.mem_cons_self))
          (KTraceDecoder.decoded_quotient_sized trace
            (valid trace List.mem_cons_self))
          KTraceDecoder.decodeModulus_length]
      rw [rowsFrom_length sharedBeta traces
        (base + traceAuxWidth trace)
        (fun item member => valid item (List.mem_cons_of_mem trace member))]
      simp only [decodedPairs, List.length_map, List.map_cons, List.sum_cons]

/-- The emitted batch length is the fold of the per-trace row receipts. -/
theorem rows_length (base : Nat) (layout : BatchLayout) :
    (rows base layout).length = layout.rowCount := by
  exact rowsFrom_length layout.sharedBeta layout.traces base layout.valid

private theorem satisfies_head
    (assignment : Nat → Nat) (sharedBeta : KColumns) (base : Nat)
    (trace : KProjectionTrace.Trace)
    (traces : List KProjectionTrace.Trace)
    (satisfied : Satisfies (rowsFrom sharedBeta base (trace :: traces))
      assignment) :
    Satisfies
      (KQuotientIdentity.identityRows
        (decodePoint sharedBeta)
        base
        (decodedPairs trace)
        (KTraceDecoder.decodeVector trace.output)
        (KTraceDecoder.decodeVector trace.quotient)
        KTraceDecoder.decodeModulus)
      assignment := by
  intro row member
  exact satisfied row (List.mem_append_left _ member)

private theorem satisfies_tail
    (assignment : Nat → Nat) (sharedBeta : KColumns) (base : Nat)
    (trace : KProjectionTrace.Trace)
    (traces : List KProjectionTrace.Trace)
    (satisfied : Satisfies (rowsFrom sharedBeta base (trace :: traces))
      assignment) :
    Satisfies
      (rowsFrom sharedBeta (base + traceAuxWidth trace) traces)
      assignment := by
  intro row member
  exact satisfied row (List.mem_append_right _ member)

/-- Satisfaction of one selected trace program constructs the previously
external `TraceAccepts` package. -/
theorem traceAccepts_of_rows
    (assignment : Nat → Nat) (sharedBeta : KColumns) (base : Nat)
    (trace : KProjectionTrace.Trace)
    (constantWire : assignment 0 = 1)
    (betaShared : trace.beta = sharedBeta)
    (valid : trace.Valid)
    (satisfied :
      Satisfies
        (KQuotientIdentity.identityRows
          (decodePoint sharedBeta)
          base
          (decodedPairs trace)
          (KTraceDecoder.decodeVector trace.output)
          (KTraceDecoder.decodeVector trace.quotient)
          KTraceDecoder.decodeModulus)
        assignment) :
    KTraceDecoder.TraceAccepts assignment (decodePoint sharedBeta) trace := by
  refine
    { valid := valid
      betaDenotes := ?_
      equation := ?_ }
  · rw [← betaShared]
    exact carriedValue_decodePoint assignment trace.beta
  · exact KQuotientIdentity.identityRows_sound
      assignment (decodePoint sharedBeta) base
      (decodedPairs trace)
      (KTraceDecoder.decodeVector trace.output)
      (KTraceDecoder.decodeVector trace.quotient)
      KTraceDecoder.decodeModulus constantWire satisfied

private theorem batchAccepted_of_rowsFrom
    (assignment : Nat → Nat) (sharedBeta : KColumns)
    (constantWire : assignment 0 = 1) :
    ∀ (traces : List KProjectionTrace.Trace) (base : Nat),
      (∀ trace ∈ traces, trace.beta = sharedBeta) →
      (∀ trace ∈ traces, trace.Valid) →
      Satisfies (rowsFrom sharedBeta base traces) assignment →
      BatchAccepted K.ops (KProjectionTrace.BatchIdentity traces assignment)
  | [], _, _, _, _ => by
      intro identity member
      simp [KProjectionTrace.BatchIdentity] at member
  | trace :: traces, base, betaShared, valid, satisfied => by
      intro identity member
      simp only [KProjectionTrace.BatchIdentity, List.map_cons,
        List.mem_cons] at member
      rcases member with image | member
      · rw [image]
        exact KTraceDecoder.accepted_of_equation
          assignment (decodePoint sharedBeta) trace constantWire
          (valid trace List.mem_cons_self)
          (by
            rw [← betaShared trace List.mem_cons_self]
            exact carriedValue_decodePoint assignment trace.beta)
          (KQuotientIdentity.identityRows_sound
            assignment (decodePoint sharedBeta) base
            (decodedPairs trace)
            (KTraceDecoder.decodeVector trace.output)
            (KTraceDecoder.decodeVector trace.quotient)
            KTraceDecoder.decodeModulus constantWire
            (satisfies_head assignment sharedBeta base trace traces satisfied))
      · exact batchAccepted_of_rowsFrom assignment sharedBeta constantWire
          traces (base + traceAuxWidth trace)
          (fun t mem => betaShared t (List.mem_cons_of_mem trace mem))
          (fun t mem => valid t (List.mem_cons_of_mem trace mem))
          (satisfies_tail assignment sharedBeta base trace traces satisfied)
          identity member

/-- Satisfaction of the selected concatenated program constructs
`BatchAccepted` for exactly the identities bound to that program. -/
theorem batchAccepted_of_rows
    (assignment : Nat → Nat) (base : Nat) (layout : BatchLayout)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows base layout) assignment) :
    BatchAccepted K.ops
      (KProjectionTrace.BatchIdentity layout.traces assignment) := by
  exact batchAccepted_of_rowsFrom assignment layout.sharedBeta constantWire
    layout.traces base layout.betaShared layout.valid (by
      simpa only [rows] using satisfied)

/-- The operational batch theorem: selected rows imply exact coefficient
identities or the exact frozen bad-root event.  No quotient equation is a
premise. -/
theorem batchExact_or_badRoot_of_rows
    (assignment : Nat → Nat) (base : Nat) (layout : BatchLayout)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows base layout) assignment) :
    BatchExact (KProjectionTrace.BatchIdentity layout.traces assignment)
      ∨ BatchBadRoot K.ops
        (KProjectionTrace.BatchIdentity layout.traces assignment) :=
  batchAccepted_implies_exact_or_badRoot _ _
    (batchAccepted_of_rows assignment base layout constantWire satisfied)

/-! ## Closed occurrence boundary -/

/-- One exact emitted projection occurrence.  The layout fixes the identity
list and the base fixes the physical auxiliary allocation. -/
structure Occurrence where
  base : Nat
  layout : BatchLayout

def Occurrence.rows (occurrence : Occurrence) : List Row :=
  KTraceProgram.rows occurrence.base occurrence.layout

def Occurrence.identities
    (occurrence : Occurrence) (assignment : Nat → Nat) :
    List (Identity K) :=
  KProjectionTrace.BatchIdentity occurrence.layout.traces assignment

/-- The only named algebraic event for this occurrence.  The witness identity
must be a member of the exact identity list decoded from this occurrence's
trace layout and assignment; an unrelated colliding identity cannot inhabit
it. -/
def Occurrence.BadRoot
    (occurrence : Occurrence) (assignment : Nat → Nat) : Prop :=
  BatchBadRoot K.ops (occurrence.identities assignment)

def Occurrence.Exact
    (occurrence : Occurrence) (assignment : Nat → Nat) : Prop :=
  BatchExact (occurrence.identities assignment)

/-- Satisfaction of this occurrence's selected rows yields exact coefficient
identities or its own closed bad-root event. -/
theorem Occurrence.exact_or_badRoot
    (occurrence : Occurrence) (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies occurrence.rows assignment) :
    occurrence.Exact assignment ∨ occurrence.BadRoot assignment :=
  batchExact_or_badRoot_of_rows assignment occurrence.base occurrence.layout
    constantWire satisfied

/-- Event inversion exposes membership in this occurrence's identity list.
This is the fail-closed distinction from the unbound existential event rejected
by `NifsRecipeShape.unbound_event_is_inhabited`. -/
theorem Occurrence.badRoot_is_bound
    {occurrence : Occurrence} {assignment : Nat → Nat}
    (event : occurrence.BadRoot assignment) :
    ∃ identity ∈ occurrence.identities assignment,
      SuperNeo.ProjectionCheck.BadRoot K.ops identity :=
  event

/-- An exact occurrence cannot simultaneously take the named event branch.
In particular, the globally available colliding fixture does not create a free
event for an unrelated exact occurrence. -/
theorem Occurrence.exact_excludes_badRoot
    {occurrence : Occurrence} {assignment : Nat → Nat}
    (exact : occurrence.Exact assignment) :
    ¬ occurrence.BadRoot assignment := by
  intro event
  rcases event with ⟨identity, member, bad⟩
  exact bad.notExact (exact identity member)

end Nightstream.Implementation.R1CS.Canonical.KTraceProgram
