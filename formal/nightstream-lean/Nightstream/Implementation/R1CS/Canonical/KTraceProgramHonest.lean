import Nightstream.Implementation.R1CS.Canonical.KTraceProgram

/-!
Contract: honest witness construction for the Lean-owned PiRLC coefficient
trace program.

Owns the converse direction omitted from `KTraceProgram`: a coefficient-exact
trace whose authoritative reads are below its auxiliary base has a concrete
assignment satisfying every emitted quotient-identity row.

This module does not assume an accepted row program, a quotient equation, or a
generic completeness callback.  The row equation is derived from frozen
coefficient exactness by `KTraceDecoder.equation_of_exact`.
-/

set_option autoImplicit false
set_option maxRecDepth 4096

namespace Nightstream.Implementation.R1CS.Canonical.KTraceProgramHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.Canonical.KProjectionTrace
open Nightstream.Implementation.R1CS.Canonical.KTraceDecoder
open Nightstream.Implementation.R1CS.Canonical.KTraceProgram

/-- The exact auxiliary assignment selected for one trace. -/
def traceWitness
    (source : Nat → Nat) (base : Nat)
    (trace : KProjectionTrace.Trace) : Nat → Nat :=
  KQuotientIdentity.identityWitness source
    (decodePoint trace.beta) base (decodedPairs trace)
    (decodeVector trace.output) (decodeVector trace.quotient) decodeModulus

private theorem beta_low
    (trace : KProjectionTrace.Trace) (base : Nat)
    (low : trace.beta.c0 < base) :
    BelowBase (decodePoint trace.beta).low base := by
  intro column member
  simp only [decodePoint, LinCombNormal.Mentions,
    List.map_cons, List.map_nil, List.mem_singleton] at member
  simpa [member] using low

private theorem beta_high
    (trace : KProjectionTrace.Trace) (base : Nat)
    (high : trace.beta.c1 < base) :
    BelowBase (decodePoint trace.beta).high base := by
  intro column member
  simp only [decodePoint, LinCombNormal.Mentions,
    List.map_cons, List.map_nil, List.mem_singleton] at member
  simpa [member] using high

private theorem decoded_pairs_below
    (trace : KProjectionTrace.Trace) (base : Nat)
    (below : trace.CoefficientsBelow base) :
    ∀ pair ∈ decodedPairs trace,
      (∀ coefficient ∈ pair.1,
          BelowBase coefficient.low base ∧ BelowBase coefficient.high base) ∧
      (∀ coefficient ∈ pair.2,
          BelowBase coefficient.low base ∧ BelowBase coefficient.high base) := by
  intro pair member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  exact
    ⟨decodeVector_belowBase source.rho base (below.1 source sourceMember).1,
      decodeVector_belowBase source.input base
        (below.1 source sourceMember).2⟩

private theorem decoded_pairs_sized
    (trace : KProjectionTrace.Trace) (valid : trace.Valid) :
    ∀ pair ∈ decodedPairs trace,
      pair.1.length = 54 ∧ pair.2.length = 54 := by
  intro pair member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  simpa only [decodeVector_length] using valid.2.1 source sourceMember

/-- Frozen coefficient exactness constructs a satisfying assignment for every
row of one selected trace. -/
theorem traceRows_honest
    (source : Nat → Nat) (sharedBeta : ProjectionProgram.KColumns)
    (base : Nat) (trace : KProjectionTrace.Trace)
    (basePositive : 0 < base)
    (constantWire : source 0 = 1)
    (betaShared : trace.beta = sharedBeta)
    (valid : trace.Valid)
    (betaBelow : trace.beta.c0 < base ∧ trace.beta.c1 < base)
    (coefficientsBelow : trace.CoefficientsBelow base)
    (exact : (trace.identity source).Exact) :
    Satisfies
      (KQuotientIdentity.identityRows
        (decodePoint sharedBeta) base
        (decodedPairs trace)
        (decodeVector trace.output)
        (decodeVector trace.quotient)
        decodeModulus)
      (traceWitness source base trace) := by
  subst sharedBeta
  apply KQuotientIdentity.identityRows_honest
    source (decodePoint trace.beta) base (decodedPairs trace)
    (decodeVector trace.output) (decodeVector trace.quotient) decodeModulus
    basePositive constantWire
  · exact decoded_pairs_sized trace valid
  · exact decoded_output_sized trace valid
  · exact decoded_quotient_sized trace valid
  · exact decodeModulus_length
  · exact beta_low trace base betaBelow.1
  · exact beta_high trace base betaBelow.2
  · exact decoded_pairs_below trace base coefficientsBelow
  · exact decodeVector_belowBase trace.output base coefficientsBelow.2.1
  · exact decodeVector_belowBase trace.quotient base coefficientsBelow.2.2
  · exact decodeModulus_belowBase base basePositive
  · exact equation_of_exact source (decodePoint trace.beta) trace
      (trace.identity source).beta constantWire
      (carriedValue_decodePoint source trace.beta) exact

theorem traceWitness_preserves_source
    (source : Nat → Nat) (base : Nat)
    (trace : KProjectionTrace.Trace)
    (column : Nat) (below : column < base) :
    traceWitness source base trace column = source column :=
  KQuotientIdentity.identityWitness_off_block source
    (decodePoint trace.beta) base (decodedPairs trace)
    (decodeVector trace.output) (decodeVector trace.quotient) decodeModulus
    column below

/-! ## Batch assembly -/

/-- Apply each trace witness in the same left-to-right auxiliary order as
`KTraceProgram.rowsFrom`. -/
def batchWitness : (Nat → Nat) → Nat → List KProjectionTrace.Trace → Nat → Nat
  | source, _, [] => source
  | source, base, trace :: traces =>
      batchWitness (traceWitness source base trace)
        (base + traceAuxWidth trace) traces

theorem batchWitness_preserves_below
    (source : Nat → Nat) (base : Nat)
    (traces : List KProjectionTrace.Trace)
    (column : Nat) (below : column < base) :
    batchWitness source base traces column = source column := by
  induction traces generalizing source base with
  | nil =>
      rfl
  | cons trace traces inductionHypothesis =>
      rw [batchWitness,
        inductionHypothesis (source := traceWitness source base trace)
          (base := base + traceAuxWidth trace) (by omega),
        traceWitness_preserves_source source base trace column below]

private theorem shared_read_below
    (beta : KMul.Carried) (coefficients : List KMul.Carried)
    (base column : Nat)
    (betaLow : BelowBase beta.low base)
    (betaHigh : BelowBase beta.high base)
    (coefficientBelow :
      ∀ coefficient ∈ coefficients,
        BelowBase coefficient.low base ∧ BelowBase coefficient.high base)
    (read : KQuotientIdentity.SharedRead beta coefficients column) :
    column < base := by
  rcases read with (low | high) | ⟨coefficient, member, low | high⟩
  · exact betaLow column low
  · exact betaHigh column high
  · exact (coefficientBelow coefficient member).1 column low
  · exact (coefficientBelow coefficient member).2 column high

private theorem trace_row_below_next
    (sharedBeta : ProjectionProgram.KColumns) (base : Nat)
    (trace : KProjectionTrace.Trace)
    (betaShared : trace.beta = sharedBeta)
    (valid : trace.Valid)
    (betaBelow : trace.beta.c0 < base ∧ trace.beta.c1 < base)
    (coefficientsBelow : trace.CoefficientsBelow base)
    (row : Row)
    (member :
      row ∈ KQuotientIdentity.identityRows
        (decodePoint sharedBeta) base (decodedPairs trace)
        (decodeVector trace.output) (decodeVector trace.quotient)
        decodeModulus)
    (column : Nat)
    (mentioned :
      LinCombNormal.Mentions row.a column ∨
        LinCombNormal.Mentions row.b column ∨
        LinCombNormal.Mentions row.c column) :
    column < base + traceAuxWidth trace := by
  subst sharedBeta
  have betaLow := beta_low trace base betaBelow.1
  have betaHigh := beta_high trace base betaBelow.2
  have pairBelow := decoded_pairs_below trace base coefficientsBelow
  have outputBelow :=
    decodeVector_belowBase trace.output base coefficientsBelow.2.1
  have quotientBelow :=
    decodeVector_belowBase trace.quotient base coefficientsBelow.2.2
  have modulusBelow := decodeModulus_belowBase base (by omega)
  rcases KQuotientIdentity.identityRows_conservation
      (decodePoint trace.beta) base (decodedPairs trace)
      (decodeVector trace.output) (decodeVector trace.quotient)
      decodeModulus
      (decoded_pairs_sized trace valid)
      (decoded_output_sized trace valid)
      (decoded_quotient_sized trace valid)
      decodeModulus_length row member column mentioned with
    zero | allocated | pairRead | outputRead | quotientRead | modulusRead
  · subst column
    omega
  · unfold KQuotientIdentity.Allocated at allocated
    change column < base + (321 * trace.pairs.length + 480)
    simpa only [decodedPairs, List.length_map] using allocated.2
  · rcases pairRead with ⟨pair, pairMember, left | right⟩
    · have := shared_read_below (decodePoint trace.beta) pair.1 base column
        betaLow betaHigh (pairBelow pair pairMember).1 left
      omega
    · have := shared_read_below (decodePoint trace.beta) pair.2 base column
        betaLow betaHigh (pairBelow pair pairMember).2 right
      omega
  · have := shared_read_below (decodePoint trace.beta)
      (decodeVector trace.output) base column betaLow betaHigh outputBelow
      outputRead
    omega
  · have := shared_read_below (decodePoint trace.beta)
      (decodeVector trace.quotient) base column betaLow betaHigh quotientBelow
      quotientRead
    omega
  · have := shared_read_below (decodePoint trace.beta) decodeModulus base
      column betaLow betaHigh modulusBelow modulusRead
    omega

/-- A coefficient-exact batch has one concrete satisfying assignment for the
entire concatenated row program.  Later trace witnesses preserve every column
used by earlier traces. -/
theorem rowsFrom_honest
    (source : Nat → Nat) (sharedBeta : ProjectionProgram.KColumns)
    (base : Nat) (traces : List KProjectionTrace.Trace)
    (basePositive : 0 < base)
    (constantWire : source 0 = 1)
    (betaShared : ∀ trace ∈ traces, trace.beta = sharedBeta)
    (valid : ∀ trace ∈ traces, trace.Valid)
    (betaBelow :
      ∀ trace ∈ traces, trace.beta.c0 < base ∧ trace.beta.c1 < base)
    (coefficientsBelow :
      ∀ trace ∈ traces, trace.CoefficientsBelow base)
    (exact :
      ∀ trace ∈ traces, (trace.identity source).Exact) :
    Satisfies (rowsFrom sharedBeta base traces)
      (batchWitness source base traces) := by
  induction traces generalizing source base with
  | nil =>
      intro row member
      simp [rowsFrom] at member
  | cons trace traces inductionHypothesis =>
      let headWitness := traceWitness source base trace
      have headSatisfied :
          Satisfies
            (KQuotientIdentity.identityRows
              (decodePoint sharedBeta) base (decodedPairs trace)
              (decodeVector trace.output) (decodeVector trace.quotient)
              decodeModulus)
            headWitness :=
        traceRows_honest source sharedBeta base trace basePositive constantWire
          (betaShared trace List.mem_cons_self)
          (valid trace List.mem_cons_self)
          (betaBelow trace List.mem_cons_self)
          (coefficientsBelow trace List.mem_cons_self)
          (exact trace List.mem_cons_self)
      have headPreserved :
          Satisfies
            (KQuotientIdentity.identityRows
              (decodePoint sharedBeta) base (decodedPairs trace)
              (decodeVector trace.output) (decodeVector trace.quotient)
              decodeModulus)
            (batchWitness headWitness
              (base + traceAuxWidth trace) traces) := by
        apply KHornerSupport.satisfies_extend _ headWitness _ _ headSatisfied
        intro row member column mentioned
        exact
          (batchWitness_preserves_below headWitness
            (base + traceAuxWidth trace) traces column
            (trace_row_below_next sharedBeta base trace
              (betaShared trace List.mem_cons_self)
              (valid trace List.mem_cons_self)
              (betaBelow trace List.mem_cons_self)
              (coefficientsBelow trace List.mem_cons_self)
              row member column mentioned)).symm
      have headWitnessConstant : headWitness 0 = 1 := by
        rw [show headWitness = traceWitness source base trace from rfl,
          traceWitness_preserves_source source base trace 0 basePositive,
          constantWire]
      have tailSatisfied :
          Satisfies
            (rowsFrom sharedBeta (base + traceAuxWidth trace) traces)
            (batchWitness headWitness
              (base + traceAuxWidth trace) traces) := by
        apply inductionHypothesis headWitness
          (base + traceAuxWidth trace) (by omega) headWitnessConstant
        · intro item member
          exact betaShared item (List.mem_cons_of_mem trace member)
        · intro item member
          exact valid item (List.mem_cons_of_mem trace member)
        · intro item member
          have bound :=
            betaBelow item (List.mem_cons_of_mem trace member)
          exact ⟨by omega, by omega⟩
        · intro item member
          exact
            ⟨fun pair pairMember =>
                ⟨fun column columnMember => by
                    have := ((coefficientsBelow item
                      (List.mem_cons_of_mem trace member)).1 pair
                        pairMember).1 column columnMember
                    omega,
                  fun column columnMember => by
                    have := ((coefficientsBelow item
                      (List.mem_cons_of_mem trace member)).1 pair
                        pairMember).2 column columnMember
                    omega⟩,
              fun column columnMember => by
                have := (coefficientsBelow item
                  (List.mem_cons_of_mem trace member)).2.1 column columnMember
                omega,
              fun column columnMember => by
                have := (coefficientsBelow item
                  (List.mem_cons_of_mem trace member)).2.2 column columnMember
                omega⟩
        · intro item member
          apply
            (item.exact_congr_below source headWitness base
              (coefficientsBelow item
                (List.mem_cons_of_mem trace member))
              (fun column below =>
                (traceWitness_preserves_source source base trace column
                  below).symm)).mp
          exact exact item (List.mem_cons_of_mem trace member)
      intro row member
      rw [rowsFrom, List.mem_append] at member
      rcases member with head | tail
      · exact headPreserved row head
      · exact tailSatisfied row tail

end Nightstream.Implementation.R1CS.Canonical.KTraceProgramHonest
