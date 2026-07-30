import Nightstream.Implementation.R1CS.Canonical.KQuotientIdentityHonest
import Nightstream.Implementation.R1CS.Canonical.KProjectionTrace

/-!
Contract: turning a canonical projection trace's columns into the recipe's
coefficient carriers.

Owns: the decoding of a base-field column list into `Carried` values, the
lengths that decoding produces, what a decoded carrier denotes, and where its
columns sit.

Also owns the evaluation bridge: a decoded projection **is**
`ProjectionProgram.Polynomial.eval` of the trace's base polynomial
(`projected_decodeVector`), and the decoded modulus is the frozen `Φ₈₁`
evaluation (`projected_decodeModulus`).

Also owns the composition that reaches the frozen check's evaluation
component: `equation_reaches_frozen_eval`.

Also owns `accepted_of_equation` and `exact_or_badRoot_of_equation`: the
emitted row program's equation gives the frozen `ProjectionCheck.Accepted`, and
therefore coefficient-exactness or the frozen `BadRoot` event.
`Identity.WellFormed` comes from
`KProjectionTrace.Trace.identity_wellFormed`, derived from the minimal
canonical width boundary rather than historical evaluation-trace metadata.

Does **not** own the emitted batch program.  The selected construction is
`KTraceProgram.rows`, which instantiates the quotient-identity rows for each
trace and derives the batch theorem from their satisfaction.

## Why this module is the gap

Every theorem in `KQuotientIdentity` quantifies over the `Carried` lists the
caller supplies. Three obligations stayed open for one reason — nothing
connected those lists to production data:

- the sizing hypotheses (`output.length = 54` and friends) had no constructor;
- `identityRows_sound`'s conclusion was an equation about projections rather
  than `ProjectionCheck.Accepted`;
- `KBatch`'s `agrees` premise stayed moved rather than closed.

`KProjectionTrace.Trace.Valid` pins `output.length = 54`,
`quotient.length = 53` and `maxDegree = 106`. Decoding turns those into
exactly the hypotheses the recipe asks for, and the evaluation bridge turns
`projected` into `Polynomial.eval`. What is left between the recipe's equation
and `ProjectionCheck.Accepted` is `toPair` injectivity and one assembly; the
`Polynomial.eval` algebra it needs already exists upstream.

## Base-field coefficients in a `K`-valued vector

The trace's coefficient columns hold *base field* elements; the recipe's
carriers are `K`-valued. Production embeds with `K.ofBase`, so a decoded
coefficient is the column in the low coordinate and nothing in the high one.
That is why `decodeBase` is asymmetric, and it is not an approximation: the
high coordinate is genuinely zero.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KTraceDecoder

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KProjectionTrace

/-! ## Decoding -/

/-- A base-field column as a `K`-valued carrier. -/
def decodeBase (column : Nat) : Carried := ⟨[(column, 1)], []⟩

/-- A coefficient vector of base-field columns. -/
def decodeVector (columns : List Nat) : List Carried := columns.map decodeBase

theorem decodeVector_length (columns : List Nat) :
    (decodeVector columns).length = columns.length :=
  List.length_map _

/-- **`Φ₈₁`'s coefficients as carriers.**  Constants, so each is either the
constant wire with coefficient one or the empty combination — no column of its
own, which is why the modulus vector allocates nothing. -/
def decodeModulus : List Carried :=
  (List.range 55).map (fun index =>
    if index = 0 ∨ index = 27 ∨ index = 54 then ⟨[(0, 1)], []⟩ else ⟨[], []⟩)

theorem decodeModulus_length : decodeModulus.length = 55 := by
  unfold decodeModulus
  rw [List.length_map, List.length_range]

/-! ## What a decoded carrier denotes

The recipe reasons about `carriedValue`; production reasons about
`K.ofBase (baseAt …)`. These are the same value, so no translation hypothesis
has to be carried through the recipe. -/

/-- **A decoded column denotes its value, embedded in `K`.** -/
theorem carriedValue_decodeBase (z : Nat → Nat) (column : Nat) :
    carriedValue z (decodeBase column)
      = KBridge.toPair
          (ProjectionProgram.K.ofBase (ProjectionProgram.baseAt z column)) := by
  unfold carriedValue decodeBase KBridge.toPair ProjectionProgram.K.ofBase
    ProjectionProgram.baseAt ProjectionProgram.residue
  simp [lcEval]

/-- The whole vector, decoded. -/
theorem carriedValue_decodeVector (z : Nat → Nat) (columns : List Nat) :
    (decodeVector columns).map (carriedValue z)
      = columns.map (fun column =>
          KBridge.toPair
            (ProjectionProgram.K.ofBase (ProjectionProgram.baseAt z column))) := by
  unfold decodeVector
  rw [List.map_map]
  exact List.map_congr_left (fun column _ => carriedValue_decodeBase z column)

/-! ## Placement

A decoded carrier mentions exactly its own column, so the recipe's `BelowBase`
hypotheses reduce to a bound on the trace's column indices — a checkable
property of the layout rather than an assumption about combinations. -/

theorem decodeBase_mentions (column other : Nat) :
    Mentions (decodeBase column).low other ↔ other = column := by
  unfold decodeBase Mentions
  simp

theorem decodeBase_high_empty (column other : Nat) :
    ¬ Mentions (decodeBase column).high other := by
  unfold decodeBase Mentions
  simp

/-- **Placement reduces to an index bound.** -/
theorem decodeVector_belowBase (columns : List Nat) (base : Nat)
    (placed : ∀ column ∈ columns, column < base) :
    ∀ c ∈ decodeVector columns,
      KHornerHonest.BelowBase c.low base
        ∧ KHornerHonest.BelowBase c.high base := by
  intro c member
  rcases List.mem_map.1 member with ⟨column, inColumns, image⟩
  refine ⟨fun other mentioned => ?_, fun other mentioned => ?_⟩
  · rw [← image] at mentioned
    rw [(decodeBase_mentions column other).1 mentioned]
    exact placed column inColumns
  · rw [← image] at mentioned
    exact absurd mentioned (decodeBase_high_empty column other)

/-- The modulus carriers mention only the constant wire, so they are below any
positive base. -/
theorem decodeModulus_belowBase (base : Nat) (positive : 0 < base) :
    ∀ c ∈ decodeModulus,
      KHornerHonest.BelowBase c.low base
        ∧ KHornerHonest.BelowBase c.high base := by
  intro c member
  unfold decodeModulus at member
  rcases List.mem_map.1 member with ⟨index, _, image⟩
  by_cases live : index = 0 ∨ index = 27 ∨ index = 54
  · rw [if_pos live] at image
    rw [← image]
    refine ⟨fun other mentioned => ?_, fun other mentioned => ?_⟩
    · have : other = 0 := by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using mentioned
      omega
    · simp only [Mentions, List.map_nil, List.not_mem_nil] at mentioned
  · rw [if_neg live] at image
    rw [← image]
    refine ⟨fun other mentioned => ?_, fun other mentioned => ?_⟩ <;>
      simp only [Mentions, List.map_nil, List.not_mem_nil] at mentioned

/-! ## The sizing hypotheses, from the frozen layout

`KProjectionTrace.Trace.Valid` pins the two widths; decoding preserves lengths;
so the recipe's sizing hypotheses are now consequences of the trace being
well-formed rather than assumptions about the caller. -/

theorem decoded_output_sized (trace : KProjectionTrace.Trace)
    (valid : trace.Valid) :
    (decodeVector trace.output).length = 54 := by
  rw [decodeVector_length]
  exact valid.2.2.1

theorem decoded_quotient_sized (trace : KProjectionTrace.Trace)
    (valid : trace.Valid) :
    (decodeVector trace.quotient).length = 53 := by
  rw [decodeVector_length]
  exact valid.2.2.2.1

/-! ## The evaluation bridge

`KBridge.toPair_eval` already says the canonical Horner reference computes what
`ProjectionCheck.eval` computes. Composing it with the decoding turns the
recipe's `projected` into the frozen `Polynomial.eval` of the trace's base
polynomial — which is what makes `identityRows_sound`'s conclusion a statement
about the frozen check rather than about the recipe's own vocabulary. -/

open ProjectionProgram in
/-- **A decoded projection is the frozen polynomial evaluation.** -/
theorem projected_decodeVector
    (z : Nat → Nat) (beta : Carried) (point : K) (columns : List Nat)
    (betaDenotes : carriedValue z beta = KBridge.toPair point) :
    KQuotientIdentity.projected z beta (decodeVector columns)
      = KBridge.toPair (Polynomial.eval (basePolynomial z columns) point) := by
  show hornerValue (carriedValue z beta)
      ((decodeVector columns).map (carriedValue z)) = _
  rw [betaDenotes, carriedValue_decodeVector,
    show Polynomial.eval (basePolynomial z columns) point
      = SuperNeo.ProjectionCheck.eval K.ops (basePolynomial z columns) point
      from rfl,
    KBridge.toPair_eval]
  unfold basePolynomial
  rw [List.map_map]
  rfl

/-! ## The modulus, decoded

`Φ₈₁`'s carriers are constants, so their denotation depends on the constant
wire holding one — the same hypothesis every equality row in the system needs.
Without it a prover could set column 0 freely and the modulus would evaluate to
something else entirely. -/

open ProjectionProgram in
/-- The decoded modulus denotes `Polynomial.phi81`. -/
theorem carriedValue_decodeModulus (z : Nat → Nat) (constantWire : z 0 = 1) :
    decodeModulus.map (carriedValue z) = Polynomial.phi81.map KBridge.toPair := by
  have live : carriedValue z (⟨[(0, 1)], []⟩ : Carried) = ⟨1, 0⟩ := by
    unfold carriedValue
    simp [lcEval, constantWire]
    exact Nat.mod_eq_of_lt (by decide)
  have dead : carriedValue z (⟨[], []⟩ : Carried) = ⟨0, 0⟩ := rfl
  have step : (List.range 55).map ((carriedValue z) ∘ (fun index =>
        if index = 0 ∨ index = 27 ∨ index = 54 then (⟨[(0, 1)], []⟩ : Carried)
        else ⟨[], []⟩))
      = (List.range 55).map (fun index =>
          if index = 0 ∨ index = 27 ∨ index = 54 then (⟨1, 0⟩ : Pair)
          else ⟨0, 0⟩) :=
    List.map_congr_left (fun index _ => by
      by_cases branch : index = 0 ∨ index = 27 ∨ index = 54
      · simp only [Function.comp_apply, if_pos branch]
        exact live
      · simp only [Function.comp_apply, if_neg branch]
        exact dead)
  unfold decodeModulus
  rw [List.map_map, step]
  rfl

open ProjectionProgram in
/-- **The decoded modulus projects to the frozen `Φ₈₁` evaluation.** -/
theorem projected_decodeModulus
    (z : Nat → Nat) (beta : Carried) (point : K) (constantWire : z 0 = 1)
    (betaDenotes : carriedValue z beta = KBridge.toPair point) :
    KQuotientIdentity.projected z beta decodeModulus
      = KBridge.toPair (Polynomial.eval Polynomial.phi81 point) := by
  show hornerValue (carriedValue z beta) (decodeModulus.map (carriedValue z)) = _
  rw [betaDenotes, carriedValue_decodeModulus z constantWire,
    show Polynomial.eval Polynomial.phi81 point
      = SuperNeo.ProjectionCheck.eval K.ops Polynomial.phi81 point from rfl,
    KBridge.toPair_eval]

/-! ## Reaching the frozen check

The recipe's equation lives in `Pair`; the frozen `Accepted` lives in `K`.
`KBridge.toPair_injective` is what makes the trip back possible, and the
`Polynomial.eval` algebra the frozen side needs — `eval_sum`, `eval_mul`,
`eval_add`, `eval_padRight` — was already proved upstream. -/

open ProjectionProgram in
/-- The recipe's sum and the frozen fold are the same sum. -/
theorem pairSum_toPair : ∀ values : List K,
    KQuotientIdentity.pairSum (values.map KBridge.toPair)
      = KBridge.toPair (values.foldr K.add K.zero)
  | [] => rfl
  | value :: rest => by
      show addPair (KBridge.toPair value)
        (KQuotientIdentity.pairSum (rest.map KBridge.toPair)) = _
      rw [pairSum_toPair rest, ← KBridge.toPair_add]
      rfl

open ProjectionProgram in
/-- **The frozen left-hand side, evaluated.** -/
theorem eval_identity_lhs
    (z : Nat → Nat) (trace : KProjectionTrace.Trace) (point : K) :
    Polynomial.eval (trace.identity z).lhs point
      = (trace.pairs.map (fun pair =>
          K.mul (Polynomial.eval (basePolynomial z pair.rho) point)
            (Polynomial.eval (basePolynomial z pair.input) point))).foldr
          K.add K.zero := by
  show Polynomial.eval (Polynomial.sum (trace.pairs.map
      (fun pair => pair.productPolynomial z))) point = _
  rw [Polynomial.eval_sum, List.foldr_map]
  induction trace.pairs with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.foldr_cons, List.map_cons, inductionHypothesis]
      rw [show head.productPolynomial z
          = Polynomial.mul (basePolynomial z head.rho)
              (basePolynomial z head.input) from rfl,
        Polynomial.eval_mul]

open ProjectionProgram in
/-- **The frozen right-hand side, evaluated.** -/
theorem eval_identity_rhs
    (z : Nat → Nat) (trace : KProjectionTrace.Trace) (point : K) :
    Polynomial.eval (trace.identity z).rhs point
      = K.add
          (K.mul (Polynomial.eval (basePolynomial z trace.quotient) point)
            (Polynomial.eval Polynomial.phi81 point))
          (Polynomial.eval (basePolynomial z trace.output) point) := by
  show Polynomial.eval (Polynomial.add
      (Polynomial.mul (basePolynomial z trace.quotient) Polynomial.phi81)
      (Polynomial.padRight (trace.maxDegree + 1)
        (basePolynomial z trace.output))) point = _
  rw [Polynomial.eval_add, Polynomial.eval_mul, Polynomial.eval_padRight]

open ProjectionProgram in
/-- **The recipe's equation is the frozen check's evaluation component.**

This is the step the whole decoder exists for: `identityRows_sound` produces an
equation about the recipe's projections, and this turns it into
`ProjectionCheck.eval lhs beta = ProjectionCheck.eval rhs beta` for the frozen
trace's own identity — the second half of `Accepted`.

The caller must have instantiated the recipe with the decoded carriers, which is
what the `equation` hypothesis says. Nothing here weakens that: the decoded
lists are exactly what `decodeVector` produces from the trace's own columns. -/
theorem equation_reaches_frozen_eval
    (z : Nat → Nat) (beta : Carried)
    (trace : KProjectionTrace.Trace) (point : K)
    (constantWire : z 0 = 1)
    (betaDenotes : carriedValue z beta = KBridge.toPair point)
    (equation : KQuotientIdentity.pairSum
        ((trace.pairs.map (fun pair =>
            (decodeVector pair.rho, decodeVector pair.input))).map
          (fun pair => mulPair (KQuotientIdentity.projected z beta pair.1)
            (KQuotientIdentity.projected z beta pair.2)))
      = addPair
          (KQuotientIdentity.projected z beta (decodeVector trace.output))
          (mulPair
            (KQuotientIdentity.projected z beta
              (decodeVector trace.quotient))
            (KQuotientIdentity.projected z beta decodeModulus))) :
    Polynomial.eval (trace.identity z).lhs point
      = Polynomial.eval (trace.identity z).rhs point := by
  refine KBridge.toPair_injective ?_
  have lhsForm : KBridge.toPair (Polynomial.eval (trace.identity z).lhs point)
      = KQuotientIdentity.pairSum
          ((trace.pairs.map (fun pair =>
              (decodeVector pair.rho, decodeVector pair.input))).map
            (fun pair => mulPair (KQuotientIdentity.projected z beta pair.1)
              (KQuotientIdentity.projected z beta pair.2))) := by
    rw [eval_identity_lhs, ← pairSum_toPair, List.map_map, List.map_map]
    congr 1
    exact List.map_congr_left (fun pair _ => by
      simp only [Function.comp_apply]
      rw [projected_decodeVector z beta point _ betaDenotes,
        projected_decodeVector z beta point _ betaDenotes,
        ← KBridge.toPair_mul])
  have rhsForm : KBridge.toPair (Polynomial.eval (trace.identity z).rhs point)
      = addPair
          (KQuotientIdentity.projected z beta (decodeVector trace.output))
          (mulPair
            (KQuotientIdentity.projected z beta
              (decodeVector trace.quotient))
            (KQuotientIdentity.projected z beta decodeModulus)) := by
    rw [eval_identity_rhs, KBridge.toPair_add, KBridge.toPair_mul,
      ← projected_decodeVector z beta point trace.quotient betaDenotes,
      ← projected_decodeModulus z beta point constantWire betaDenotes,
      ← projected_decodeVector z beta point trace.output betaDenotes]
    exact KPairLaws.addPair_comm _ _
  rw [lhsForm, rhsForm]
  exact equation

open ProjectionProgram in
/-- **Coefficient exactness supplies the equation needed by the honest row
constructor.**

This is the converse transport used only for completeness.  It does not turn
one-point acceptance into coefficient exactness: the premise is the frozen
coefficient equality itself. -/
theorem equation_of_exact
    (z : Nat → Nat) (beta : Carried)
    (trace : KProjectionTrace.Trace) (point : K)
    (constantWire : z 0 = 1)
    (betaDenotes : carriedValue z beta = KBridge.toPair point)
    (exact : (trace.identity z).Exact) :
    KQuotientIdentity.pairSum
        ((trace.pairs.map (fun pair =>
            (decodeVector pair.rho, decodeVector pair.input))).map
          (fun pair => mulPair (KQuotientIdentity.projected z beta pair.1)
            (KQuotientIdentity.projected z beta pair.2)))
      = addPair
          (KQuotientIdentity.projected z beta (decodeVector trace.output))
          (mulPair
            (KQuotientIdentity.projected z beta
              (decodeVector trace.quotient))
            (KQuotientIdentity.projected z beta decodeModulus)) := by
  have frozen :
      Polynomial.eval (trace.identity z).lhs point =
        Polynomial.eval (trace.identity z).rhs point := by
    rw [exact]
  have lhsForm : KBridge.toPair (Polynomial.eval (trace.identity z).lhs point)
      = KQuotientIdentity.pairSum
          ((trace.pairs.map (fun pair =>
              (decodeVector pair.rho, decodeVector pair.input))).map
            (fun pair => mulPair (KQuotientIdentity.projected z beta pair.1)
              (KQuotientIdentity.projected z beta pair.2))) := by
    rw [eval_identity_lhs, ← pairSum_toPair, List.map_map, List.map_map]
    congr 1
    exact List.map_congr_left (fun pair _ => by
      simp only [Function.comp_apply]
      rw [projected_decodeVector z beta point _ betaDenotes,
        projected_decodeVector z beta point _ betaDenotes,
        ← KBridge.toPair_mul])
  have rhsForm : KBridge.toPair (Polynomial.eval (trace.identity z).rhs point)
      = addPair
          (KQuotientIdentity.projected z beta (decodeVector trace.output))
          (mulPair
            (KQuotientIdentity.projected z beta
              (decodeVector trace.quotient))
            (KQuotientIdentity.projected z beta decodeModulus)) := by
    rw [eval_identity_rhs, KBridge.toPair_add, KBridge.toPair_mul,
      ← projected_decodeVector z beta point trace.quotient betaDenotes,
      ← projected_decodeModulus z beta point constantWire betaDenotes,
      ← projected_decodeVector z beta point trace.output betaDenotes]
    exact KPairLaws.addPair_comm _ _
  rw [← lhsForm, ← rhsForm, frozen]

/-! ## The frozen relation

`Accepted` is `WellFormed` and the evaluation equality.
`equation_reaches_frozen_eval` gives the second;
`KProjectionTrace.Trace.identity_wellFormed` gives the first. Composing them is
the point at which the emitted row program reaches the frozen relation rather
than an equation in the recipe's own vocabulary. -/

open ProjectionProgram in
/-- **The emitted program's equation gives the frozen `Accepted`.** -/
theorem accepted_of_equation
    (z : Nat → Nat) (betaCarried : Carried)
    (trace : KProjectionTrace.Trace)
    (constantWire : z 0 = 1)
    (valid : trace.Valid)
    (betaDenotes : carriedValue z betaCarried
      = KBridge.toPair (trace.identity z).beta)
    (equation : KQuotientIdentity.pairSum
        ((trace.pairs.map (fun pair =>
            (decodeVector pair.rho, decodeVector pair.input))).map
          (fun pair => mulPair (KQuotientIdentity.projected z betaCarried pair.1)
            (KQuotientIdentity.projected z betaCarried pair.2)))
      = addPair
          (KQuotientIdentity.projected z betaCarried
            (decodeVector trace.output))
          (mulPair
            (KQuotientIdentity.projected z betaCarried
              (decodeVector trace.quotient))
            (KQuotientIdentity.projected z betaCarried decodeModulus))) :
    SuperNeo.ProjectionCheck.Accepted K.ops (trace.identity z) :=
  ⟨trace.identity_wellFormed z valid,
   equation_reaches_frozen_eval z betaCarried trace (trace.identity z).beta
     constantWire betaDenotes equation⟩

open ProjectionProgram in
/-- **The emitted program is coefficient-exact, or the named root event fires.**

This is the frozen soundness statement, reached from a row program whose every
count, column and witness was derived in Lean. `BadRoot` is the only escape, and
it is `SuperNeo.ProjectionCheck`'s own event, not one invented here. -/
theorem exact_or_badRoot_of_equation
    (z : Nat → Nat) (betaCarried : Carried)
    (trace : KProjectionTrace.Trace)
    (constantWire : z 0 = 1)
    (valid : trace.Valid)
    (betaDenotes : carriedValue z betaCarried
      = KBridge.toPair (trace.identity z).beta)
    (equation : KQuotientIdentity.pairSum
        ((trace.pairs.map (fun pair =>
            (decodeVector pair.rho, decodeVector pair.input))).map
          (fun pair => mulPair (KQuotientIdentity.projected z betaCarried pair.1)
            (KQuotientIdentity.projected z betaCarried pair.2)))
      = addPair
          (KQuotientIdentity.projected z betaCarried
            (decodeVector trace.output))
          (mulPair
            (KQuotientIdentity.projected z betaCarried
              (decodeVector trace.quotient))
            (KQuotientIdentity.projected z betaCarried decodeModulus))) :
    (trace.identity z).Exact
      ∨ SuperNeo.ProjectionCheck.BadRoot K.ops (trace.identity z) :=
  SuperNeo.ProjectionCheck.accepted_implies_exact_or_badRoot _ _
    (accepted_of_equation z betaCarried trace constantWire valid
      betaDenotes equation)

/-! ## The per-trace boundary -/

open ProjectionProgram in
/-- What one trace needs to give `Accepted`.

This structure is not an external premise of the selected batch theorem.
`KTraceProgram.traceAccepts_of_rows` constructs it from the occurrence's
emitted quotient-identity rows. -/
structure TraceAccepts (z : Nat → Nat) (betaCarried : Carried)
    (trace : KProjectionTrace.Trace) : Prop where
  valid : trace.Valid
  betaDenotes : carriedValue z betaCarried
    = KBridge.toPair (trace.identity z).beta
  equation : KQuotientIdentity.pairSum
      ((trace.pairs.map (fun pair =>
          (decodeVector pair.rho, decodeVector pair.input))).map
        (fun pair => mulPair (KQuotientIdentity.projected z betaCarried pair.1)
          (KQuotientIdentity.projected z betaCarried pair.2)))
    = addPair
        (KQuotientIdentity.projected z betaCarried
          (decodeVector trace.output))
        (mulPair
          (KQuotientIdentity.projected z betaCarried
            (decodeVector trace.quotient))
          (KQuotientIdentity.projected z betaCarried decodeModulus))

end Nightstream.Implementation.R1CS.Canonical.KTraceDecoder
