import Nightstream.Implementation.R1CS.Canonical.KBridge
import Nightstream.Implementation.R1CS.Canonical.KHornerHonest
import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.Lowering.Typed.Cost
import Nightstream.SuperNeo.SumCheck.FixedPhase

/-!
Contract: a Lean-owned row program for the fixed-width SumCheck chain used by
the paper `Pi_CCS` verifier.

Owns:
- constant-first, fixed-width round coefficient columns;
- the optimized round equation `current = p(0) + p(1)` as a coefficient
  combination, without materializing either evaluation;
- one Horner program for the verifier challenge;
- exact rejection of round/challenge shape mismatch;
- the derived row count; and
- soundness to `SumCheck.Finite.FixedPhase.Chain` over the concrete
  Goldilocks-quadratic carrier.

Does not own: transcript challenge generation, the protocol-specific initial
or terminal expression, the surrounding `Pi_CCS` call, or probability bounds.

The program is independent of Rust and generated row artifacts.  Its only
nonlinear rows are the existing three-row extension multiplications used by
`KHorner`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.SumCheck.Finite

/-- SumCheck's operation record, instantiated by the same concrete extension
operations used by the projection program. -/
def sumCheckOps : Ops K where
  zero := K.zero
  one := K.one
  add := K.add
  mul := K.mul

/-- One verifier-visible fixed-width round, represented by carried extension
values rather than semantic field elements. -/
structure Round (degree : Nat) where
  coefficients : List Carried
  coefficients_length : coefficients.length = degree + 1

/-- Decode one carried value from the canonical residues read by an
assignment. -/
def decodeCarried (assignment : Nat -> Nat) (value : Carried) : K where
  c0 := ⟨lcEval assignment value.low, by
    unfold lcEval
    exact Nat.mod_lt _ (by decide)⟩
  c1 := ⟨lcEval assignment value.high, by
    unfold lcEval
    exact Nat.mod_lt _ (by decide)⟩

@[simp] theorem toPair_decodeCarried
    (assignment : Nat -> Nat) (value : Carried) :
    KBridge.toPair (decodeCarried assignment value) =
      carriedValue assignment value := by
  rfl

/-- Decode a round to the exact fixed polynomial consumed by the paper
verifier. -/
def Round.polynomial
    {degree : Nat} (round : Round degree) (assignment : Nat -> Nat) :
    FixedPolynomial K degree where
  coefficients := round.coefficients.map (decodeCarried assignment)
  coefficients_length := by
    rw [List.length_map, round.coefficients_length]

/-- Coordinatewise addition is a coefficient concatenation and emits no row. -/
def addCarried (left right : Carried) : Carried where
  low := left.low ++ right.low
  high := left.high ++ right.high

/-- The zero carried value. -/
def zeroCarried : Carried := ⟨[], []⟩

/-- Sum a list of carried values without materializing an intermediate. -/
def sumCarried : List Carried -> Carried
  | [] => zeroCarried
  | head :: tail => addCarried head (sumCarried tail)

/-- Totalized first coefficient.  A `Round` is statically nonempty because its
length is `degree + 1`; totalization keeps the row generator structurally
simple. -/
def firstCarried : List Carried -> Carried
  | [] => zeroCarried
  | head :: _ => head

/-- The optimized linear combination for `p(0) + p(1)`.

For constant-first coefficients, `p(0)` is the first coefficient and `p(1)`
is the sum of all coefficients. -/
def roundInitial (coefficients : List Carried) : Carried :=
  addCarried (firstCarried coefficients) (sumCarried coefficients)

theorem decodeCarried_add
    (assignment : Nat -> Nat) (left right : Carried) :
    decodeCarried assignment (addCarried left right) =
      K.add (decodeCarried assignment left) (decodeCarried assignment right) := by
  apply KBridge.toPair_injective
  simp only [toPair_decodeCarried, carriedValue, addCarried, KBridge.toPair_add,
    addPair, Pair.mk.injEq]
  exact ⟨lcEval_append assignment left.low right.low,
    lcEval_append assignment left.high right.high⟩

theorem decodeCarried_zero (assignment : Nat -> Nat) :
    decodeCarried assignment zeroCarried = K.zero := by
  apply KBridge.toPair_injective
  rfl

theorem decodeCarried_sum
    (assignment : Nat -> Nat) :
    forall values : List Carried,
      decodeCarried assignment (sumCarried values) =
        values.foldr
          (fun value suffix =>
            K.add (decodeCarried assignment value) suffix)
          K.zero
  | [] => decodeCarried_zero assignment
  | head :: tail => by
      rw [sumCarried, decodeCarried_add, decodeCarried_sum assignment tail]
      rfl

theorem decodeCarried_first
    (assignment : Nat -> Nat) :
    forall values : List Carried,
      decodeCarried assignment (firstCarried values) =
        match values with
        | [] => K.zero
        | head :: _ => decodeCarried assignment head
  | [] => decodeCarried_zero assignment
  | _ :: _ => rfl

private theorem eval_zero :
    forall coefficients : List K,
      Message.evaluateCoefficients sumCheckOps K.zero coefficients =
        match coefficients with
        | [] => K.zero
        | head :: _ => head
  | [] => rfl
  | head :: tail => by
      simp [Message.evaluateCoefficients, sumCheckOps]

private theorem eval_one :
    forall coefficients : List K,
      Message.evaluateCoefficients sumCheckOps K.one coefficients =
        coefficients.foldr K.add K.zero
  | [] => rfl
  | head :: tail => by
      simp only [Message.evaluateCoefficients, sumCheckOps, K.one_mul,
        List.foldr_cons]
      change K.add head
          (Message.evaluateCoefficients sumCheckOps K.one tail) =
        K.add head (List.foldr K.add K.zero tail)
      rw [eval_one tail]

/-- The SumCheck evaluator and the canonical row reference are the same
constant-first Horner machine. -/
theorem toPair_evaluateCoefficients (point : K) :
    forall coefficients : List K,
      KBridge.toPair
          (Message.evaluateCoefficients sumCheckOps point coefficients) =
        hornerValue (KBridge.toPair point)
          (coefficients.map KBridge.toPair)
  | [] => rfl
  | [coefficient] => by
      simp only [Message.evaluateCoefficients, sumCheckOps,
        List.map_cons, List.map_nil]
      rw [K.mul_zero, K.add_zero]
      rfl
  | coefficient :: next :: rest => by
      change
        KBridge.toPair
            (K.add coefficient
              (K.mul point
                (Message.evaluateCoefficients sumCheckOps point
                  (next :: rest)))) =
          addPair (KBridge.toPair coefficient)
            (mulPair (KBridge.toPair point)
              (hornerValue (KBridge.toPair point)
                ((next :: rest).map KBridge.toPair)))
      rw [KBridge.toPair_add, KBridge.toPair_mul,
        toPair_evaluateCoefficients point (next :: rest)]

theorem toPair_fixedPolynomial_evaluate
    {degree : Nat} (polynomial : FixedPolynomial K degree) (point : K) :
    KBridge.toPair (FixedPolynomial.evaluate sumCheckOps polynomial point) =
      hornerValue (KBridge.toPair point)
        (polynomial.coefficients.map KBridge.toPair) := by
  unfold FixedPolynomial.evaluate FixedPolynomial.toMessage Message.evaluate
  exact toPair_evaluateCoefficients point polynomial.coefficients

private theorem decodeCarried_roundInitial_coefficients
    (assignment : Nat -> Nat) (coefficients : List Carried) :
    decodeCarried assignment (roundInitial coefficients) =
      K.add
        (Message.evaluateCoefficients sumCheckOps K.zero
          (coefficients.map (decodeCarried assignment)))
        (Message.evaluateCoefficients sumCheckOps K.one
          (coefficients.map (decodeCarried assignment))) := by
  unfold roundInitial
  rw [decodeCarried_add, decodeCarried_first, decodeCarried_sum,
    eval_zero, eval_one]
  cases coefficients with
  | nil => rfl
  | cons head tail =>
      simp only [List.map_cons, List.foldr_cons, List.foldr_map,
        Function.comp_apply]

/-- The optimized carried combination is exactly the paper verifier's
`p(0) + p(1)` expression. -/
theorem decodeCarried_roundInitial
    {degree : Nat} (round : Round degree) (assignment : Nat -> Nat) :
    decodeCarried assignment (roundInitial round.coefficients) =
      K.add
        (FixedPolynomial.evaluate sumCheckOps
          (round.polynomial assignment) K.zero)
        (FixedPolynomial.evaluate sumCheckOps
          (round.polynomial assignment) K.one) := by
  simpa [Round.polynomial, FixedPolynomial.evaluate,
    FixedPolynomial.toMessage, Message.evaluate] using
      decodeCarried_roundInitial_coefficients assignment round.coefficients

/-- A fail-closed row used only when the round and challenge lists have
different lengths.  Under the required constant-one wire it asserts
`1 * 1 = 0`. -/
def rejectRow : Row :=
  ⟨[(0, 1)], [(0, 1)], []⟩

/-- Complete fixed-phase row program.

Each round emits two equality rows for `current = p(0)+p(1)`, `3*degree`
Horner rows for `p(challenge)`, then carries that result into the next round.
The final claim costs two equality rows. -/
def chainRows {degree : Nat} :
    Carried -> List (Round degree) -> List Carried -> Carried -> Nat -> List Row
  | current, [], [], terminal, _ =>
      KEquality.rows current terminal
  | current, round :: rounds, challenge :: challenges, terminal, base =>
      KEquality.rows current (roundInitial round.coefficients) ++
        KHorner.hornerRows challenge (KFrames.frameAt base)
          round.coefficients 0 ++
        chainRows
          (KHorner.hornerCarried challenge (KFrames.frameAt base)
            round.coefficients 0)
          rounds challenges terminal (base + 3 * degree)
  | _, _, _, _, _ => [rejectRow]

private theorem rejectRow_not_satisfied
    (assignment : Nat -> Nat) (constantWire : assignment 0 = 1) :
    ¬ Satisfies [rejectRow] assignment := by
  intro satisfied
  have row := satisfied rejectRow (by simp)
  unfold RowHolds rejectRow at row
  change
    lcEval assignment [(0, 1)] * lcEval assignment [(0, 1)] %
        goldilocksP =
      lcEval assignment [] at row
  simp only [KEquality.one_wire assignment constantWire,
    KHorner.lcEval_nil] at row
  exact (by decide : 1 * 1 % goldilocksP ≠ 0) row

private theorem satisfies_append_left
    {left right : List Row} {assignment : Nat -> Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies left assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

private theorem satisfies_append_right
    {left right : List Row} {assignment : Nat -> Nat}
    (satisfied : Satisfies (left ++ right) assignment) :
    Satisfies right assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

/-- A satisfying Horner block carries exactly the fixed polynomial's
evaluation at the decoded challenge. -/
theorem horner_decodes
    {degree : Nat}
    (assignment : Nat -> Nat)
    (round : Round degree) (challenge : Carried) (base : Nat)
    (satisfied : Satisfies
      (KHorner.hornerRows challenge (KFrames.frameAt base)
        round.coefficients 0) assignment) :
    decodeCarried assignment
        (KHorner.hornerCarried challenge (KFrames.frameAt base)
          round.coefficients 0) =
      FixedPolynomial.evaluate sumCheckOps (round.polynomial assignment)
        (decodeCarried assignment challenge) := by
  apply KBridge.toPair_injective
  rw [toPair_decodeCarried,
    toPair_fixedPolynomial_evaluate, toPair_decodeCarried]
  have sound := KHorner.hornerRows_sound assignment challenge
    (KFrames.frameAt base) round.coefficients 0 satisfied
  rw [sound]
  apply congrArg (hornerValue (carriedValue assignment challenge))
  simp only [Round.polynomial, List.map_map]
  apply List.map_congr_left
  intro coefficient _
  exact (toPair_decodeCarried assignment coefficient).symm

/-- **Soundness of the complete emitted chain.**

No claimed equation is supplied by the caller.  Every `FixedPhase.Chain`
conjunct is reconstructed from row satisfaction, and mismatched list shapes
are rejected by an unsatisfiable row. -/
theorem chainRows_sound
    {degree : Nat}
    (assignment : Nat -> Nat) (constantWire : assignment 0 = 1) :
    forall
      (current : Carried)
      (rounds : List (Round degree))
      (challenges : List Carried)
      (terminal : Carried)
      (base : Nat),
      Satisfies (chainRows current rounds challenges terminal base) assignment ->
      FixedPhase.Chain sumCheckOps
        (decodeCarried assignment current)
        (rounds.map fun round => round.polynomial assignment)
        (challenges.map (decodeCarried assignment))
        (decodeCarried assignment terminal)
  | current, [], [], terminal, _, satisfied => by
      simp only [chainRows, List.map_nil, FixedPhase.Chain]
      rcases KEquality.rows_sound assignment current terminal constantWire
        satisfied with ⟨low, high⟩
      apply KBridge.toPair_injective
      simp only [toPair_decodeCarried, carriedValue, Pair.mk.injEq]
      exact ⟨low, high⟩
  | current, [], _ :: _, terminal, _, satisfied =>
      absurd satisfied (rejectRow_not_satisfied assignment constantWire)
  | current, _ :: _, [], terminal, _, satisfied =>
      absurd satisfied (rejectRow_not_satisfied assignment constantWire)
  | current, round :: rounds, challenge :: challenges, terminal, base,
      satisfied => by
      simp only [chainRows, List.map_cons, FixedPhase.Chain]
      have rebracketed :
          Satisfies
            (KEquality.rows current (roundInitial round.coefficients) ++
              (KHorner.hornerRows challenge (KFrames.frameAt base)
                  round.coefficients 0 ++
                chainRows
                  (KHorner.hornerCarried challenge (KFrames.frameAt base)
                    round.coefficients 0)
                  rounds challenges terminal (base + 3 * degree)))
            assignment := by
        simpa only [List.append_assoc] using satisfied
      have first :
          Satisfies
            (KEquality.rows current (roundInitial round.coefficients))
            assignment :=
        satisfies_append_left rebracketed
      have remaining :
          Satisfies
            (KHorner.hornerRows challenge (KFrames.frameAt base)
                round.coefficients 0 ++
              chainRows
                (KHorner.hornerCarried challenge (KFrames.frameAt base)
                  round.coefficients 0)
                rounds challenges terminal (base + 3 * degree))
            assignment :=
        satisfies_append_right rebracketed
      have horner :
          Satisfies
            (KHorner.hornerRows challenge (KFrames.frameAt base)
              round.coefficients 0) assignment :=
        satisfies_append_left remaining
      have rest :
          Satisfies
            (chainRows
              (KHorner.hornerCarried challenge (KFrames.frameAt base)
                round.coefficients 0)
              rounds challenges terminal (base + 3 * degree)) assignment :=
        satisfies_append_right remaining
      constructor
      · rcases KEquality.rows_sound assignment current
          (roundInitial round.coefficients) constantWire first with
          ⟨low, high⟩
        apply KBridge.toPair_injective
        change carriedValue assignment current =
          KBridge.toPair
            (K.add
              (FixedPolynomial.evaluate sumCheckOps
                (round.polynomial assignment) K.zero)
              (FixedPolynomial.evaluate sumCheckOps
                (round.polynomial assignment) K.one))
        rw [← decodeCarried_roundInitial round assignment,
          toPair_decodeCarried]
        simp only [carriedValue, Pair.mk.injEq]
        exact ⟨low, high⟩
      · rw [← horner_decodes assignment round challenge base horner]
        exact chainRows_sound assignment constantWire
          (KHorner.hornerCarried challenge (KFrames.frameAt base)
            round.coefficients 0)
          rounds challenges terminal (base + 3 * degree) rest

/-- Exact row count for well-shaped fixed-phase executions. -/
theorem chainRows_length
    {degree : Nat}
    (current : Carried)
    (rounds : List (Round degree))
    (challenges : List Carried)
    (terminal : Carried)
    (base : Nat)
    (sameLength : rounds.length = challenges.length) :
    (chainRows current rounds challenges terminal base).length =
      rounds.length * (3 * degree + 2) + 2 := by
  induction rounds generalizing current challenges base with
  | nil =>
      cases challenges with
      | nil => simp [chainRows, KEquality.rows_length]
      | cons _ _ => simp at sameLength
  | cons round rounds inductionHypothesis =>
      cases challenges with
      | nil => simp at sameLength
      | cons challenge challenges =>
          simp only [List.length_cons, Nat.succ.injEq] at sameLength
          simp only [chainRows, List.length_append,
            KEquality.rows_length,
            KHorner.hornerRows_length_of_degree challenge
              (KFrames.frameAt base) round.coefficients degree
              round.coefficients_length 0,
            inductionHypothesis
              (KHorner.hornerCarried challenge (KFrames.frameAt base)
                round.coefficients 0)
              challenges (base + 3 * degree) sameLength]
          simp only [List.length_cons, Nat.succ_eq_add_one, Nat.add_mul,
            Nat.one_mul]
          omega

/-- Cost of a fixed-phase chain, parameterized only by its statically selected
round count and degree.  It excludes source/message columns and owns only the
Horner auxiliaries. -/
def chainCost (degree roundCount : Nat) :
    Nightstream.Implementation.Lowering.Typed.Cost where
  recurringRows := roundCount * (3 * degree + 2) + 2
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := roundCount * (3 * degree)

theorem chainCost_rows
    {degree : Nat}
    (current : Carried)
    (rounds : List (Round degree))
    (challenges : List Carried)
    (terminal : Carried)
    (base : Nat)
    (sameLength : rounds.length = challenges.length) :
    (chainRows current rounds challenges terminal base).length =
      (chainCost degree rounds.length).recurringRows :=
  chainRows_length current rounds challenges terminal base sameLength

end Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
