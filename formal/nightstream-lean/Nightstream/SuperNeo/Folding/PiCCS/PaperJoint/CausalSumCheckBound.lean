import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords

/-!
Finite-uniform soundness for a causal fixed-width SumCheck process.

Assurance tier: model-level.

Owns: an indexed message-before-challenge process, its exact Boolean collision
event, a fiberwise finite root bound, successive-coordinate independence from
the recursive Cartesian word enumeration, and the explicit multi-round union
bound `rounds * degree / |alphabet|`.

Does not own: a concrete protocol polynomial, a prover implementation,
Fiat--Shamir, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

| Owned object | Exact equation or bound |
|---|---|
| per-prefix bad challenges | `count <= degree` |
| all bad challenge words | `count <= rounds * degree * |alphabet|^(rounds - 1)` |
| bad-word probability | `Pr[detects] <= rounds * degree / |alphabet|` |

The process state after `i` rounds may contain every prior message and
challenge.  Its current polynomial pair is selected from that state before
the next challenge is supplied to `advance`; future challenges are absent
from the interface.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CausalSumCheckBound

open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck.Finite

universe uField uState uHead uTail

/-- A finite causal process whose two round polynomials are fixed before the
current challenge is sampled. `none` is a malformed or unavailable pair and
cannot itself create a bad-challenge event. -/
structure Process
    (Field : Type uField)
    (degree rounds : Nat) where
  State : Nat -> Type uState
  initial : State 0
  polynomials :
    forall round : Nat, round < rounds -> State round ->
      Option
        (FixedPolynomial Field degree × FixedPolynomial Field degree)
  advance :
    forall round : Nat, round < rounds -> State round -> Field ->
      State (round + 1)

noncomputable def currentBad
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds)
    (round : Nat)
    (within : round < rounds)
    (state : process.State round)
    (challenge : Field) : Bool :=
  match process.polynomials round within state with
  | none => false
  | some (claimed, expected) =>
      letI : Decidable
          ((fun point => claimed.evaluate ops.toOps point) ≠
            fun point => expected.evaluate ops.toOps point) :=
        Classical.propDecidable _
      if
          (fun point => claimed.evaluate ops.toOps point) ≠
            fun point => expected.evaluate ops.toOps point then
        decide
          (claimed.evaluate ops.toOps challenge =
            expected.evaluate ops.toOps challenge)
      else
        false

/-- Proof receipts and propositionally equal round indices do not alter the
current collision event. -/
theorem currentBad_congr
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds)
    {leftRound rightRound : Nat}
    (roundEqual : leftRound = rightRound)
    (leftWithin : leftRound < rounds)
    (rightWithin : rightRound < rounds)
    (leftState : process.State leftRound)
    (rightState : process.State rightRound)
    (stateEqual : HEq leftState rightState)
    (challenge : Field) :
    currentBad ops process leftRound leftWithin leftState challenge =
      currentBad ops process rightRound rightWithin rightState challenge := by
  subst rightRound
  cases stateEqual
  rfl

/-- Execute the collision monitor over exactly `remaining` successive
challenges. The equality receipt prevents truncation or silent extra rounds. -/
noncomputable def detectsFrom
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds) :
    (round remaining : Nat) ->
      round + remaining = rounds ->
      process.State round ->
      (Fin remaining -> Field) ->
      Bool
  | _, 0, _, _, _ => false
  | round, remaining + 1, total, state, word =>
      let within : round < rounds := by omega
      currentBad ops process round within state (word 0) ||
        detectsFrom ops process (round + 1) remaining (by omega)
          (process.advance round within state (word 0))
          (fun index => word index.succ)

/-- The complete causal collision event on a verifier word. -/
noncomputable def detects
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds)
    (word : Fin rounds -> Field) : Bool :=
  detectsFrom ops process 0 rounds (by simp) process.initial word

/-- List presentation of the same causal monitor. This form is used for
exact transport from protocol transcript lists; the finite sampling theorem
continues to enumerate typed verifier words. -/
noncomputable def detectsListFrom
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds) :
    (round : Nat) ->
      (challenges : List Field) ->
      round + challenges.length = rounds ->
      process.State round ->
      Bool
  | _, [], _, _ => false
  | round, challenge :: challenges, total, state =>
      let within : round < rounds := by
        have totalLength := total
        simp only [List.length_cons] at totalLength
        omega
      currentBad ops process round within state challenge ||
        detectsListFrom ops process (round + 1) challenges (by
          simp only [List.length_cons] at total
          omega)
          (process.advance round within state challenge)

private theorem detectsListFrom_congr
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds)
    (round : Nat)
    (left right : List Field)
    (equal : left = right)
    (leftTotal : round + left.length = rounds)
    (rightTotal : round + right.length = rounds)
    (state : process.State round) :
    detectsListFrom ops process round left leftTotal state =
      detectsListFrom ops process round right rightTotal state := by
  subst right
  rfl

/-- Reindexing a typed word as a list changes neither order nor the monitored
event. -/
theorem detectsFrom_eq_detectsListFrom
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds)
    (round remaining : Nat)
    (total : round + remaining = rounds)
    (state : process.State round)
    (word : Fin remaining -> Field) :
    detectsFrom ops process round remaining total state word =
      detectsListFrom ops process round (List.ofFn word) (by
        simp [total]) state := by
  induction remaining generalizing round state with
  | zero =>
      simp [detectsFrom, detectsListFrom]
  | succ remaining inductionHypothesis =>
      let tailWord : Fin remaining -> Field := fun index => word index.succ
      have wordList :
          List.ofFn word = word 0 :: List.ofFn tailWord := by
        rw [List.ofFn_succ]
      calc
        detectsFrom ops process round (remaining + 1) total state word =
            (currentBad ops process round (by omega) state (word 0) ||
              detectsFrom ops process (round + 1) remaining (by omega)
                (process.advance round (by omega) state (word 0))
                tailWord) := by
          rfl
        _ =
            (currentBad ops process round (by omega) state (word 0) ||
              detectsListFrom ops process (round + 1)
                (List.ofFn tailWord) (by simp; omega)
                (process.advance round (by omega) state (word 0))) := by
          rw [inductionHypothesis]
        _ =
            detectsListFrom ops process round
              (word 0 :: List.ofFn tailWord) (by simp; omega) state := by
          rfl
        _ =
            detectsListFrom ops process round (List.ofFn word) (by
              simp [total]) state := by
          exact detectsListFrom_congr ops process round
            (word 0 :: List.ofFn tailWord) (List.ofFn word) wordList.symm
            (by simp; omega) (by simp [total]) state

theorem detects_eq_detectsListFrom
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds)
    (word : Fin rounds -> Field) :
    detects ops process word =
      detectsListFrom ops process 0 (List.ofFn word) (by simp)
        process.initial :=
  detectsFrom_eq_detectsListFrom ops process 0 rounds (by simp)
    process.initial word

private theorem currentBad_count_le
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (process : Process Field degree rounds)
    (round : Nat)
    (within : round < rounds)
    (state : process.State round)
    (alphabet : List Field)
    (alphabetNodup : alphabet.Nodup) :
    alphabet.countP (currentBad ops process round within state) <= degree := by
  unfold currentBad
  cases pairEqual : process.polynomials round within state with
  | none => simp
  | some pair =>
      rcases pair with ⟨claimed, expected⟩
      by_cases different :
          (fun point => claimed.evaluate ops.toOps point) ≠
            fun point => expected.evaluate ops.toOps point
      · simpa [pairEqual, different] using
          FiniteRootCounting.collisions_count_le_degree ops laws
            noZeroDivisors degree claimed expected alphabet alphabetNodup
            different
      · simp [different]

private theorem sum_countP_or_le
    {Head : Type uHead}
    {Tail : Type uTail}
    (heads : List Head)
    (tails : List Tail)
    (current : Head -> Bool)
    (future : Head -> Tail -> Bool)
    (futureBound : Nat)
    (bounded :
      forall head, head ∈ heads ->
        tails.countP (future head) <= futureBound) :
    (heads.map fun head =>
        tails.countP (fun tail => current head || future head tail)).sum <=
      heads.countP current * tails.length +
        heads.length * futureBound := by
  induction heads with
  | nil => simp
  | cons head heads inductionHypothesis =>
      have headBound := bounded head (by simp)
      have tailBound :
          forall value, value ∈ heads ->
            tails.countP (future value) <= futureBound := by
        intro value member
        exact bounded value (by simp [member])
      have tailResult := inductionHypothesis tailBound
      cases currentAtHead : current head with
      | false =>
          have combined := Nat.add_le_add headBound tailResult
          simpa [currentAtHead, Nat.add_mul, Nat.mul_add, Nat.add_assoc,
            Nat.add_comm, Nat.add_left_comm] using combined
      | true =>
          have headWithSlack :
              tails.length <= tails.length + futureBound :=
            Nat.le_add_right _ _
          have combined := Nat.add_le_add headWithSlack tailResult
          simpa [currentAtHead, Nat.add_mul, Nat.mul_add, Nat.add_assoc,
            Nat.add_comm, Nat.add_left_comm] using combined

@[simp] private theorem detectsFrom_prepend
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (process : Process Field degree rounds)
    (round remaining : Nat)
    (total : round + (remaining + 1) = rounds)
    (state : process.State round)
    (head : Field)
    (tail : Fin remaining -> Field) :
    detectsFrom ops process round (remaining + 1) total state
        (prepend head tail) =
      (currentBad ops process round (by omega) state head ||
        detectsFrom ops process (round + 1) remaining (by omega)
          (process.advance round (by omega) state head) tail) := by
  rfl

/-- Exact count form of the explicit multi-round union bound.  The factor
`|alphabet|^(remaining-1)` is the number of independent completions after
fixing one bad current challenge. -/
theorem detectsFrom_count_le
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (process : Process Field degree rounds)
    (alphabet : List Field)
    (alphabetNodup : alphabet.Nodup)
    (round remaining : Nat)
    (total : round + remaining = rounds)
    (state : process.State round) :
    (vectors alphabet remaining).countP
        (detectsFrom ops process round remaining total state) <=
      remaining * degree * alphabet.length ^ remaining.pred := by
  induction remaining generalizing round state with
  | zero => simp [vectors, detectsFrom]
  | succ remaining inductionHypothesis =>
      let within : round < rounds := by omega
      let tails := vectors alphabet remaining
      let futureBound :=
        remaining * degree * alphabet.length ^ remaining.pred
      have tailBound :
          forall head, head ∈ alphabet ->
            tails.countP (fun tail =>
              detectsFrom ops process (round + 1) remaining (by omega)
                (process.advance round within state head) tail) <=
              futureBound := by
        intro head _member
        exact inductionHypothesis
          (round := round + 1)
          (state := process.advance round within state head)
          (total := by omega)
      have splitBound :=
        sum_countP_or_le alphabet tails
          (currentBad ops process round within state)
          (fun head tail =>
            detectsFrom ops process (round + 1) remaining (by omega)
              (process.advance round within state head) tail)
          futureBound tailBound
      have currentBound :=
        currentBad_count_le ops laws noZeroDivisors process round within state
          alphabet alphabetNodup
      have combined :
          (vectors alphabet (remaining + 1)).countP
              (detectsFrom ops process round (remaining + 1) total state) <=
            degree * alphabet.length ^ remaining +
              alphabet.length * futureBound := by
        calc
          _ =
              (alphabet.map fun head =>
                tails.countP (fun tail =>
                  currentBad ops process round within state head ||
                    detectsFrom ops process (round + 1) remaining (by omega)
                      (process.advance round within state head) tail)).sum := by
                rw [vectors, List.countP_flatMap]
                apply congrArg List.sum
                apply List.map_congr_left
                intro head _member
                simp only [Function.comp_apply]
                rw [List.countP_map]
                apply List.countP_congr
                intro tail _tailMember
                simp only [Function.comp_apply]
                rw [detectsFrom_prepend ops process round remaining total
                  state head tail]
          _ <=
              alphabet.countP
                    (currentBad ops process round within state) *
                  tails.length +
                alphabet.length * futureBound :=
            splitBound
          _ <=
              degree * alphabet.length ^ remaining +
                alphabet.length * futureBound := by
            apply Nat.add_le_add
            · rw [show tails.length = alphabet.length ^ remaining by
                exact vectors_length alphabet remaining]
              exact Nat.mul_le_mul_right _ currentBound
            · exact Nat.le_refl _
      refine Nat.le_trans combined ?_
      cases remaining with
      | zero => simp [futureBound]
      | succ prior =>
          simp only [futureBound, Nat.pred_succ]
          apply Nat.le_of_eq
          simp [Nat.pow_succ, Nat.add_mul, Nat.mul_add]
          ac_rfl

/-- Successive challenges from the recursive Cartesian support obey the
standard SumCheck bound. No coordinate-independence proposition is assumed:
the proof counts every suffix independently under every possible prior
state. -/
theorem probability_detects_le_ratio
    {Field : Type uField}
    [DecidableEq Field]
    {degree rounds : Nat}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (process : Process Field degree rounds)
    (alphabet : Support Field) :
    ((FiniteWords.Support.challengeVectors alphabet rounds).uniform).probabilityBool
        (detects ops process) <=
      ratio (rounds * degree) alphabet.cardinality := by
  have countBound :=
    detectsFrom_count_le ops laws noZeroDivisors process alphabet.values
      alphabet.nodup 0 rounds (by simp) process.initial
  cases rounds with
  | zero =>
      simp [detects, detectsFrom, Experiment.probabilityBool,
        Experiment.countBool, ratio, Rat.div_def]
  | succ rounds =>
      have alphabetPosRat : 0 < (alphabet.cardinality : Rat) :=
        Rat.natCast_pos.mpr alphabet.cardinality_pos
      have alphabetNeZero : (alphabet.cardinality : Rat) ≠ 0 :=
        Rat.ne_of_gt alphabetPosRat
      have denominatorPos :
          0 <
            (((FiniteWords.Support.challengeVectors alphabet
              (rounds + 1)).cardinality : Nat) : Rat) := by
        rw [FiniteWords.Support.challengeVectors_cardinality]
        exact Rat.natCast_pos.mpr
          (Nat.pow_pos alphabet.cardinality_pos)
      unfold Experiment.probabilityBool Experiment.countBool
      simp only [Support.uniform, id_eq,
        FiniteWords.Support.challengeVectors_values]
      apply (div_le_iff_of_pos denominatorPos).2
      have castCountBound :
          ((vectors alphabet.values (rounds + 1)).countP
              (detects ops process) : Rat) <=
            ((rounds + 1) * degree *
              alphabet.cardinality ^ rounds : Nat) := by
        exact Rat.natCast_le_natCast.mpr (by
          simpa [detects, Support.cardinality] using countBound)
      refine Rat.le_trans castCountBound ?_
      have ratioTimesDenominator :
          ratio ((rounds + 1) * degree) alphabet.cardinality *
              (((FiniteWords.Support.challengeVectors alphabet
                (rounds + 1)).cardinality : Nat) : Rat) =
            (((rounds + 1) * degree *
              alphabet.cardinality ^ rounds : Nat) : Rat) := by
        unfold ratio
        rw [FiniteWords.Support.challengeVectors_cardinality, Nat.pow_succ]
        simp only [Rat.natCast_mul, Rat.natCast_pow]
        calc
          (((rounds + 1 : Nat) : Rat) * (degree : Rat) /
                (alphabet.cardinality : Rat)) *
              ((alphabet.cardinality : Rat) ^ rounds *
                (alphabet.cardinality : Rat)) =
            ((((rounds + 1 : Nat) : Rat) * (degree : Rat) /
                (alphabet.cardinality : Rat)) *
              (alphabet.cardinality : Rat)) *
                (alphabet.cardinality : Rat) ^ rounds := by
            rw [Rat.mul_comm
              ((alphabet.cardinality : Rat) ^ rounds)
              (alphabet.cardinality : Rat)]
            rw [← Rat.mul_assoc]
          _ =
              (((rounds + 1 : Nat) : Rat) * (degree : Rat)) *
                (alphabet.cardinality : Rat) ^ rounds := by
            rw [Rat.div_mul_cancel alphabetNeZero]
      rw [ratioTimesDenominator]
      exact Rat.le_refl

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CausalSumCheckBound
