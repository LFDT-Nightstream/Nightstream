import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CausalSumCheckBound
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra
import Nightstream.SuperNeo.SumCheck.FixedPhase
import Nightstream.SuperNeo.SumCheck.FixedPhase.Sequential

/-!
Causal finite-uniform soundness adapter for one typed fixed-phase SumCheck.

Assurance tier: model-level.

Owns: a message-before-challenge generator, exact generated certificate,
transport from the repository `FixedPhase.BadChallenge` event to the generic
causal collision monitor, and the finite root-counting probability bound.

Does not own: a concrete FE/NC polynomial, mixing challenges, Fiat--Shamir,
Rust/R1CS, costs, or rows.

Emits constraints: none.

Authority boundary: `Generator.claimed` sees exactly the prior challenge
prefix. The current and future challenges are absent from its input.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `sumcheck.causal.claimed` | fix each submitted polynomial before its current challenge | checked | `Generator.claimed`, `certificate` |
| `sumcheck.causal.expected` | represent the semantic completion polynomial at the same prior prefix | derived | `Generator.expectedRepresents` |
| `sumcheck.causal.transport` | map the exact fixed-phase collision into the causal detector | derived | `badChallenge_implies_detects` |
| `sumcheck.causal.probability` | bound all detected rounds by finite root counting and a union bound | derived | `detects_probability_le` |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.CausalFixedPhase

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.SumCheck
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws

/-- Exact prior-challenge state at one causal round. -/
structure Prefix (round : Nat) where
  values : List K
  length : values.length = round

namespace Prefix

@[ext] theorem ext
    {round : Nat}
    (left right : Prefix round)
    (values : left.values = right.values) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Equality of round indices and prefix values gives heterogeneous equality
of the indexed prefix receipts. -/
theorem heq_of_round_eq
    {leftRound rightRound : Nat}
    (left : Prefix leftRound)
    (right : Prefix rightRound)
    (roundEqual : leftRound = rightRound)
    (valuesEqual : left.values = right.values) :
    HEq left right := by
  subst rightRound
  exact heq_of_eq (ext left right valuesEqual)

end Prefix

/-- A generated fixed-width proof. Both polynomial functions are fixed by the
prior prefix; `expectedRepresents` connects the verifier-owned expected
polynomial to the semantic completion of `q`. -/
structure Generator (degree rounds : Nat) where
  q : List K -> K
  initial : K
  challengeSetSize : Nat
  claimed :
    forall round : Nat, round < rounds -> Prefix round ->
      FixedPolynomial K degree
  expected :
    forall round : Nat, round < rounds -> Prefix round ->
      FixedPolynomial K degree
  expectedRepresents :
    forall round within prior,
      FixedPhase.Represents ops.toOps
        (expected round within prior)
        (fun point =>
          HypercubeTruth.sumCompletions ops.toOps q
            (prior.values ++ [point])
            (rounds - (round + 1)))

namespace Generator

/-- Build the verifier-owned expected round directly from a prefix-local
round-representability theorem. The claimed message remains an arbitrary
causal strategy, while the expected message is chosen solely from `q` and the
same prior prefix. -/
noncomputable def ofRoundRepresentable
    {degree rounds : Nat}
    (q : List K -> K)
    (initial : K)
    (challengeSetSize : Nat)
    (claimed :
      forall round : Nat, round < rounds -> Prefix round ->
        FixedPolynomial K degree)
    (represented :
      FixedPhase.Sequential.RoundRepresentable ops.toOps q degree rounds) :
    Generator degree rounds where
  q := q
  initial := initial
  challengeSetSize := challengeSetSize
  claimed := claimed
  expected := fun round within prior =>
    Classical.choose (represented prior.values
      (rounds - (round + 1)) (by
        rw [prior.length]
        omega))
  expectedRepresents := fun round within prior =>
    Classical.choose_spec (represented prior.values
      (rounds - (round + 1)) (by
        rw [prior.length]
        omega))

end Generator

/-- Generic causal monitor induced by one generated proof. -/
noncomputable def process
    {degree rounds : Nat}
    (generator : Generator degree rounds) :
    CausalSumCheckBound.Process K degree rounds where
  State := Prefix
  initial := ⟨[], rfl⟩
  polynomials := fun round within prior =>
    some
      (generator.claimed round within prior,
        generator.expected round within prior)
  advance := fun round _ prior challenge => {
    values := prior.values ++ [challenge]
    length := by simp [prior.length]
  }

/-- Prefix of a complete verifier word at one round. -/
def prefixAt
    {rounds : Nat}
    (word : Fin rounds -> K)
    (round : Nat)
    (within : round <= rounds) :
    Prefix round where
  values := (List.ofFn word).take round
  length := by
    rw [List.length_take, List.length_ofFn]
    omega

/-- The exact fixed-phase certificate generated from a complete verifier
word. Each round message is evaluated only on the word prefix preceding that
round. -/
noncomputable def certificate
    {degree rounds : Nat}
    (generator : Generator degree rounds)
    (word : Fin rounds -> K) :
    FixedPhase.Certificate K degree where
  rounds := List.ofFn fun round : Fin rounds =>
    generator.claimed round.val round.isLt
      (prefixAt word round.val (Nat.le_of_lt round.isLt))

@[simp] theorem certificate_rounds_length
    {degree rounds : Nat}
    (generator : Generator degree rounds)
    (word : Fin rounds -> K) :
    (certificate generator word).rounds.length = rounds := by
  simp [certificate]

/-- Root counting applies directly to the generated causal proof. -/
theorem detects_probability_le
    {degree rounds : Nat}
    (generator : Generator degree rounds)
    (noZeroDivisors :
      FiniteRootCounting.NoZeroDivisors ops)
    (alphabet : Support K) :
    ((Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords.Support.challengeVectors
        alphabet rounds).uniform
      ).probabilityBool
        (CausalSumCheckBound.detects ops (process generator)) <=
      ratio (rounds * degree) alphabet.cardinality := by
  exact CausalSumCheckBound.probability_detects_le_ratio
    ops laws noZeroDivisors (process generator) alphabet

private theorem detectsListFrom_true_of_current
    {degree rounds : Nat}
    (generator : Generator degree rounds)
    (round : Nat)
    (prior : Prefix round)
    (before : List K)
    (challenge : K)
    (after : List K)
    (total : round + (before ++ challenge :: after).length = rounds)
    (bad :
      CausalSumCheckBound.currentBad ops (process generator)
        (round + before.length) (by
          simp only [List.length_append, List.length_cons] at total
          omega)
        {
          values := prior.values ++ before
          length := by
            simp only [List.length_append, prior.length]
        }
        challenge = true) :
    CausalSumCheckBound.detectsListFrom ops (process generator)
      round (before ++ challenge :: after) total prior = true := by
  induction before generalizing round prior with
  | nil =>
      simp only [List.nil_append] at bad ⊢
      unfold CausalSumCheckBound.detectsListFrom
      rw [Bool.or_eq_true]
      exact Or.inl (by simpa [process] using bad)
  | cons head before inductionHypothesis =>
      have tailTotal :
          (round + 1) + (before ++ challenge :: after).length = rounds := by
        simp only [List.length_append, List.length_cons] at total ⊢
        omega
      let next : Prefix (round + 1) :=
        (process generator).advance round (by
          simp only [List.length_append, List.length_cons] at total
          omega) prior head
      have tailBad :
          CausalSumCheckBound.currentBad ops (process generator)
            ((round + 1) + before.length) (by
              simp only [List.length_append, List.length_cons] at tailTotal
              omega)
            {
              values := next.values ++ before
              length := by
                simp only [List.length_append, next.length]
            }
            challenge = true := by
        have roundEqual :
            round + (before.length + 1) =
              (round + 1) + before.length := by
          omega
        let oldState : Prefix (round + (before.length + 1)) := {
          values := prior.values ++ head :: before
          length := by
            simp only [List.length_append, List.length_cons, prior.length]
        }
        let newState : Prefix ((round + 1) + before.length) := {
          values := next.values ++ before
          length := by
            simp only [List.length_append, next.length]
        }
        have stateEqual : HEq oldState newState :=
          Prefix.heq_of_round_eq oldState newState roundEqual (by
            simp [oldState, newState, next, process, List.append_assoc])
        have eventEqual :=
          CausalSumCheckBound.currentBad_congr ops (process generator)
            roundEqual (by
              simp only [List.length_append, List.length_cons] at total
              omega) (by
              simp only [List.length_append, List.length_cons] at tailTotal
              omega)
            oldState newState stateEqual challenge
        rw [← eventEqual]
        simpa [oldState] using bad
      have tailResult :=
        inductionHypothesis (round := round + 1) (prior := next)
          tailTotal tailBad
      change
        (CausalSumCheckBound.currentBad ops (process generator)
            round (by
              simp only [List.length_append, List.length_cons] at total
              omega)
            prior head ||
          CausalSumCheckBound.detectsListFrom ops (process generator)
            (round + 1) (before ++ challenge :: after) tailTotal next) = true
      rw [Bool.or_eq_true]
      exact Or.inr tailResult

/-- Exact event transport from the repository fixed-phase event on the
generated certificate to the causal root-counting monitor. -/
theorem badChallenge_implies_detects
    {degree rounds : Nat}
    (generator : Generator degree rounds)
    (word : Fin rounds -> K)
    (bad :
      ∃ round,
        FixedPhase.BadChallenge ops.toOps generator.q degree
          generator.challengeSetSize generator.initial
          (List.ofFn word) (certificate generator word) round) :
    CausalSumCheckBound.detects ops (process generator) word = true := by
  classical
  obtain ⟨before, challenge, after, beforePolynomials, claimedPolynomial,
      afterPolynomials, wordEqual, polynomialsEqual, prefixLengths,
      functionsDifferent, valuesEqual⟩ :=
    FixedPhase.badChallenge_implies_causal_decomposition ops.toOps
      generator.q generator.challengeSetSize generator.initial
      (List.ofFn word) (certificate generator word) bad
  have beforeWithin : before.length < rounds := by
    have wordLength := List.length_ofFn (f := word)
    rw [wordEqual] at wordLength
    simp only [List.length_append, List.length_cons] at wordLength
    omega
  let selectedIndex :
      Fin (certificate generator word).rounds.length :=
    ⟨before.length, by
      rw [certificate_rounds_length]
      exact beforeWithin⟩
  have selectedFromCertificate :
      (certificate generator word).rounds.get selectedIndex =
        claimedPolynomial := by
    calc
      (certificate generator word).rounds.get selectedIndex =
          (beforePolynomials ++ claimedPolynomial :: afterPolynomials).get
            ⟨selectedIndex.val, polynomialsEqual ▸ selectedIndex.isLt⟩ :=
        List.get_of_eq polynomialsEqual selectedIndex
      _ = claimedPolynomial := by
        simp only [selectedIndex, List.get_eq_getElem]
        rw [List.getElem_append_right (by omega)]
        simp [prefixLengths]
  let selectedPrefix : Prefix before.length := ⟨before, rfl⟩
  have generatedClaimed :
      generator.claimed before.length beforeWithin selectedPrefix =
        claimedPolynomial := by
    have prefixValue :
        (prefixAt word before.length
          (Nat.le_of_lt beforeWithin)).values = before := by
      simp [prefixAt, wordEqual]
    have prefixEqual :
        prefixAt word before.length (Nat.le_of_lt beforeWithin) =
          selectedPrefix :=
      Prefix.ext _ _ prefixValue
    simpa [certificate, selectedIndex, prefixEqual] using
      selectedFromCertificate
  have expectedFunction :
      (fun point =>
        (generator.expected before.length beforeWithin selectedPrefix
          ).evaluate ops.toOps point) =
        (fun point =>
          HypercubeTruth.sumCompletions ops.toOps generator.q
            (before ++ [point]) after.length) := by
    funext point
    have totalLength :
        before.length + 1 + after.length = rounds := by
      have wordLength := List.length_ofFn (f := word)
      rw [wordEqual] at wordLength
      simp only [List.length_append, List.length_cons] at wordLength
      omega
    have afterLength :
        rounds - (before.length + 1) = after.length := by
      omega
    simpa [selectedPrefix, afterLength] using
      generator.expectedRepresents before.length beforeWithin
        selectedPrefix point
  have selectedDifferent :
      (fun point => claimedPolynomial.evaluate ops.toOps point) ≠
        (fun point =>
          (generator.expected before.length beforeWithin selectedPrefix
            ).evaluate ops.toOps point) := by
    intro equal
    exact functionsDifferent (equal.trans expectedFunction)
  have selectedCollision :
      claimedPolynomial.evaluate ops.toOps challenge =
        (generator.expected before.length beforeWithin selectedPrefix
          ).evaluate ops.toOps challenge := by
    exact valuesEqual.trans (congrFun expectedFunction challenge).symm
  have current :
      CausalSumCheckBound.currentBad ops (process generator)
        before.length beforeWithin selectedPrefix challenge = true := by
    unfold CausalSumCheckBound.currentBad
    simp only [process]
    rw [generatedClaimed]
    simp [selectedDifferent, selectedCollision]
  rw [CausalSumCheckBound.detects_eq_detectsListFrom]
  have transported :=
    detectsListFrom_true_of_current generator 0
      (process generator).initial before challenge after (by
        have wordLength := List.length_ofFn (f := word)
        rw [wordEqual] at wordLength
        simpa using wordLength)
      (by
        have eventEqual :=
          CausalSumCheckBound.currentBad_congr ops (process generator)
            (leftRound := before.length)
            (rightRound := 0 + before.length)
            (by omega) beforeWithin (by omega)
            selectedPrefix
            {
              values := (process generator).initial.values ++ before
              length := by simp [(process generator).initial.length]
            }
            (Prefix.heq_of_round_eq _ _
              (by omega) (by rfl))
            challenge
        exact eventEqual.symm.trans current)
  simpa [wordEqual] using transported

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.CausalFixedPhase
