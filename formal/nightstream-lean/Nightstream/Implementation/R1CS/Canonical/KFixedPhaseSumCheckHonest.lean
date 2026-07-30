import Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck

/-!
Contract: honest completeness for the Lean-owned fixed-phase SumCheck row
program.

Owns:
- one sequential witness over the disjoint per-round Horner blocks;
- preservation of the constant wire and all earlier blocks;
- the placement proof for every source combination and allocated frame; and
- satisfaction of the complete chain whenever the exact paper
  `FixedPhase.Chain` holds.

Does not own transcript generation, the protocol-specific initial or terminal
expression, or any application/NIFS selection.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KHornerSupport
open Nightstream.Implementation.R1CS.Canonical.KHornerHonest
open Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheck
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.SuperNeo.SumCheck.Finite

/-- Both coordinates of a carried extension value are placed below a base. -/
def CarriedBelow (value : Carried) (base : Nat) : Prop :=
  BelowBase value.low base ∧ BelowBase value.high base

/-- Every coefficient of one verifier message is placed below a base. -/
def RoundBelow {degree : Nat} (round : Round degree) (base : Nat) : Prop :=
  ∀ coefficient ∈ round.coefficients, CarriedBelow coefficient base

theorem below_mono {combination : LinComb} {base nextBase : Nat}
    (below : BelowBase combination base) (ordered : base ≤ nextBase) :
    BelowBase combination nextBase :=
  fun column mentioned => Nat.lt_of_lt_of_le (below column mentioned) ordered

theorem carriedBelow_mono {value : Carried} {base nextBase : Nat}
    (below : CarriedBelow value base) (ordered : base ≤ nextBase) :
    CarriedBelow value nextBase :=
  ⟨below_mono below.1 ordered, below_mono below.2 ordered⟩

theorem addCarried_below {left right : Carried} {base : Nat}
    (leftBelow : CarriedBelow left base)
    (rightBelow : CarriedBelow right base) :
    CarriedBelow (addCarried left right) base := by
  constructor <;> intro column mentioned
  · simp only [addCarried, BelowBase, Mentions, List.map_append,
      List.mem_append] at mentioned
    exact mentioned.elim (leftBelow.1 column) (rightBelow.1 column)
  · simp only [addCarried, BelowBase, Mentions, List.map_append,
      List.mem_append] at mentioned
    exact mentioned.elim (leftBelow.2 column) (rightBelow.2 column)

theorem zeroCarried_below (base : Nat) :
    CarriedBelow zeroCarried base := by
  constructor <;> intro column mentioned <;>
    simp [zeroCarried, BelowBase, Mentions] at mentioned

theorem sumCarried_below {base : Nat} :
    ∀ (values : List Carried),
      (∀ value ∈ values, CarriedBelow value base) →
      CarriedBelow (sumCarried values) base
  | [], _ => zeroCarried_below base
  | head :: tail, allBelow =>
      addCarried_below
        (allBelow head (by simp))
        (sumCarried_below tail
          (fun value member => allBelow value (List.mem_cons_of_mem _ member)))

theorem firstCarried_below {base : Nat} :
    ∀ (values : List Carried),
      (∀ value ∈ values, CarriedBelow value base) →
      CarriedBelow (firstCarried values) base
  | [], _ => zeroCarried_below base
  | head :: _, allBelow => allBelow head (by simp)

theorem roundInitial_below {degree base : Nat} (round : Round degree)
    (below : RoundBelow round base) :
    CarriedBelow (roundInitial round.coefficients) base :=
  addCarried_below
    (firstCarried_below round.coefficients below)
    (sumCarried_below round.coefficients below)

/-- The output combination of one degree-`d` Horner block is contained in
that block's `3*d` allocated columns plus its source columns. -/
theorem hornerCarried_below_next
    {degree base : Nat} (round : Round degree) (challenge : Carried)
    (roundBelow : RoundBelow round base) :
    CarriedBelow
      (hornerCarried challenge (KFrames.frameAt base)
        round.coefficients 0)
      (base + 3 * degree) := by
  have classify :
      ∀ column,
        (Mentions
            (hornerCarried challenge (KFrames.frameAt base)
              round.coefficients 0).low column
          ∨ Mentions
            (hornerCarried challenge (KFrames.frameAt base)
              round.coefficients 0).high column) →
        column < base + 3 * degree := by
    intro column mentioned
    rcases hornerCarried_mentions challenge (KFrames.frameAt base)
        round.coefficients 0 column mentioned with
      ⟨coefficient, member, inCoefficient⟩ |
      ⟨later, _, bounded, inFrame⟩
    · rcases inCoefficient with low | high
      · exact Nat.lt_of_lt_of_le
          ((roundBelow coefficient member).1 column low)
          (Nat.le_add_right base (3 * degree))
      · exact Nat.lt_of_lt_of_le
          ((roundBelow coefficient member).2 column high)
          (Nat.le_add_right base (3 * degree))
    · have laterLt : later < degree := by
        rw [round.coefficients_length] at bounded
        omega
      rcases inFrame with rfl | rfl | rfl <;>
        simp only [KFrames.frameAt, KFrames.frameColumn,
          KFrames.columnsPerFrame] <;> omega
  exact
    ⟨fun column mentioned => classify column (Or.inl mentioned),
      fun column mentioned => classify column (Or.inr mentioned)⟩

/-- Every column used by one Horner block lies below the next block's base. -/
theorem hornerRows_below_next
    {degree base : Nat} (round : Round degree) (challenge : Carried)
    (challengeBelow : CarriedBelow challenge base)
    (roundBelow : RoundBelow round base)
    (row : Row)
    (member : row ∈
      hornerRows challenge (KFrames.frameAt base) round.coefficients 0)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column < base + 3 * degree := by
  rcases hornerRows_mentions challenge (KFrames.frameAt base)
      round.coefficients 0 row member column mentioned with
    inChallenge | ⟨coefficient, memberCoefficient, inCoefficient⟩ |
      ⟨later, _, bounded, inFrame⟩
  · rcases inChallenge with low | high
    · exact Nat.lt_of_lt_of_le (challengeBelow.1 column low)
        (Nat.le_add_right base (3 * degree))
    · exact Nat.lt_of_lt_of_le (challengeBelow.2 column high)
        (Nat.le_add_right base (3 * degree))
  · rcases inCoefficient with low | high
    · exact Nat.lt_of_lt_of_le
        ((roundBelow coefficient memberCoefficient).1 column low)
        (Nat.le_add_right base (3 * degree))
    · exact Nat.lt_of_lt_of_le
        ((roundBelow coefficient memberCoefficient).2 column high)
        (Nat.le_add_right base (3 * degree))
  · have laterLt : later < degree := by
      rw [round.coefficients_length] at bounded
      omega
    rcases inFrame with rfl | rfl | rfl <;>
      simp only [KFrames.frameAt, KFrames.frameColumn,
        KFrames.columnsPerFrame] <;> omega

/-! ## Sequential witness -/

/-- Fill each round's Horner block from left to right.  Equality rows allocate
nothing, so the final branch leaves the assignment unchanged. -/
def chainWitness {degree : Nat} (assignment : Nat → Nat) :
    List (Round degree) → List Carried → Nat → (Nat → Nat)
  | [], [], _ => assignment
  | round :: rounds, challenge :: challenges, base =>
      let afterRound :=
        KHornerHonest.hornerWitness assignment challenge base
          round.coefficients 0
      chainWitness afterRound rounds challenges (base + 3 * degree)
  | _, _, _ => assignment

/-- A chain witness writes only at or above its current allocation base. -/
theorem chainWitness_off_block
    {degree : Nat} (assignment : Nat → Nat) :
    ∀ (rounds : List (Round degree)) (challenges : List Carried)
      (base column : Nat),
      column < base →
      chainWitness assignment rounds challenges base column =
        assignment column
  | [], [], _, _, _ => rfl
  | [], _ :: _, _, _, _ => rfl
  | _ :: _, [], _, _, _ => rfl
  | round :: rounds, challenge :: challenges, base, column, below => by
      unfold chainWitness
      rw [chainWitness_off_block
          (KHornerHonest.hornerWitness assignment challenge base
            round.coefficients 0)
          rounds challenges (base + 3 * degree) column
          (by omega),
        KHornerHonest.hornerWitness_off_block assignment challenge base
          round.coefficients 0 column (by simpa using below)]

/-- Sequential fixed-phase completion preserves canonical representatives. -/
theorem chainWitness_residues
    {degree : Nat} (assignment : Nat → Nat) :
    ∀ (rounds : List (Round degree)) (challenges : List Carried)
      (base : Nat),
      (∀ column, assignment column < goldilocksP) →
      ∀ column,
        chainWitness assignment rounds challenges base column < goldilocksP
  | [], [], _, residues => residues
  | [], _ :: _, _, residues => residues
  | _ :: _, [], _, residues => residues
  | round :: rounds, challenge :: challenges, base, residues => by
      let afterRound :=
        KHornerHonest.hornerWitness assignment challenge base
          round.coefficients 0
      exact chainWitness_residues afterRound rounds challenges
        (base + 3 * degree)
        (KHornerHonest.hornerWitness_residues assignment challenge base
          round.coefficients 0 residues)

private theorem satisfies_append
    {left right : List Row} {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

theorem equalityRows_below
    {left right : Carried} {base : Nat}
    (basePositive : 0 < base)
    (leftBelow : CarriedBelow left base)
    (rightBelow : CarriedBelow right base)
    (row : Row) (member : row ∈ KEquality.rows left right)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column < base := by
  rcases KEquality.rows_conservation left right row member column mentioned with
    rfl | low | high | low | high
  · exact basePositive
  · exact leftBelow.1 column low
  · exact leftBelow.2 column high
  · exact rightBelow.1 column low
  · exact rightBelow.2 column high

private theorem fixedPolynomial_eq_of_coefficients_eq
    {degree : Nat} {left right : FixedPolynomial K degree}
    (equal : left.coefficients = right.coefficients) :
    left = right := by
  cases left with
  | mk leftCoefficients leftLength =>
      cases right with
      | mk rightCoefficients rightLength =>
          simp only at equal
          subst rightCoefficients
          rfl

/-- **Honest completeness of the full fixed-phase chain.**

The source-placement hypotheses contain only physical column bounds.  The
semantic hypothesis is exactly the frozen `FixedPhase.Chain`; no row equation
or acceptance conclusion is supplied separately. -/
theorem chainWitness_satisfies
    {degree : Nat}
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1) :
    ∀
      (current : Carried)
      (rounds : List (Round degree))
      (challenges : List Carried)
      (terminal : Carried)
      (base : Nat),
      0 < base →
      CarriedBelow current base →
      (∀ round ∈ rounds, RoundBelow round base) →
      (∀ challenge ∈ challenges, CarriedBelow challenge base) →
      CarriedBelow terminal base →
      FixedPhase.Chain sumCheckOps
        (decodeCarried assignment current)
        (rounds.map fun round => round.polynomial assignment)
        (challenges.map (decodeCarried assignment))
        (decodeCarried assignment terminal) →
      Satisfies (chainRows current rounds challenges terminal base)
        (chainWitness assignment rounds challenges base)
  | current, [], [], terminal, base, _, _, _, _, _, chain => by
      have pairEqual := congrArg KBridge.toPair chain
      simp only [toPair_decodeCarried, carriedValue, Pair.mk.injEq] at pairEqual
      exact KEquality.rows_complete assignment current terminal constantWire
        pairEqual.1 pairEqual.2
  | _, [], _ :: _, _, _, _, _, _, _, _, chain => by
      simp [FixedPhase.Chain] at chain
  | _, _ :: _, [], _, _, _, _, _, _, _, chain => by
      simp [FixedPhase.Chain] at chain
  | current, round :: rounds, challenge :: challenges, terminal, base,
      basePositive, currentBelow, roundsBelow, challengesBelow, terminalBelow,
      chain => by
      simp only [List.map_cons, FixedPhase.Chain] at chain
      let afterRound :=
        KHornerHonest.hornerWitness assignment challenge base
          round.coefficients 0
      let nextBase := base + 3 * degree
      let nextCurrent :=
        hornerCarried challenge (KFrames.frameAt base)
          round.coefficients 0
      have roundBelow : RoundBelow round base :=
        roundsBelow round (by simp)
      have challengeBelow : CarriedBelow challenge base :=
        challengesBelow challenge (by simp)
      have nextCurrentBelow : CarriedBelow nextCurrent nextBase := by
        exact hornerCarried_below_next round challenge roundBelow
      have afterRoundConstant : afterRound 0 = 1 := by
        rw [show afterRound 0 = assignment 0 by
          exact KHornerHonest.hornerWitness_off_block assignment challenge base
            round.coefficients 0 0 (by simpa [nextBase] using basePositive)]
        exact constantWire
      have hornerSatisfied :
          Satisfies
            (hornerRows challenge (KFrames.frameAt base)
              round.coefficients 0)
            afterRound := by
        exact KHornerHonest.hornerWitness_satisfies assignment challenge base
          challengeBelow.1 challengeBelow.2 round.coefficients 0 roundBelow
      have nextChain :
          FixedPhase.Chain sumCheckOps
            (decodeCarried afterRound nextCurrent)
            (rounds.map fun next => next.polynomial afterRound)
            (challenges.map (decodeCarried afterRound))
            (decodeCarried afterRound terminal) := by
        have preserveSource :
            ∀ value : Carried, CarriedBelow value base →
              decodeCarried afterRound value =
                decodeCarried assignment value := by
          intro value below
          apply KBridge.toPair_injective
          simp only [toPair_decodeCarried, carriedValue, Pair.mk.injEq]
          constructor <;>
            apply KMulHonest.lcEval_congr <;>
            intro column mentioned
          · exact KHornerHonest.hornerWitness_off_block assignment challenge
              base round.coefficients 0 column (below.1 column mentioned)
          · exact KHornerHonest.hornerWitness_off_block assignment challenge
              base round.coefficients 0 column (below.2 column mentioned)
        have roundPolynomialEqual :
            round.polynomial afterRound =
              round.polynomial assignment := by
          apply fixedPolynomial_eq_of_coefficients_eq
          simp only [Round.polynomial]
          apply List.map_congr_left
          intro coefficient member
          exact preserveSource coefficient (roundBelow coefficient member)
        have currentEqual :=
          horner_decodes afterRound round challenge base hornerSatisfied
        rw [roundPolynomialEqual,
          preserveSource challenge challengeBelow] at currentEqual
        rw [currentEqual]
        have tailRounds :
            rounds.map (fun next => next.polynomial afterRound) =
              rounds.map (fun next => next.polynomial assignment) := by
          apply List.map_congr_left
          intro next member
          apply fixedPolynomial_eq_of_coefficients_eq
          simp only [Round.polynomial]
          apply List.map_congr_left
          intro coefficient coefficientMember
          exact preserveSource coefficient
            (roundsBelow next (List.mem_cons_of_mem _ member)
              coefficient coefficientMember)
        have tailChallenges :
            challenges.map (decodeCarried afterRound) =
              challenges.map (decodeCarried assignment) := by
          apply List.map_congr_left
          intro next member
          exact preserveSource next
            (challengesBelow next (List.mem_cons_of_mem _ member))
        rw [tailRounds, tailChallenges,
          preserveSource terminal terminalBelow]
        exact chain.2
      have recursiveSatisfied :
          Satisfies
            (chainRows nextCurrent rounds challenges terminal nextBase)
            (chainWitness afterRound rounds challenges nextBase) := by
        apply chainWitness_satisfies afterRound afterRoundConstant
          nextCurrent rounds challenges terminal nextBase
        · unfold nextBase
          omega
        · exact nextCurrentBelow
        · intro next member
          exact fun coefficient coefficientMember =>
            carriedBelow_mono
              (roundsBelow next (List.mem_cons_of_mem _ member)
                coefficient coefficientMember)
              (by unfold nextBase; omega)
        · intro next member
          exact carriedBelow_mono
            (challengesBelow next (List.mem_cons_of_mem _ member))
            (by unfold nextBase; omega)
        · exact carriedBelow_mono terminalBelow (by unfold nextBase; omega)
        · exact nextChain
      have finalAssignment :
          chainWitness assignment (round :: rounds)
              (challenge :: challenges) base =
            chainWitness afterRound rounds challenges nextBase := by
        rfl
      have equalityAtSource :
          Satisfies
            (KEquality.rows current (roundInitial round.coefficients))
            assignment := by
        have decodedEqual :
            decodeCarried assignment current =
              decodeCarried assignment
                (roundInitial round.coefficients) := by
          rw [decodeCarried_roundInitial round assignment]
          exact chain.1
        have pairEqual := congrArg KBridge.toPair decodedEqual
        simp only [toPair_decodeCarried, carriedValue, Pair.mk.injEq] at pairEqual
        exact KEquality.rows_complete assignment current
          (roundInitial round.coefficients) constantWire
          pairEqual.1 pairEqual.2
      have equalityFinal :
          Satisfies
            (KEquality.rows current (roundInitial round.coefficients))
            (chainWitness afterRound rounds challenges nextBase) := by
        refine satisfies_extend _ assignment
          (chainWitness afterRound rounds challenges nextBase) ?_
          equalityAtSource
        intro row member column mentioned
        have belowBase :=
          equalityRows_below basePositive currentBelow
            (roundInitial_below round roundBelow)
            row member column mentioned
        have afterEqual :
            afterRound column = assignment column := by
          exact KHornerHonest.hornerWitness_off_block assignment challenge base
            round.coefficients 0 column (by simpa using belowBase)
        have finalEqual :
            chainWitness afterRound rounds challenges nextBase column =
              afterRound column :=
          chainWitness_off_block afterRound rounds challenges nextBase
            column (Nat.lt_of_lt_of_le belowBase
              (by unfold nextBase; omega))
        exact afterEqual.symm.trans finalEqual.symm
      have hornerFinal :
          Satisfies
            (hornerRows challenge (KFrames.frameAt base)
              round.coefficients 0)
            (chainWitness afterRound rounds challenges nextBase) := by
        refine satisfies_extend _ afterRound
          (chainWitness afterRound rounds challenges nextBase) ?_
          hornerSatisfied
        intro row member column mentioned
        exact (chainWitness_off_block afterRound rounds challenges nextBase
          column
          (hornerRows_below_next round challenge challengeBelow roundBelow
            row member column mentioned)).symm
      rw [finalAssignment]
      exact satisfies_append
        (satisfies_append equalityFinal hornerFinal)
        recursiveSatisfied

end Nightstream.Implementation.R1CS.Canonical.KFixedPhaseSumCheckHonest
