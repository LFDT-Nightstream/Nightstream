import Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity

/-!
Contract: honest completeness for the projected quotient identity.

Owns: the witness constructions and the proof that an honest execution
satisfies the emitted rows.

Continues `KQuotientIdentity`'s namespace rather than opening a new one: this is
a file split for size, not a responsibility boundary — the row program and its
witness are one recipe.

Also owns the whole-program assembly: `identityRows_honest`. The equality part
is the one that is different in kind — its rows are equalities rather than
writes, so satisfying them is a fact about the *values*, supplied by the
caller's honest identity and transported by `projected_preserved`.

Does **not** own: the decoder that turns a frozen `ProjectionTrace` into the
`Carried` lists these theorems take. Until that exists the recipe is complete
about *its own* inputs rather than about the frozen trace's.

## Why every proof here consumes conservation

Each witness extension has to miss everything the earlier rows mention. That
bound is exactly what `KQuotientIdentity`'s conservation theorems state, in the
interval form they were deliberately written in, so no placement arithmetic is
re-derived here.
-/

set_option autoImplicit false
-- Matches `KQuotientIdentity`: the placement arithmetic rewrites through
-- 321-column atom blocks.
set_option maxRecDepth 8000

namespace Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-! ## Honest completeness, for the atom

Built left to right: the left evaluation's witness, extended by the right
evaluation's, extended by the product's. Each extension writes strictly to the
right of everything the previous rows mention, which is exactly what
conservation established — so completeness consumes the conservation proof
rather than repeating its arithmetic. -/

/-- **The honest witness for one atom.** -/
def atomWitness (z : Nat → Nat) (beta : Carried) (atomBase : Nat)
    (left right : List Carried) : Nat → Nat :=
  KMulHonest.witness
    (KHornerHonest.hornerWitness
      (KHornerHonest.hornerWitness z beta atomBase left 0)
      beta (atomBase + projectionWidth) right 0)
    (hornerCarried beta (KFrames.frameAt atomBase) left 0)
    (hornerCarried beta (KFrames.frameAt (atomBase + projectionWidth)) right 0)
    (KFrames.frameAt (atomBase + 2 * projectionWidth) 0)

/-- **An honest execution satisfies the atom.** -/
theorem productRows_honest
    (z : Nat → Nat) (beta : Carried) (atomBase : Nat)
    (left right : List Carried)
    (leftSized : left.length = 54) (rightSized : right.length = 54)
    (betaLow : KHornerHonest.BelowBase beta.low atomBase)
    (betaHigh : KHornerHonest.BelowBase beta.high atomBase)
    (leftBelow : ∀ c ∈ left, KHornerHonest.BelowBase c.low atomBase
      ∧ KHornerHonest.BelowBase c.high atomBase)
    (rightBelow : ∀ c ∈ right, KHornerHonest.BelowBase c.low atomBase
      ∧ KHornerHonest.BelowBase c.high atomBase) :
    Satisfies (productRows beta atomBase (atomBase + projectionWidth)
        (KFrames.frameAt (atomBase + 2 * projectionWidth) 0) left right)
      (atomWitness z beta atomBase left right) := by
  have widthPos : projectionWidth = 159 := rfl
  -- the two blocks' rows, bounded by conservation
  have blockBelow : ∀ (blockBase : Nat) (coefficients : List Carried),
      coefficients.length = 54 →
      (∀ c ∈ coefficients, KHornerHonest.BelowBase c.low atomBase
        ∧ KHornerHonest.BelowBase c.high atomBase) →
      atomBase ≤ blockBase →
      ∀ row ∈ hornerRows beta (KFrames.frameAt blockBase) coefficients 0,
        ∀ column, (Mentions row.a column ∨ Mentions row.b column
          ∨ Mentions row.c column) →
        column < blockBase + projectionWidth := by
    intro blockBase coefficients sizedC below placed row member column mentioned
    rcases hornerBlock_conservation beta blockBase coefficients row member column
      mentioned with interval | shared
    · have upper := interval.2
      rw [sizedC] at upper
      rw [widthPos]
      omega
    · rcases shared with (bl | bh) | ⟨c, memberC, inC⟩
      · have := betaLow column bl
        rw [widthPos]; omega
      · have := betaHigh column bh
        rw [widthPos]; omega
      · rcases inC with cl | ch
        · have := (below c memberC).1 column cl
          rw [widthPos]; omega
        · have := (below c memberC).2 column ch
          rw [widthPos]; omega
  -- the carried results, bounded by conservation
  have carrBelow : ∀ (blockBase : Nat) (coefficients : List Carried),
      coefficients.length = 54 →
      (∀ c ∈ coefficients, KHornerHonest.BelowBase c.low atomBase
        ∧ KHornerHonest.BelowBase c.high atomBase) →
      atomBase ≤ blockBase → blockBase + projectionWidth
        ≤ atomBase + 2 * projectionWidth →
      ∀ column,
        (Mentions (hornerCarried beta (KFrames.frameAt blockBase)
            coefficients 0).low column
          ∨ Mentions (hornerCarried beta (KFrames.frameAt blockBase)
              coefficients 0).high column) →
        column < atomBase + 2 * projectionWidth := by
    intro blockBase coefficients sizedC below placed room column mentioned
    rcases hornerCarried_conservation beta blockBase coefficients column
      mentioned with interval | ⟨c, memberC, inC⟩
    · have upper := interval.2
      rw [sizedC] at upper
      rw [widthPos] at room ⊢
      omega
    · rcases inC with cl | ch
      · have := (below c memberC).1 column cl
        rw [widthPos]; omega
      · have := (below c memberC).2 column ch
        rw [widthPos]; omega
  have betaShift : KHornerHonest.BelowBase beta.low (atomBase + projectionWidth)
      ∧ KHornerHonest.BelowBase beta.high (atomBase + projectionWidth) :=
    ⟨fun column m => by have := betaLow column m; omega,
     fun column m => by have := betaHigh column m; omega⟩
  have rightShift : ∀ c ∈ right,
      KHornerHonest.BelowBase c.low (atomBase + projectionWidth)
        ∧ KHornerHonest.BelowBase c.high (atomBase + projectionWidth) :=
    fun c memberC =>
      ⟨fun column m => by have := (rightBelow c memberC).1 column m; omega,
       fun column m => by have := (rightBelow c memberC).2 column m; omega⟩
  -- the three pieces
  have leftSat := KHornerHonest.hornerWitness_satisfies z beta atomBase betaLow
    betaHigh left 0 leftBelow
  have rightSat := KHornerHonest.hornerWitness_satisfies
    (KHornerHonest.hornerWitness z beta atomBase left 0) beta
    (atomBase + projectionWidth) betaShift.1 betaShift.2 right 0 rightShift
  have leftUnderRight := KHornerSupport.satisfies_extend _ _ _
    (fun row member column mentioned =>
      (KHornerHonest.hornerWitness_off_block _ beta (atomBase + projectionWidth)
        right 0 column (by
          have := blockBelow atomBase left leftSized leftBelow (Nat.le_refl _)
            row member column mentioned
          omega)).symm)
    leftSat
  have frameFresh : ∀ (comb : LinComb),
      (∀ column, Mentions comb column → column < atomBase + 2 * projectionWidth) →
      KMulHonest.Fresh comb
        (KFrames.frameAt (atomBase + 2 * projectionWidth) 0) :=
    fun comb below => KHornerHonest.fresh_of_belowBase comb
      (atomBase + 2 * projectionWidth) 0 below
  have preserve : ∀ (program : List Row),
      (∀ row ∈ program, ∀ column, (Mentions row.a column ∨ Mentions row.b column
        ∨ Mentions row.c column) → column < atomBase + 2 * projectionWidth) →
      Satisfies program (KHornerHonest.hornerWitness
        (KHornerHonest.hornerWitness z beta atomBase left 0) beta
        (atomBase + projectionWidth) right 0) →
      Satisfies program (atomWitness z beta atomBase left right) := by
    intro program bounded satisfied
    refine KHornerSupport.satisfies_extend _ _ _
      (fun row member column mentioned => ?_) satisfied
    have below := bounded row member column mentioned
    exact (KMulHonest.witness_off_frame _ _ _ _ column
      (by simp only [KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)
      (by simp only [KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)
      (by simp only [KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)).symm
  intro row member
  unfold productRows at member
  simp only [List.mem_append] at member
  rcases member with (inLeft | inRight) | inMul
  · exact preserve _ (fun r m column mentioned => by
      have := blockBelow atomBase left leftSized leftBelow (Nat.le_refl _) r m
        column mentioned
      rw [widthPos] at this ⊢
      omega) leftUnderRight row inLeft
  · exact preserve _ (fun r m column mentioned => by
      have := blockBelow (atomBase + projectionWidth) right rightSized rightBelow
        (by omega) r m column mentioned
      omega) rightSat row inRight
  · exact KMulHonest.witness_satisfies _ _ _ _
      (KMulHonest.canonical_distinct (atomBase + 2 * projectionWidth) 0)
      (frameFresh _ (fun column m => carrBelow atomBase left leftSized leftBelow
        (Nat.le_refl _) (by omega) column (Or.inl m)))
      (frameFresh _ (fun column m => carrBelow atomBase left leftSized leftBelow
        (Nat.le_refl _) (by omega) column (Or.inr m)))
      (frameFresh _ (fun column m => carrBelow (atomBase + projectionWidth) right
        rightSized rightBelow (by omega) (by omega) column (Or.inl m)))
      (frameFresh _ (fun column m => carrBelow (atomBase + projectionWidth) right
        rightSized rightBelow (by omega) (by omega) column (Or.inr m)))
      row inMul

/-- **An atom's witness writes only inside its atom.**  The composition
direction, as `hornerWitness_off_block` is for one block. -/
theorem atomWitness_off_block
    (z : Nat → Nat) (beta : Carried) (atomBase : Nat)
    (left right : List Carried) (column : Nat) (below : column < atomBase) :
    atomWitness z beta atomBase left right column = z column := by
  unfold atomWitness
  rw [KMulHonest.witness_off_frame _ _ _ _ column
      (by simp only [KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)
      (by simp only [KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)
      (by simp only [KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega),
    KHornerHonest.hornerWitness_off_block _ beta (atomBase + projectionWidth)
      right 0 column (by omega),
    KHornerHonest.hornerWitness_off_block z beta atomBase left 0 column
      (by omega)]

/-- **The honest witness for the left-hand side**, one atom at a time, left to
right. -/
def pairsWitness (beta : Carried) :
    (Nat → Nat) → List (List Carried × List Carried) → Nat → (Nat → Nat)
  | z, [], _ => z
  | z, (left, right) :: rest, base =>
      pairsWitness beta (atomWitness z beta base left right) rest
        (base + atomWidth)

/-- **The left-hand side's witness writes only inside the atoms' block.** -/
theorem pairsWitness_off_block (beta : Carried) :
    ∀ (pairs : List (List Carried × List Carried)) (base : Nat)
      (z : Nat → Nat) (column : Nat), column < base →
      pairsWitness beta z pairs base column = z column
  | [], _, _, _, _ => rfl
  | (left, right) :: rest, base, z, column, below => by
      show pairsWitness beta (atomWitness z beta base left right) rest
        (base + atomWidth) column = z column
      rw [pairsWitness_off_block beta rest (base + atomWidth) _ column (by omega),
        atomWitness_off_block z beta base left right column below]

/-- **An honest execution satisfies the whole left-hand side.**

Each atom's witness is extended by the atoms to its right, and every such
extension writes at a strictly larger column than anything the earlier atom's
rows mention — which `productRows_conservation` already established. -/
theorem pairsRows_honest (beta : Carried) :
    ∀ (pairs : List (List Carried × List Carried)) (base : Nat) (z : Nat → Nat),
      (∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54) →
      KHornerHonest.BelowBase beta.low base →
      KHornerHonest.BelowBase beta.high base →
      (∀ pair ∈ pairs,
        (∀ c ∈ pair.1, KHornerHonest.BelowBase c.low base
          ∧ KHornerHonest.BelowBase c.high base)
        ∧ (∀ c ∈ pair.2, KHornerHonest.BelowBase c.low base
          ∧ KHornerHonest.BelowBase c.high base)) →
      Satisfies (pairsRows beta pairs base) (pairsWitness beta z pairs base)
  | [], _, _, _, _, _, _ => by intro row member; simp [pairsRows] at member
  | (left, right) :: rest, base, z, sized, betaLow, betaHigh, below => by
      have head : left.length = 54 ∧ right.length = 54 :=
        sized (left, right) (by simp)
      have headBelow := below (left, right) (by simp)
      have headBounded : ∀ row ∈ productRows beta base (base + projectionWidth)
          (KFrames.frameAt (base + 2 * projectionWidth) 0) left right,
          ∀ column, (Mentions row.a column ∨ Mentions row.b column
            ∨ Mentions row.c column) → column < base + atomWidth := by
        intro row member column mentioned
        rcases productRows_conservation beta base left right head.1 head.2 row
          member column mentioned with interval | shared | shared
        · exact interval.2
        · rcases shared with (bl | bh) | ⟨c, memberC, inC⟩
          · have := betaLow column bl; omega
          · have := betaHigh column bh; omega
          · rcases inC with cl | ch
            · have := (headBelow.1 c memberC).1 column cl; omega
            · have := (headBelow.1 c memberC).2 column ch; omega
        · rcases shared with (bl | bh) | ⟨c, memberC, inC⟩
          · have := betaLow column bl; omega
          · have := betaHigh column bh; omega
          · rcases inC with cl | ch
            · have := (headBelow.2 c memberC).1 column cl; omega
            · have := (headBelow.2 c memberC).2 column ch; omega
      intro row member
      rw [show pairsRows beta ((left, right) :: rest) base
          = productRows beta base (base + projectionWidth)
              (KFrames.frameAt (base + 2 * projectionWidth) 0) left right
            ++ pairsRows beta rest (base + atomWidth) from rfl,
        List.mem_append] at member
      rcases member with inHead | inTail
      · exact KHornerSupport.satisfies_extend _ _ _
          (fun r m column mentioned =>
            (pairsWitness_off_block beta rest (base + atomWidth) _ column
              (headBounded r m column mentioned)).symm)
          (productRows_honest z beta base left right head.1 head.2 betaLow
            betaHigh headBelow.1 headBelow.2)
          row inHead
      · exact pairsRows_honest beta rest (base + atomWidth)
          (atomWitness z beta base left right)
          (fun pair pairMember => sized pair (List.mem_cons_of_mem _ pairMember))
          (fun column m => by have := betaLow column m; omega)
          (fun column m => by have := betaHigh column m; omega)
          (fun pair pairMember =>
            ⟨fun c memberC =>
              ⟨fun column m => by
                have := ((below pair (List.mem_cons_of_mem _ pairMember)).1 c
                  memberC).1 column m
                omega,
               fun column m => by
                have := ((below pair (List.mem_cons_of_mem _ pairMember)).1 c
                  memberC).2 column m
                omega⟩,
             fun c memberC =>
              ⟨fun column m => by
                have := ((below pair (List.mem_cons_of_mem _ pairMember)).2 c
                  memberC).1 column m
                omega,
               fun column m => by
                have := ((below pair (List.mem_cons_of_mem _ pairMember)).2 c
                  memberC).2 column m
                omega⟩⟩)
          row inTail

/-! ## The whole-program witness

The five writing parts, left to right. The `K`-equality writes nothing — it is
the one part whose satisfaction is a fact about the *values*, not about where
the witness put them, which is why it needs the honest identity rather than
another placement lemma. -/

/-- **The honest witness for the whole check.** -/
def identityWitness (z : Nat → Nat) (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried) : Nat → Nat :=
  KMulHonest.witness
    (KHornerHonest.hornerWitness
      (KHornerHonest.hornerWitness
        (KHornerHonest.hornerWitness (pairsWitness beta z pairs base) beta
          (outBase base pairs.length) output 0)
        beta (quotientBase base pairs.length) quotient 0)
      beta (modulusBase base pairs.length) modulus 0)
    (hornerCarried beta (KFrames.frameAt (quotientBase base pairs.length))
      quotient 0)
    (hornerCarried beta (KFrames.frameAt (modulusBase base pairs.length))
      modulus 0)
    (productFrameAt base pairs.length)

/-- **The whole witness writes only inside the program's block.**  Every input
the caller placed below `base` keeps its value, which is what lets the honest
identity be stated about the caller's assignment rather than about the
witness. -/
theorem identityWitness_off_block
    (z : Nat → Nat) (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried) (column : Nat) (below : column < base) :
    identityWitness z beta base pairs output quotient modulus column
      = z column := by
  have layout : base ≤ outBase base pairs.length
      ∧ outBase base pairs.length ≤ quotientBase base pairs.length
      ∧ quotientBase base pairs.length ≤ modulusBase base pairs.length := by
    simp only [outBase, quotientBase, modulusBase, projectionWidth, quotientWidth]
    omega
  unfold identityWitness
  rw [KMulHonest.witness_off_frame _ _ _ _ column
      (by simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame, modulusWidth]; omega)
      (by simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame, modulusWidth]; omega)
      (by simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame, modulusWidth]; omega),
    KHornerHonest.hornerWitness_off_block _ beta
      (modulusBase base pairs.length) modulus 0 column (by omega),
    KHornerHonest.hornerWitness_off_block _ beta
      (quotientBase base pairs.length) quotient 0 column (by omega),
    KHornerHonest.hornerWitness_off_block _ beta
      (outBase base pairs.length) output 0 column (by omega),
    pairsWitness_off_block beta pairs base z column below]

/-- **A projection is unchanged by a witness that stays above the inputs.**
What lets the honest identity be a hypothesis about `z` rather than about the
constructed assignment. -/
theorem projected_preserved
    (z w : Nat → Nat) (beta : Carried) (base : Nat)
    (agree : ∀ column, column < base → w column = z column)
    (betaLow : KHornerHonest.BelowBase beta.low base)
    (betaHigh : KHornerHonest.BelowBase beta.high base)
    (coefficients : List Carried)
    (below : ∀ c ∈ coefficients, KHornerHonest.BelowBase c.low base
      ∧ KHornerHonest.BelowBase c.high base) :
    projected w beta coefficients = projected z beta coefficients := by
  have carr : ∀ x : Carried, KHornerHonest.BelowBase x.low base →
      KHornerHonest.BelowBase x.high base →
      carriedValue w x = carriedValue z x := by
    intro x lowB highB
    unfold carriedValue
    simp only [Pair.mk.injEq]
    exact ⟨KMulHonest.lcEval_congr w z x.low (fun col m => agree col (lowB col m)),
      KMulHonest.lcEval_congr w z x.high (fun col m => agree col (highB col m))⟩
  unfold projected
  rw [carr beta betaLow betaHigh,
    List.map_congr_left (fun c memberC =>
      carr c (below c memberC).1 (below c memberC).2)]

/-! ## The assembly

Five satisfaction facts and ten preservation steps. Every preservation step is
`KHornerSupport.satisfies_extend` fed by two things that already exist: a bound
on what the part's rows mention (conservation) and an agreement below the next
block's base (`hornerWitness_off_block`, `witness_off_frame`). Because the bases
chain, an earlier part's bound clears every later stage at once.

The sixth part is different in kind: `KEquality`'s rows are equalities, not
writes, so satisfying them is a fact about the *values*. That fact is the
caller's honest identity, transported to the witness by
`projected_preserved`. -/

/-- **An honest execution satisfies the whole check.**

`basePositive` is not bookkeeping: the constant wire is column 0, so a program
based at 0 would allocate over it. -/
theorem identityRows_honest
    (z : Nat → Nat) (beta : Carried) (base : Nat)
    (pairs : List (List Carried × List Carried))
    (output quotient modulus : List Carried)
    (basePositive : 0 < base) (constantWire : z 0 = 1)
    (sized : ∀ pair ∈ pairs, pair.1.length = 54 ∧ pair.2.length = 54)
    (outputSized : output.length = 54) (quotientSized : quotient.length = 53)
    (modulusSized : modulus.length = 55)
    (betaLow : KHornerHonest.BelowBase beta.low base)
    (betaHigh : KHornerHonest.BelowBase beta.high base)
    (pairsBelow : ∀ pair ∈ pairs,
      (∀ c ∈ pair.1, KHornerHonest.BelowBase c.low base
        ∧ KHornerHonest.BelowBase c.high base)
      ∧ (∀ c ∈ pair.2, KHornerHonest.BelowBase c.low base
        ∧ KHornerHonest.BelowBase c.high base))
    (outputBelow : ∀ c ∈ output, KHornerHonest.BelowBase c.low base
      ∧ KHornerHonest.BelowBase c.high base)
    (quotientBelow : ∀ c ∈ quotient, KHornerHonest.BelowBase c.low base
      ∧ KHornerHonest.BelowBase c.high base)
    (modulusBelow : ∀ c ∈ modulus, KHornerHonest.BelowBase c.low base
      ∧ KHornerHonest.BelowBase c.high base)
    (honest : pairSum (pairs.map (fun pair =>
        mulPair (projected z beta pair.1) (projected z beta pair.2)))
      = addPair (projected z beta output)
          (mulPair (projected z beta quotient) (projected z beta modulus))) :
    Satisfies (identityRows beta base pairs output quotient modulus)
      (identityWitness z beta base pairs output quotient modulus) := by
  obtain ⟨outEq, quotientEq, modulusEq⟩ := layout_bases base pairs.length
  have widthEq : atomWidth = 321 := rfl
  have weaken : ∀ (comb : LinComb) (higher : Nat), base ≤ higher →
      KHornerHonest.BelowBase comb base → KHornerHonest.BelowBase comb higher :=
    fun comb higher le below column m => Nat.lt_of_lt_of_le (below column m) le
  have leOut : base ≤ outBase base pairs.length := by rw [outEq]; omega
  have leQuotient : base ≤ quotientBase base pairs.length := by
    rw [quotientEq]; omega
  have leModulus : base ≤ modulusBase base pairs.length := by rw [modulusEq]; omega
  -- the five stage witnesses satisfy their own parts
  have satA := pairsRows_honest beta pairs base z sized betaLow betaHigh pairsBelow
  have satB := KHornerHonest.hornerWitness_satisfies
    (pairsWitness beta z pairs base) beta (outBase base pairs.length)
    (weaken _ _ leOut betaLow) (weaken _ _ leOut betaHigh) output 0
    (fun c m => ⟨weaken _ _ leOut (outputBelow c m).1,
      weaken _ _ leOut (outputBelow c m).2⟩)
  have satC := KHornerHonest.hornerWitness_satisfies
    (KHornerHonest.hornerWitness (pairsWitness beta z pairs base) beta
      (outBase base pairs.length) output 0)
    beta (quotientBase base pairs.length)
    (weaken _ _ leQuotient betaLow) (weaken _ _ leQuotient betaHigh) quotient 0
    (fun c m => ⟨weaken _ _ leQuotient (quotientBelow c m).1,
      weaken _ _ leQuotient (quotientBelow c m).2⟩)
  have satD := KHornerHonest.hornerWitness_satisfies
    (KHornerHonest.hornerWitness
      (KHornerHonest.hornerWitness (pairsWitness beta z pairs base) beta
        (outBase base pairs.length) output 0)
      beta (quotientBase base pairs.length) quotient 0)
    beta (modulusBase base pairs.length)
    (weaken _ _ leModulus betaLow) (weaken _ _ leModulus betaHigh) modulus 0
    (fun c m => ⟨weaken _ _ leModulus (modulusBelow c m).1,
      weaken _ _ leModulus (modulusBelow c m).2⟩)
  -- what each part may mention
  have bA : ∀ row ∈ pairsRows beta pairs base, ∀ column,
      (Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column) →
      column < outBase base pairs.length := by
    intro row member column mentioned
    rcases pairsRows_conservation beta pairs base sized row member column
      mentioned with interval | ⟨pair, pairMember, shared⟩
    · have upper := interval.2
      rw [widthEq] at upper
      rw [outEq]
      omega
    · rcases shared with ((bl | bh) | ⟨c, memberC, inC⟩) | ((bl | bh) | ⟨c, memberC, inC⟩)
      · have := betaLow column bl; omega
      · have := betaHigh column bh; omega
      · rcases inC with cl | ch
        · have := ((pairsBelow pair pairMember).1 c memberC).1 column cl; omega
        · have := ((pairsBelow pair pairMember).1 c memberC).2 column ch; omega
      · have := betaLow column bl; omega
      · have := betaHigh column bh; omega
      · rcases inC with cl | ch
        · have := ((pairsBelow pair pairMember).2 c memberC).1 column cl; omega
        · have := ((pairsBelow pair pairMember).2 c memberC).2 column ch; omega
  have blockBound : ∀ (blockBase : Nat) (coefficients : List Carried) (width : Nat),
      3 * (coefficients.length - 1) = width →
      (∀ c ∈ coefficients, KHornerHonest.BelowBase c.low base
        ∧ KHornerHonest.BelowBase c.high base) →
      base ≤ blockBase →
      ∀ row ∈ hornerRows beta (KFrames.frameAt blockBase) coefficients 0,
        ∀ column, (Mentions row.a column ∨ Mentions row.b column
          ∨ Mentions row.c column) → column < blockBase + width := by
    intro blockBase coefficients width sizedC below placed row member column mentioned
    rcases hornerBlock_conservation beta blockBase coefficients row member column
      mentioned with interval | shared
    · have upper := interval.2
      rw [sizedC] at upper
      omega
    · rcases shared with (bl | bh) | ⟨c, memberC, inC⟩
      · have := betaLow column bl; omega
      · have := betaHigh column bh; omega
      · rcases inC with cl | ch
        · have := (below c memberC).1 column cl; omega
        · have := (below c memberC).2 column ch; omega
  -- the four cumulative agreements
  have aE : ∀ column, column < modulusBase base pairs.length + modulusWidth →
      identityWitness z beta base pairs output quotient modulus column
        = KHornerHonest.hornerWitness
            (KHornerHonest.hornerWitness
              (KHornerHonest.hornerWitness (pairsWitness beta z pairs base) beta
                (outBase base pairs.length) output 0)
              beta (quotientBase base pairs.length) quotient 0)
            beta (modulusBase base pairs.length) modulus 0 column := by
    intro column below
    unfold identityWitness
    exact KMulHonest.witness_off_frame _ _ _ _ column
      (by simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)
      (by simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)
      (by simp only [productFrameAt, KFrames.frameAt, KFrames.frameColumn,
            KFrames.columnsPerFrame]; omega)
  have aD : ∀ column, column < modulusBase base pairs.length →
      identityWitness z beta base pairs output quotient modulus column
        = KHornerHonest.hornerWitness
            (KHornerHonest.hornerWitness (pairsWitness beta z pairs base) beta
              (outBase base pairs.length) output 0)
            beta (quotientBase base pairs.length) quotient 0 column := by
    intro column below
    rw [aE column (by simp only [modulusWidth]; omega),
      KHornerHonest.hornerWitness_off_block _ beta
        (modulusBase base pairs.length) modulus 0 column (by omega)]
  have aC : ∀ column, column < quotientBase base pairs.length →
      identityWitness z beta base pairs output quotient modulus column
        = KHornerHonest.hornerWitness (pairsWitness beta z pairs base) beta
            (outBase base pairs.length) output 0 column := by
    intro column below
    rw [aD column (by rw [modulusEq]; rw [quotientEq] at below; omega),
      KHornerHonest.hornerWitness_off_block _ beta
        (quotientBase base pairs.length) quotient 0 column (by omega)]
  have aB : ∀ column, column < outBase base pairs.length →
      identityWitness z beta base pairs output quotient modulus column
        = pairsWitness beta z pairs base column := by
    intro column below
    rw [aC column (by rw [quotientEq]; rw [outEq] at below; omega),
      KHornerHonest.hornerWitness_off_block _ beta
        (outBase base pairs.length) output 0 column (by omega)]
  -- the five parts, all at the final witness
  have satA5 := KHornerSupport.satisfies_extend _ _ _
    (fun r m column mentioned => (aB column (bA r m column mentioned)).symm) satA
  have satB5 := KHornerSupport.satisfies_extend _ _ _
    (fun r m column mentioned => (aC column
      (by have := blockBound (outBase base pairs.length) output projectionWidth
            (by rw [outputSized]; rfl) outputBelow leOut r m column mentioned
          rw [quotientEq]; rw [outEq] at this; simp only [projectionWidth] at this
          omega)).symm) satB
  have satC5 := KHornerSupport.satisfies_extend _ _ _
    (fun r m column mentioned => (aD column
      (by have := blockBound (quotientBase base pairs.length) quotient quotientWidth
            (by rw [quotientSized]; rfl) quotientBelow leQuotient r m column mentioned
          rw [modulusEq]; rw [quotientEq] at this
          simp only [quotientWidth] at this
          omega)).symm) satC
  have satD5 := KHornerSupport.satisfies_extend _ _ _
    (fun r m column mentioned => (aE column
      (by have := blockBound (modulusBase base pairs.length) modulus modulusWidth
            (by rw [modulusSized]; rfl) modulusBelow leModulus r m column mentioned
          omega)).symm) satD
  have carrBound : ∀ (blockBase : Nat) (coefficients : List Carried) (width : Nat),
      3 * (coefficients.length - 1) = width →
      (∀ c ∈ coefficients, KHornerHonest.BelowBase c.low base
        ∧ KHornerHonest.BelowBase c.high base) →
      base ≤ blockBase → ∀ column,
        (Mentions (hornerCarried beta (KFrames.frameAt blockBase)
            coefficients 0).low column
          ∨ Mentions (hornerCarried beta (KFrames.frameAt blockBase)
              coefficients 0).high column) →
        column < blockBase + width := by
    intro blockBase coefficients width sizedC below placed column m
    rcases hornerCarried_conservation beta blockBase coefficients column m with
      interval | ⟨c, memberC, inC⟩
    · have upper := interval.2
      rw [sizedC] at upper
      omega
    · rcases inC with cl | ch
      · have := (below c memberC).1 column cl; omega
      · have := (below c memberC).2 column ch; omega
  have quotientCarr := carrBound (quotientBase base pairs.length) quotient
    quotientWidth (by rw [quotientSized]; rfl) quotientBelow leQuotient
  have modulusCarr := carrBound (modulusBase base pairs.length) modulus
    modulusWidth (by rw [modulusSized]; rfl) modulusBelow leModulus
  have frameFresh : ∀ (comb : LinComb),
      KHornerHonest.BelowBase comb
        (modulusBase base pairs.length + modulusWidth) →
      KMulHonest.Fresh comb (productFrameAt base pairs.length) :=
    fun comb below => KHornerHonest.fresh_of_belowBase comb _ 0 below
  have satE5 : Satisfies (KMul.rows
      (hornerCarried beta (KFrames.frameAt (quotientBase base pairs.length))
        quotient 0)
      (hornerCarried beta (KFrames.frameAt (modulusBase base pairs.length))
        modulus 0)
      (productFrameAt base pairs.length))
      (identityWitness z beta base pairs output quotient modulus) :=
    KMulHonest.witness_satisfies _ _ _ _
      (KMulHonest.canonical_distinct _ 0)
      (frameFresh _ (fun column m => by
        have := quotientCarr column (Or.inl m)
        rw [modulusEq]; rw [quotientEq] at this
        simp only [quotientWidth, modulusWidth] at this ⊢
        omega))
      (frameFresh _ (fun column m => by
        have := quotientCarr column (Or.inr m)
        rw [modulusEq]; rw [quotientEq] at this
        simp only [quotientWidth, modulusWidth] at this ⊢
        omega))
      (frameFresh _ (fun column m => by
        have := modulusCarr column (Or.inl m); omega))
      (frameFresh _ (fun column m => by
        have := modulusCarr column (Or.inr m); omega))
  -- the equality part: a fact about values, not placement
  have preserve : ∀ (coefficients : List Carried),
      (∀ c ∈ coefficients, KHornerHonest.BelowBase c.low base
        ∧ KHornerHonest.BelowBase c.high base) →
      projected (identityWitness z beta base pairs output quotient modulus) beta
          coefficients
        = projected z beta coefficients :=
    fun coefficients below => projected_preserved z _ beta base
      (fun column lt => identityWitness_off_block z beta base pairs output
        quotient modulus column lt) betaLow betaHigh coefficients below
  have leftValue := pairsRows_sound _ beta pairs base satA5
  have outValue := hornerRows_sound _ beta
    (KFrames.frameAt (outBase base pairs.length)) output 0 satB5
  have quotientValue := hornerRows_sound _ beta
    (KFrames.frameAt (quotientBase base pairs.length)) quotient 0 satC5
  have modulusValue := hornerRows_sound _ beta
    (KFrames.frameAt (modulusBase base pairs.length)) modulus 0 satD5
  have productValue := mulRows_sound _ _ _ (productFrameAt base pairs.length) satE5
  have equalValue : carriedValue
      (identityWitness z beta base pairs output quotient modulus)
      (pairsCarried pairs base)
    = carriedValue (identityWitness z beta base pairs output quotient modulus)
        (concatCarried
          (hornerCarried beta (KFrames.frameAt (outBase base pairs.length))
            output 0)
          (productCarried (productFrameAt base pairs.length))) := by
    rw [carriedValue_concat, leftValue, outValue, productValue, quotientValue,
      modulusValue]
    show pairSum (pairs.map (fun pair =>
        mulPair (projected _ beta pair.1) (projected _ beta pair.2)))
      = addPair (projected _ beta output)
          (mulPair (projected _ beta quotient) (projected _ beta modulus))
    rw [preserve output outputBelow, preserve quotient quotientBelow,
      preserve modulus modulusBelow,
      List.map_congr_left (fun pair pairMember => by
        rw [preserve pair.1 (pairsBelow pair pairMember).1,
          preserve pair.2 (pairsBelow pair pairMember).2])]
    exact honest
  intro row member
  simp only [identityRows, List.mem_append] at member
  rcases member with ((((inA | inB) | inC) | inD) | inE) | inF
  · exact satA5 row inA
  · exact satB5 row inB
  · exact satC5 row inC
  · exact satD5 row inD
  · exact satE5 row inE
  · exact KEquality.rows_complete _ _ _
      (by rw [identityWitness_off_block z beta base pairs output quotient modulus
            0 basePositive]; exact constantWire)
      (congrArg Pair.low equalValue) (congrArg Pair.high equalValue) row inF

end Nightstream.Implementation.R1CS.Canonical.KQuotientIdentity
