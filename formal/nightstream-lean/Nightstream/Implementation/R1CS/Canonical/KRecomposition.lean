import Nightstream.Implementation.R1CS.Canonical.KEquality
import Nightstream.Implementation.R1CS.Core.LinearSubstitution
import Nightstream.Implementation.Lowering.Typed.Cost

/-!
Contract: the emitted row program for Π_DEC's radix-`b` recomposition.

Owns: the never-materialized combination `Σ_i b^i · childᵢ`, its agreement with
the verifier's own accumulator loop, the equality rows against the parent, the
derived row count, conservation, soundness, honest completeness, and cost.

## Π_DEC's algebraic core is one shape

`neo_reductions::api::dec::verify_dec_public` is mostly shape guards. Its
algebraic content — the part that is a constraint rather than a decoder-side
length test — is the single relation

```text
parent = Σ_{i < k} b^i · childᵢ
```

instantiated on four different carriers:

| carrier | site in `verify_dec_public` |
|---|---|
| every entry of the public `X` matrix | `split_b_matrix_k(parent.X, k, b)` |
| every lane of every `y_ring` row | the `y_lhs` loop, `t · d_pad` values |
| every lane of `y_zcol` | guarded by `enforce_y_zcol_recomposition` |
| every `aux_openings` entry | the final scalar loop |

So this module is not one of Π_DEC's checks; it is the check that the others
are instances of. What remains outside it: the digit-range constraint on `X`
(`KLowNorm`, since `b = 2` produces balanced digits in `{-1, 0, 1}`), the
`ct = y_ring[j][0]` consistency (`KConsistency`), the `y_zcol` tail zeroing
(`KZeroCheck.paddingRows`), and the Ajtai commitment fold, which is not
arithmetic this layer owns.

## Zero rows for the recomposition itself

`b` is public, so `b^i` are public constants and scaling is a **coefficient
rewrite**, not a multiplication. Nothing is allocated and no product row is
emitted for the sum. The whole cost of a recomposition is the equality against
the parent: two rows, one per extension coordinate, and no columns.

That is worth stating precisely because it is easy to get wrong in the other
direction. A `K`-valued recomposition looks like it needs `k` extension
multiplications; it needs none, because the scalars live in the base field and
act coordinatewise.

## Horner, and why it matches the Rust loop

The combination is built by Horner's rule — `c₀ + b·(c₁ + b·(…))` — which is
what a never-materializing encoder wants, since each level reuses the previous
level's combination instead of tabulating powers.

`verify_dec_public` does not do that. It builds an explicit table with
`p_k *= b_k` and then takes `Σ pow_i · child_i`. Those are the same function but
not the same expression, so `powerSumFrom_eq_hornerValue` proves it rather than
assuming it. Without that theorem the encoding would be checking a relation the
verifier does not check.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KRecomposition

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-! ## Modular plumbing

Two rewrites used throughout: pushing a reduction through a product, and
absorbing a reduction inside a sum. -/

theorem scaled_term_mod (scalar coefficient value : Nat) :
    (scalar * coefficient % goldilocksP) * value % goldilocksP
      = scalar * (coefficient * value) % goldilocksP := by
  rw [Nat.mul_mod, Nat.mod_mod, ← Nat.mul_mod, Nat.mul_assoc]

theorem add_inner_mod (left right : Nat) :
    (left + right % goldilocksP) % goldilocksP = (left + right) % goldilocksP := by
  rw [Nat.add_mod, Nat.mod_mod, ← Nat.add_mod]

theorem mul_inner_mod (left right : Nat) :
    left * (right % goldilocksP) % goldilocksP = left * right % goldilocksP := by
  rw [Nat.mul_mod, Nat.mod_mod, ← Nat.mul_mod]

/-! ## Scaling by a public constant

`LinearSubstitution.scaleTerms` already denotes exactly this rewrite, so it is
reused rather than redefined. What is new here is its effect on `rawSum`. -/

/-- **Scaling is a coefficient rewrite.**  It emits no row and allocates no
column, which is the whole reason a recomposition costs only its equality. -/
theorem rawSum_scaleTerms (z : Nat → Nat) (scalar : Nat) (comb : LinComb) :
    rawSum z (LinearSubstitution.scaleTerms scalar comb) % goldilocksP
      = scalar * rawSum z comb % goldilocksP := by
  induction comb with
  | nil => simp [LinearSubstitution.scaleTerms, rawSum]
  | cons term rest inductionHypothesis =>
      have unfoldCons :
          LinearSubstitution.scaleTerms scalar (term :: rest)
            = (term.1, scalar * term.2 % goldilocksP)
              :: LinearSubstitution.scaleTerms scalar rest := rfl
      rw [unfoldCons, rawSum_cons, rawSum_cons, Nat.add_mod,
        inductionHypothesis, scaled_term_mod, Nat.mul_add, Nat.add_mod,
        Nat.mod_mod]
      simp only [Nat.mod_mod, ← Nat.add_mod]

theorem lcEval_scaleTerms (z : Nat → Nat) (scalar : Nat) (comb : LinComb) :
    lcEval z (LinearSubstitution.scaleTerms scalar comb)
      = scalar * lcEval z comb % goldilocksP := by
  rw [lcEval_eq_rawSum, rawSum_scaleTerms, lcEval_eq_rawSum, mul_inner_mod]

/-! ## The combination

Horner's rule over the children, one coordinate at a time. -/

/-- **The emitted combination.**  `c₀ + b·(c₁ + b·(…))`, built by concatenation
and coefficient rewrites only. -/
def recomposeComb (base : Nat) : List LinComb → LinComb
  | [] => []
  | head :: rest =>
      head ++ LinearSubstitution.scaleTerms base (recomposeComb base rest)

/-- The value the combination denotes, in the same Horner shape. -/
def hornerValue (base : Nat) : List Nat → Nat
  | [] => 0
  | head :: rest => (head + base * hornerValue base rest) % goldilocksP

/-- **The combination denotes the Horner value.**  This is the whole content of
"the recomposition emits no row": the sum already exists as a combination. -/
theorem lcEval_recomposeComb (z : Nat → Nat) (base : Nat) (combs : List LinComb) :
    lcEval z (recomposeComb base combs)
      = hornerValue base (combs.map (lcEval z)) := by
  induction combs with
  | nil => simp [recomposeComb, hornerValue, lcEval]
  | cons head rest inductionHypothesis =>
      rw [recomposeComb, lcEval_append, lcEval_scaleTerms,
        inductionHypothesis, List.map_cons, hornerValue, add_inner_mod]

/-! ## Agreement with the verifier's accumulator loop

`verify_dec_public` carries an explicit power `p_k`, multiplies it into each
child, then scales it by `b`.  That is a different expression from Horner's
rule, so the agreement is proved. -/

/-- The verifier's own shape: an explicit power accumulator, scaled after each
child exactly as `p_k *= b_k`. -/
def powerSumFrom (base power : Nat) : List Nat → Nat
  | [] => 0
  | head :: rest =>
      (power * head
        + powerSumFrom base (power * base % goldilocksP) rest) % goldilocksP

/-- **The accumulator loop and Horner's rule compute the same value.**

Stated from an arbitrary starting power, which is what makes the induction go
through; the verifier's case is `power = 1`. -/
theorem powerSumFrom_eq_hornerValue (base : Nat) (values : List Nat) :
    ∀ power : Nat,
      powerSumFrom base power values = power * hornerValue base values
        % goldilocksP := by
  induction values with
  | nil => intro power; simp [powerSumFrom, hornerValue]
  | cons head rest inductionHypothesis =>
      intro power
      rw [powerSumFrom, inductionHypothesis (power * base % goldilocksP),
        hornerValue, mul_inner_mod, Nat.mul_add, Nat.add_mod,
        scaled_term_mod, ← Nat.add_mod]
      exact add_inner_mod _ _

/-- **The verifier's loop, at its own starting power.** -/
theorem powerSum_one (base : Nat) (values : List Nat) :
    powerSumFrom base 1 values = hornerValue base values % goldilocksP := by
  rw [powerSumFrom_eq_hornerValue, Nat.one_mul]

/-! ## The `K`-valued recomposition

The scalars are base-field constants, so they act coordinatewise and the
recomposition needs no extension multiplication at all. -/

/-- **The recomposed carrier.**  One Horner combination per coordinate. -/
def recompose (base : Nat) (children : List Carried) : Carried where
  low := recomposeComb base (children.map Carried.low)
  high := recomposeComb base (children.map Carried.high)

/-- The pair the recomposition denotes. -/
def hornerPair (base : Nat) : List Pair → Pair
  | [] => ⟨0, 0⟩
  | head :: rest =>
      ⟨(head.low + base * (hornerPair base rest).low) % goldilocksP,
        (head.high + base * (hornerPair base rest).high) % goldilocksP⟩

theorem carriedValue_recompose (z : Nat → Nat) (base : Nat)
    (children : List Carried) :
    carriedValue z (recompose base children)
      = hornerPair base (children.map (carriedValue z)) := by
  induction children with
  | nil => rfl
  | cons child rest inductionHypothesis =>
      have lowStep :
          lcEval z (recompose base (child :: rest)).low
            = (lcEval z child.low
                + base * lcEval z (recompose base rest).low) % goldilocksP := by
        simp only [recompose, List.map_cons, recomposeComb]
        rw [lcEval_append, lcEval_scaleTerms, add_inner_mod]
      have highStep :
          lcEval z (recompose base (child :: rest)).high
            = (lcEval z child.high
                + base * lcEval z (recompose base rest).high) % goldilocksP := by
        simp only [recompose, List.map_cons, recomposeComb]
        rw [lcEval_append, lcEval_scaleTerms, add_inner_mod]
      have restLow : lcEval z (recompose base rest).low
          = (hornerPair base (rest.map (carriedValue z))).low := by
        rw [show lcEval z (recompose base rest).low
              = (carriedValue z (recompose base rest)).low from rfl,
          inductionHypothesis]
      have restHigh : lcEval z (recompose base rest).high
          = (hornerPair base (rest.map (carriedValue z))).high := by
        rw [show lcEval z (recompose base rest).high
              = (carriedValue z (recompose base rest)).high from rfl,
          inductionHypothesis]
      unfold carriedValue
      rw [lowStep, highStep, restLow, restHigh, List.map_cons, hornerPair]
      rfl

/-! ## The emitted check -/

/-- **The emitted recomposition check.**  The combination is free; the equality
against the parent is what costs rows. -/
def recompositionRows (base : Nat) (children : List Carried) (parent : Carried) :
    List Row :=
  KEquality.rows (recompose base children) parent

/-- **The derived row count.**  Two — one per extension coordinate — however
many children there are. -/
theorem recompositionRows_length
    (base : Nat) (children : List Carried) (parent : Carried) :
    (recompositionRows base children parent).length = 2 :=
  KEquality.rows_length _ _

/-- The check allocates nothing. -/
def recompositionColumns : List Nat := []

theorem recompositionColumns_length : recompositionColumns.length = 0 := rfl

theorem recompositionColumns_nodup : recompositionColumns.Nodup := List.nodup_nil

/-- **Satisfaction forces the parent to be the radix-`b` recomposition.** -/
theorem recompositionRows_sound
    (z : Nat → Nat) (base : Nat) (children : List Carried) (parent : Carried)
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (recompositionRows base children parent) z) :
    hornerPair base (children.map (carriedValue z)) = carriedValue z parent := by
  rcases KEquality.rows_sound z (recompose base children) parent constantWire
    satisfied with ⟨lowEqual, highEqual⟩
  rw [← carriedValue_recompose]
  unfold carriedValue
  simp only [Pair.mk.injEq]
  exact ⟨lowEqual, highEqual⟩

/-- **An honest decomposition satisfies the check**, under the caller's own
assignment.  Nothing is allocated, so there is no witness to extend.

The consumer that constructs this premise is a Π_DEC prover: it produced the
children as balanced radix-`b` digits of the parent, so the parent's carried
value *is* the Horner value of theirs. -/
theorem recompositionRows_honest
    (z : Nat → Nat) (base : Nat) (children : List Carried) (parent : Carried)
    (constantWire : z 0 = 1)
    (decomposed :
      hornerPair base (children.map (carriedValue z)) = carriedValue z parent) :
    Satisfies (recompositionRows base children parent) z := by
  rw [← carriedValue_recompose] at decomposed
  unfold carriedValue at decomposed
  simp only [Pair.mk.injEq] at decomposed
  exact KEquality.rows_complete z (recompose base children) parent constantWire
    decomposed.1 decomposed.2

/-! ## Conservation

Every column of every emitted row is a child's, the parent's, or the constant
wire.  Scaling changes no column and concatenation introduces none, so the
recomposition's support is exactly the children's. -/

/-- Scaling and concatenation introduce no column. -/
theorem mentions_recomposeComb
    (base : Nat) (combs : List LinComb) (column : Nat)
    (mentioned : Mentions (recomposeComb base combs) column) :
    ∃ comb ∈ combs, Mentions comb column := by
  induction combs with
  | nil => simp [recomposeComb, Mentions] at mentioned
  | cons head rest inductionHypothesis =>
      rw [recomposeComb, mentions_append] at mentioned
      rcases mentioned with inHead | inScaled
      · exact ⟨head, List.mem_cons_self, inHead⟩
      · have inRest : Mentions (recomposeComb base rest) column :=
          (mentions_map_scale base (recomposeComb base rest) column).1 inScaled
        rcases inductionHypothesis inRest with ⟨comb, member, mentions⟩
        exact ⟨comb, List.mem_cons_of_mem head member, mentions⟩

/-- **Every column is a child's, the parent's, or the constant wire.** -/
theorem recompositionRows_conservation
    (base : Nat) (children : List Carried) (parent : Carried)
    (row : Row) (member : row ∈ recompositionRows base children parent)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0 ∨ Mentions parent.low column ∨ Mentions parent.high column
      ∨ ∃ child ∈ children,
          Mentions child.low column ∨ Mentions child.high column := by
  rcases KEquality.rows_conservation (recompose base children) parent row member
    column mentioned with wire | lowIn | highIn | parentLow | parentHigh
  · exact Or.inl wire
  · refine Or.inr (Or.inr (Or.inr ?_))
    rcases mentions_recomposeComb base (children.map Carried.low) column lowIn
      with ⟨comb, comment, mentions⟩
    rcases List.mem_map.1 comment with ⟨child, childMember, isLow⟩
    exact ⟨child, childMember, Or.inl (isLow ▸ mentions)⟩
  · refine Or.inr (Or.inr (Or.inr ?_))
    rcases mentions_recomposeComb base (children.map Carried.high) column highIn
      with ⟨comb, comment, mentions⟩
    rcases List.mem_map.1 comment with ⟨child, childMember, isHigh⟩
    exact ⟨child, childMember, Or.inr (isHigh ▸ mentions)⟩
  · exact Or.inr (Or.inl parentLow)
  · exact Or.inr (Or.inr (Or.inl parentHigh))

/-! ## Cost -/

/-- **One recomposition's cost.**  Two rows, nothing allocated, at every child
count — the children are read, not created here. -/
def recompositionCost : Lowering.Typed.Cost where
  recurringRows := 2
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem recompositionCost_rows
    (base : Nat) (children : List Carried) (parent : Carried) :
    (recompositionRows base children parent).length
      = recompositionCost.recurringRows :=
  recompositionRows_length base children parent

theorem recompositionCost_columns :
    recompositionColumns.length = recompositionCost.auxiliaryColumns :=
  recompositionColumns_length

/-! ## The check, over every recomposed carrier

`verify_dec_public` recomposes `t · d_pad` ring lanes, `d_pad` `y_zcol` lanes,
`|aux_openings|` scalars, and `D · m_in` public `X` entries. How many there are
is a property of the claim's shape, so the count is a fold over per-carrier
receipts rather than a closed formula. -/

/-- **The emitted program for a list of recompositions.** -/
def recompositionsRows (base : Nat)
    (checks : List (List Carried × Carried)) : List Row :=
  checks.flatMap (fun check => recompositionRows base check.1 check.2)

/-- **Every column belongs to some check**, or is the constant wire.

The fold of `recompositionRows_conservation` over the checks, which is what a
caller assembling several recompositions needs — the single-check form does not
compose on its own. -/
theorem recompositionsRows_conservation
    (base : Nat) (checks : List (List Carried × Carried))
    (row : Row) (member : row ∈ recompositionsRows base checks)
    (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ check ∈ checks,
      column = 0 ∨ Mentions check.2.low column ∨ Mentions check.2.high column
        ∨ ∃ child ∈ check.1,
            Mentions child.low column ∨ Mentions child.high column := by
  unfold recompositionsRows at member
  rcases List.mem_flatMap.1 member with ⟨check, checkMember, rowMember⟩
  exact ⟨check, checkMember,
    recompositionRows_conservation base check.1 check.2 row rowMember column
      mentioned⟩

/-- **The derived row count, as a fold over per-carrier receipts.** -/
theorem recompositionsRows_length (base : Nat)
    (checks : List (List Carried × Carried)) :
    (recompositionsRows base checks).length = (checks.map (fun _ => 2)).sum := by
  unfold recompositionsRows
  rw [List.length_flatMap]
  exact congrArg List.sum
    (List.map_congr_left (fun check _ =>
      recompositionRows_length base check.1 check.2))

/-- Two rows per recomposed carrier, once the fold is evaluated. -/
theorem recompositionsRows_length_eq (base : Nat)
    (checks : List (List Carried × Carried)) :
    (recompositionsRows base checks).length = 2 * checks.length := by
  rw [recompositionsRows_length]
  induction checks with
  | nil => rfl
  | cons check rest inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis]
      omega

/-- **Satisfaction forces every carrier to recompose.** -/
theorem recompositionsRows_sound
    (z : Nat → Nat) (base : Nat) (checks : List (List Carried × Carried))
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (recompositionsRows base checks) z)
    (check : List Carried × Carried) (member : check ∈ checks) :
    hornerPair base (check.1.map (carriedValue z)) = carriedValue z check.2 :=
  recompositionRows_sound z base check.1 check.2 constantWire
    (fun row rowMember =>
      satisfied row (List.mem_flatMap.2 ⟨check, member, rowMember⟩))

/-- **An honest decomposition of every carrier satisfies the check.** -/
theorem recompositionsRows_honest
    (z : Nat → Nat) (base : Nat) (checks : List (List Carried × Carried))
    (constantWire : z 0 = 1)
    (decomposed : ∀ check ∈ checks,
      hornerPair base (check.1.map (carriedValue z)) = carriedValue z check.2) :
    Satisfies (recompositionsRows base checks) z := by
  intro row member
  rcases List.mem_flatMap.1 member with ⟨check, checkMember, rowMember⟩
  exact recompositionRows_honest z base check.1 check.2 constantWire
    (decomposed check checkMember) row rowMember

/-- **The check's cost**, folded over carriers.  Nothing is allocated at any
carrier, so the auxiliary component stays zero however many there are. -/
def recompositionsCost (checks : List (List Carried × Carried)) :
    Lowering.Typed.Cost where
  recurringRows := 2 * checks.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem recompositionsCost_rows (base : Nat)
    (checks : List (List Carried × Carried)) :
    (recompositionsRows base checks).length
      = (recompositionsCost checks).recurringRows :=
  recompositionsRows_length_eq base checks

/-! ## Row ownership fails here, and the witness says why

Section 2 item 3 asks that every emitted row belong to exactly **one** receipt.
For this recipe that is **false without a hypothesis**, and the reason is
structural rather than incidental.

A check emits *two* rows carrying *different* coordinates: `⟨recomposed.low,
[(0,1)], parent.low⟩` and `⟨recomposed.high, [(0,1)], parent.high⟩`.  Nothing
stops one check's low row from coinciding with another check's high row, because
the two rows expose unrelated halves of unrelated carriers.

Contrast the recipes where uniqueness came cheaply:

| recipe | why uniqueness holds |
|---|---|
| `KLowNormBatch` | each receipt **allocates** a column, and the row exposes it |
| `FoldDigestRecipe` | each receipt emits **one** row, and the row *is* the receipt |
| `CommitmentMixerRecipe` | each receipt emits **one** row, carrying the whole parent |
| `KRecomposition` | each receipt emits **two** rows carrying different halves — none of the above |

So the obligation is real and belongs to the caller: the emitted rows must be
pairwise distinct across the program.  This module states the obstruction rather
than assuming it away. -/

/-- A check whose low row is about to be shared. -/
def witnessCheckA : List Carried × Carried :=
  ([⟨[(1, 1)], [(2, 1)]⟩], ⟨[(3, 1)], [(4, 1)]⟩)

/-- A different check whose **high** row is `witnessCheckA`'s **low** row. -/
def witnessCheckB : List Carried × Carried :=
  ([⟨[(9, 1)], [(1, 1)]⟩], ⟨[(9, 1)], [(3, 1)]⟩)

/-- The two checks are different: their children's low combinations differ. -/
theorem witnessChecks_differ :
    (witnessCheckA.1.head?.map Carried.low) ≠ (witnessCheckB.1.head?.map Carried.low) := by
  decide

/-! ## Row ownership, positionally

Section 2 item 3.  `recompositionRows_owner_not_unique` below is the value-level
negative: two checks constraining the same relation emit the same row, which they
should.  `Poseidon2Ownership`'s header settled that structural `Row` equality is
the wrong ABI, and `PiDecOwnership` applied the positional contract to the recipe
that *contains* these rows.

This is the analogue here, and it is what Π_DEC's inherits from: a receipt is a
check index paired with one of `KEquality`'s two halves, and each such receipt
emits exactly one row. -/

/-- The receipt that emits a row: which check, and which coordinate of it. -/
structure RowOwner where
  check : Nat
  half : KEquality.RowOwner
deriving DecidableEq

private def blankCheck : List Carried × Carried := ([], ⟨[], []⟩)

/-- The row a receipt emits. -/
def ownedRow (base : Nat) (checks : List (List Carried × Carried))
    (owner : RowOwner) : Row :=
  KEquality.ownedRow (recompose base (checks.getD owner.check blankCheck).1)
    (checks.getD owner.check blankCheck).2 owner.half

/-- Every receipt, in program order. -/
def owners (checks : List (List Carried × Carried)) : List RowOwner :=
  (List.range checks.length).flatMap
    (fun index => KEquality.allOwners.map (RowOwner.mk index))

/-- Reading a list back from its positions, through a `flatMap`. -/
theorem flatMap_getD_range {α β : Type} (fallback : α) (f : α → List β) :
    ∀ list : List α,
      (List.range list.length).flatMap
          (fun index => f (list.getD index fallback))
        = list.flatMap f
  | [] => rfl
  | head :: tail => by
      rw [List.length_cons, List.range_succ_eq_map, List.flatMap_cons,
        List.flatMap_map]
      exact congrArg (f head ++ ·) (flatMap_getD_range fallback f tail)

/-- **The emitted program is the receipt list's image.** -/
theorem recompositionsRows_eq_map_owners
    (base : Nat) (checks : List (List Carried × Carried)) :
    recompositionsRows base checks
      = (owners checks).map (ownedRow base checks) := by
  rw [owners, List.map_flatMap]
  simp only [List.map_map, Function.comp_def, ownedRow]
  rw [flatMap_getD_range blankCheck
    (fun check => KEquality.allOwners.map
      (fun half => KEquality.ownedRow (recompose base check.1) check.2 half))]
  simp only [recompositionsRows, recompositionRows,
    KEquality.rows_eq_map_owners]

theorem owners_nodup (checks : List (List Carried × Carried)) :
    (owners checks).Nodup := by
  rw [owners]
  induction checks.length with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.range_succ, List.flatMap_append]
      refine List.nodup_append.2 ⟨inductionHypothesis, ?_, ?_⟩
      · simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil,
          KEquality.allOwners, List.map_cons, List.map_nil]
        simp
      · intro left leftMember right rightMember
        have leftLt : left.check < count := by
          rcases List.mem_flatMap.1 leftMember with ⟨index, indexMember, inBlock⟩
          rcases List.mem_map.1 inBlock with ⟨_, _, rfl⟩
          exact List.mem_range.1 indexMember
        have rightAt : right.check = count := by
          simp only [List.flatMap_cons, List.flatMap_nil,
            List.append_nil] at rightMember
          rcases List.mem_map.1 rightMember with ⟨_, _, rfl⟩
          rfl
        intro equal
        rw [equal, rightAt] at leftLt
        omega

/-- **Exactly one receipt per emitted row.** -/
theorem ownership_is_positional
    (base : Nat) (checks : List (List Carried × Carried)) :
    (recompositionsRows base checks).length = (owners checks).length
      ∧ (owners checks).Nodup
      ∧ recompositionsRows base checks
          = (owners checks).map (ownedRow base checks) := by
  refine ⟨?_, owners_nodup checks, recompositionsRows_eq_map_owners base checks⟩
  rw [recompositionsRows_eq_map_owners, List.length_map]

/-- **A row belonging to two distinct checks.**

`witnessCheckA`'s low row is `witnessCheckB`'s high row, so attributing this row
to a receipt is not possible from the row alone.  Item 3 therefore needs a
caller obligation for this recipe, and the obligation is not vacuous. -/
theorem recompositionRows_owner_not_unique :
    ∃ row : Row,
      row ∈ recompositionRows 2 witnessCheckA.1 witnessCheckA.2
        ∧ row ∈ recompositionRows 2 witnessCheckB.1 witnessCheckB.2 := by
  refine ⟨⟨[(1, 1)], [(0, 1)], [(3, 1)]⟩, ?_, ?_⟩ <;> decide

end Nightstream.Implementation.R1CS.Canonical.KRecomposition
