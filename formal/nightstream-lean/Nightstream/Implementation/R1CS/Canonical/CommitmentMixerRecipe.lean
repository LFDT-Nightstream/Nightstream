import Nightstream.Implementation.R1CS.Canonical.KRecomposition
import Nightstream.Implementation.R1CS.Canonical.FoldDigestRecipe

/-!
Contract: the emitted row program for the Π_DEC commitment mixer.

Owns: the coordinatewise radix-`b` recomposition of commitments, its derived row
count, soundness, honest completeness, conservation and cost.

Does **not** own: binding.  That is the Ajtai security contract and no amount of
arithmetic here establishes it.

## The mixer is the recomposition, not a commitment scheme

`combine_b_pows` is

```text
acc = 0; pow = 1
for c in cs { scale_commitment_add_inplace(acc, pow, c); pow *= b }
```

and `scale_commitment_add_inplace(acc, scalar, c)` is `acc += scalar · c`
coordinatewise — the `ZERO`, `ONE` and `-1` cases in `commit.rs` are fast paths
for the same map.  A `Commitment` is a flat `d × kappa` vector of field elements
(`neo-ajtai/src/types.rs`), so the mixer is

```text
Σ_i b^i · c_i,  coordinate by coordinate
```

which is exactly the relation `KRecomposition` already owns.  Its
`powerSumFrom_eq_hornerValue` proves the accumulator loop and Horner form agree,
and that proof is reused here rather than repeated.

## The correction this records

Every ledger entry from cycle 326 to 340 said the mixer needed "the Ajtai
commitment layer".  `PIDEC-ADV-PRESENCE` narrowed that to one of three branches.
This narrows it further: **the arithmetic of that branch is not blocked either.**

What genuinely needs Ajtai is *binding* — that a commitment determines its
opening.  That is a security contract, and prompt section 4.6's rule applies
directly: arithmetic correctness proves neither binding nor random-oracle
soundness.  `COMMITMENT-MIXER-NOT-BINDING` records the boundary.

## One row per coordinate

A commitment coordinate is a **base-field** value, not an extension element, so
each coordinate equality costs one row — as with the fold-digest lanes, and
unlike a `K` equality, which costs two.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal

/-- One commitment coordinate: the children's columns at that coordinate, and
the parent's. -/
structure Coordinate where
  children : List LinComb
  parent : LinComb

/-- **The emitted mixer program.**  One equality per coordinate; the
recomposition itself is a combination and emits nothing. -/
def mixerRows (base : Nat) (coordinates : List Coordinate) : List Row :=
  coordinates.map (fun coordinate =>
    KEquality.equalityRow
      (KRecomposition.recomposeComb base coordinate.children) coordinate.parent)

/-- **The derived row count**, from the emitted list.  One per coordinate,
however many children each has — the scaling is a coefficient rewrite. -/
theorem mixerRows_length (base : Nat) (coordinates : List Coordinate) :
    (mixerRows base coordinates).length = coordinates.length :=
  List.length_map _

/-- The mixer allocates nothing. -/
def mixerColumns : List Nat := []

theorem mixerColumns_length : mixerColumns.length = 0 := rfl

theorem mixerColumns_nodup : mixerColumns.Nodup := List.nodup_nil

/-- **Satisfaction forces every coordinate to recompose.**

The value is stated in Horner form, which `KRecomposition.powerSum_one` already
proves equal to the verifier's own power-accumulator loop.  So this is the
relation `combine_b_pows` computes, not merely one that resembles it. -/
theorem mixerRows_sound
    (z : Nat → Nat) (base : Nat) (coordinates : List Coordinate)
    (constantWire : z 0 = 1)
    (satisfied : Satisfies (mixerRows base coordinates) z)
    (coordinate : Coordinate) (member : coordinate ∈ coordinates) :
    KRecomposition.hornerValue base (coordinate.children.map (lcEval z))
      = lcEval z coordinate.parent := by
  have row := (KEquality.equalityRow_iff z _ coordinate.parent constantWire).1
    (satisfied _ (List.mem_map.2 ⟨coordinate, member, rfl⟩))
  rw [← KRecomposition.lcEval_recomposeComb z base coordinate.children]
  exact row

/-- **An honest mix satisfies the check**, under the caller's own assignment.
Nothing is allocated, so there is no witness to extend. -/
theorem mixerRows_honest
    (z : Nat → Nat) (base : Nat) (coordinates : List Coordinate)
    (constantWire : z 0 = 1)
    (mixed : ∀ coordinate ∈ coordinates,
      KRecomposition.hornerValue base (coordinate.children.map (lcEval z))
        = lcEval z coordinate.parent) :
    Satisfies (mixerRows base coordinates) z := by
  intro row member
  rcases List.mem_map.1 member with ⟨coordinate, coordinateMember, rfl⟩
  refine (KEquality.equalityRow_iff z _ coordinate.parent constantWire).2 ?_
  rw [KRecomposition.lcEval_recomposeComb z base coordinate.children]
  exact mixed coordinate coordinateMember

/-- **Every column is a child's, the parent's, or the constant wire.** -/
theorem mixerRows_conservation
    (base : Nat) (coordinates : List Coordinate) (row : Row)
    (member : row ∈ mixerRows base coordinates) (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    ∃ coordinate ∈ coordinates,
      column = 0 ∨ Mentions coordinate.parent column
        ∨ ∃ child ∈ coordinate.children, Mentions child column := by
  rcases List.mem_map.1 member with ⟨coordinate, coordinateMember, rfl⟩
  refine ⟨coordinate, coordinateMember, ?_⟩
  simp only [KEquality.equalityRow] at mentioned
  rcases mentioned with inA | inB | inC
  · exact Or.inr (Or.inr
      (KRecomposition.mentions_recomposeComb base coordinate.children column inA))
  · exact Or.inl (by
      simpa only [Mentions, List.map_cons, List.map_nil,
        List.mem_singleton] using inB)
  · exact Or.inr (Or.inl inC)

/-! ## Cost -/

/-- **The mixer's cost**, folded over coordinates.  One row each, nothing
allocated at any of them. -/
def mixerCost (coordinates : List Coordinate) : Lowering.Typed.Cost where
  recurringRows := coordinates.length
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem mixerCost_rows (base : Nat) (coordinates : List Coordinate) :
    (mixerRows base coordinates).length = (mixerCost coordinates).recurringRows :=
  mixerRows_length base coordinates

theorem mixerCost_columns (coordinates : List Coordinate) :
    mixerColumns.length = (mixerCost coordinates).auxiliaryColumns := rfl

/-! ## What this does not establish

`COMMITMENT-MIXER-NOT-BINDING`.  The rows above say the parent commitment is the
radix-`b` combination of the children's, coordinate by coordinate.  They say
nothing about whether a commitment determines its opening.

That is the Ajtai binding contract, it is a hardness assumption rather than an
arithmetic fact, and no row program establishes it.  Prompt section 4.6 makes the
same point for the two hashes: shared arithmetic does not transfer a security
contract. -/

/-- **Different children mix to the same parent.**

At base two, the digit lists `[2]` and `[0, 1]` both recompose to `2`.  Neither
is degenerate and neither is all-zero, so this is a real collision rather than a
statement about empty lists.

The mixer is linear, so collisions are expected and are not a defect in the
rows: `KLowNorm` is what confines digits to the centered window, and Ajtai
binding is what makes a commitment determine its opening.  Recomposition alone
does neither, which is why this module claims neither. -/
theorem mixing_alone_does_not_bind :
    KRecomposition.hornerValue 2 [2] = KRecomposition.hornerValue 2 [0, 1]
      ∧ ([2] : List Nat) ≠ [0, 1] := by
  constructor
  · decide
  · decide

/-! ## Row ownership

Section 2 item 3.  Existence is `List.mem_map`.  Uniqueness needs distinct
coordinates to emit distinct rows, and the parent supplies it: a coordinate's
row carries its parent combination in the `c` field, so **distinct parents give
distinct rows**.

The hypothesis is `(coordinates.map Coordinate.parent).Nodup`, which a decoder
meets by construction — each coordinate is a distinct commitment position with
its own parent column. -/

/-- A `Nodup` parent list makes the coordinate recoverable from its parent. -/
theorem parent_determines_coordinate
    (coordinates : List Coordinate)
    (distinctParents : (coordinates.map Coordinate.parent).Nodup)
    (first second : Coordinate)
    (firstMember : first ∈ coordinates) (secondMember : second ∈ coordinates)
    (sameParent : first.parent = second.parent) :
    first = second := by
  induction coordinates with
  | nil => cases firstMember
  | cons head rest inductionHypothesis =>
      rw [List.map_cons, List.nodup_cons] at distinctParents
      rcases List.mem_cons.1 firstMember with rfl | firstTail
      · rcases List.mem_cons.1 secondMember with rfl | secondTail
        · rfl
        · refine absurd ?_ distinctParents.1
          rw [sameParent]
          exact List.mem_map.2 ⟨second, secondTail, rfl⟩
      · rcases List.mem_cons.1 secondMember with rfl | secondTail
        · refine absurd ?_ distinctParents.1
          rw [← sameParent]
          exact List.mem_map.2 ⟨first, firstTail, rfl⟩
        · exact inductionHypothesis distinctParents.2 firstTail secondTail

/-- A coordinate's row exposes its parent. -/
theorem mixerRow_parent
    (base : Nat) (coordinate : Coordinate) :
    (KEquality.equalityRow
      (KRecomposition.recomposeComb base coordinate.children)
      coordinate.parent).c = coordinate.parent := rfl

/-- **Every emitted row belongs to exactly one coordinate**, given distinct
parents. -/
theorem mixerRows_owned
    (base : Nat) (coordinates : List Coordinate)
    (distinctParents : (coordinates.map Coordinate.parent).Nodup)
    (row : Row) (member : row ∈ mixerRows base coordinates) :
    ∃ coordinate, coordinate ∈ coordinates
      ∧ row = KEquality.equalityRow
          (KRecomposition.recomposeComb base coordinate.children)
          coordinate.parent
      ∧ ∀ other ∈ coordinates,
          row = KEquality.equalityRow
              (KRecomposition.recomposeComb base other.children) other.parent →
            other = coordinate := by
  rcases List.mem_map.1 member with ⟨coordinate, coordinateMember, rfl⟩
  refine ⟨coordinate, coordinateMember, rfl,
    fun other otherMember sameRow => ?_⟩
  have parents := congrArg Row.c sameRow
  simp only [KEquality.equalityRow] at parents
  exact parent_determines_coordinate coordinates distinctParents other
    coordinate otherMember coordinateMember parents.symm

end Nightstream.Implementation.R1CS.Canonical.CommitmentMixerRecipe
