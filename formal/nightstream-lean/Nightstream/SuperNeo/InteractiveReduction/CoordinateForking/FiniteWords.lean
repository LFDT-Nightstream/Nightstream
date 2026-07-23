import Init.Data.Fin.Lemmas
import Init.Data.List.Nat.Sum
import Init.Data.List.Pairwise
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

/-!
Exact finite product enumeration for coordinate-forking challenges.

Owns: the recursive enumeration of every length-`n` word over an explicit
finite alphabet, together with kernel proofs of membership, duplicate freedom,
cardinality, and the resulting uniform finite support.

Does not own: a verifier, an extractor, a rejection sampler, a probability
bound, a query schedule, Rust, R1CS, or constraints.

The enumeration is actual data.  Later coordinate-forking experiments can use
it as their seed support instead of receiving an abstract challenge
distribution from the caller.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords

open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uScalar

variable {Scalar : Type uScalar}

/-- Add one leading coordinate to a finite word. -/
def prepend {coordinates : Nat}
    (head : Scalar)
    (tail : Fin coordinates -> Scalar) :
    Fin (coordinates + 1) -> Scalar :=
  fun index => Fin.cases head tail index

@[simp] theorem prepend_zero
    {coordinates : Nat}
    (head : Scalar)
    (tail : Fin coordinates -> Scalar) :
    prepend head tail (0 : Fin (coordinates + 1)) = head :=
  rfl

@[simp] theorem prepend_succ
    {coordinates : Nat}
    (head : Scalar)
    (tail : Fin coordinates -> Scalar)
    (index : Fin coordinates) :
    prepend head tail index.succ = tail index :=
  rfl

/-- Equality of prepended words recovers both pieces of data. -/
theorem prepend_injective_pair
    {coordinates : Nat}
    {leftHead rightHead : Scalar}
    {leftTail rightTail : Fin coordinates -> Scalar}
    (wordsEqual : prepend leftHead leftTail = prepend rightHead rightTail) :
    leftHead = rightHead ∧ leftTail = rightTail := by
  constructor
  · have coordinateEqual := congrFun wordsEqual (0 : Fin (coordinates + 1))
    simpa only [prepend_zero] using coordinateEqual
  · funext index
    have coordinateEqual := congrFun wordsEqual index.succ
    simpa only [prepend_succ] using coordinateEqual

/-- Lexicographic finite-product enumeration.  The first coordinate is the
outer loop; the remaining coordinates retain their recursive order. -/
def vectors (alphabet : List Scalar) :
    (coordinates : Nat) -> List (Fin coordinates -> Scalar)
  | 0 => [fun index => Fin.elim0 index]
  | coordinates + 1 =>
      alphabet.flatMap fun head =>
        (vectors alphabet coordinates).map (prepend head)

/-- A duplicate-free alphabet yields a duplicate-free product enumeration. -/
theorem vectors_nodup
    (alphabet : List Scalar)
    (alphabetNodup : alphabet.Nodup)
    (coordinates : Nat) :
    (vectors alphabet coordinates).Nodup := by
  induction coordinates with
  | zero => simp [vectors]
  | succ coordinates inductionHypothesis =>
      change List.Pairwise
        (fun left right : Fin (Nat.succ coordinates) -> Scalar =>
          left ≠ right)
        (vectors alphabet (Nat.succ coordinates))
      rw [vectors, List.pairwise_flatMap]
      constructor
      · intro head _
        rw [List.pairwise_map]
        exact inductionHypothesis.imp (by
          intro leftTail rightTail tailsDistinct wordsEqual
          exact tailsDistinct (prepend_injective_pair wordsEqual).2)
      · apply alphabetNodup.imp
        intro leftHead rightHead headsDistinct
        intro leftWord leftMember rightWord rightMember
        rcases List.mem_map.mp leftMember with
          ⟨leftTail, _, rfl⟩
        rcases List.mem_map.mp rightMember with
          ⟨rightTail, _, rfl⟩
        intro wordsEqual
        exact headsDistinct (prepend_injective_pair wordsEqual).1

/-- The product enumeration contains exactly `|alphabet|^coordinates` words. -/
theorem vectors_length
    (alphabet : List Scalar)
    (coordinates : Nat) :
    (vectors alphabet coordinates).length =
      alphabet.length ^ coordinates := by
  induction coordinates with
  | zero => rfl
  | succ coordinates inductionHypothesis =>
      simp only [vectors, List.length_flatMap, List.length_map,
        inductionHypothesis, List.map_const', List.sum_replicate_nat,
        Nat.pow_succ]
      exact Nat.mul_comm alphabet.length (alphabet.length ^ coordinates)

/-- Membership is pointwise alphabet membership, with no hidden decoding. -/
theorem mem_vectors_iff
    (alphabet : List Scalar)
    (coordinates : Nat)
    (word : Fin coordinates -> Scalar) :
    word ∈ vectors alphabet coordinates <->
      forall index, word index ∈ alphabet := by
  induction coordinates with
  | zero =>
      constructor
      · intro _ index
        exact Fin.elim0 index
      · intro _
        have wordEqual : word = (fun index => Fin.elim0 index) := by
          funext index
          exact Fin.elim0 index
        rw [wordEqual]
        simp [vectors]
  | succ coordinates inductionHypothesis =>
      constructor
      · intro wordMember
        rcases List.mem_flatMap.mp wordMember with
          ⟨head, headMember, mappedMember⟩
        rcases List.mem_map.mp mappedMember with
          ⟨tail, tailMember, rfl⟩
        intro index
        refine Fin.cases headMember ?_ index
        intro tailIndex
        exact (inductionHypothesis tail).mp tailMember tailIndex
      · intro coordinatesMember
        let head : Scalar := word (0 : Fin (coordinates + 1))
        let tail : Fin coordinates -> Scalar := fun index => word index.succ
        have headMember : head ∈ alphabet :=
          coordinatesMember (0 : Fin (coordinates + 1))
        have tailMember : tail ∈ vectors alphabet coordinates :=
          (inductionHypothesis tail).mpr (by
            intro index
            exact coordinatesMember index.succ)
        apply List.mem_flatMap.mpr
        refine ⟨head, headMember, ?_⟩
        apply List.mem_map.mpr
        refine ⟨tail, tailMember, ?_⟩
        funext index
        refine Fin.cases ?_ ?_ index
        · rfl
        · intro tailIndex
          rfl

namespace Support

/-- The exact uniform challenge-vector support induced by an alphabet. -/
def challengeVectors
    (alphabet : Support Scalar)
    (coordinates : Nat) : Support (Fin coordinates -> Scalar) where
  values := vectors alphabet.values coordinates
  nodup := vectors_nodup alphabet.values alphabet.nodup coordinates
  nonempty := by
    apply List.length_pos_iff.mp
    rw [vectors_length]
    exact Nat.pow_pos (by
      simpa only [Support.cardinality] using alphabet.cardinality_pos)

@[simp] theorem challengeVectors_values
    (alphabet : Support Scalar)
    (coordinates : Nat) :
    (challengeVectors alphabet coordinates).values =
      vectors alphabet.values coordinates :=
  rfl

@[simp] theorem challengeVectors_cardinality
    (alphabet : Support Scalar)
    (coordinates : Nat) :
    (challengeVectors alphabet coordinates).cardinality =
      alphabet.cardinality ^ coordinates := by
  exact vectors_length alphabet.values coordinates

theorem mem_challengeVectors_iff
    (alphabet : Support Scalar)
    (coordinates : Nat)
    (word : Fin coordinates -> Scalar) :
    word ∈ (challengeVectors alphabet coordinates).values <->
      forall index, word index ∈ alphabet.values := by
  exact mem_vectors_iff alphabet.values coordinates word

end Support

end Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
