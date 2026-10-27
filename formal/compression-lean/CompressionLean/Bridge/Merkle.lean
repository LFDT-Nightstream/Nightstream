import CompressionLean.Bridge.Setup
import Mathlib.Data.Nat.Bitwise

/-!
Owns the honest setup tree and the non-vacuity of `MerkleBinds`. The tree
of leaf values `vals` hashes leaf `p` with `hashLeaf` and compresses the
nodes `2p` and `2p + 1` (`treeNode`); the honest path of index `i` holds the
sibling of each node on the way up. An honest leaf passes `pathOk` at its
own index (`pathOk_honest`). A leaf that passes `pathOk` at the tree's root
holds the tree's values, or Poseidon2 collides at explicit inputs
(`pathOk_binds`). So `MerkleBinds` follows from an honest key root unless
such a collision occurs (`merkleBinds_of_root`), and honest leaves meet it
(`honest_leaves_bind`).
-/

namespace CompressionLean.Bridge

open NightstreamFPrime.Spec
open CompressionLean.Replay CompressionLean.Layer1 CompressionLean.Spec CompressionLean.Lib

/-! ## Digest sizes -/

theorem size_hashLeaf (values : Array Word) : (hashLeaf values).size = 4 := by
  have fold : ∀ (chunks : List (Array Word)) (s : Array Word), s.size = width →
      (chunks.foldl (fun state chunk =>
        permute ((List.range chunk.size).foldl (fun s i => s.set! i chunk[i]!) state)) s).size = width := by
    intro chunks
    induction chunks with
    | nil => intro s same; exact same
    | cons _ rest ih => intro _ _; exact ih _ rfl
  simp only [hashLeaf, Array.size_extract, fold _ _ (Array.size_replicate ..)]
  rfl

theorem size_compress (left right : Array Word) : (compress left right).size = 4 := by
  simp only [compress, Array.size_extract]
  rfl

/-- Four lanes equal under `==` make two digests equal. -/
theorem eq_of_lanes {a b : Array Word} (ha : a.size = 4) (hb : b.size = 4)
    (lanes : ((List.range 4).all fun lane => a[lane]! == b[lane]!) = true) : a = b := by
  apply Array.ext (ha.trans hb.symm)
  intro lane la lb
  have same := Word.eq_of_beq (List.all_eq_true.mp lanes lane (List.mem_range.mpr (by omega)))
  rwa [getElem!_pos a lane la, getElem!_pos b lane lb] at same

/-! ## The climb of `merkleRoot` -/

/-- The node of `merkleRoot` at `index` along `leaf.path` after `k` levels. -/
def climb (leaf : Leaf) (index : ℕ) : ℕ → Array Word
  | 0 => hashLeaf leaf.values
  | k + 1 =>
      if index / 2 ^ k % 2 = 0 then compress (climb leaf index k) leaf.path[k]!
      else compress leaf.path[k]! (climb leaf index k)

/-- The pair that `merkleRoot` compresses at level `k`: bit `k` of `index`
puts the running node on the left or on the right. -/
def climbPair (leaf : Leaf) (index k : ℕ) : Array Word × Array Word :=
  if index / 2 ^ k % 2 = 0 then (climb leaf index k, leaf.path[k]!) else (leaf.path[k]!, climb leaf index k)

theorem climb_succ (leaf : Leaf) (index k : ℕ) :
    climb leaf index (k + 1) = Function.uncurry compress (climbPair leaf index k) := by
  rw [climb, climbPair]
  split_ifs <;> rfl

theorem size_climb (leaf : Leaf) (index : ℕ) : ∀ k, (climb leaf index k).size = 4
  | 0 => size_hashLeaf _
  | k + 1 => by rw [climb_succ]; exact size_compress _ _

theorem merkleRoot_eq_climb (leaf : Leaf) (index : ℕ) :
    merkleRoot (hashLeaf leaf.values) index leaf.path = climb leaf index leaf.path.size := by
  unfold merkleRoot
  generalize leaf.path.size = n
  induction n with
  | zero => rfl
  | succ n ih => rw [List.range_succ, List.foldl_append, ih]; rfl

/-! ## The tree of leaf values `vals` -/

section Tree

variable (vals : ℕ → Array Word)

/-- Node `p` at level `k`: the leaf hash at level `0`, then the compression
of nodes `2p` and `2p + 1` of the level below. -/
def treeNode : ℕ → ℕ → Array Word
  | 0, p => hashLeaf (vals p)
  | k + 1, p => compress (treeNode k (2 * p)) (treeNode k (2 * p + 1))

/-- The pair that node `p` at level `k + 1` compresses. -/
def treePair (k p : ℕ) : Array Word × Array Word := (treeNode vals k (2 * p), treeNode vals k (2 * p + 1))

theorem size_treeNode : ∀ k p, (treeNode vals k p).size = 4
  | 0, _ => size_hashLeaf _
  | _ + 1, _ => size_compress _ _

/-- The path of index `i` in a tree of depth `d`: the sibling node at each level. -/
def honestPath (d i : ℕ) : Array (Array Word) := (Array.range d).map fun k => treeNode vals k ((i / 2 ^ k) ^^^ 1)

/-- A Poseidon2 collision at explicit inputs on the climb of `leaf` at
`index` in the tree of depth `d`: at a level `k < d` the leaf's pair and the
tree's pair differ and compress alike, or the leaf's values and `vals index`
differ and hash alike. Both inputs are functions of `(vals, leaf, index, k)`.
A bare "some collision exists" holds for every compressing hash, so it would
not bind. -/
def PathCollision (d : ℕ) (leaf : Leaf) (index : ℕ) : Prop :=
  (∃ k < d, climbPair leaf index k ≠ treePair vals k (index / 2 ^ (k + 1)) ∧
      Function.uncurry compress (climbPair leaf index k) =
        Function.uncurry compress (treePair vals k (index / 2 ^ (k + 1)))) ∨
    (leaf.values ≠ vals index ∧ hashLeaf leaf.values = hashLeaf (vals index))

theorem half_index (index k : ℕ) : index / 2 ^ (k + 1) = index / 2 ^ k / 2 := by
  rw [pow_succ, Nat.div_div_eq_div_mul]

theorem climb_honest (d i : ℕ) : ∀ k ≤ d, climb ⟨vals i, honestPath vals d i⟩ i k = treeNode vals k (i / 2 ^ k)
  | 0, _ => by simp [climb, treeNode]
  | k + 1, le => by
      rw [climb, climb_honest d i k (by omega)]
      simp only [honestPath, getElem!_map_range _ _ (show k < d by omega), half_index, treeNode]
      split_ifs with even
      · rw [Nat.xor_one_of_even (Nat.even_iff.mpr even)]
        congr 2 <;> omega
      · rw [Nat.xor_one_of_odd (Nat.odd_iff.mpr (by omega))]
        congr 2 <;> omega

/-- An honest leaf passes the path check at its own index. -/
theorem pathOk_honest (d i : ℕ) (inside : i < 2 ^ d) :
    pathOk (treeNode vals d 0) i ⟨vals i, honestPath vals d i⟩ = true := by
  unfold pathOk
  rw [merkleRoot_eq_climb, show (honestPath vals d i).size = d by simp [honestPath],
    climb_honest vals d i d le_rfl, Nat.div_eq_of_lt inside]
  exact List.all_eq_true.mpr fun lane _ => beq_self_eq_true (treeNode vals d 0)[lane]!.word

/-- **Binding by extraction**: a leaf that passes the path check at the
tree's root holds the tree's values at its index, or gives a `PathCollision`. -/
theorem pathOk_binds {root : Array Word} (d index : ℕ) (leaf : Leaf) (isRoot : root = treeNode vals d 0)
    (size : leaf.path.size = d) (inside : index < 2 ^ d) (ok : pathOk root index leaf = true) :
    leaf.values = vals index ∨ PathCollision vals d leaf index := by
  have top : climb leaf index d = treeNode vals d (index / 2 ^ d) := by
    unfold pathOk at ok
    rw [merkleRoot_eq_climb, size, isRoot] at ok
    rw [Nat.div_eq_of_lt inside]
    exact eq_of_lanes (size_climb _ _ _) (size_treeNode _ _ _) ok
  have down : ∀ n ≤ d, climb leaf index (d - n) = treeNode vals (d - n) (index / 2 ^ (d - n)) ∨
      PathCollision vals d leaf index := by
    intro n
    induction n with
    | zero => intro _; exact .inl top
    | succ n ih =>
        intro le
        obtain ⟨k, above⟩ : ∃ k, d - n = k + 1 := ⟨d - n - 1, by omega⟩
        rw [show d - (n + 1) = k by omega]
        rcases ih (by omega) with same | broken
        · rw [above, climb_succ] at same
          by_cases pairs : climbPair leaf index k = treePair vals k (index / 2 ^ (k + 1))
          · left
            rw [climbPair, treePair, half_index] at pairs
            split_ifs at pairs with even
            · rw [(Prod.mk.inj pairs).1]
              congr 1
              omega
            · rw [(Prod.mk.inj pairs).2]
              congr 1
              omega
          · exact .inr (.inl ⟨k, by omega, pairs, same⟩)
        · exact .inr broken
  rcases down d le_rfl with same | broken
  · by_cases values : leaf.values = vals index
    · exact .inl values
    · rw [Nat.sub_self] at same
      exact .inr (.inr ⟨values, by simpa [climb, treeNode] using same⟩)
  · exact .inr broken

end Tree

/-! ## The honest setup tree -/

/-- The honest values of leaf `i`: value `o·2^foldVars + v` is setup oracle
`o`'s codeword at point `v` of the coset `y_i·ω_8^v`. -/
noncomputable def honestValues (rel : Replay.Relation) (S : Setup Gl) (i : ℕ) : Array Word :=
  (Array.range ((rel.kappa + runOracles) * 2 ^ foldVars)).map fun t =>
    Word.ofNat ((univ rel.setupVars (setupOracle rel S (t / 2 ^ foldVars))).eval
      (leafY rel.setupVars i * twoAdic 3 ^ (t % 2 ^ foldVars))).val

/-- The root of the honest setup tree. -/
noncomputable def honestRoot (rel : Replay.Relation) (S : Setup Gl) : Array Word :=
  treeNode (honestValues rel S) (rel.setupVars - 2) 0

/-- The honest leaf of index `i` and its path. -/
noncomputable def honestLeaf (rel : Replay.Relation) (S : Setup Gl) (i : ℕ) : Leaf :=
  ⟨honestValues rel S i, honestPath (honestValues rel S) (rel.setupVars - 2) i⟩

theorem toF_honestValues (rel : Replay.Relation) (S : Setup Gl) (i o v : ℕ) (oracle : o < rel.kappa + runOracles)
    (point : v < 2 ^ foldVars) : ((honestValues rel S i)[o * 2 ^ foldVars + v]!).toF =
      (univ rel.setupVars (setupOracle rel S o)).eval (leafY rel.setupVars i * twoAdic 3 ^ v) := by
  have eight : (2 : ℕ) ^ foldVars = 8 := rfl
  rw [honestValues, getElem!_map_range _ _ (by rw [eight] at point ⊢; nlinarith), Word.toF_ofNat,
    ZMod.natCast_zmod_val, show (o * 2 ^ foldVars + v) / 2 ^ foldVars = o by rw [eight]; omega,
    show (o * 2 ^ foldVars + v) % 2 ^ foldVars = v by rw [eight]; omega]

/-- **`MerkleBinds` from the root**: under the honest key root, leaves of
the replay's path length meet `MerkleBinds`, or one of them passes the path
check at an index with a `PathCollision`. -/
theorem merkleBinds_of_root (rel : Replay.Relation) (S : Setup Gl) (leaves : Array Leaf)
    (root : rel.key.root = honestRoot rel S) (sizes : ∀ leaf ∈ leaves.toList, leaf.path.size = rel.setupVars - 2) :
    MerkleBinds rel S leaves ∨ ∃ leaf ∈ leaves.toList, ∃ i < 2 ^ (rel.setupVars - 2),
      pathOk rel.key.root i leaf = true ∧ PathCollision (honestValues rel S) (rel.setupVars - 2) leaf i := by
  by_cases collision : ∃ leaf ∈ leaves.toList, ∃ i < 2 ^ (rel.setupVars - 2),
      pathOk rel.key.root i leaf = true ∧ PathCollision (honestValues rel S) (rel.setupVars - 2) leaf i
  · exact .inr collision
  · left
    intro leaf member i inside ok o v oracle point
    rcases pathOk_binds _ _ i leaf root (sizes leaf member) inside ok with same | broken
    · rw [same]
      exact toF_honestValues rel S i o v oracle point
    · exact absurd ⟨leaf, member, i, inside, ok, broken⟩ collision

/-- **Non-vacuity of `MerkleBinds`**: under the honest key root, the honest
leaves at indices `qs` pass the path check at their own index and meet
`MerkleBinds` unless Poseidon2 collides at explicit inputs. An unbounded
index would break it: index `q + 2^(setupVars - 2)` passes the same check
with no collision and asks for the coset turned by `ω_8`. -/
theorem honest_leaves_bind (rel : Replay.Relation) (S : Setup Gl) (root : rel.key.root = honestRoot rel S)
    (qs : Array ℕ) (inside : ∀ q ∈ qs.toList, q < 2 ^ (rel.setupVars - 2)) :
    (∀ q ∈ qs.toList, pathOk rel.key.root q (honestLeaf rel S q) = true) ∧
      (MerkleBinds rel S (qs.map (honestLeaf rel S)) ∨
        ∃ leaf ∈ (qs.map (honestLeaf rel S)).toList, ∃ i < 2 ^ (rel.setupVars - 2),
          pathOk rel.key.root i leaf = true ∧ PathCollision (honestValues rel S) (rel.setupVars - 2) leaf i) := by
  refine ⟨fun q member => ?_, merkleBinds_of_root rel S _ root fun leaf member => ?_⟩
  · rw [root]
    exact pathOk_honest _ _ q (inside q member)
  · obtain ⟨q, _, rfl⟩ : ∃ q ∈ qs, honestLeaf rel S q = leaf := by simpa using member
    simp [honestLeaf, honestPath]

end CompressionLean.Bridge
