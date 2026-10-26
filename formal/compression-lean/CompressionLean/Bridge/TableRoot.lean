import CompressionLean.Bridge.Merkle

/-!
Owns the computable honest setup root. `tableLeaf n tables i` is leaf `i` of
the honest setup tree of the oracle tables on the `n`-cube: the coset points
in `Gl`, the extension in `Fp`. When the tables hold the setup oracles of
`(rel, S)` on the cube, these leaves are `honestValues rel S`
(`tableLeaf_eq`), so `tableRoot` is `honestRoot rel S` (`tableRoot_eq`). The
`setupRootCheck` executable compares `tableRoot` with the root of
`setup::build`.
-/

namespace CompressionLean.Bridge

open NightstreamFPrime.Spec
open CompressionLean.Replay CompressionLean.Layer1

/-- Leaf `i` of the setup tree of `tables`: value `o·2^foldVars + v` is the
extension of table `o` at the power point of `7·ω_{2^(n+1)}^i·ω_8^v`. -/
def tableLeaf (n : ℕ) (tables : Array (Array Gl)) (i : ℕ) : Array Gl :=
  let y := generator * (twoAdicGenerator (n + 1)).pow i
  (Array.range (tables.size * 2 ^ foldVars)).map fun t =>
    let point := y * (twoAdicGenerator 3).pow (t % 2 ^ foldVars)
    Gl.ofNat (mle (powerPoint point.toF n) fun x => ((tables[t / 2 ^ foldVars]!)[x]!).toF).val

/-- The root of the setup tree of `tables` on the `n`-cube. -/
def tableRoot (n : ℕ) (tables : Array (Array Gl)) : Array Gl :=
  treeNode (tableLeaf n tables) (n - 2) 0

theorem length_powerPoint {K : Type} [Field K] (q : K) (n : ℕ) : (powerPoint q n).length = n := by
  induction n generalizing q with
  | zero => rfl
  | succ n ih => simp [powerPoint, ih]

/-- Tables that hold the setup oracles on the cube give the honest leaves. -/
theorem tableLeaf_eq (rel : Replay.Relation) (S : Setup Fp) (tables : Array (Array Gl))
    (size : tables.size = rel.kappa + runOracles)
    (agree : ∀ o < rel.kappa + runOracles, ∀ x < 2 ^ rel.setupVars, setupOracle rel S o x = ((tables[o]!)[x]!).toF) :
    tableLeaf rel.setupVars tables = honestValues rel S := by
  funext i
  have eight : (2 : ℕ) ^ foldVars = 8 := rfl
  apply Array.ext (by simp [tableLeaf, honestValues, size])
  intro t lt _
  simp only [tableLeaf, honestValues, Array.getElem_map, Array.getElem_range, univ_eval]
  have oracle : t / 2 ^ foldVars < rel.kappa + runOracles := by
    simp only [tableLeaf, Array.size_map, Array.size_range, size] at lt
    rw [eight] at lt ⊢
    omega
  have point : (generator * (twoAdicGenerator (rel.setupVars + 1)).pow i *
      (twoAdicGenerator 3).pow (t % 2 ^ foldVars)).toF = leafY rel.setupVars i * twoAdic 3 ^ (t % 2 ^ foldVars) := by
    rw [Gl.toF_mul, Gl.toF_mul, Gl.toF_pow, Gl.toF_pow, toF_twoAdicGenerator, toF_twoAdicGenerator,
      show generator.toF = 7 from Gl.toF_lit 7, leafY]
  rw [point, mle_congr _ fun x inside => (agree _ oracle x (by rwa [length_powerPoint] at inside)).symm]

/-- **The computable root is the honest root**: the root of tables that hold
the setup oracles of `(rel, S)` on the cube is `honestRoot rel S`. -/
theorem tableRoot_eq (rel : Replay.Relation) (S : Setup Fp) (tables : Array (Array Gl))
    (size : tables.size = rel.kappa + runOracles)
    (agree : ∀ o < rel.kappa + runOracles, ∀ x < 2 ^ rel.setupVars, setupOracle rel S o x = ((tables[o]!)[x]!).toF) :
    tableRoot rel.setupVars tables = honestRoot rel S := by
  rw [tableRoot, honestRoot, tableLeaf_eq rel S tables size agree]

end CompressionLean.Bridge
