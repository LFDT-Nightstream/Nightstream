import CompressionLean.Bridge.Merkle

namespace CompressionLean.Bridge

open NightstreamFPrime.Spec
open CompressionLean.Replay CompressionLean.Layer1 CompressionLean.Spec CompressionLean.Lib

def tableLeaf (n : ℕ) (tables : Array (Array Word)) (i : ℕ) : Array Word :=
  let y := generator * (twoAdicGenerator (n + 1)).pow i
  (Array.range (tables.size * 2 ^ foldVars)).map fun t =>
    let point := y * (twoAdicGenerator 3).pow (t % 2 ^ foldVars)
    Word.ofNat (mle (powerPoint point.toF n) fun x => ((tables[t / 2 ^ foldVars]!)[x]!).toF).val

def tableRoot (n : ℕ) (tables : Array (Array Word)) : Array Word :=
  treeNode (tableLeaf n tables) (n - 2) 0

theorem length_powerPoint {K : Type} [Field K] (q : K) (n : ℕ) : (powerPoint q n).length = n := by
  induction n generalizing q with
  | zero => rfl
  | succ n ih => simp [powerPoint, ih]

theorem tableLeaf_eq (rel : Replay.Relation) (S : Setup Gl) (tables : Array (Array Word))
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
    rw [Word.toF_mul, Word.toF_mul, Word.toF_pow, Word.toF_pow, toF_twoAdicGenerator, toF_twoAdicGenerator,
      show generator.toF = 7 from Word.toF_lit 7, leafY]
  rw [point, mle_congr _ fun x inside => (agree _ oracle x (by rwa [length_powerPoint] at inside)).symm]

theorem tableRoot_eq (rel : Replay.Relation) (S : Setup Gl) (tables : Array (Array Word))
    (size : tables.size = rel.kappa + runOracles)
    (agree : ∀ o < rel.kappa + runOracles, ∀ x < 2 ^ rel.setupVars, setupOracle rel S o x = ((tables[o]!)[x]!).toF) :
    tableRoot rel.setupVars tables = honestRoot rel S := by
  rw [tableRoot, honestRoot, tableLeaf_eq rel S tables size agree]

end CompressionLean.Bridge
