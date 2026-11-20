import NightstreamFPrime.Spec.Checkpoint.Extraction
import NightstreamFPrime.Spec.Checkpoint.Runs
import NightstreamFPrime.Spec.Compression.Akita.Extract.Level

/-! Akita Theorem 10.17 for one packing fold (ePrint 2026/1983, revision of 2026-10-15, with
Corollary 10.32 at the fold challenge): the rewinding extractor of `Checkpoint.extract` on the
checkpoint structure `Level.extraction`.

- `Level.checkError_eq`, `Level.loopsFrom_eq` and `Level.treeSize_eq` are `ε_ck`, the number of
  retry loops and the tree size `K̂` of one fold.
- `Level.extract_sound` states that an output of the extractor is a weak opening of the statement
  or a solution of Module-SIS.
- `Level.extraction_le` bounds the probability that the verifier accepts and the extractor
  returns no output, and `Level.runs_le` bounds the expected number of runs of the prover. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Extract

open scoped BigOperators
open NightstreamFPrime.Spec.Checkpoint
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

/-- The sum of the entries of a list by index. -/
theorem sum_range_getD (costs : List ℝ) :
    ∑ index ∈ Finset.range costs.length, costs.getD index 0 = costs.sum := by
  induction costs with
  | nil => simp
  | cons cost rest induction =>
      rw [List.length_cons, Finset.sum_range_succ', List.sum_cons, ← induction, add_comm]
      simp

namespace Level

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor} {E Wire Coord Slot Result : Type}
  [CommRing E] [Module Zq E] [Fintype Coord] [DecidableEq Coord] [Fintype Slot] [Nonempty Slot]
  (L : Level O fold E Wire Coord Slot Result)

theorem charge_zero : L.extraction.charge 0 = S.blocks * L.chance := by
  unfold Structure.charge
  rw [ite_eq_left (show 0 < L.extraction.protocol.stages from Nat.succ_pos _)]
  show (((List.finRange S.blocks).map L.block).map fun _ => L.chance).sum = _
  rw [List.map_map]
  show ((List.finRange S.blocks).map fun _ => L.chance).sum = _
  simp

theorem charge_succ {index : ℕ} (inside : index < L.costs.length) :
    L.extraction.charge (index + 1) = L.costs.getD index 0 := by
  unfold Structure.charge
  rw [ite_eq_left (show index + 1 < L.extraction.protocol.stages from Nat.succ_lt_succ inside)]
  rcases index with _ | index <;> rfl

/-- `ε_ck` of one fold: the weights of the fold challenges of the blocks, and the bounds of the
draws of the checks. -/
theorem checkError_eq : L.extraction.checkError = S.blocks * L.chance + L.costs.sum := by
  unfold Structure.checkError
  show ∑ level ∈ Finset.range (L.costs.length + 1), L.extraction.charge level = _
  rw [Finset.sum_range_succ', L.charge_zero, add_comm, ← sum_range_getD]
  exact congrArg _ (Finset.sum_congr rfl fun index member =>
    L.charge_succ (Finset.mem_range.mp member))

/-- The extractor has one retry loop for each block and one for the checkpoint. -/
theorem loopsFrom_eq (draws : L.costs ≠ []) : L.extraction.loopsFrom 0 = S.blocks + 1 := by
  have long : 0 < L.costs.length := List.length_pos_iff.mpr draws
  unfold Structure.loopsFrom
  show ∑ later ∈ Finset.Ico 0 (L.costs.length + 1), L.extraction.loops later = _
  rw [Nat.Ico_zero_eq_range, Finset.sum_range_succ']
  have first : L.extraction.loops 0 = S.blocks := by
    show ((List.finRange S.blocks).map L.block).length = _
    simp
  have later : ∀ index, L.extraction.loops (index + 1) = if index = 0 then 1 else 0 := by
    intro index
    rcases index with _ | index <;> rfl
  rw [first, Finset.sum_congr rfl fun index _ => later index, Finset.sum_ite_eq',
    ite_eq_left (Finset.mem_range.mpr long), add_comm]

/-- The tree size `K̂` of one fold: `1 + r` children at the fold challenge of `r` blocks, and two
at the checkpoint. -/
theorem treeSize_eq (draws : L.costs ≠ []) : L.extraction.treeSize = 2 * (S.blocks + 1) := by
  have long : 0 < L.costs.length := List.length_pos_iff.mpr draws
  unfold Structure.treeSize Structure.treeFrom
  show ∏ later ∈ Finset.Ico 0 (L.costs.length + 1), L.extraction.arity later = _
  rw [Nat.Ico_zero_eq_range, Finset.prod_range_succ']
  have first : L.extraction.arity 0 = 1 + S.blocks := by
    show 1 + ((List.finRange S.blocks).map L.block).length = _
    simp
  have later : ∀ index, L.extraction.arity (index + 1) = if index = 0 then 2 else 1 := by
    intro index
    rcases index with _ | index <;> rfl
  rw [first, Finset.prod_congr rfl fun index _ => later index, Finset.prod_ite_eq',
    ite_eq_left (Finset.mem_range.mpr long)]
  ring

/-- An output of the extractor for one fold is correct: the verifier accepts the transcript of the
prover, and the output is a weak opening of the statement with `κ̄_1 = 2ω` and
`β̄_∞ = Δ^cert_f` (Akita Definition 10.22), or a solution of Module-SIS for `B` or `D` with the
bound `b⋆ − 1` or for a compression map with the bound 1 (Akita Corollary 10.32). -/
theorem extract_sound {peak : ℕ} (valid : L.Valid peak)
    (prover : OracleComp L.Stage (Coord → Slot) (ℕ → Wire)) (budget : ℕ)
    (table : L.Stage → Coord → Slot)
    (tape : Tape L.Stage Coord Slot budget L.extraction.protocol.stages)
    (out : L.Cert ⊕ L.Break) (result : Checkpoint.extract L.extraction prover budget table tape = some out) :
    L.Accepts (L.extraction.protocol.final prover table) ∧
      Sum.elim (fun certificate => L.Opens certificate.opening)
        (fun solution => solution.isSome) out :=
  Checkpoint.extract_sound L.extraction prover (L.lawful valid) budget table tape out result

variable [Finite Wire]

/-- Akita Theorem 10.17, in the form (161), for one fold. For each prover with at most `Q`
queries, the probability that the verifier accepts and the extractor returns no output is at most
`(Q + 1)·(r·p + Σ_j ε_j) + (Q + 1)·(r + 1)/budget`: `r` is the number of blocks, `p` is the largest
probability of one fold challenge, `ε_j` is the bound of draw `j` of the checks, and `budget` is
the limit of the attempts of a retry loop. The probability is over the table of the oracle and
the tape of the extractor. -/
theorem extraction_le {peak : ℕ} (valid : L.Valid peak)
    (prover : OracleComp L.Stage (Coord → Slot) (ℕ → Wire)) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape L.Stage Coord Slot budget L.extraction.protocol.stages,
        (if L.Accepts (L.extraction.protocol.final prover table) ∧
          Checkpoint.extract L.extraction prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * (S.blocks * L.chance + L.costs.sum) +
        (queries + 1) * (S.blocks + 1) / budget := by
  have bound := checkpoint_extraction_le L.extraction prover (L.lawful valid) bounded positive
  rw [L.checkError_eq, L.loopsFrom_eq valid.draws] at bound
  exact_mod_cast bound

/-- The expected number of runs of the prover in the extraction of one fold is at most
`K̂ + (Q + 1)·(K̂ − 1)` with `K̂ = 2·(r + 1)` (Akita Theorem 10.17, the cost). -/
theorem runs_le (draws : L.costs ≠ [])
    (prover : OracleComp L.Stage (Coord → Slot) (ℕ → Wire)) {queries : ℕ}
    (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape L.Stage Coord Slot budget L.extraction.protocol.stages,
        (extractRuns L.extraction prover budget table tape : ℝ) ≤
      2 * (S.blocks + 1) + (queries + 1) * (2 * (S.blocks + 1) - 1) := by
  have bound := extract_runs_le L.extraction prover bounded budget
  rw [L.treeSize_eq draws] at bound
  exact_mod_cast bound

end Level

end NightstreamFPrime.Spec.Compression.Akita.Extract
