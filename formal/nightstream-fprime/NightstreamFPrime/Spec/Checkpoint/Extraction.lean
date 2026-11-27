import NightstreamFPrime.Spec.Checkpoint.Indexed
import NightstreamFPrime.Spec.Checkpoint.Segment

/-! Akita Theorem 10.17, checkpoint extraction (ePrint 2026/1983, p. 78, proof in Appendix E.2),
in the restricted form of the proof path of Akita: indexed binary extraction stages, checkpoints
and plain stages. The environment and the tape of the prover are fixed, as in (284). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

variable {Addr Point Coord Slot Proof Cert Break Cand Key Witness : Type}
  [Fintype Addr] [DecidableEq Addr] [Fintype Point] [DecidableEq Point] [Fintype Coord]
  [DecidableEq Coord] [Fintype Slot] [Nonempty Slot]
  (structure_ : Structure Addr Point Coord Slot Proof Cert Break Cand Key Witness)
  (prover : OracleComp Addr (Coord → Slot) Proof)

/-- The probability that the verifier accepts and the node returns no output. -/
noncomputable def miss (budget depth level : ℕ) : ℝ :=
  𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget depth,
    if structure_.protocol.accept (structure_.protocol.final prover table) ∧
      node structure_ prover budget depth level table tape = none then (1 : ℝ) else 0

/-- The loss of one stage in the success recurrences of Appendix E.2. -/
noncomputable def stageLoss (level : ℕ) (table : Addr → Coord → Slot) : ℝ :=
  match structure_.kind level with
  | .indexed _ => ∑ key, if key ∈ structure_.protocol.asked prover table ∧
      structure_.protocol.rank key = level then structure_.charge level else 0
  | .checkpoint span => ∑ key, if key ∈ structure_.protocol.asked prover table ∧
      level ≤ structure_.protocol.rank key ∧ structure_.protocol.rank key ≤ level + span
      then structure_.error (structure_.protocol.rank key) else 0
  | .plain => 0

noncomputable def loss : ℕ → ℕ → (Addr → Coord → Slot) → ℝ
  | 0, _, _ => 0
  | depth + 1, level, table =>
      stageLoss structure_ prover level table + loss depth (level + 1) table

/-- `Σ_{a ∈ Addr(J)} ϵ(a)` over the keys of the stages from `level`. -/
noncomputable def chargeFrom (level : ℕ) (table : Addr → Coord → Slot) : ℝ :=
  ∑ key, if key ∈ structure_.protocol.asked prover table ∧
    level ≤ structure_.protocol.rank key then structure_.charge (structure_.protocol.rank key)
    else 0

omit [Fintype Point] [DecidableEq Point] [Nonempty Slot] in
private theorem chargeFrom_nonnegative (lawful : structure_.Lawful) (level : ℕ)
    (table : Addr → Coord → Slot) : 0 ≤ chargeFrom structure_ prover level table :=
  Finset.sum_nonneg fun _ _ => by split <;> simp [structure_.charge_nonnegative lawful]

omit [Fintype Point] [DecidableEq Point] [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Nonempty Slot] in
private theorem loss_plain (table : Addr → Coord → Slot) (count : ℕ) :
    ∀ depth level, count ≤ depth →
      (∀ later, level ≤ later → later < level + count → structure_.kind later = .plain) →
      loss structure_ prover depth level table =
        loss structure_ prover (depth - count) (level + count) table := by
  induction count with
  | zero => exact fun depth level _ _ => rfl
  | succ count induction =>
      intro depth level enough plain
      obtain ⟨depth, rfl⟩ : ∃ smaller, depth = smaller + 1 := ⟨depth - 1, by omega⟩
      have here : stageLoss structure_ prover level table = 0 := by
        unfold stageLoss
        rw [plain level le_rfl (by omega)]
      rw [loss, here, zero_add, induction depth (level + 1) (by omega)
        fun later low high => plain later (by omega) (by omega)]
      congr 1
      · omega
      · omega

omit [Fintype Addr] [DecidableEq Addr] [Fintype Point] [DecidableEq Point] [Nonempty Slot] in
private theorem charge_segment (lawful : structure_.Lawful) {level span : ℕ}
    (checkpoint : structure_.kind level = .checkpoint span) {stage : ℕ} (low : level ≤ stage)
    (high : stage ≤ level + span) : structure_.charge stage = structure_.error stage := by
  obtain ⟨inside, plain⟩ := lawful.segment level span checkpoint
  unfold Structure.charge
  rw [ite_eq_left (by omega)]
  rcases Nat.eq_or_lt_of_le low with rfl | above
  · rw [checkpoint]
  · rw [plain stage above high]

omit [Fintype Point] [DecidableEq Point] [Nonempty Slot] in
/-- The telescoped losses are at most the sum of the weights of the queried addresses: the error
of each plain stage appears once, because the stage lies in one segment (Appendix E.2,
"Conclusion"). -/
theorem loss_le_chargeFrom (lawful : structure_.Lawful) (table : Addr → Coord → Slot) :
    ∀ depth level, level + depth = structure_.protocol.stages →
      loss structure_ prover depth level table ≤ chargeFrom structure_ prover level table := by
  intro depth
  induction depth using Nat.strong_induction_on with
  | _ depth induction =>
      intro level full
      rcases depth with _ | depth
      · exact chargeFrom_nonnegative structure_ prover lawful level table
      · have inside : level < structure_.protocol.stages := by omega
        rw [loss]
        rcases kindEq : structure_.kind level with coordinates | span | _
        · refine (add_le_add le_rfl (induction depth (Nat.lt_succ_self depth) (level + 1)
            (by omega))).trans ?_
          unfold stageLoss chargeFrom
          rw [kindEq, ← Finset.sum_add_distrib]
          refine Finset.sum_le_sum fun key _ => ?_
          have nonnegative := structure_.charge_nonnegative lawful (structure_.protocol.rank key)
          by_cases queried : key ∈ structure_.protocol.asked prover table
          · rcases Nat.lt_trichotomy (structure_.protocol.rank key) level with less | equal | more
            · simp [queried, Nat.ne_of_lt less, Nat.not_le.mpr less,
                Nat.not_le.mpr (Nat.lt_succ_of_lt less)]
            · simp [queried, equal]
            · simp [queried, Nat.ne_of_gt more, Nat.le_of_lt more, Nat.succ_le_of_lt more]
          · simp [queried]
        · obtain ⟨short, plain⟩ := lawful.segment level span kindEq
          rw [loss_plain structure_ prover table span depth (level + 1) (by omega)
            fun later low high => plain later (by omega) (by omega)]
          refine (add_le_add le_rfl (induction (depth - span) (by omega) (level + 1 + span)
            (by omega))).trans ?_
          unfold stageLoss chargeFrom
          rw [kindEq, ← Finset.sum_add_distrib]
          refine Finset.sum_le_sum fun key _ => ?_
          have nonnegative := structure_.charge_nonnegative lawful (structure_.protocol.rank key)
          by_cases queried : key ∈ structure_.protocol.asked prover table
          · by_cases low : level ≤ structure_.protocol.rank key
            · by_cases high : structure_.protocol.rank key ≤ level + span
              · have none : ¬ level + 1 + span ≤ structure_.protocol.rank key := by omega
                simp [queried, low, high, none,
                  charge_segment structure_ lawful kindEq low high]
              · have far : level + 1 + span ≤ structure_.protocol.rank key := by omega
                simp [queried, low, high, far]
            · have none : ¬ level + 1 + span ≤ structure_.protocol.rank key := by omega
              simp [queried, low, none]
          · simp [queried]
        · refine (add_le_add le_rfl (induction depth (Nat.lt_succ_self depth) (level + 1)
            (by omega))).trans ?_
          unfold stageLoss chargeFrom
          rw [kindEq, zero_add]
          refine Finset.sum_le_sum fun key _ => ?_
          have nonnegative := structure_.charge_nonnegative lawful (structure_.protocol.rank key)
          by_cases queried : key ∈ structure_.protocol.asked prover table ∧
              level + 1 ≤ structure_.protocol.rank key
          · rw [ite_eq_left queried, ite_eq_left ⟨queried.1, Nat.le_of_succ_le queried.2⟩]
          · rw [ite_eq_right queried]
            split <;> linarith

omit [Fintype Point] in
private theorem miss_succ_le (budget depth level : ℕ)
    (parent : (Addr → Coord → Slot) → Tape Addr Coord Slot budget depth →
      Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth) → Option (Cert ⊕ Break))
    (same : ∀ table tape, node structure_ prover budget (depth + 1) level table tape =
      parent table tape.own tape.loops) :
    miss structure_ prover budget (depth + 1) level ≤
      miss structure_ prover budget depth (level + 1) +
        𝔼 table, 𝔼 own, 𝔼 loops,
          (if (node structure_ prover budget depth (level + 1) table own).isSome ∧
            parent table own loops = none then (1 : ℝ) else 0) := by
  unfold miss
  simp only [same]
  calc
    _ = 𝔼 table, 𝔼 own, 𝔼 loops,
          (if structure_.protocol.accept (structure_.protocol.final prover table) ∧
            parent table own loops = none then (1 : ℝ) else 0) :=
      Finset.expect_congr rfl fun table _ => Tape.expect_succ fun own loops =>
        if structure_.protocol.accept (structure_.protocol.final prover table) ∧
          parent table own loops = none then (1 : ℝ) else 0
    _ ≤ 𝔼 table, 𝔼 own, 𝔼 loops,
          ((if structure_.protocol.accept (structure_.protocol.final prover table) ∧
              node structure_ prover budget depth (level + 1) table own = none
              then (1 : ℝ) else 0) +
            (if (node structure_ prover budget depth (level + 1) table own).isSome ∧
              parent table own loops = none then (1 : ℝ) else 0)) := by
      refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun own _ =>
        Finset.expect_le_expect fun loops _ => ?_
      rcases child : node structure_ prover budget depth (level + 1) table own with _ | out
      · by_cases accepts : structure_.protocol.accept (structure_.protocol.final prover table) <;>
          by_cases missed : parent table own loops = none <;> simp [accepts, missed]
      · by_cases accepts : structure_.protocol.accept (structure_.protocol.final prover table) <;>
          by_cases missed : parent table own loops = none <;> simp [accepts, missed]
    _ = _ := by
      simp only [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty]

/-- The telescoped success recurrences of Appendix E.2 for the stages from `level`. -/
theorem miss_le (lawful : structure_.Lawful) {queries : ℕ} (spread : structure_.protocol.Spread prover queries)
    {budget : ℕ} (positive : 0 < budget) :
    ∀ depth level, level + depth = structure_.protocol.stages →
      miss structure_ prover budget depth level ≤
        (𝔼 table, loss structure_ prover depth level table) +
          (queries + 1) * structure_.loopsFrom level / budget := by
  intro depth
  induction depth with
  | zero =>
      intro level _
      have zero : miss structure_ prover budget 0 level = 0 := by
        unfold miss
        refine Finset.expect_eq_zero fun table _ => Finset.expect_eq_zero fun tape _ =>
          ite_eq_right ?_
        rintro ⟨accepts, missed⟩
        simp [node, accepts] at missed
      rw [zero]
      simp only [loss, Finset.expect_const_zero, zero_add]
      positivity
  | succ depth induction =>
      intro level full
      have inside : level < structure_.protocol.stages := by omega
      have below := induction (level + 1) (by omega)
      have childSound := node_sound structure_ prover lawful budget depth (level + 1) (by omega)
      have accepts : ∀ table tape,
          (node structure_ prover budget depth (level + 1) table tape).isSome →
            structure_.protocol.accept (structure_.protocol.final prover table) := by
        intro table tape ran
        obtain ⟨out, result⟩ := Option.isSome_iff_exists.mp ran
        exact (childSound table tape out result).1
      have split : (𝔼 table, loss structure_ prover (depth + 1) level table) =
          (𝔼 table, stageLoss structure_ prover level table) +
            𝔼 table, loss structure_ prover depth (level + 1) table := by
        simp only [loss, Finset.expect_add_distrib]
      have loops : ((structure_.loopsFrom level : ℕ) : ℝ) =
          structure_.loops level + structure_.loopsFrom (level + 1) := by
        rw [structure_.loopsFrom_succ inside]
        push_cast
        rfl
      have budgetPositive : (0 : ℝ) < budget := by exact_mod_cast positive
      rw [split, loops, mul_add, add_div]
      rcases kindEq : structure_.kind level with coordinates | span | _
      · refine (miss_succ_le structure_ prover budget depth level _
          (node_indexed structure_ prover budget kindEq)).trans ?_
        have stage := indexed_miss_le structure_ prover lawful kindEq inside spread positive
          accepts
        have here : stageLoss structure_ prover level = fun table =>
            ∑ key, if key ∈ structure_.protocol.asked prover table ∧
              structure_.protocol.rank key = level then structure_.charge level else 0 := by
          funext table
          unfold stageLoss
          rw [kindEq]
        rw [here]
        linarith
      · obtain ⟨short, plain⟩ := lawful.segment level span kindEq
        refine (miss_succ_le structure_ prover budget depth level _
          (node_checkpoint structure_ prover budget kindEq)).trans ?_
        have deep : level + span + 1 ≤ structure_.frontier depth (level + 1) := by
          have := structure_.plain_le_frontier span depth (level + 1) (by omega)
            fun later low high => plain later (by omega) (by omega)
          omega
        have stage := checkpoint_miss_le structure_ prover lawful kindEq deep spread positive
          childSound
        have here : stageLoss structure_ prover level = fun table =>
            ∑ key, if key ∈ structure_.protocol.asked prover table ∧
              level ≤ structure_.protocol.rank key ∧
              structure_.protocol.rank key ≤ level + span
              then structure_.error (structure_.protocol.rank key) else 0 := by
          funext table
          unfold stageLoss
          rw [kindEq]
        have one : ((structure_.loops level : ℕ) : ℝ) = 1 := by
          unfold Structure.loops
          rw [kindEq]
          norm_num
        rw [here, one, mul_one]
        linarith
      · refine (miss_succ_le structure_ prover budget depth level
          (fun table own _ => node structure_ prover budget depth (level + 1) table own)
          (node_plain structure_ prover budget kindEq)).trans ?_
        have none : (𝔼 table, 𝔼 own : Tape Addr Coord Slot budget depth,
            𝔼 _loops : Loops Addr Coord Slot budget (Tape Addr Coord Slot budget depth),
            (if (node structure_ prover budget depth (level + 1) table own).isSome ∧
              node structure_ prover budget depth (level + 1) table own = none
              then (1 : ℝ) else 0)) = 0 := by
          refine Finset.expect_eq_zero fun table _ => Finset.expect_eq_zero fun own _ =>
            Finset.expect_eq_zero fun _ _ => ite_eq_right ?_
          rintro ⟨ran, missed⟩
          rw [missed] at ran
          exact absurd ran (by simp)
        have here : stageLoss structure_ prover level = fun _ => 0 := by
          funext table
          unfold stageLoss
          rw [kindEq]
        have zero : ((structure_.loops level : ℕ) : ℝ) = 0 := by
          unfold Structure.loops
          rw [kindEq]
          norm_num
        rw [none, here, zero]
        simp only [Finset.expect_const_zero, mul_zero, zero_div, zero_add, add_zero]
        exact below

/-- Akita Theorem 10.17, first inequality of (160), for a fixed environment (this is (284)), with
a limit `budget` on the attempts of each retry loop of the extractor. The probability that the
verifier accepts and the extractor returns no output is at most `E[Σ_{a ∈ Addr(J)} ϵ(a)]` plus the
loss of the limit, `(Q + 1)·L/budget`, where `L` is the number of retry loops of one node of each
stage. By `extract_sound`, an output is a witness of the relation or a witness of `R_bind`. -/
theorem checkpoint_extraction (lawful : structure_.Lawful) {queries : ℕ}
    (spread : structure_.protocol.Spread prover queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget structure_.protocol.stages,
        (if structure_.protocol.accept (structure_.protocol.final prover table) ∧
          extract structure_ prover budget table tape = none then (1 : ℝ) else 0) ≤
      (𝔼 table, ∑ key, if key ∈ structure_.protocol.asked prover table
        then structure_.charge (structure_.protocol.rank key) else 0) +
        (queries + 1) * structure_.loopsFrom 0 / budget := by
  refine (miss_le structure_ prover lawful spread positive structure_.protocol.stages 0
    (Nat.zero_add _)).trans (add_le_add (Finset.expect_le_expect fun table _ => ?_) le_rfl)
  refine (loss_le_chargeFrom structure_ prover lawful table structure_.protocol.stages 0
    (Nat.zero_add _)).trans (le_of_eq ?_)
  unfold chargeFrom
  simp

omit [Fintype Point] [DecidableEq Point] in
/-- Akita Theorem 10.17, second inequality of (160): the addresses of the output transcript have
total weight `ε_ck`, and the prover queries at most `Q` further addresses, each of weight at most
`ϵ_max`. -/
theorem address_sum_le (lawful : structure_.Lawful) {queries : ℕ}
    (bounded : prover.QueryBound queries) (top : ℝ)
    (below : ∀ level, structure_.charge level ≤ top) :
    𝔼 table, (∑ key, if key ∈ structure_.protocol.asked prover table
        then structure_.charge (structure_.protocol.rank key) else 0) ≤
      structure_.checkError + queries * top :=
  (Finset.expect_le_expect fun table _ => structure_.protocol.sum_asked_le prover bounded table
    structure_.charge top (structure_.charge_nonnegative lawful) below).trans
    (Finset.expect_const Finset.univ_nonempty _).le

/-- The familiar form (161) of Akita Theorem 10.17, with the loss of the limit of the loops:
`(Q + 1)·ε_ck + (Q + 1)·L/budget`, for a prover with at most `Q` different queried keys of each
stage (`Protocol.Spread`). -/
theorem checkpoint_extraction_spread_le (lawful : structure_.Lawful) {queries : ℕ}
    (spread : structure_.protocol.Spread prover queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget structure_.protocol.stages,
        (if structure_.protocol.accept (structure_.protocol.final prover table) ∧
          extract structure_ prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * structure_.checkError + (queries + 1) * structure_.loopsFrom 0 / budget := by
  have outside : ∀ level, structure_.protocol.stages ≤ level → structure_.charge level = 0 := by
    intro level beyond
    unfold Structure.charge
    rw [ite_eq_right (Nat.not_lt.mpr beyond)]
  have first := checkpoint_extraction structure_ prover lawful spread positive
  have second : 𝔼 table, (∑ key, if key ∈ structure_.protocol.asked prover table
      then structure_.charge (structure_.protocol.rank key) else 0) ≤
        (queries + 1) * structure_.checkError :=
    (Finset.expect_le_expect fun table _ => structure_.protocol.sum_ranked_le prover spread table
      structure_.charge (structure_.charge_nonnegative lawful) outside).trans
      (Finset.expect_const Finset.univ_nonempty _).le
  linarith

/-- The familiar form (161) of Akita Theorem 10.17, with the loss of the limit of the loops:
`(Q + 1)·ε_ck + (Q + 1)·L/budget`, for a prover with at most `Q` queries. -/
theorem checkpoint_extraction_le (lawful : structure_.Lawful) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget structure_.protocol.stages,
        (if structure_.protocol.accept (structure_.protocol.final prover table) ∧
          extract structure_ prover budget table tape = none then (1 : ℝ) else 0) ≤
      (queries + 1) * structure_.checkError + (queries + 1) * structure_.loopsFrom 0 / budget :=
  checkpoint_extraction_spread_le structure_ prover lawful (bounded.spread _) positive

end NightstreamFPrime.Spec.Checkpoint
