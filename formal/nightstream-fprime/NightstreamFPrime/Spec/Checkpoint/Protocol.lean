import NightstreamFPrime.Spec.RandomOracle

/-! The Fiat–Shamir protocol shape of Akita Section 10.3 (ePrint 2026/1983, revision of 2026-11-23,
"Extraction with checkpoints", p. 76), and the run `A^J` of a prover against a finite oracle
table.

The oracle reads the key `⟨ξ_r⟩` of a stage input, not the stage input. The answer of the table at
one key is one long output. A prover query to a key returns the whole answer and counts once. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle

attribute [local instance low] Classical.propDecidable

/-- The stage inputs `ξ_r` of a protocol with stages `0, …, stages − 1`, and their keys. A `Point`
is a stage input: it encodes its stage and the transcript before the answer of that stage. A point
of stage `stages` is a complete transcript. `root` is `ξ_0`, `step` extends a stage input by the
answer at it and by the next message of the proof, `cut level` is the stage-`level` input that a
point extends, and `last` is the answer that the last step added. An `Addr` is a key: the stage and
the messages of a stage input. `address` is `⟨ξ⟩`, `rank` is the stage of a key, `trim level` is
`k_{≤ level}`, and `rebuild J` is `T_J`. The laws state that the encoding is injective in the way
that the paper uses; `address_step` states that a key reads no answer, and `rebuild_congr` is (F1).
The laws of `step` on the stage and on the last answer are for a stage input, not for a complete
transcript: with these laws for each point, no finite type of points satisfies them. -/
structure Protocol (Addr Point Reply Proof : Type) where
  stages : ℕ
  root : Proof → Point
  step : Proof → Point → Reply → Point
  stage : Point → ℕ
  cut : ℕ → Point → Option Point
  last : Point → Reply
  accept : Point → Prop
  address : Point → Addr
  rank : Addr → ℕ
  trim : ℕ → Addr → Option Addr
  rebuild : (Addr → Reply) → Addr → Point
  stage_root : ∀ proof, stage (root proof) = 0
  stage_step : ∀ proof point reply, stage point < stages →
    stage (step proof point reply) = stage point + 1
  cut_self : ∀ point, cut (stage point) point = some point
  cut_step : ∀ proof point reply level, level ≤ stage point →
    cut level (step proof point reply) = cut level point
  cut_stage : ∀ level point anchor, cut level point = some anchor → level ≤ stage point
  last_step : ∀ proof point reply, stage point < stages → last (step proof point reply) = reply
  rank_address : ∀ point, rank (address point) = stage point
  trim_address : ∀ level point, trim level (address point) = (cut level point).map address
  trim_rank : ∀ level key anchor, trim level key = some anchor →
    rank anchor = level ∧ level ≤ rank key
  address_rebuild : ∀ table key, address (rebuild table key) = key
  address_step : ∀ proof point other reply answer, address point = address other →
    address (step proof point reply) = address (step proof other answer)
  rebuild_root : ∀ table proof, rebuild table (address (root proof)) = root proof
  rebuild_step : ∀ table proof point, stage point < stages → rebuild table (address point) = point →
    rebuild table (address (step proof point (table (address point)))) =
      step proof point (table (address point))
  rebuild_congr : ∀ table other key,
    (∀ earlier, rank earlier < rank key → table earlier = other earlier) →
      rebuild table key = rebuild other key

variable {Addr Point Reply Proof : Type}

namespace Protocol

variable (protocol : Protocol Addr Point Reply Proof)

/-- The stage inputs of the transcript that a proof and a table give. -/
def path (proof : Proof) (table : Addr → Reply) : ℕ → Point
  | 0 => protocol.root proof
  | level + 1 => protocol.step proof (path proof table level)
      (table (protocol.address (path proof table level)))

/-- The verifier queries the key of each stage input of the transcript in turn. -/
def walk (proof : Proof) : ℕ → Point → OracleComp Addr Reply Point
  | 0, point => .done point
  | count + 1, point =>
      .query (protocol.address point) fun reply =>
        walk proof count (protocol.step proof point reply)

/-- `A^J`: the prover, then the verifier, which queries every challenge address of the output
transcript. The output is the complete transcript. -/
def attack : OracleComp Addr Reply Proof → OracleComp Addr Reply Point
  | .done proof => protocol.walk proof protocol.stages (protocol.root proof)
  | .query key next => .query key fun reply => attack (next reply)

theorem run_walk (proof : Proof) (table : Addr → Reply) (count level : ℕ) :
    (protocol.walk proof count (protocol.path proof table level)).run table =
      protocol.path proof table (level + count) := by
  induction count generalizing level with
  | zero => rfl
  | succ count induction =>
      change (protocol.walk proof count (protocol.path proof table (level + 1))).run table = _
      rw [induction]
      congr 1
      omega

theorem queries_walk (proof : Proof) (table : Addr → Reply) (count level : ℕ) :
    (protocol.walk proof count (protocol.path proof table level)).queries table =
      (List.range count).map fun offset =>
        protocol.address (protocol.path proof table (level + offset)) := by
  induction count generalizing level with
  | zero => rfl
  | succ count induction =>
      change protocol.address (protocol.path proof table level) ::
        (protocol.walk proof count (protocol.path proof table (level + 1))).queries table = _
      rw [induction, List.range_succ_eq_map, List.map_cons, List.map_map]
      congr 1
      exact List.map_congr_left fun offset _ => by
        change protocol.address (protocol.path proof table (level + 1 + offset)) =
          protocol.address (protocol.path proof table (level + (offset + 1)))
        congr 2
        omega

theorem path_congr (proof : Proof) (table other : Addr → Reply) (level : ℕ)
    (agree : ∀ earlier < level, table (protocol.address (protocol.path proof table earlier)) =
      other (protocol.address (protocol.path proof table earlier))) :
    protocol.path proof other level = protocol.path proof table level := by
  induction level with
  | zero => rfl
  | succ level induction =>
      have before := induction fun earlier less => agree earlier (Nat.lt_succ_of_lt less)
      simp only [path]
      rw [before, agree level (Nat.lt_succ_self level)]

theorem stage_path (proof : Proof) (table : Addr → Reply) {level : ℕ}
    (inside : level ≤ protocol.stages) :
    protocol.stage (protocol.path proof table level) = level := by
  induction level with
  | zero => exact protocol.stage_root proof
  | succ level induction =>
      have before := induction (Nat.le_of_succ_le inside)
      rw [path, protocol.stage_step _ _ _ (by rw [before]; exact inside), before]

theorem cut_path (proof : Proof) (table : Addr → Reply) {level deep : ℕ} (below : level ≤ deep)
    (inside : deep ≤ protocol.stages) :
    protocol.cut level (protocol.path proof table deep) = some (protocol.path proof table level) := by
  induction deep with
  | zero =>
      obtain rfl : level = 0 := Nat.le_zero.mp below
      have same := protocol.cut_self (protocol.path proof table 0)
      rwa [protocol.stage_path proof table inside] at same
  | succ deep induction =>
      rcases Nat.lt_or_ge deep level with above | under
      · obtain rfl : level = deep + 1 := by omega
        have same := protocol.cut_self (protocol.path proof table (deep + 1))
        rwa [protocol.stage_path proof table inside] at same
      · rw [path, protocol.cut_step _ _ _ _
          (by rw [protocol.stage_path proof table (Nat.le_of_succ_le inside)]; exact under)]
        exact induction under (Nat.le_of_succ_le inside)

/-- `ξ_r = T_J(⟨ξ_r⟩)` on the transcript of a proof and a table. -/
theorem rebuild_path (proof : Proof) (table : Addr → Reply) {level : ℕ}
    (inside : level ≤ protocol.stages) :
    protocol.rebuild table (protocol.address (protocol.path proof table level)) =
      protocol.path proof table level := by
  induction level with
  | zero => exact protocol.rebuild_root table proof
  | succ level induction =>
      exact protocol.rebuild_step table proof _
        (by rw [protocol.stage_path proof table (Nat.le_of_succ_le inside)]; exact inside)
        (induction (Nat.le_of_succ_le inside))

/-- A key reads no answer (Section 10.3): the keys of the transcript of a proof do not depend on
the table. -/
theorem address_path (proof : Proof) (table other : Addr → Reply) (level : ℕ) :
    protocol.address (protocol.path proof table level) =
      protocol.address (protocol.path proof other level) := by
  induction level with
  | zero => rfl
  | succ level induction => exact protocol.address_step proof _ _ _ _ induction

theorem stage_rebuild (table : Addr → Reply) (key : Addr) :
    protocol.stage (protocol.rebuild table key) = protocol.rank key := by
  rw [← protocol.rank_address, protocol.address_rebuild]

/-- (F1) for a table that differs at one key: the input rebuilt from that key does not change. -/
theorem rebuild_update [DecidableEq Addr] (table : Addr → Reply) (key : Addr) (reply : Reply) :
    protocol.rebuild (Function.update table key reply) key = protocol.rebuild table key :=
  protocol.rebuild_congr _ _ key fun earlier less =>
    Function.update_of_ne (fun same => by subst same; exact lt_irrefl _ less) reply table

/-- For a fixed table, a key has one reconstructed input (Section 10.3): the stage inputs `ξ` with
`T_J(⟨ξ⟩) = ξ` are in bijection with the keys. -/
theorem sum_rebuild [Fintype Addr] [Fintype Point] [DecidableEq Point] (table : Addr → Reply)
    (value : Addr → ℝ) :
    ∑ point, (if protocol.rebuild table (protocol.address point) = point
      then value (protocol.address point) else 0) = ∑ key, value key := by
  rw [← Finset.sum_filter]
  exact Finset.sum_nbij' protocol.address (protocol.rebuild table) (fun _ _ => Finset.mem_univ _)
    (fun key _ => by simp [protocol.address_rebuild])
    (fun point member => (Finset.mem_filter.mp member).2)
    (fun key _ => protocol.address_rebuild table key)
    fun _ _ => rfl

variable (prover : OracleComp Addr Reply Proof)

/-- `ξ_level(J)`: the stage-`level` input of the transcript of `A^J`. -/
def input (table : Addr → Reply) (level : ℕ) : Point :=
  protocol.path (prover.run table) table level

/-- The complete transcript of `A^J`. -/
def final (table : Addr → Reply) : Point :=
  protocol.input prover table protocol.stages

/-- `Addr(J)`: the keys that `A^J` queries, the queries of the prover first. -/
def asked (table : Addr → Reply) : List Addr :=
  (protocol.attack prover).queries table

theorem asked_eq (table : Addr → Reply) :
    protocol.asked prover table =
      prover.queries table ++ (List.range protocol.stages).map
        fun level => protocol.address (protocol.input prover table level) := by
  unfold asked input
  induction prover with
  | done proof =>
      have walked := protocol.queries_walk proof table protocol.stages 0
      simp only [Nat.zero_add] at walked
      exact walked
  | query key next induction =>
      exact congrArg (key :: ·) (induction (table key))

theorem mem_asked_iff (table : Addr → Reply) (key : Addr) :
    key ∈ protocol.asked prover table ↔ key ∈ prover.queries table ∨
      ∃ level < protocol.stages, protocol.address (protocol.input prover table level) = key := by
  simp [asked_eq, List.mem_append, List.mem_map, List.mem_range]

theorem input_mem_asked (table : Addr → Reply) {level : ℕ} (inside : level < protocol.stages) :
    protocol.address (protocol.input prover table level) ∈ protocol.asked prover table :=
  (protocol.mem_asked_iff prover table _).mpr (Or.inr ⟨level, inside, rfl⟩)

theorem asked_update [DecidableEq Addr] (table : Addr → Reply) (key : Addr) (reply : Reply) :
    key ∈ protocol.asked prover (Function.update table key reply) ↔
      key ∈ protocol.asked prover table :=
  OracleComp.mem_queries_update_iff table key reply (protocol.attack prover)

theorem stage_input (table : Addr → Reply) {level : ℕ} (inside : level ≤ protocol.stages) :
    protocol.stage (protocol.input prover table level) = level :=
  protocol.stage_path _ table inside

theorem cut_input (table : Addr → Reply) {level deep : ℕ} (below : level ≤ deep)
    (inside : deep ≤ protocol.stages) :
    protocol.cut level (protocol.input prover table deep) =
      some (protocol.input prover table level) :=
  protocol.cut_path _ table below inside

theorem last_input (table : Addr → Reply) {level : ℕ} (inside : level < protocol.stages) :
    protocol.last (protocol.input prover table (level + 1)) =
      table (protocol.address (protocol.input prover table level)) :=
  protocol.last_step _ _ _ (by
    rw [show protocol.path (prover.run table) table level = protocol.input prover table level
      from rfl, protocol.stage_input prover table (Nat.le_of_lt inside)]
    exact inside)

/-- `ξ_r(J) = T_J(⟨ξ_r(J)⟩)`. -/
theorem rebuild_input (table : Addr → Reply) {level : ℕ} (inside : level ≤ protocol.stages) :
    protocol.rebuild table (protocol.address (protocol.input prover table level)) =
      protocol.input prover table level :=
  protocol.rebuild_path _ table inside

theorem input_congr (table other : Addr → Reply) (level : ℕ)
    (agreeProver : ∀ key ∈ prover.queries table, table key = other key)
    (agreePath : ∀ earlier < level,
      table (protocol.address (protocol.input prover table earlier)) =
        other (protocol.address (protocol.input prover table earlier))) :
    protocol.input prover other level = protocol.input prover table level := by
  unfold input at agreePath ⊢
  rw [OracleComp.run_congr table other prover agreeProver]
  exact protocol.path_congr _ table other level agreePath

/-- `A^J` queries at most one key of each stage besides the queries of the prover. -/
theorem stage_asked_le [Fintype Addr] [DecidableEq Addr] {queries : ℕ}
    (bounded : prover.QueryBound queries) (table : Addr → Reply) (level : ℕ) :
    ∑ key, (if key ∈ protocol.asked prover table ∧ protocol.rank key = level
      then (1 : ℝ) else 0) ≤ queries + 1 := by
  have inside : (Finset.univ.filter fun key =>
      key ∈ protocol.asked prover table ∧ protocol.rank key = level) ⊆
        (prover.queries table).toFinset ∪
          {protocol.address (protocol.input prover table level)} := by
    intro key member
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, mem_asked_iff] at member
    simp only [Finset.mem_union, List.mem_toFinset, Finset.mem_singleton]
    rcases member.1 with queried | ⟨other, inside, rfl⟩
    · exact Or.inl queried
    · rw [protocol.rank_address, protocol.stage_input prover table (Nat.le_of_lt inside)] at member
      exact Or.inr (by rw [member.2])
  rw [Finset.sum_boole]
  have count := (Finset.card_le_card inside).trans (Finset.card_union_le _ _)
  have short := (List.toFinset_card_le (prover.queries table)).trans
    (bounded.queries_length_le table)
  rw [Finset.card_singleton] at count
  exact_mod_cast count.trans (Nat.add_le_add_right short 1)

/-- A sum over the queried keys is at most the sum over the queries of the prover and the stages
of the transcript. -/
theorem sum_asked_le [Fintype Addr] [DecidableEq Addr] {queries : ℕ}
    (bounded : prover.QueryBound queries) (table : Addr → Reply) (charge : ℕ → ℝ) (top : ℝ)
    (nonnegative : ∀ level, 0 ≤ charge level) (below : ∀ level, charge level ≤ top) :
    ∑ key, (if key ∈ protocol.asked prover table then charge (protocol.rank key) else 0) ≤
      (∑ level ∈ Finset.range protocol.stages, charge level) + queries * top := by
  have topNonnegative : 0 ≤ top := (nonnegative 0).trans (below 0)
  have split (key : Addr) :
      (if key ∈ protocol.asked prover table then charge (protocol.rank key) else 0) ≤
        (if ∃ level < protocol.stages, protocol.address (protocol.input prover table level) = key
          then charge (protocol.rank key) else 0) +
        (if key ∈ prover.queries table then top else 0) := by
    by_cases member : key ∈ protocol.asked prover table
    · rw [ite_eq_left member]
      rcases (protocol.mem_asked_iff prover table key).mp member with queried | onPath
      · rw [ite_eq_left queried]
        have := below (protocol.rank key)
        split <;> linarith [nonnegative (protocol.rank key)]
      · rw [ite_eq_left onPath]
        split <;> linarith
    · rw [ite_eq_right member]
      have := nonnegative (protocol.rank key)
      split <;> split <;> linarith
  refine (Finset.sum_le_sum fun key _ => split key).trans ?_
  rw [Finset.sum_add_distrib]
  refine add_le_add ?_ ?_
  · calc
      _ ≤ ∑ key, ∑ level ∈ Finset.range protocol.stages,
            (if protocol.address (protocol.input prover table level) = key
              then charge level else 0) := by
        refine Finset.sum_le_sum fun key _ => ?_
        split
        · rename_i onPath
          obtain ⟨level, inside, rfl⟩ := onPath
          rw [protocol.rank_address, protocol.stage_input prover table (Nat.le_of_lt inside)]
          refine le_trans ?_ (Finset.single_le_sum (fun other _ => ?_) (Finset.mem_range.mpr inside))
          · simp
          · split <;> simp [nonnegative]
        · exact Finset.sum_nonneg fun other _ => by split <;> simp [nonnegative]
      _ = ∑ level ∈ Finset.range protocol.stages, charge level := by
        rw [Finset.sum_comm]
        simp
  · have count : (∑ key, if key ∈ prover.queries table then (1 : ℝ) else 0) =
        ((prover.queries table).toFinset.card : ℝ) := by
      rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
      congr 2
      ext key
      simp
    calc
      _ = (∑ key, if key ∈ prover.queries table then (1 : ℝ) else 0) * top := by
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun _ _ => by split <;> simp
      _ ≤ queries * top := by
        refine mul_le_mul_of_nonneg_right ?_ topNonnegative
        rw [count]
        exact_mod_cast (List.toFinset_card_le _).trans (bounded.queries_length_le table)

end Protocol

section Table

variable {Coord Slot : Type} [Fintype Addr] [DecidableEq Addr] [Fintype Coord] [DecidableEq Coord]
  [Fintype Slot] [Nonempty Slot]

/-- The table with one slot of one answer replaced. -/
def setSlot (table : Addr → Coord → Slot) (key : Addr) (coordinate : Coord) (value : Slot) :
    Addr → Coord → Slot :=
  Function.update table key (Function.update (table key) coordinate value)

omit [Fintype Addr] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem setSlot_setSlot (table : Addr → Coord → Slot) (key : Addr) (coordinate : Coord)
    (value other : Slot) :
    setSlot (setSlot table key coordinate value) key coordinate other =
      setSlot table key coordinate other := by
  simp [setSlot]

omit [Fintype Addr] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem setSlot_apply (table : Addr → Coord → Slot) (key : Addr) (coordinate : Coord)
    (value : Slot) : setSlot table key coordinate value key coordinate = value := by
  simp [setSlot]

omit [Fintype Addr] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem setSlot_point (table : Addr → Coord → Slot) (key : Addr) (coordinate : Coord)
    (value : Slot) :
    setSlot table key coordinate value key = Function.update (table key) coordinate value := by
  simp [setSlot]

omit [Fintype Addr] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem Protocol.rebuild_setSlot (protocol : Protocol Addr Point (Coord → Slot) Proof)
    (table : Addr → Coord → Slot) (key : Addr) (coordinate : Coord) (value : Slot) :
    protocol.rebuild (setSlot table key coordinate value) key = protocol.rebuild table key :=
  protocol.rebuild_update table key _

/-- A uniform table is a uniform table with one slot drawn again: the slots of one answer are
independent. -/
theorem expect_setSlot (key : Addr) (coordinate : Coord)
    (value : (Addr → Coord → Slot) → ℝ) :
    𝔼 table, value table = 𝔼 table, 𝔼 slot, value (setSlot table key coordinate slot) :=
  expect_lens (fun table => table key coordinate)
    (fun table slot => setSlot table key coordinate slot)
    (fun table slot => setSlot_apply table key coordinate slot)
    (fun table => by simp [setSlot])
    (fun table slot other => setSlot_setSlot table key coordinate slot other) value

theorem expect_slot (key : Addr) (coordinate : Coord) (value : Slot → ℝ) :
    𝔼 table : Addr → Coord → Slot, value (table key coordinate) = 𝔼 slot, value slot := by
  rw [expect_eval key (fun reply : Coord → Slot => value (reply coordinate)),
    expect_eval coordinate value]

end Table

section Regrow

variable [Fintype Addr] [DecidableEq Addr] [Fintype Reply] [Nonempty Reply]
  (protocol : Protocol Addr Point Reply Proof)

/-- The table with a fresh table on the message subtree `Sub(anchor)`: the keys whose
stage-`level` key is `anchor`. -/
noncomputable def Protocol.regrow (level : ℕ) (anchor : Addr) (table fresh : Addr → Reply) :
    Addr → Reply :=
  fun key => if protocol.trim level key = some anchor then fresh key else table key

omit [Fintype Addr] [Fintype Reply] [Nonempty Reply] in
theorem Protocol.regrow_regrow (level : ℕ) (anchor : Addr) (table fresh other : Addr → Reply) :
    protocol.regrow level anchor (protocol.regrow level anchor table fresh) other =
      protocol.regrow level anchor table other := by
  funext key
  by_cases inside : protocol.trim level key = some anchor <;> simp [Protocol.regrow, inside]

omit [Fintype Addr] [Fintype Reply] [Nonempty Reply] in
theorem Protocol.regrow_update (level : ℕ) (anchor : Addr) (table fresh : Addr → Reply)
    {key : Addr} (inside : protocol.trim level key = some anchor) (reply : Reply) :
    protocol.regrow level anchor table (Function.update fresh key reply) =
      Function.update (protocol.regrow level anchor table fresh) key reply := by
  funext other
  by_cases same : other = key
  · subst same
    simp [Protocol.regrow, inside]
  · simp [Protocol.regrow, same]

omit [Fintype Addr] [Fintype Reply] [Nonempty Reply] in
/-- (F1): `Sub(anchor)` holds no key of a stage before `level`, so a fresh table on it keeps the
inputs rebuilt from the keys of the stages up to `level`. -/
theorem Protocol.rebuild_regrow {level : ℕ} {key : Addr} (low : protocol.rank key ≤ level)
    (anchor : Addr) (table fresh : Addr → Reply) :
    protocol.rebuild (protocol.regrow level anchor table fresh) key = protocol.rebuild table key :=
  protocol.rebuild_congr _ _ key fun earlier less => by
    have outside : protocol.trim level earlier ≠ some anchor := fun inside =>
      absurd (protocol.trim_rank level earlier anchor inside).2 (by omega)
    simp [Protocol.regrow, outside]

/-- A uniform table is a uniform table with a fresh uniform table on `Sub(anchor)`. -/
theorem Protocol.expect_regrow (level : ℕ) (anchor : Addr) (value : (Addr → Reply) → ℝ) :
    𝔼 table, value table =
      𝔼 table, 𝔼 fresh, value (protocol.regrow level anchor table fresh) := by
  rw [expect_resample
    (fun _ => Finset.univ.filter fun key => protocol.trim level key ≠ some anchor)
    (fun _ _ _ => rfl) value]
  refine Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun fresh _ => ?_
  congr 1
  funext key
  by_cases inside : protocol.trim level key = some anchor <;>
    simp [Protocol.regrow, overlay, inside]

end Regrow

end NightstreamFPrime.Spec.Checkpoint
