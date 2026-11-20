import NightstreamFPrime.Spec.RandomOracle

/-! The Fiat–Shamir protocol shape of Akita Section 10.3, "Extraction with checkpoints: conventions"
(ePrint 2026/1983, pp. 75–76), and the run `A^J` of a prover against a finite oracle table.

The oracle model is the model of one squeeze run. The answer of the table at one stage input is
one long output. A prover query to a stage input returns the whole answer and counts once. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle

attribute [local instance low] Classical.propDecidable

/-- The stage inputs `ξ_r` of a protocol with stages `0, …, stages − 1`. A `Point` is a stage
input: it encodes its stage and the transcript before the answer of that stage. A point of stage
`stages` is a complete transcript. `root` is `ξ_0`, `step` extends a stage input by the answer at
it and by the next message of the proof, `cut level` is the stage-`level` input that a point
extends, and `last` is the answer that the last step added. The laws state that the encoding is
injective in the way that the paper uses. -/
structure Protocol (Point Reply Proof : Type) where
  stages : ℕ
  root : Proof → Point
  step : Proof → Point → Reply → Point
  stage : Point → ℕ
  cut : ℕ → Point → Option Point
  last : Point → Reply
  accept : Point → Prop
  stage_root : ∀ proof, stage (root proof) = 0
  stage_step : ∀ proof point reply, stage (step proof point reply) = stage point + 1
  cut_self : ∀ point, cut (stage point) point = some point
  cut_step : ∀ proof point reply level, level ≤ stage point →
    cut level (step proof point reply) = cut level point
  cut_stage : ∀ level point anchor, cut level point = some anchor → level ≤ stage point
  last_step : ∀ proof point reply, last (step proof point reply) = reply

variable {Point Reply Proof : Type}

namespace Protocol

variable (protocol : Protocol Point Reply Proof)

/-- The stage inputs of the transcript that a proof and a table give. -/
def path (proof : Proof) (table : Point → Reply) : ℕ → Point
  | 0 => protocol.root proof
  | level + 1 => protocol.step proof (path proof table level) (table (path proof table level))

/-- The verifier queries each stage input of the transcript in turn. -/
def walk (proof : Proof) : ℕ → Point → OracleComp Point Reply Point
  | 0, point => .done point
  | count + 1, point =>
      .query point fun reply => walk proof count (protocol.step proof point reply)

/-- `A^J`: the prover, then the verifier, which queries every challenge address of the output
transcript. The output is the complete transcript. -/
def attack : OracleComp Point Reply Proof → OracleComp Point Reply Point
  | .done proof => protocol.walk proof protocol.stages (protocol.root proof)
  | .query point next => .query point fun reply => attack (next reply)

theorem run_walk (proof : Proof) (table : Point → Reply) (count level : ℕ) :
    (protocol.walk proof count (protocol.path proof table level)).run table =
      protocol.path proof table (level + count) := by
  induction count generalizing level with
  | zero => rfl
  | succ count induction =>
      change (protocol.walk proof count (protocol.path proof table (level + 1))).run table = _
      rw [induction]
      congr 1
      omega

theorem queries_walk (proof : Proof) (table : Point → Reply) (count level : ℕ) :
    (protocol.walk proof count (protocol.path proof table level)).queries table =
      (List.range count).map fun offset => protocol.path proof table (level + offset) := by
  induction count generalizing level with
  | zero => rfl
  | succ count induction =>
      change protocol.path proof table level ::
        (protocol.walk proof count (protocol.path proof table (level + 1))).queries table = _
      rw [induction, List.range_succ_eq_map, List.map_cons, List.map_map]
      congr 1
      exact List.map_congr_left fun offset _ => by
        change protocol.path proof table (level + 1 + offset) = protocol.path proof table (level + (offset + 1))
        congr 1
        omega

theorem path_congr (proof : Proof) (table other : Point → Reply) (level : ℕ)
    (agree : ∀ earlier < level,
      table (protocol.path proof table earlier) = other (protocol.path proof table earlier)) :
    protocol.path proof other level = protocol.path proof table level := by
  induction level with
  | zero => rfl
  | succ level induction =>
      have before := induction fun earlier less => agree earlier (Nat.lt_succ_of_lt less)
      simp only [path]
      rw [before, agree level (Nat.lt_succ_self level)]

theorem stage_path (proof : Proof) (table : Point → Reply) (level : ℕ) :
    protocol.stage (protocol.path proof table level) = level := by
  induction level with
  | zero => exact protocol.stage_root proof
  | succ level induction => rw [path, protocol.stage_step, induction]

theorem cut_path (proof : Proof) (table : Point → Reply) {level deep : ℕ} (below : level ≤ deep) :
    protocol.cut level (protocol.path proof table deep) = some (protocol.path proof table level) := by
  induction deep with
  | zero =>
      obtain rfl : level = 0 := Nat.le_zero.mp below
      simpa [protocol.stage_path] using protocol.cut_self (protocol.path proof table 0)
  | succ deep induction =>
      rcases Nat.lt_or_ge deep level with above | inside
      · obtain rfl : level = deep + 1 := by omega
        simpa [protocol.stage_path] using protocol.cut_self (protocol.path proof table (deep + 1))
      · rw [path, protocol.cut_step _ _ _ _ (by rw [protocol.stage_path]; exact inside)]
        exact induction inside

variable (prover : OracleComp Point Reply Proof)

/-- `ξ_level(J)`: the stage-`level` input of the transcript of `A^J`. -/
def input (table : Point → Reply) (level : ℕ) : Point :=
  protocol.path (prover.run table) table level

/-- The complete transcript of `A^J`. -/
def final (table : Point → Reply) : Point :=
  protocol.input prover table protocol.stages

/-- `Addr(J)`: the stage inputs that `A^J` queries, the queries of the prover first. -/
def asked (table : Point → Reply) : List Point :=
  (protocol.attack prover).queries table

theorem asked_eq (table : Point → Reply) :
    protocol.asked prover table =
      prover.queries table ++ (List.range protocol.stages).map (protocol.input prover table) := by
  unfold asked input
  induction prover with
  | done proof =>
      have walked := protocol.queries_walk proof table protocol.stages 0
      simp only [Nat.zero_add] at walked
      exact walked
  | query point next induction =>
      exact congrArg (point :: ·) (induction (table point))

theorem mem_asked_iff (table : Point → Reply) (point : Point) :
    point ∈ protocol.asked prover table ↔ point ∈ prover.queries table ∨
      ∃ level < protocol.stages, protocol.input prover table level = point := by
  simp [asked_eq, List.mem_append, List.mem_map, List.mem_range]

theorem input_mem_asked (table : Point → Reply) {level : ℕ} (inside : level < protocol.stages) :
    protocol.input prover table level ∈ protocol.asked prover table :=
  (protocol.mem_asked_iff prover table _).mpr (Or.inr ⟨level, inside, rfl⟩)

theorem asked_update [DecidableEq Point] (table : Point → Reply) (point : Point) (reply : Reply) :
    point ∈ protocol.asked prover (Function.update table point reply) ↔
      point ∈ protocol.asked prover table :=
  OracleComp.mem_queries_update_iff table point reply (protocol.attack prover)

theorem stage_input (table : Point → Reply) (level : ℕ) :
    protocol.stage (protocol.input prover table level) = level :=
  protocol.stage_path _ table level

theorem cut_input (table : Point → Reply) {level deep : ℕ} (below : level ≤ deep) :
    protocol.cut level (protocol.input prover table deep) =
      some (protocol.input prover table level) :=
  protocol.cut_path _ table below

theorem last_input (table : Point → Reply) (level : ℕ) :
    protocol.last (protocol.input prover table (level + 1)) =
      table (protocol.input prover table level) :=
  protocol.last_step _ _ _

theorem input_congr (table other : Point → Reply) (level : ℕ)
    (agreeProver : ∀ point ∈ prover.queries table, table point = other point)
    (agreePath : ∀ earlier < level,
      table (protocol.input prover table earlier) = other (protocol.input prover table earlier)) :
    protocol.input prover other level = protocol.input prover table level := by
  unfold input at agreePath ⊢
  rw [OracleComp.run_congr table other prover agreeProver]
  exact protocol.path_congr _ table other level agreePath

/-- `A^J` queries at most one stage input of each stage besides the queries of the prover. -/
theorem stage_asked_le [Fintype Point] [DecidableEq Point] {queries : ℕ} (bounded : prover.QueryBound queries)
    (table : Point → Reply) (level : ℕ) :
    ∑ point, (if point ∈ protocol.asked prover table ∧ protocol.stage point = level
      then (1 : ℝ) else 0) ≤ queries + 1 := by
  have inside : (Finset.univ.filter fun point =>
      point ∈ protocol.asked prover table ∧ protocol.stage point = level) ⊆
        (prover.queries table).toFinset ∪ {protocol.input prover table level} := by
    intro point member
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, mem_asked_iff] at member
    simp only [Finset.mem_union, List.mem_toFinset, Finset.mem_singleton]
    rcases member.1 with queried | ⟨other, _, rfl⟩
    · exact Or.inl queried
    · rw [stage_input] at member
      exact Or.inr (by rw [member.2])
  rw [Finset.sum_boole]
  have count := (Finset.card_le_card inside).trans (Finset.card_union_le _ _)
  have short := (List.toFinset_card_le (prover.queries table)).trans
    (bounded.queries_length_le table)
  rw [Finset.card_singleton] at count
  exact_mod_cast count.trans (Nat.add_le_add_right short 1)

/-- A sum over the queried stage inputs is at most the sum over the queries of the prover and the
stages of the transcript. -/
theorem sum_asked_le [Fintype Point] [DecidableEq Point] {queries : ℕ} (bounded : prover.QueryBound queries)
    (table : Point → Reply) (charge : ℕ → ℝ) (top : ℝ) (nonnegative : ∀ level, 0 ≤ charge level)
    (below : ∀ level, charge level ≤ top) :
    ∑ point, (if point ∈ protocol.asked prover table then charge (protocol.stage point) else 0) ≤
      (∑ level ∈ Finset.range protocol.stages, charge level) + queries * top := by
  have topNonnegative : 0 ≤ top := (nonnegative 0).trans (below 0)
  have split (point : Point) :
      (if point ∈ protocol.asked prover table then charge (protocol.stage point) else 0) ≤
        (if ∃ level < protocol.stages, protocol.input prover table level = point
          then charge (protocol.stage point) else 0) +
        (if point ∈ prover.queries table then top else 0) := by
    by_cases member : point ∈ protocol.asked prover table
    · rw [ite_eq_left member]
      rcases (protocol.mem_asked_iff prover table point).mp member with queried | onPath
      · rw [ite_eq_left queried]
        have := below (protocol.stage point)
        split <;> linarith [nonnegative (protocol.stage point)]
      · rw [ite_eq_left onPath]
        split <;> linarith
    · rw [ite_eq_right member]
      have := nonnegative (protocol.stage point)
      split <;> split <;> linarith
  refine (Finset.sum_le_sum fun point _ => split point).trans ?_
  rw [Finset.sum_add_distrib]
  refine add_le_add ?_ ?_
  · calc
      _ ≤ ∑ point, ∑ level ∈ Finset.range protocol.stages,
            (if protocol.input prover table level = point then charge level else 0) := by
        refine Finset.sum_le_sum fun point _ => ?_
        split
        · rename_i onPath
          obtain ⟨level, inside, rfl⟩ := onPath
          rw [protocol.stage_input]
          refine le_trans ?_ (Finset.single_le_sum (fun other _ => ?_) (Finset.mem_range.mpr inside))
          · simp
          · split <;> simp [nonnegative]
        · exact Finset.sum_nonneg fun other _ => by split <;> simp [nonnegative]
      _ = ∑ level ∈ Finset.range protocol.stages, charge level := by
        rw [Finset.sum_comm]
        simp
  · have count : (∑ point, if point ∈ prover.queries table then (1 : ℝ) else 0) =
        ((prover.queries table).toFinset.card : ℝ) := by
      rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_one]
      congr 2
      ext point
      simp
    calc
      _ = (∑ point, if point ∈ prover.queries table then (1 : ℝ) else 0) * top := by
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun _ _ => by split <;> simp
      _ ≤ queries * top := by
        refine mul_le_mul_of_nonneg_right ?_ topNonnegative
        rw [count]
        exact_mod_cast (List.toFinset_card_le _).trans (bounded.queries_length_le table)

end Protocol

section Table

variable {Coord Slot : Type} [Fintype Point] [DecidableEq Point] [Fintype Coord] [DecidableEq Coord]
  [Fintype Slot] [Nonempty Slot]

/-- The table with one slot of one answer replaced. -/
def setSlot (table : Point → Coord → Slot) (point : Point) (coordinate : Coord) (value : Slot) :
    Point → Coord → Slot :=
  Function.update table point (Function.update (table point) coordinate value)

omit [Fintype Point] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem setSlot_setSlot (table : Point → Coord → Slot) (point : Point) (coordinate : Coord)
    (value other : Slot) :
    setSlot (setSlot table point coordinate value) point coordinate other =
      setSlot table point coordinate other := by
  simp [setSlot]

omit [Fintype Point] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem setSlot_apply (table : Point → Coord → Slot) (point : Point) (coordinate : Coord)
    (value : Slot) : setSlot table point coordinate value point coordinate = value := by
  simp [setSlot]

omit [Fintype Point] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem setSlot_point (table : Point → Coord → Slot) (point : Point) (coordinate : Coord)
    (value : Slot) :
    setSlot table point coordinate value point = Function.update (table point) coordinate value := by
  simp [setSlot]

/-- A uniform table is a uniform table with one slot drawn again: the slots of one answer are
independent. -/
theorem expect_setSlot (point : Point) (coordinate : Coord)
    (value : (Point → Coord → Slot) → ℝ) :
    𝔼 table, value table = 𝔼 table, 𝔼 slot, value (setSlot table point coordinate slot) :=
  expect_lens (fun table => table point coordinate)
    (fun table slot => setSlot table point coordinate slot)
    (fun table slot => setSlot_apply table point coordinate slot)
    (fun table => by simp [setSlot])
    (fun table slot other => setSlot_setSlot table point coordinate slot other) value

theorem expect_slot (point : Point) (coordinate : Coord) (value : Slot → ℝ) :
    𝔼 table : Point → Coord → Slot, value (table point coordinate) = 𝔼 slot, value slot := by
  rw [expect_eval point (fun reply : Coord → Slot => value (reply coordinate)),
    expect_eval coordinate value]

end Table

section Regrow

variable [Fintype Point] [DecidableEq Point] [Fintype Reply] [Nonempty Reply]
  (protocol : Protocol Point Reply Proof)

/-- The table with a fresh table on `Sub(anchor)`, the stage inputs that extend the
stage-`level` input `anchor`. -/
noncomputable def Protocol.regrow (level : ℕ) (anchor : Point) (table fresh : Point → Reply) :
    Point → Reply :=
  fun point => if protocol.cut level point = some anchor then fresh point else table point

omit [Fintype Point] [Fintype Reply] [Nonempty Reply] in
theorem Protocol.regrow_regrow (level : ℕ) (anchor : Point) (table fresh other : Point → Reply) :
    protocol.regrow level anchor (protocol.regrow level anchor table fresh) other =
      protocol.regrow level anchor table other := by
  funext point
  by_cases inside : protocol.cut level point = some anchor <;> simp [Protocol.regrow, inside]

omit [Fintype Point] [Fintype Reply] [Nonempty Reply] in
theorem Protocol.regrow_update (level : ℕ) (anchor : Point) (table fresh : Point → Reply)
    {point : Point} (inside : protocol.cut level point = some anchor) (reply : Reply) :
    protocol.regrow level anchor table (Function.update fresh point reply) =
      Function.update (protocol.regrow level anchor table fresh) point reply := by
  funext other
  by_cases same : other = point
  · subst same
    simp [Protocol.regrow, inside]
  · simp [Protocol.regrow, same]

/-- A uniform table is a uniform table with a fresh uniform table on `Sub(anchor)`. -/
theorem Protocol.expect_regrow (level : ℕ) (anchor : Point) (value : (Point → Reply) → ℝ) :
    𝔼 table, value table =
      𝔼 table, 𝔼 fresh, value (protocol.regrow level anchor table fresh) := by
  rw [expect_resample
    (fun _ => Finset.univ.filter fun point => protocol.cut level point ≠ some anchor)
    (fun _ _ _ => rfl) value]
  refine Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun fresh _ => ?_
  congr 1
  funext point
  by_cases inside : protocol.cut level point = some anchor <;>
    simp [Protocol.regrow, overlay, inside]

end Regrow

end NightstreamFPrime.Spec.Checkpoint
