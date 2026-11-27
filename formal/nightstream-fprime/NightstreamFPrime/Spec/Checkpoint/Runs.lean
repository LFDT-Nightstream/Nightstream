import NightstreamFPrime.Spec.Checkpoint.Segment

/-! The number of runs of `A` that the extractor of Akita Theorem 10.17 makes (ePrint 2026/1983,
Appendix E.2, "Cost" and "Conclusion"): at most `K̂ + (Q + 1)(K̂ − 1)` in expectation. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (OracleComp)

attribute [local instance low] Classical.propDecidable

variable {Addr Point Coord Slot Proof Cert Break Cand Key Witness : Type}
  [Fintype Addr] [DecidableEq Addr] [Fintype Point] [DecidableEq Point] [Fintype Coord]
  [DecidableEq Coord] [Fintype Slot] [Nonempty Slot]
  (structure_ : Structure Addr Point Coord Slot Proof Cert Break Cand Key Witness)
  (prover : OracleComp Addr (Coord → Slot) Proof) {Inner : Type}

/-- The runs of one attempt: the attempt stops after its first run of `A` when the run selects
another stage input or the verifier rejects. -/
noncomputable def guardedRuns (level : ℕ) (anchor : Point)
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (table : Addr → Coord → Slot)
    (tape : Inner) : ℕ :=
  if structure_.protocol.input prover table level = anchor ∧
    structure_.protocol.accept (structure_.protocol.final prover table) then childRuns table tape
  else 1

/-- The runs of the loop of one coordinate of an indexed stage. The node enters the loop when
the reference continuation returns a certificate. -/
noncomputable def indexedLoopRuns (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (table : Addr → Coord → Slot) (own : Inner)
    (anchor : Point) (coordinate : Coord)
    (attempts : Fin budget → (Addr → Coord → Slot) × Inner) : ℕ :=
  match child table own with
  | some (.inl _) =>
      loopCost (fun sample =>
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true)
        (fun sample => guardedRuns structure_ prover level anchor childRuns
          (setSlot table (structure_.protocol.address anchor) coordinate
            (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2) budget attempts
  | _ => 0

noncomputable def indexedRuns (level : ℕ) (coordinates : List Coord) (budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (table : Addr → Coord → Slot) (own : Inner)
    (loops : Loops Addr Coord Slot budget Inner) : ℕ :=
  childRuns table own + (coordinates.map fun coordinate =>
    indexedLoopRuns structure_ prover level budget child childRuns table own
      (structure_.protocol.input prover table level) coordinate (loops (some coordinate))).sum

/-- The runs of the loop of step 3 of a checkpoint. The node enters the loop when step 1 returns
a certificate that fails the test. -/
noncomputable def checkpointLoopRuns (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (table : Addr → Coord → Slot) (own : Inner)
    (anchor : Point) (attempts : Fin budget → (Addr → Coord → Slot) × Inner) : ℕ :=
  match child table own with
  | some (.inl found) =>
      if structure_.good level found then 0
      else loopCost (fun sample =>
          (checkpointTry structure_ prover level child table anchor sample).isSome = true)
        (fun sample => guardedRuns structure_ prover level anchor childRuns
          (structure_.protocol.regrow level (structure_.protocol.address anchor) table sample.1)
            sample.2) budget attempts
  | _ => 0

noncomputable def checkpointRuns (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (table : Addr → Coord → Slot) (own : Inner)
    (loops : Loops Addr Coord Slot budget Inner) : ℕ :=
  childRuns table own + checkpointLoopRuns structure_ prover level budget child childRuns table
    own (structure_.protocol.input prover table level) (loops none)

/-- The number of runs of `A` that the node `E_level` makes. -/
noncomputable def nodeRuns (budget : ℕ) :
    (depth : ℕ) → ℕ → (Addr → Coord → Slot) → Tape Addr Coord Slot budget depth → ℕ
  | 0, _, _, _ => 1
  | depth + 1, level, table, tape =>
      match structure_.kind level with
      | .indexed coordinates =>
          indexedRuns structure_ prover level coordinates budget
            (node structure_ prover budget depth (level + 1)) (nodeRuns budget depth (level + 1))
            table tape.own tape.loops
      | .checkpoint _ =>
          checkpointRuns structure_ prover level budget
            (node structure_ prover budget depth (level + 1)) (nodeRuns budget depth (level + 1))
            table tape.own tape.loops
      | .plain => nodeRuns budget depth (level + 1) table tape.own

/-- The number of runs of `A` that the extractor makes. -/
noncomputable def extractRuns (budget : ℕ) (table : Addr → Coord → Slot)
    (tape : Tape Addr Coord Slot budget structure_.protocol.stages) : ℕ :=
  nodeRuns structure_ prover budget structure_.protocol.stages 0 table tape

omit [Fintype Addr] [Fintype Point] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
private theorem indexedTry_setSlot (level : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Addr → Coord → Slot) (anchor : Point) (coordinate : Coord) (value : Slot) :
    indexedTry structure_ prover level child
        (setSlot table (structure_.protocol.address anchor) coordinate value) anchor coordinate =
      indexedTry structure_ prover level child table anchor coordinate := by
  funext sample
  simp only [indexedTry, setSlot_setSlot]

omit [Fintype Addr] [DecidableEq Addr] [Fintype Point] [Fintype Coord] [DecidableEq Coord]
  [Fintype Slot] [Nonempty Slot] in
private theorem guardedRuns_le (level : ℕ) (anchor : Point)
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (table : Addr → Coord → Slot)
    (tape : Inner) :
    (guardedRuns structure_ prover level anchor childRuns table tape : ℝ) ≤
      1 + if structure_.protocol.input prover table level = anchor
        then (childRuns table tape : ℝ) else 0 := by
  unfold guardedRuns
  by_cases same : structure_.protocol.input prover table level = anchor
  · by_cases accepts : structure_.protocol.accept (structure_.protocol.final prover table)
    · simp [same, accepts]
    · simp [same, accepts]
  · simp [same]

variable [Fintype Inner] [Nonempty Inner]

omit [Fintype Point] in
private theorem indexed_runs_anchor {level : ℕ} (inside : level < structure_.protocol.stages)
    (budget : ℕ) (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (anchor : Point) (coordinate : Coord) :
    𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
        (if structure_.protocol.input prover table level = anchor ∧ (child table own).isSome
          then (loopCost (fun sample => (indexedTry structure_ prover level child table anchor
              coordinate sample).isSome = true)
            (fun sample => guardedRuns structure_ prover level anchor childRuns
              (setSlot table (structure_.protocol.address anchor) coordinate
                (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2)
            budget attempts : ℝ) else 0) ≤
      𝔼 table, ((if structure_.protocol.rebuild table (structure_.protocol.address anchor) =
            anchor ∧
          (structure_.protocol.address anchor ∈ structure_.protocol.asked prover table ∧
            structure_.protocol.rank (structure_.protocol.address anchor) = level)
          then (1 : ℝ) else 0) +
        𝔼 own : Inner, if structure_.protocol.input prover table level = anchor
          then (childRuns table own : ℝ) else 0) := by
  have right : (𝔼 table, ((if structure_.protocol.rebuild table
        (structure_.protocol.address anchor) = anchor ∧
        (structure_.protocol.address anchor ∈ structure_.protocol.asked prover table ∧
          structure_.protocol.rank (structure_.protocol.address anchor) = level)
        then (1 : ℝ) else 0) +
      𝔼 own : Inner, if structure_.protocol.input prover table level = anchor
        then (childRuns table own : ℝ) else 0)) =
      𝔼 table, ((if structure_.protocol.rebuild table
        (structure_.protocol.address anchor) = anchor ∧
        (structure_.protocol.address anchor ∈ structure_.protocol.asked prover table ∧
          structure_.protocol.rank (structure_.protocol.address anchor) = level)
        then (1 : ℝ) else 0) +
      𝔼 slot, 𝔼 own : Inner, if structure_.protocol.input prover
          (setSlot table (structure_.protocol.address anchor) coordinate slot) level = anchor
        then (childRuns (setSlot table (structure_.protocol.address anchor) coordinate slot)
          own : ℝ) else 0) := by
    rw [Finset.expect_add_distrib, Finset.expect_add_distrib]
    congr 1
    exact expect_setSlot (structure_.protocol.address anchor) coordinate fun table =>
      𝔼 own : Inner, if structure_.protocol.input prover table level = anchor
        then (childRuns table own : ℝ) else 0
  rw [expect_setSlot (structure_.protocol.address anchor) coordinate, right]
  refine Finset.expect_le_expect fun table _ => ?_
  have chargeNonnegative : 0 ≤ 𝔼 slot, 𝔼 own : Inner, if structure_.protocol.input prover
      (setSlot table (structure_.protocol.address anchor) coordinate slot) level = anchor
        then (childRuns (setSlot table (structure_.protocol.address anchor) coordinate slot)
          own : ℝ) else 0 :=
    Finset.expect_nonneg fun _ _ => Finset.expect_nonneg fun _ _ => by split <;> simp
  calc
    _ = 𝔼 fresh : Addr → Coord → Slot, 𝔼 own : Inner,
          𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
          (if structure_.protocol.input prover
              (setSlot table (structure_.protocol.address anchor) coordinate
                (fresh (structure_.protocol.address anchor) coordinate)) level = anchor ∧
            (child (setSlot table (structure_.protocol.address anchor) coordinate
              (fresh (structure_.protocol.address anchor) coordinate)) own).isSome
          then (loopCost (fun sample => (indexedTry structure_ prover level child
              (setSlot table (structure_.protocol.address anchor) coordinate
                (fresh (structure_.protocol.address anchor) coordinate)) anchor
              coordinate sample).isSome = true)
            (fun sample => guardedRuns structure_ prover level anchor childRuns
              (setSlot (setSlot table (structure_.protocol.address anchor) coordinate
                (fresh (structure_.protocol.address anchor) coordinate))
                (structure_.protocol.address anchor) coordinate
                (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2)
            budget attempts : ℝ) else 0) :=
      (expect_slot (structure_.protocol.address anchor) coordinate fun slot => 𝔼 own : Inner,
        𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
          (if structure_.protocol.input prover
              (setSlot table (structure_.protocol.address anchor) coordinate slot) level =
              anchor ∧ (child (setSlot table (structure_.protocol.address anchor) coordinate slot)
                own).isSome
          then (loopCost (fun sample => (indexedTry structure_ prover level child
              (setSlot table (structure_.protocol.address anchor) coordinate slot) anchor
              coordinate sample).isSome = true)
            (fun sample => guardedRuns structure_ prover level anchor childRuns
              (setSlot (setSlot table (structure_.protocol.address anchor) coordinate slot)
                (structure_.protocol.address anchor) coordinate
                (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2)
            budget attempts : ℝ) else 0)).symm
    _ = 𝔼 base : (Addr → Coord → Slot) × Inner,
          𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
          (if (indexedTry structure_ prover level child table anchor coordinate base).isSome = true
          then (loopCost (fun sample => (indexedTry structure_ prover level child table anchor
              coordinate sample).isSome = true)
            (fun sample => guardedRuns structure_ prover level anchor childRuns
              (setSlot table (structure_.protocol.address anchor) coordinate
                (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2)
            budget attempts : ℝ) else 0) := by
      rw [expect_pair]
      refine Finset.expect_congr rfl fun fresh _ => Finset.expect_congr rfl fun own _ =>
        Finset.expect_congr rfl fun attempts _ => ?_
      simp only [indexedTry_setSlot, setSlot_setSlot]
      refine if_congr ?_ rfl rfl
      unfold indexedTry guarded
      split <;> simp [*]
    _ ≤ if ∃ sample : (Addr → Coord → Slot) × Inner,
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true
        then 𝔼 sample : (Addr → Coord → Slot) × Inner,
          (guardedRuns structure_ prover level anchor childRuns
            (setSlot table (structure_.protocol.address anchor) coordinate
              (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2 : ℝ)
        else 0 := by
      refine (le_of_eq ?_).trans ((loop_cost_le
        (fun sample : (Addr → Coord → Slot) × Inner =>
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true)
        (fun sample => guardedRuns structure_ prover level anchor childRuns
          (setSlot table (structure_.protocol.address anchor) coordinate
            (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2) budget).trans
        (le_of_eq ?_))
      · congr!
      · congr!
    _ ≤ _ := by
      by_cases possible : ∃ sample : (Addr → Coord → Slot) × Inner,
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true
      · obtain ⟨sample, passes⟩ := possible
        obtain ⟨out, ran⟩ := Option.isSome_iff_exists.mp passes
        obtain ⟨same, _⟩ := (guarded_eq_some structure_ prover).mp ran
        have queried : structure_.protocol.address anchor ∈
            structure_.protocol.asked prover table := by
          have fresh := structure_.protocol.input_mem_asked prover
            (setSlot table (structure_.protocol.address anchor) coordinate
              (sample.1 (structure_.protocol.address anchor) coordinate)) inside
          rw [same] at fresh
          exact (structure_.protocol.asked_update prover table _ _).mp fresh
        have staged : structure_.protocol.rank (structure_.protocol.address anchor) = level := by
          rw [structure_.protocol.rank_address, ← same]
          exact structure_.protocol.stage_input prover _ (Nat.le_of_lt inside)
        have rebuilt : structure_.protocol.rebuild table (structure_.protocol.address anchor) =
            anchor := by
          have again := structure_.protocol.rebuild_input prover
            (setSlot table (structure_.protocol.address anchor) coordinate
              (sample.1 (structure_.protocol.address anchor) coordinate)) (Nat.le_of_lt inside)
          rw [same, structure_.protocol.rebuild_setSlot] at again
          exact again
        rw [ite_eq_left ⟨sample, passes⟩, ite_eq_left ⟨rebuilt, queried, staged⟩]
        calc
          _ ≤ 𝔼 sample : (Addr → Coord → Slot) × Inner,
                (1 + if structure_.protocol.input prover
                    (setSlot table (structure_.protocol.address anchor) coordinate
                      (sample.1 (structure_.protocol.address anchor) coordinate)) level = anchor
                  then (childRuns (setSlot table (structure_.protocol.address anchor) coordinate
                    (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2 : ℝ)
                  else 0) :=
            Finset.expect_le_expect fun sample _ => guardedRuns_le structure_ prover _ _ _ _ _
          _ = _ := by
            rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty, expect_pair]
            congr 1
            exact expect_slot (structure_.protocol.address anchor) coordinate fun slot =>
              𝔼 own : Inner, if structure_.protocol.input prover
                  (setSlot table (structure_.protocol.address anchor) coordinate slot) level =
                anchor then (childRuns (setSlot table (structure_.protocol.address anchor)
                  coordinate slot) own : ℝ) else 0
      · rw [ite_eq_right possible]
        have : (0 : ℝ) ≤ if structure_.protocol.rebuild table
            (structure_.protocol.address anchor) = anchor ∧
            (structure_.protocol.address anchor ∈ structure_.protocol.asked prover table ∧
              structure_.protocol.rank (structure_.protocol.address anchor) = level)
            then (1 : ℝ) else 0 := by
          split <;> norm_num
        linarith

/-- The cost recurrence of an indexed stage (Appendix E.2):
`C_r ≤ b_r·C_{r+1} + (b_r − 1)·(Q + 1)` with `b_r = 1 + |U_r|`. -/
theorem indexed_runs_le {level : ℕ} (inside : level < structure_.protocol.stages)
    (coordinates : List Coord) {queries : ℕ} (spread : structure_.protocol.Spread prover queries) (budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) :
    𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Addr Coord Slot budget Inner,
        (indexedRuns structure_ prover level coordinates budget child childRuns table own loops :
          ℝ) ≤
      (1 + coordinates.length) * (𝔼 table, 𝔼 own : Inner, (childRuns table own : ℝ)) +
        coordinates.length * (queries + 1) := by
  have each (coordinate : Coord) :
      𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Addr Coord Slot budget Inner,
        (indexedLoopRuns structure_ prover level budget child childRuns table own
          (structure_.protocol.input prover table level) coordinate (loops (some coordinate)) : ℝ) ≤
        (𝔼 table, 𝔼 own : Inner, (childRuns table own : ℝ)) + (queries + 1) := by
    calc
      _ = 𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
            (indexedLoopRuns structure_ prover level budget child childRuns table own
              (structure_.protocol.input prover table level) coordinate attempts : ℝ) :=
        Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
          RandomOracle.expect_eval (some coordinate) fun attempts =>
            (indexedLoopRuns structure_ prover level budget child childRuns table own
              (structure_.protocol.input prover table level) coordinate attempts : ℝ)
      _ ≤ 𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
            ∑ anchor, (if structure_.protocol.input prover table level = anchor ∧
                (child table own).isSome
              then (loopCost (fun sample => (indexedTry structure_ prover level child table anchor
                  coordinate sample).isSome = true)
                (fun sample => guardedRuns structure_ prover level anchor childRuns
                  (setSlot table (structure_.protocol.address anchor) coordinate
                    (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2)
                budget attempts : ℝ) else 0) := by
        refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun own _ =>
          Finset.expect_le_expect fun attempts _ => ?_
        rw [Finset.sum_eq_single (structure_.protocol.input prover table level)]
        · unfold indexedLoopRuns
          rcases child table own with _ | (reference | broken) <;> simp
        · intro anchor _ different
          simp [Ne.symm different]
        · simp
      _ = ∑ anchor, 𝔼 table, 𝔼 own : Inner,
            𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
            (if structure_.protocol.input prover table level = anchor ∧ (child table own).isSome
              then (loopCost (fun sample => (indexedTry structure_ prover level child table anchor
                  coordinate sample).isSome = true)
                (fun sample => guardedRuns structure_ prover level anchor childRuns
                  (setSlot table (structure_.protocol.address anchor) coordinate
                    (sample.1 (structure_.protocol.address anchor) coordinate)) sample.2)
                budget attempts : ℝ) else 0) := by
        simp only [Finset.expect_sum_comm]
      _ ≤ ∑ anchor, 𝔼 table, ((if structure_.protocol.rebuild table
              (structure_.protocol.address anchor) = anchor ∧
              (structure_.protocol.address anchor ∈ structure_.protocol.asked prover table ∧
                structure_.protocol.rank (structure_.protocol.address anchor) = level)
              then (1 : ℝ) else 0) +
            𝔼 own : Inner, if structure_.protocol.input prover table level = anchor
              then (childRuns table own : ℝ) else 0) :=
        Finset.sum_le_sum fun anchor _ =>
          indexed_runs_anchor structure_ prover inside budget child childRuns anchor coordinate
      _ = 𝔼 table, ((∑ key, if key ∈ structure_.protocol.asked prover table ∧
              structure_.protocol.rank key = level then (1 : ℝ) else 0) +
            𝔼 own : Inner, (childRuns table own : ℝ)) := by
        rw [← Finset.expect_sum_comm]
        refine Finset.expect_congr rfl fun table _ => ?_
        rw [Finset.sum_add_distrib, ← Finset.expect_sum_comm]
        congr 1
        · exact (Finset.sum_congr rfl fun anchor _ => ite_and _ _ _ _).trans
            (structure_.protocol.sum_rebuild table fun key =>
              if key ∈ structure_.protocol.asked prover table ∧
                structure_.protocol.rank key = level then (1 : ℝ) else 0)
        · exact Finset.expect_congr rfl fun own _ => by simp
      _ ≤ _ := by
        rw [Finset.expect_add_distrib, add_comm]
        exact add_le_add le_rfl ((Finset.expect_le_expect fun table _ =>
          structure_.protocol.stage_asked_le prover spread table level).trans
          (Finset.expect_const Finset.univ_nonempty _).le)
  have listed : ∀ sub : List Coord,
      𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Addr Coord Slot budget Inner,
        (((sub.map fun coordinate => indexedLoopRuns structure_ prover level budget child
          childRuns table own (structure_.protocol.input prover table level) coordinate
          (loops (some coordinate))).sum : ℕ) : ℝ) ≤
        sub.length * ((𝔼 table, 𝔼 own : Inner, (childRuns table own : ℝ)) + (queries + 1)) := by
    intro sub
    induction sub with
    | nil => simp
    | cons coordinate rest induction =>
        refine le_trans (le_of_eq ?_) ((add_le_add (each coordinate) induction).trans
          (le_of_eq ?_))
        · simp only [List.map_cons, List.sum_cons, Nat.cast_add, Finset.expect_add_distrib]
        · rw [List.length_cons]
          push_cast
          ring
  unfold indexedRuns
  simp only [Nat.cast_add, Finset.expect_add_distrib]
  rw [Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
    Finset.expect_const Finset.univ_nonempty _]
  have tail := listed coordinates
  linarith

omit [Fintype Point] in
private theorem checkpoint_runs_anchor (level budget : ℕ)
    (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) (anchor : Point) :
    𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
        (if structure_.protocol.input prover table level = anchor ∧ (child table own).isSome
          then (loopCost (fun sample => (checkpointTry structure_ prover level child table anchor
              sample).isSome = true)
            (fun sample => guardedRuns structure_ prover level anchor childRuns
              (structure_.protocol.regrow level (structure_.protocol.address anchor) table
                sample.1) sample.2)
            budget attempts : ℝ) else 0) ≤
      𝔼 table, ((if ∃ fresh, structure_.protocol.input prover
          (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
            level = anchor
          then (1 : ℝ) else 0) +
        𝔼 own : Inner, if structure_.protocol.input prover table level = anchor
          then (childRuns table own : ℝ) else 0) := by
  have right : (𝔼 table, ((if ∃ fresh, structure_.protocol.input prover
        (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
          level = anchor
        then (1 : ℝ) else 0) +
      𝔼 own : Inner, if structure_.protocol.input prover table level = anchor
        then (childRuns table own : ℝ) else 0)) =
      𝔼 table, ((if ∃ fresh, structure_.protocol.input prover
        (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
          level = anchor
        then (1 : ℝ) else 0) +
      𝔼 fresh, 𝔼 own : Inner, if structure_.protocol.input prover
          (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
            level = anchor
        then (childRuns (structure_.protocol.regrow level (structure_.protocol.address anchor) table
          fresh) own : ℝ) else 0) := by
    rw [Finset.expect_add_distrib, Finset.expect_add_distrib]
    congr 1
    exact structure_.protocol.expect_regrow level (structure_.protocol.address anchor)
      fun table => 𝔼 own : Inner,
      if structure_.protocol.input prover table level = anchor
        then (childRuns table own : ℝ) else 0
  rw [structure_.protocol.expect_regrow level (structure_.protocol.address anchor), right]
  refine Finset.expect_le_expect fun table _ => ?_
  have tryEq (fresh : Addr → Coord → Slot) : checkpointTry structure_ prover level child
      (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh) anchor =
        checkpointTry structure_ prover level child table anchor := by
    funext sample
    simp only [checkpointTry, Protocol.regrow_regrow]
  calc
    _ = 𝔼 base : (Addr → Coord → Slot) × Inner,
          𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
          (if (checkpointTry structure_ prover level child table anchor base).isSome = true
          then (loopCost (fun sample => (checkpointTry structure_ prover level child table anchor
              sample).isSome = true)
            (fun sample => guardedRuns structure_ prover level anchor childRuns
              (structure_.protocol.regrow level (structure_.protocol.address anchor) table
                sample.1) sample.2)
            budget attempts : ℝ) else 0) := by
      rw [expect_pair]
      refine Finset.expect_congr rfl fun fresh _ => Finset.expect_congr rfl fun own _ =>
        Finset.expect_congr rfl fun attempts _ => ?_
      simp only [tryEq, Protocol.regrow_regrow]
      refine if_congr ?_ rfl rfl
      unfold checkpointTry guarded
      split <;> simp [*]
    _ ≤ if ∃ sample : (Addr → Coord → Slot) × Inner,
          (checkpointTry structure_ prover level child table anchor sample).isSome = true
        then 𝔼 sample : (Addr → Coord → Slot) × Inner,
          (guardedRuns structure_ prover level anchor childRuns
            (structure_.protocol.regrow level (structure_.protocol.address anchor) table sample.1)
              sample.2 : ℝ)
        else 0 := by
      refine (le_of_eq ?_).trans ((loop_cost_le
        (fun sample : (Addr → Coord → Slot) × Inner =>
          (checkpointTry structure_ prover level child table anchor sample).isSome = true)
        (fun sample => guardedRuns structure_ prover level anchor childRuns
          (structure_.protocol.regrow level (structure_.protocol.address anchor) table sample.1)
            sample.2) budget).trans
        (le_of_eq ?_))
      · congr!
      · congr!
    _ ≤ _ := by
      have chargeNonnegative : 0 ≤ 𝔼 fresh, 𝔼 own : Inner, if structure_.protocol.input prover
          (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
            level = anchor
          then (childRuns (structure_.protocol.regrow level
            (structure_.protocol.address anchor) table fresh) own : ℝ) else 0 :=
        Finset.expect_nonneg fun _ _ => Finset.expect_nonneg fun _ _ => by split <;> simp
      by_cases possible : ∃ sample : (Addr → Coord → Slot) × Inner,
          (checkpointTry structure_ prover level child table anchor sample).isSome = true
      · obtain ⟨sample, passes⟩ := possible
        obtain ⟨out, ran⟩ := Option.isSome_iff_exists.mp passes
        obtain ⟨same, _⟩ := (guarded_eq_some structure_ prover).mp ran
        rw [ite_eq_left ⟨sample, passes⟩, ite_eq_left ⟨sample.1, same⟩]
        calc
          _ ≤ 𝔼 sample : (Addr → Coord → Slot) × Inner,
                (1 + if structure_.protocol.input prover
                    (structure_.protocol.regrow level (structure_.protocol.address anchor) table
                      sample.1) level = anchor
                  then (childRuns (structure_.protocol.regrow level
                    (structure_.protocol.address anchor) table sample.1) sample.2 : ℝ) else 0) :=
            Finset.expect_le_expect fun sample _ => guardedRuns_le structure_ prover _ _ _ _ _
          _ = _ := by
            rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty, expect_pair]
      · rw [ite_eq_right possible]
        have : (0 : ℝ) ≤ if ∃ fresh, structure_.protocol.input prover
            (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
              level = anchor
            then (1 : ℝ) else 0 := by
          split <;> norm_num
        linarith

/-- The cost recurrence of a checkpoint (Appendix E.2, "Cost"): `C_ℓ ≤ 2·C_{ℓ+1} + (Q + 1)`. -/
theorem checkpoint_runs_le {level : ℕ} (inside : level < structure_.protocol.stages)
    {queries : ℕ} (spread : structure_.protocol.Spread prover queries)
    (budget : ℕ) (child : (Addr → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (childRuns : (Addr → Coord → Slot) → Inner → ℕ) :
    𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Addr Coord Slot budget Inner,
        (checkpointRuns structure_ prover level budget child childRuns table own loops : ℝ) ≤
      2 * (𝔼 table, 𝔼 own : Inner, (childRuns table own : ℝ)) + (queries + 1) := by
  have loop : 𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Addr Coord Slot budget Inner,
      (checkpointLoopRuns structure_ prover level budget child childRuns table own
        (structure_.protocol.input prover table level) (loops none) : ℝ) ≤
      (𝔼 table, 𝔼 own : Inner, (childRuns table own : ℝ)) + (queries + 1) := by
    calc
      _ = 𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
            (checkpointLoopRuns structure_ prover level budget child childRuns table own
              (structure_.protocol.input prover table level) attempts : ℝ) :=
        Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
          RandomOracle.expect_eval (none : Option Coord) fun attempts =>
            (checkpointLoopRuns structure_ prover level budget child childRuns table own
              (structure_.protocol.input prover table level) attempts : ℝ)
      _ ≤ 𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
            ∑ anchor, (if structure_.protocol.input prover table level = anchor ∧
                (child table own).isSome
              then (loopCost (fun sample => (checkpointTry structure_ prover level child table
                  anchor sample).isSome = true)
                (fun sample => guardedRuns structure_ prover level anchor childRuns
                  (structure_.protocol.regrow level (structure_.protocol.address anchor) table
                    sample.1) sample.2)
                budget attempts : ℝ) else 0) := by
        refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun own _ =>
          Finset.expect_le_expect fun attempts _ => ?_
        rw [Finset.sum_eq_single (structure_.protocol.input prover table level)]
        · unfold checkpointLoopRuns
          rcases child table own with _ | (found | broken)
          · simp
          · by_cases isGood : structure_.good level found <;> simp [isGood]
          · simp
        · intro anchor _ different
          simp [Ne.symm different]
        · simp
      _ = ∑ anchor, 𝔼 table, 𝔼 own : Inner,
            𝔼 attempts : Fin budget → (Addr → Coord → Slot) × Inner,
            (if structure_.protocol.input prover table level = anchor ∧ (child table own).isSome
              then (loopCost (fun sample => (checkpointTry structure_ prover level child table
                  anchor sample).isSome = true)
                (fun sample => guardedRuns structure_ prover level anchor childRuns
                  (structure_.protocol.regrow level (structure_.protocol.address anchor) table
                    sample.1) sample.2)
                budget attempts : ℝ) else 0) := by
        simp only [Finset.expect_sum_comm]
      _ ≤ ∑ anchor, 𝔼 table, ((if ∃ fresh, structure_.protocol.input prover
              (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
                level = anchor
              then (1 : ℝ) else 0) +
            𝔼 own : Inner, if structure_.protocol.input prover table level = anchor
              then (childRuns table own : ℝ) else 0) :=
        Finset.sum_le_sum fun anchor _ =>
          checkpoint_runs_anchor structure_ prover level budget child childRuns anchor
      _ = 𝔼 table, ((∑ anchor, if ∃ fresh, structure_.protocol.input prover
              (structure_.protocol.regrow level (structure_.protocol.address anchor) table fresh)
                level = anchor
              then (1 : ℝ) else 0) +
            𝔼 own : Inner, (childRuns table own : ℝ)) := by
        rw [← Finset.expect_sum_comm]
        refine Finset.expect_congr rfl fun table _ => ?_
        rw [Finset.sum_add_distrib, ← Finset.expect_sum_comm]
        congr 1
        exact Finset.expect_congr rfl fun own _ => by simp
      _ ≤ _ := by
        rw [Finset.expect_add_distrib, add_comm]
        exact add_le_add le_rfl ((Finset.expect_le_expect fun table _ =>
          selected_count_le structure_ prover spread (Nat.le_of_lt inside) table).trans
          (Finset.expect_const Finset.univ_nonempty _).le)
  unfold checkpointRuns
  simp only [Nat.cast_add, Finset.expect_add_distrib]
  rw [Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
    Finset.expect_const Finset.univ_nonempty _]
  linarith

/-- The expected number of runs of `A` of the node `E_level`: `C_level` of Appendix E.2. -/
noncomputable def expectedRuns (budget depth level : ℕ) : ℝ :=
  𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget depth,
    (nodeRuns structure_ prover budget depth level table tape : ℝ)

/-- The cost recurrences `C_r ≤ k·C_{r+1} + (k − 1)·(Q + 1)` of Appendix E.2 solve to
`C_r + (Q + 1) ≤ (Q + 2)·K̂_r`, where `K̂_r` is the tree size of the stages from `r`. -/
theorem runs_le {queries : ℕ} (spread : structure_.protocol.Spread prover queries) (budget : ℕ) :
    ∀ depth level, level + depth = structure_.protocol.stages →
      expectedRuns structure_ prover budget depth level + (queries + 1) ≤
        (queries + 2) * structure_.treeFrom level := by
  intro depth
  induction depth with
  | zero =>
      intro level full
      obtain rfl : level = structure_.protocol.stages := by omega
      have one : expectedRuns structure_ prover budget 0 structure_.protocol.stages = 1 := by
        unfold expectedRuns
        simp [nodeRuns]
      have tree : structure_.treeFrom structure_.protocol.stages = 1 := by
        unfold Structure.treeFrom
        simp
      rw [one, tree]
      push_cast
      linarith
  | succ depth induction =>
      intro level full
      have inside : level < structure_.protocol.stages := by omega
      have below := induction (level + 1) (by omega)
      have tree : ((structure_.treeFrom level : ℕ) : ℝ) =
          structure_.arity level * structure_.treeFrom (level + 1) := by
        rw [structure_.treeFrom_succ inside]
        push_cast
        rfl
      rw [tree]
      unfold expectedRuns at below ⊢
      rcases kindEq : structure_.kind level with coordinates | span | _
      · have same : (𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget (depth + 1),
            (nodeRuns structure_ prover budget (depth + 1) level table tape : ℝ)) =
            𝔼 table, 𝔼 own, 𝔼 loops,
              (indexedRuns structure_ prover level coordinates budget
                (node structure_ prover budget depth (level + 1))
                (nodeRuns structure_ prover budget depth (level + 1)) table own loops : ℝ) := by
          refine Finset.expect_congr rfl fun table _ => ?_
          have unfolded : ∀ tape : Tape Addr Coord Slot budget (depth + 1),
              nodeRuns structure_ prover budget (depth + 1) level table tape =
                indexedRuns structure_ prover level coordinates budget
                  (node structure_ prover budget depth (level + 1))
                  (nodeRuns structure_ prover budget depth (level + 1)) table tape.own
                  tape.loops := fun tape => by simp only [nodeRuns, kindEq]
          simp only [unfolded]
          exact Tape.expect_succ fun own loops =>
            (indexedRuns structure_ prover level coordinates budget
              (node structure_ prover budget depth (level + 1))
              (nodeRuns structure_ prover budget depth (level + 1)) table own loops : ℝ)
        have stage := indexed_runs_le structure_ prover inside coordinates spread budget
          (node structure_ prover budget depth (level + 1))
          (nodeRuns structure_ prover budget depth (level + 1))
        have arity : ((structure_.arity level : ℕ) : ℝ) = 1 + coordinates.length := by
          unfold Structure.arity
          rw [kindEq]
          push_cast
          rfl
        rw [same, arity]
        nlinarith [mul_le_mul_of_nonneg_left below
          (by positivity : (0 : ℝ) ≤ 1 + coordinates.length)]
      · have same : (𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget (depth + 1),
            (nodeRuns structure_ prover budget (depth + 1) level table tape : ℝ)) =
            𝔼 table, 𝔼 own, 𝔼 loops,
              (checkpointRuns structure_ prover level budget
                (node structure_ prover budget depth (level + 1))
                (nodeRuns structure_ prover budget depth (level + 1)) table own loops : ℝ) := by
          refine Finset.expect_congr rfl fun table _ => ?_
          have unfolded : ∀ tape : Tape Addr Coord Slot budget (depth + 1),
              nodeRuns structure_ prover budget (depth + 1) level table tape =
                checkpointRuns structure_ prover level budget
                  (node structure_ prover budget depth (level + 1))
                  (nodeRuns structure_ prover budget depth (level + 1)) table tape.own
                  tape.loops := fun tape => by simp only [nodeRuns, kindEq]
          simp only [unfolded]
          exact Tape.expect_succ fun own loops =>
            (checkpointRuns structure_ prover level budget
              (node structure_ prover budget depth (level + 1))
              (nodeRuns structure_ prover budget depth (level + 1)) table own loops : ℝ)
        have stage := checkpoint_runs_le structure_ prover inside spread budget
          (node structure_ prover budget depth (level + 1))
          (nodeRuns structure_ prover budget depth (level + 1))
        have arity : ((structure_.arity level : ℕ) : ℝ) = 2 := by
          unfold Structure.arity
          rw [kindEq]
          norm_num
        rw [same, arity]
        linarith
      · have same : (𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget (depth + 1),
            (nodeRuns structure_ prover budget (depth + 1) level table tape : ℝ)) =
            𝔼 table, 𝔼 own : Tape Addr Coord Slot budget depth,
              (nodeRuns structure_ prover budget depth (level + 1) table own : ℝ) := by
          refine Finset.expect_congr rfl fun table _ => ?_
          have unfolded : ∀ tape : Tape Addr Coord Slot budget (depth + 1),
              nodeRuns structure_ prover budget (depth + 1) level table tape =
                nodeRuns structure_ prover budget depth (level + 1) table tape.own :=
            fun tape => by simp only [nodeRuns, kindEq]
          simp only [unfolded]
          rw [Tape.expect_succ fun own _ =>
            (nodeRuns structure_ prover budget depth (level + 1) table own : ℝ)]
          exact Finset.expect_congr rfl fun own _ => Finset.expect_const Finset.univ_nonempty _
        have arity : ((structure_.arity level : ℕ) : ℝ) = 1 := by
          unfold Structure.arity
          rw [kindEq]
          norm_num
        rw [same, arity, one_mul]
        exact below

/-- Akita Theorem 10.17, the number of runs: the extractor runs `A` at most
`K̂ + (Q + 1)(K̂ − 1)` times in expectation, for each limit on the attempts of its loops. `Q` is the
count of `Protocol.Spread`. -/
theorem extract_runs_spread_le {queries : ℕ} (spread : structure_.protocol.Spread prover queries)
    (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget structure_.protocol.stages,
        (extractRuns structure_ prover budget table tape : ℝ) ≤
      structure_.treeSize + (queries + 1) * (structure_.treeSize - 1) := by
  have solved := runs_le structure_ prover spread budget structure_.protocol.stages 0
    (Nat.zero_add _)
  unfold expectedRuns at solved
  unfold extractRuns Structure.treeSize
  linarith

/-- Akita Theorem 10.17, the number of runs, for a prover with at most `Q` queries. -/
theorem extract_runs_le {queries : ℕ} (bounded : prover.QueryBound queries) (budget : ℕ) :
    𝔼 table, 𝔼 tape : Tape Addr Coord Slot budget structure_.protocol.stages,
        (extractRuns structure_ prover budget table tape : ℝ) ≤
      structure_.treeSize + (queries + 1) * (structure_.treeSize - 1) :=
  extract_runs_spread_le structure_ prover (bounded.spread _) budget

end NightstreamFPrime.Spec.Checkpoint
