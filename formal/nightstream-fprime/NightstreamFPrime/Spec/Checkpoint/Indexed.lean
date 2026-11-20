import NightstreamFPrime.Spec.Checkpoint.Extractor

/-! The success step of an indexed extraction stage of Akita Theorem 10.17 (ePrint 2026/1983,
Appendix E.2, "Indexed extraction stages"): the binary sampling game on one slot of one answer,
for each coordinate in turn. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (OracleComp mass)

attribute [local instance low] Classical.propDecidable

variable {Point Coord Slot Proof Cert Break Cand Key Witness : Type}
  [Fintype Point] [DecidableEq Point] [Fintype Coord] [DecidableEq Coord] [Fintype Slot]
  [Nonempty Slot]
  (structure_ : Structure Point Coord Slot Proof Cert Break Cand Key Witness)
  (prover : OracleComp Point (Coord → Slot) Proof) {Inner : Type}

omit [Fintype Point] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
theorem indexedNode_eq_none {level : ℕ} {coordinates : List Coord} {budget : ℕ}
    {child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    {table : Point → Coord → Slot} {own : Inner} {loops : Loops Point Coord Slot budget Inner}
    (ran : (child table own).isSome)
    (missed : indexedNode structure_ prover level coordinates budget child table own loops =
      none) :
    ∃ coordinate ∈ coordinates, indexedFails structure_ prover level budget child table
      (structure_.protocol.input prover table level) coordinate (loops (some coordinate)) := by
  unfold indexedNode at missed
  rcases first : child table own with _ | (reference | broken)
  · simp [first] at ran
  · simp only [first] at missed
    unfold indexedCombine at missed
    split at missed
    · assumption
    · split at missed <;> exact absurd missed (by simp)
  · simp [first] at missed

omit [Fintype Point] [Fintype Coord] [Fintype Slot] [Nonempty Slot] in
private theorem event_setSlot (level budget : ℕ)
    (child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Point → Coord → Slot) (anchor : Point) (coordinate : Coord)
    (base : (Point → Coord → Slot) × Inner)
    (attempts : Fin budget → (Point → Coord → Slot) × Inner) :
    (structure_.protocol.input prover
        (setSlot table anchor coordinate (base.1 anchor coordinate)) level = anchor ∧
      (child (setSlot table anchor coordinate (base.1 anchor coordinate)) base.2).isSome ∧
      indexedFails structure_ prover level budget child
        (setSlot table anchor coordinate (base.1 anchor coordinate)) anchor coordinate attempts) ↔
    ((indexedTry structure_ prover level child table anchor coordinate base).isSome = true ∧
      (firstPass (fun sample =>
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true)
          budget attempts = none ∨
        True ∧ ∃ sample, firstPass (fun sample =>
            (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true)
            budget attempts = some sample ∧
          structure_.key level coordinate (sample.1 anchor coordinate) =
            structure_.key level coordinate (base.1 anchor coordinate))) := by
  have same : indexedTry structure_ prover level child
      (setSlot table anchor coordinate (base.1 anchor coordinate)) anchor coordinate =
        indexedTry structure_ prover level child table anchor coordinate := by
    funext sample
    simp only [indexedTry, setSlot_setSlot]
  have passes : (indexedTry structure_ prover level child table anchor coordinate base).isSome =
      true ↔ structure_.protocol.input prover
        (setSlot table anchor coordinate (base.1 anchor coordinate)) level = anchor ∧
      (child (setSlot table anchor coordinate (base.1 anchor coordinate)) base.2).isSome := by
    unfold indexedTry guarded
    split <;> simp [*]
  unfold indexedFails
  rw [same, setSlot_apply, passes, and_assoc]
  refine and_congr_right fun _ => and_congr_right fun _ => ?_
  rcases firstPass (fun sample =>
      (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true)
      budget attempts with _ | sample <;> simp

variable [Fintype Inner] [Nonempty Inner]

/-- The binary game on the slot of one coordinate at one stage input (Appendix E.2): the reference
run succeeds and the loop fails with probability at most `1/|S_{r,u}| + 1/budget`, and only when
`A^J` queries the stage input. -/
theorem indexed_anchor_le (lawful : structure_.Lawful) {level : ℕ} {coordinates : List Coord}
    (indexed : structure_.kind level = .indexed coordinates)
    (inside : level < structure_.protocol.stages) {budget : ℕ} (positive : 0 < budget)
    {child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    (accepts : ∀ table tape, (child table tape).isSome →
      structure_.protocol.accept (structure_.protocol.final prover table))
    (anchor : Point) {coordinate : Coord} (member : coordinate ∈ coordinates) :
    𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
        (if structure_.protocol.input prover table level = anchor ∧ (child table own).isSome ∧
          indexedFails structure_ prover level budget child table anchor coordinate attempts
          then (1 : ℝ) else 0) ≤
      𝔼 table, if anchor ∈ structure_.protocol.asked prover table ∧
          structure_.protocol.stage anchor = level
        then structure_.weight level coordinate + 1 / budget else 0 := by
  have margin : 0 ≤ structure_.weight level coordinate + 1 / (budget : ℝ) :=
    add_nonneg (lawful.weight_nonnegative level coordinate) (by positivity)
  rw [expect_setSlot anchor coordinate]
  refine Finset.expect_le_expect fun table _ => ?_
  calc
    _ = 𝔼 fresh : Point → Coord → Slot, 𝔼 own : Inner,
          𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
          (if structure_.protocol.input prover
              (setSlot table anchor coordinate (fresh anchor coordinate)) level = anchor ∧
            (child (setSlot table anchor coordinate (fresh anchor coordinate)) own).isSome ∧
            indexedFails structure_ prover level budget child
              (setSlot table anchor coordinate (fresh anchor coordinate)) anchor coordinate attempts
            then (1 : ℝ) else 0) :=
      (expect_slot anchor coordinate fun slot => 𝔼 own : Inner,
        𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
          (if structure_.protocol.input prover (setSlot table anchor coordinate slot) level =
              anchor ∧ (child (setSlot table anchor coordinate slot) own).isSome ∧
            indexedFails structure_ prover level budget child
              (setSlot table anchor coordinate slot) anchor coordinate attempts
            then (1 : ℝ) else 0)).symm
    _ = 𝔼 base : (Point → Coord → Slot) × Inner,
          𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
          (if (indexedTry structure_ prover level child table anchor coordinate base).isSome =
              true ∧
            (firstPass (fun sample => (indexedTry structure_ prover level child table anchor
                coordinate sample).isSome = true) budget attempts = none ∨
              True ∧ ∃ sample, firstPass (fun sample => (indexedTry structure_ prover level child
                  table anchor coordinate sample).isSome = true) budget attempts = some sample ∧
                structure_.key level coordinate (sample.1 anchor coordinate) =
                  structure_.key level coordinate (base.1 anchor coordinate))
            then (1 : ℝ) else 0) := by
      rw [expect_pair]
      exact Finset.expect_congr rfl fun fresh _ => Finset.expect_congr rfl fun own _ =>
        Finset.expect_congr rfl fun attempts _ => if_congr (event_setSlot structure_ prover level
          budget child table anchor coordinate (fresh, own) attempts) rfl rfl
    _ ≤ if ∃ sample : (Point → Coord → Slot) × Inner,
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true
          then structure_.weight level coordinate + 1 / budget else 0 := by
      refine (le_of_eq ?_).trans ((fork_le
        (fun sample : (Point → Coord → Slot) × Inner =>
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true)
        (fun _ => True)
        (fun sample => structure_.key level coordinate (sample.1 anchor coordinate))
        (structure_.weight level coordinate)
        (lawful.weight_nonnegative level coordinate) positive fun base passes _ => ?_).trans
        (le_of_eq ?_))
      · congr!
      rotate_left
      · congr!
      obtain ⟨out, ran⟩ := Option.isSome_iff_exists.mp passes
      obtain ⟨same, childRan⟩ := (guarded_eq_some structure_ prover).mp ran
      have accepted := accepts _ _ (Option.isSome_iff_exists.mpr ⟨out, childRan⟩)
      have decoded := lawful.key_some level coordinates coordinate _ _ indexed member accepted
        (structure_.protocol.cut_input prover _ (Nat.succ_le_of_lt inside) le_rfl)
      rw [structure_.protocol.last_input prover _ inside, same, setSlot_apply] at decoded
      obtain ⟨value, isValue⟩ := Option.isSome_iff_exists.mp decoded
      calc
        _ ≤ 𝔼 sample : (Point → Coord → Slot) × Inner,
              (if structure_.key level coordinate (sample.1 anchor coordinate) = some value
                then (1 : ℝ) else 0) := by
          refine Finset.expect_le_expect fun sample _ => ?_
          rw [isValue]
          by_cases equal : structure_.key level coordinate (sample.1 anchor coordinate) = some value
          · simp only [equal, and_true, ite_true]
            split <;> norm_num
          · simp [equal]
        _ = 𝔼 slot, (if structure_.key level coordinate slot = some value then (1 : ℝ) else 0) := by
          rw [expect_fst fun fresh : Point → Coord → Slot =>
            if structure_.key level coordinate (fresh anchor coordinate) = some value
              then (1 : ℝ) else 0]
          exact expect_slot anchor coordinate fun slot =>
            if structure_.key level coordinate slot = some value then (1 : ℝ) else 0
        _ ≤ structure_.weight level coordinate := by
          have small := lawful.key_mass level coordinates coordinate value indexed member
          unfold mass at small
          simp only [Set.mem_ofPred_eq] at small
          convert small
    _ ≤ _ := by
      by_cases possible : ∃ sample : (Point → Coord → Slot) × Inner,
          (indexedTry structure_ prover level child table anchor coordinate sample).isSome = true
      · obtain ⟨sample, passes⟩ := possible
        obtain ⟨out, ran⟩ := Option.isSome_iff_exists.mp passes
        obtain ⟨same, _⟩ := (guarded_eq_some structure_ prover).mp ran
        have queried : anchor ∈ structure_.protocol.asked prover table := by
          have fresh := structure_.protocol.input_mem_asked prover
            (setSlot table anchor coordinate (sample.1 anchor coordinate)) inside
          rw [same] at fresh
          exact (structure_.protocol.asked_update prover table anchor _).mp fresh
        have staged : structure_.protocol.stage anchor = level := by
          rw [← same]
          exact structure_.protocol.stage_input prover _ (Nat.le_of_lt inside)
        rw [ite_eq_left ⟨sample, passes⟩, ite_eq_left ⟨queried, staged⟩]
      · rw [ite_eq_right possible]
        split <;> linarith

/-- The success recurrence of an indexed stage (Appendix E.2):
`s_r ≥ s_{r+1} − E[Σ_{a ∈ Addr_r(J)} ϵ(a)]`, with the loss of the limit of the loops. -/
theorem indexed_miss_le (lawful : structure_.Lawful) {level : ℕ} {coordinates : List Coord}
    (indexed : structure_.kind level = .indexed coordinates)
    (inside : level < structure_.protocol.stages) {queries : ℕ}
    (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget)
    {child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    (accepts : ∀ table tape, (child table tape).isSome →
      structure_.protocol.accept (structure_.protocol.final prover table)) :
    𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Point Coord Slot budget Inner,
        (if (child table own).isSome ∧
          indexedNode structure_ prover level coordinates budget child table own loops = none
          then (1 : ℝ) else 0) ≤
      (𝔼 table, ∑ point, if point ∈ structure_.protocol.asked prover table ∧
          structure_.protocol.stage point = level then structure_.charge level else 0) +
        (queries + 1) * structure_.loops level / budget := by
  have count : 𝔼 table, (∑ point, if point ∈ structure_.protocol.asked prover table ∧
      structure_.protocol.stage point = level then (1 : ℝ) else 0) ≤ queries + 1 :=
    (Finset.expect_le_expect fun table _ =>
      structure_.protocol.stage_asked_le prover bounded table level).trans
      (Finset.expect_const Finset.univ_nonempty _).le
  have countNonnegative : 0 ≤ 𝔼 table, (∑ point,
      if point ∈ structure_.protocol.asked prover table ∧
        structure_.protocol.stage point = level then (1 : ℝ) else 0) :=
    Finset.expect_nonneg fun _ _ => Finset.sum_nonneg fun _ _ => by split <;> norm_num
  have each : ∀ coordinate ∈ coordinates,
      𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Point Coord Slot budget Inner,
        (if (child table own).isSome ∧ indexedFails structure_ prover level budget child table
            (structure_.protocol.input prover table level) coordinate (loops (some coordinate))
          then (1 : ℝ) else 0) ≤
        (structure_.weight level coordinate + 1 / budget) *
          𝔼 table, (∑ point, if point ∈ structure_.protocol.asked prover table ∧
            structure_.protocol.stage point = level then (1 : ℝ) else 0) := by
    intro coordinate member
    calc
      _ = 𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
            (if (child table own).isSome ∧ indexedFails structure_ prover level budget child table
                (structure_.protocol.input prover table level) coordinate attempts
              then (1 : ℝ) else 0) :=
        Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
          RandomOracle.expect_eval (some coordinate) fun attempts =>
            if (child table own).isSome ∧ indexedFails structure_ prover level budget child table
                (structure_.protocol.input prover table level) coordinate attempts
              then (1 : ℝ) else 0
      _ = 𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
            ∑ anchor, (if structure_.protocol.input prover table level = anchor ∧
                (child table own).isSome ∧ indexedFails structure_ prover level budget child table
                  anchor coordinate attempts then (1 : ℝ) else 0) := by
        refine Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
          Finset.expect_congr rfl fun attempts _ => ?_
        rw [Finset.sum_eq_single (structure_.protocol.input prover table level)]
        · simp
        · intro anchor _ different
          simp [Ne.symm different]
        · simp
      _ = ∑ anchor, 𝔼 table, 𝔼 own : Inner,
            𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
            (if structure_.protocol.input prover table level = anchor ∧
                (child table own).isSome ∧ indexedFails structure_ prover level budget child table
                  anchor coordinate attempts then (1 : ℝ) else 0) := by
        simp only [Finset.expect_sum_comm]
      _ ≤ ∑ anchor, 𝔼 table, (if anchor ∈ structure_.protocol.asked prover table ∧
            structure_.protocol.stage anchor = level
            then structure_.weight level coordinate + 1 / budget else 0) :=
        Finset.sum_le_sum fun anchor _ => indexed_anchor_le structure_ prover lawful indexed
          inside positive accepts anchor member
      _ = _ := by
        rw [← Finset.expect_sum_comm, Finset.mul_expect]
        refine Finset.expect_congr rfl fun table _ => ?_
        rw [Finset.mul_sum]
        exact Finset.sum_congr rfl fun _ _ => by split <;> simp
  have listed : ∀ sub : List Coord, (∀ coordinate ∈ sub, coordinate ∈ coordinates) →
      𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Point Coord Slot budget Inner,
        (if (child table own).isSome ∧ ∃ coordinate ∈ sub, indexedFails structure_ prover level
            budget child table (structure_.protocol.input prover table level) coordinate
            (loops (some coordinate)) then (1 : ℝ) else 0) ≤
        ((sub.map (structure_.weight level)).sum + sub.length / budget) *
          𝔼 table, (∑ point, if point ∈ structure_.protocol.asked prover table ∧
            structure_.protocol.stage point = level then (1 : ℝ) else 0) := by
    intro sub
    induction sub with
    | nil => intro _; simp
    | cons coordinate rest induction =>
        intro members
        have head := each coordinate (members coordinate List.mem_cons_self)
        have tail := induction fun other inside => members other (List.mem_cons_of_mem _ inside)
        calc
          _ ≤ 𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Point Coord Slot budget Inner,
                ((if (child table own).isSome ∧ indexedFails structure_ prover level budget child
                    table (structure_.protocol.input prover table level) coordinate
                    (loops (some coordinate)) then (1 : ℝ) else 0) +
                  (if (child table own).isSome ∧ ∃ other ∈ rest, indexedFails structure_ prover
                    level budget child table (structure_.protocol.input prover table level) other
                    (loops (some other)) then (1 : ℝ) else 0)) := by
            refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun own _ =>
              Finset.expect_le_expect fun loops _ => ?_
            by_cases ran : (child table own).isSome
            · by_cases first : indexedFails structure_ prover level budget child table
                  (structure_.protocol.input prover table level) coordinate
                  (loops (some coordinate))
              · simp only [ran, first, List.mem_cons, true_and, exists_eq_or_imp, true_or,
                  ite_true]
                split <;> norm_num
              · simp [ran, first]
            · simp [ran]
          _ ≤ _ := by
            refine le_trans (le_of_eq ?_) ((add_le_add head tail).trans (le_of_eq ?_))
            · simp only [Finset.expect_add_distrib]
            · rw [List.map_cons, List.sum_cons, List.length_cons]
              push_cast
              ring
  calc
    _ ≤ 𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Point Coord Slot budget Inner,
          (if (child table own).isSome ∧ ∃ coordinate ∈ coordinates, indexedFails structure_
            prover level budget child table (structure_.protocol.input prover table level)
            coordinate (loops (some coordinate)) then (1 : ℝ) else 0) := by
      refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun own _ =>
        Finset.expect_le_expect fun loops _ => ?_
      by_cases event : (child table own).isSome ∧
          indexedNode structure_ prover level coordinates budget child table own loops = none
      · rw [ite_eq_left event,
          ite_eq_left ⟨event.1, indexedNode_eq_none structure_ prover event.1 event.2⟩]
      · rw [ite_eq_right event]
        split <;> norm_num
    _ ≤ _ := (listed coordinates fun _ member => member).trans ?_
  have charged : structure_.charge level = (coordinates.map (structure_.weight level)).sum := by
    unfold Structure.charge
    rw [ite_eq_left inside, indexed]
  have looped : structure_.loops level = coordinates.length := by
    unfold Structure.loops
    rw [indexed]
  have moved : 𝔼 table, (∑ point, if point ∈ structure_.protocol.asked prover table ∧
      structure_.protocol.stage point = level then structure_.charge level else 0) =
        structure_.charge level * 𝔼 table, (∑ point,
          if point ∈ structure_.protocol.asked prover table ∧
            structure_.protocol.stage point = level then (1 : ℝ) else 0) := by
    rw [Finset.mul_expect]
    refine Finset.expect_congr rfl fun table _ => ?_
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun _ _ => by split <;> simp
  have share : (coordinates.length : ℝ) / budget * 𝔼 table, (∑ point,
      if point ∈ structure_.protocol.asked prover table ∧
        structure_.protocol.stage point = level then (1 : ℝ) else 0) ≤
      coordinates.length / budget * (queries + 1) :=
    mul_le_mul_of_nonneg_left count (by positivity)
  rw [moved, charged, looped]
  have reorder : ((queries : ℝ) + 1) * (coordinates.length : ℕ) / budget =
      coordinates.length / budget * (queries + 1) := by ring
  rw [reorder]
  linarith [countNonnegative]

end NightstreamFPrime.Spec.Checkpoint
