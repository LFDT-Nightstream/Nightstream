import NightstreamFPrime.Spec.Checkpoint.Extractor

/-! The success step of a checkpoint of Akita Theorem 10.17 (ePrint 2026/1983, Appendix E.2,
"Checkpoint stages", Claim 1, Claim 2 and "Success"). -/

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

/-- The event `Fail_ℓ` of Appendix E.2 and the event that the loop of step 3 finds no attempt:
step 1 returns a certificate that fails the test, and the loop finds nothing or a certificate
with the same candidate. -/
def checkpointFails (level budget : ℕ)
    (child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Point → Coord → Slot) (anchor : Point) (own : Inner)
    (attempts : Fin budget → (Point → Coord → Slot) × Inner) : Prop :=
  ∃ found, child table own = some (.inl found) ∧ ¬ structure_.good level found ∧
    (checkpointChild structure_ prover level budget child table anchor attempts = none ∨
      ∃ other, checkpointChild structure_ prover level budget child table anchor attempts =
          some (.inl other) ∧
        structure_.candidate level other = structure_.candidate level found)

omit [Fintype Point] [Fintype Coord] [DecidableEq Coord] [Fintype Slot] [Nonempty Slot] in
theorem checkpointNode_eq_none {level budget : ℕ}
    {child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    {table : Point → Coord → Slot} {own : Inner} {loops : Loops Point Coord Slot budget Inner}
    (ran : (child table own).isSome)
    (missed : checkpointNode structure_ prover level budget child table own loops = none) :
    checkpointFails structure_ prover level budget child table
      (structure_.protocol.input prover table level) own (loops none) := by
  unfold checkpointNode at missed
  rcases first : child table own with _ | (found | broken)
  · simp [first] at ran
  · simp only [first] at missed
    unfold checkpointRetry at missed
    split at missed
    · exact absurd missed (by simp)
    · rename_i notGood
      refine ⟨found, first, notGood, ?_⟩
      split at missed
      · rename_i other tried
        split at missed
        · rename_i same
          exact Or.inr ⟨other, tried, same⟩
        · exact absurd missed (by simp)
      · exact absurd missed (by simp)
      · rename_i nothing
        exact Or.inl nothing
  · simp [first] at missed

/-- The candidate of the certificate of an attempt. -/
def candidateKey (level : ℕ) : Option (Cert ⊕ Break) → Option Cand
  | some (.inl certificate) => some (structure_.candidate level certificate)
  | _ => none

omit [Fintype Point] [Fintype Coord] [DecidableEq Coord] [Fintype Slot] [Nonempty Slot] in
private theorem event_regrow (level budget : ℕ)
    (child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break))
    (table : Point → Coord → Slot) (anchor : Point) (base : (Point → Coord → Slot) × Inner)
    (attempts : Fin budget → (Point → Coord → Slot) × Inner)
    (event : structure_.protocol.input prover
        (structure_.protocol.regrow level anchor table base.1) level = anchor ∧
      checkpointFails structure_ prover level budget child
        (structure_.protocol.regrow level anchor table base.1) anchor base.2 attempts) :
    (checkpointTry structure_ prover level child table anchor base).isSome = true ∧
      (firstPass (fun sample =>
          (checkpointTry structure_ prover level child table anchor sample).isSome = true)
          budget attempts = none ∨
        (∃ found, checkpointTry structure_ prover level child table anchor base =
            some (.inl found) ∧ ¬ structure_.good level found) ∧
          ∃ sample, firstPass (fun sample =>
              (checkpointTry structure_ prover level child table anchor sample).isSome = true)
              budget attempts = some sample ∧
            candidateKey structure_ level
                (checkpointTry structure_ prover level child table anchor sample) =
              candidateKey structure_ level
                (checkpointTry structure_ prover level child table anchor base)) := by
  obtain ⟨same, found, ran, notGood, loop⟩ := event
  have tryEq : checkpointTry structure_ prover level child
      (structure_.protocol.regrow level anchor table base.1) anchor =
        checkpointTry structure_ prover level child table anchor := by
    funext sample
    simp only [checkpointTry, Protocol.regrow_regrow]
  have baseRan : checkpointTry structure_ prover level child table anchor base =
      some (.inl found) := (guarded_eq_some structure_ prover).mpr ⟨same, ran⟩
  refine ⟨by rw [baseRan]; rfl, ?_⟩
  unfold checkpointChild at loop
  rw [tryEq] at loop
  rcases first : firstPass (fun sample =>
      (checkpointTry structure_ prover level child table anchor sample).isSome = true)
      budget attempts with _ | sample
  · exact Or.inl rfl
  · refine Or.inr ⟨⟨found, baseRan, notGood⟩, sample, rfl, ?_⟩
    rw [first] at loop
    rcases loop with missed | ⟨other, tried, equal⟩
    · have passed : (checkpointTry structure_ prover level child table anchor sample).isSome =
          true := firstPass_pass (pass := fun sample =>
            (checkpointTry structure_ prover level child table anchor sample).isSome = true) first
      have nothing : checkpointTry structure_ prover level child table anchor sample = none :=
        missed
      rw [nothing] at passed
      exact absurd passed (by simp)
    · have tried' : checkpointTry structure_ prover level child table anchor sample =
          some (.inl other) := tried
      rw [tried', baseRan]
      simp [candidateKey, equal]

omit [Fintype Point] [DecidableEq Point] [Nonempty Slot] in
/-- Claim 1 of Appendix E.2, the path part: claims that are false at the checkpoint input and true
at the end of the segment pass an answer of a bad set at a stage of the segment. -/
theorem segment_path (lawful : structure_.Lawful) {level span : ℕ}
    (checkpoint : structure_.kind level = .checkpoint span) (table : Point → Coord → Slot)
    (candidate : Cand)
    (start : structure_.refuted level (structure_.protocol.input prover table level) candidate)
    (finish : ¬ structure_.refuted level
      (structure_.protocol.input prover table (level + span + 1)) candidate) :
    ∃ stage, level ≤ stage ∧ stage ≤ level + span ∧
      table (structure_.protocol.input prover table stage) ∈
        structure_.bad level (structure_.protocol.input prover table stage) candidate := by
  by_contra none
  push Not at none
  apply finish
  have inside := (lawful.segment level span checkpoint).1
  have reach : ∀ offset, offset ≤ span + 1 → structure_.refuted level
      (structure_.protocol.input prover table (level + offset)) candidate := by
    intro offset
    induction offset with
    | zero => exact fun _ => start
    | succ offset induction =>
        intro short
        have before := induction (by omega)
        exact lawful.refuted_step level span
          (structure_.protocol.input prover table (level + offset)) candidate
          (table (structure_.protocol.input prover table (level + offset))) (prover.run table)
          checkpoint (by rw [structure_.protocol.stage_input prover table (by omega)]; omega)
          (by rw [structure_.protocol.stage_input prover table (by omega)]; omega) before
          (none (level + offset) (by omega) (by omega))
  exact reach (span + 1) le_rfl

variable [Fintype Inner] [Nonempty Inner]

/-- The total error of the queried stage inputs of the segment in `Sub(anchor)`:
`Σ_{j ∈ seg(ℓ)} ε_j·N_{i,j}(J)`. -/
noncomputable def segmentCharge (level span : ℕ) (anchor : Point)
    (table : Point → Coord → Slot) : ℝ :=
  ∑ point, if point ∈ structure_.protocol.asked prover table ∧
      structure_.protocol.cut level point = some anchor ∧
      structure_.protocol.stage point ≤ level + span
    then structure_.error (structure_.protocol.stage point) else 0

/-- Claim 1 of Appendix E.2: for a candidate with false claims at the checkpoint input, a fresh
extraction below the input selects the input and returns the candidate with probability at most
`Σ_{j ∈ seg(ℓ)} ε_j·E_K[N_{i,j}(c, K)]`. -/
theorem segment_bad_le (lawful : structure_.Lawful) {level span frontier : ℕ}
    (checkpoint : structure_.kind level = .checkpoint span) (deep : level + span + 1 ≤ frontier)
    {child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    (sound : Sound structure_ prover child (level + 1) frontier)
    (table : Point → Coord → Slot) (anchor : Point) (candidate : Cand)
    (refuted : structure_.refuted level anchor candidate) :
    𝔼 sample : (Point → Coord → Slot) × Inner,
        (if (checkpointTry structure_ prover level child table anchor sample).isSome = true ∧
          candidateKey structure_ level
            (checkpointTry structure_ prover level child table anchor sample) = some candidate
          then (1 : ℝ) else 0) ≤
      𝔼 fresh : Point → Coord → Slot, segmentCharge structure_ prover level span anchor
        (structure_.protocol.regrow level anchor table fresh) := by
  have pointwise (sample : (Point → Coord → Slot) × Inner) :
      (if (checkpointTry structure_ prover level child table anchor sample).isSome = true ∧
          candidateKey structure_ level
            (checkpointTry structure_ prover level child table anchor sample) = some candidate
          then (1 : ℝ) else 0) ≤
        ∑ point, if point ∈ structure_.protocol.asked prover
              (structure_.protocol.regrow level anchor table sample.1) ∧
            structure_.protocol.cut level point = some anchor ∧
            structure_.protocol.stage point ≤ level + span ∧
            structure_.protocol.regrow level anchor table sample.1 point ∈
              structure_.bad level point candidate then (1 : ℝ) else 0 := by
    by_cases event : (checkpointTry structure_ prover level child table anchor sample).isSome =
        true ∧ candidateKey structure_ level
          (checkpointTry structure_ prover level child table anchor sample) = some candidate
    · rw [ite_eq_left event]
      obtain ⟨_, keyed⟩ := event
      rcases tried : checkpointTry structure_ prover level child table anchor sample
        with _ | (certificate | broken)
      · simp [candidateKey, tried] at keyed
      · rw [tried] at keyed
        have equal : structure_.candidate level certificate = candidate := Option.some.inj keyed
        obtain ⟨same, ran⟩ := (guarded_eq_some structure_ prover).mp tried
        obtain ⟨_, held, agrees⟩ := sound _ _ _ ran
        have finish := lawful.refuted_end level span certificate _ checkpoint held
          (agrees (level + span + 1) deep)
        rw [equal] at finish
        have start : structure_.refuted level (structure_.protocol.input prover
            (structure_.protocol.regrow level anchor table sample.1) level) candidate := by
          rw [same]
          exact refuted
        obtain ⟨stage, low, high, isBad⟩ :=
          segment_path structure_ prover lawful checkpoint _ candidate start finish
        have inside : stage < structure_.protocol.stages := by
          have := (lawful.segment level span checkpoint).1
          omega
        refine le_trans ?_ (Finset.single_le_sum (fun _ _ => by split <;> norm_num)
          (Finset.mem_univ (structure_.protocol.input prover
            (structure_.protocol.regrow level anchor table sample.1) stage)))
        rw [ite_eq_left ⟨structure_.protocol.input_mem_asked prover _ inside,
          by rw [structure_.protocol.cut_input prover _ low (Nat.le_of_lt inside), same],
          by rw [structure_.protocol.stage_input prover _ (Nat.le_of_lt inside)]; exact high,
          isBad⟩]
      · simp [candidateKey, tried] at keyed
    · rw [ite_eq_right event]
      exact Finset.sum_nonneg fun _ _ => by split <;> norm_num
  calc
    _ ≤ 𝔼 sample : (Point → Coord → Slot) × Inner,
          ∑ point, (if point ∈ structure_.protocol.asked prover
              (structure_.protocol.regrow level anchor table sample.1) ∧
            structure_.protocol.cut level point = some anchor ∧
            structure_.protocol.stage point ≤ level + span ∧
            structure_.protocol.regrow level anchor table sample.1 point ∈
              structure_.bad level point candidate then (1 : ℝ) else 0) :=
      Finset.expect_le_expect fun sample _ => pointwise sample
    _ = 𝔼 fresh : Point → Coord → Slot,
          ∑ point, (if point ∈ structure_.protocol.asked prover
              (structure_.protocol.regrow level anchor table fresh) ∧
            structure_.protocol.cut level point = some anchor ∧
            structure_.protocol.stage point ≤ level + span ∧
            structure_.protocol.regrow level anchor table fresh point ∈
              structure_.bad level point candidate then (1 : ℝ) else 0) :=
      expect_fst fun fresh : Point → Coord → Slot =>
          ∑ point, (if point ∈ structure_.protocol.asked prover
              (structure_.protocol.regrow level anchor table fresh) ∧
            structure_.protocol.cut level point = some anchor ∧
            structure_.protocol.stage point ≤ level + span ∧
            structure_.protocol.regrow level anchor table fresh point ∈
              structure_.bad level point candidate then (1 : ℝ) else 0)
    _ ≤ _ := by
      unfold segmentCharge
      rw [Finset.expect_sum_comm, Finset.expect_sum_comm]
      refine Finset.sum_le_sum fun point _ => ?_
      by_cases inside : structure_.protocol.cut level point = some anchor
      · rw [RandomOracle.expect_update point]
        refine Finset.expect_le_expect fun fresh _ => ?_
        simp only [structure_.protocol.regrow_update level anchor table fresh inside,
          structure_.protocol.asked_update, Function.update_self, inside, true_and]
        by_cases queried : point ∈ structure_.protocol.asked prover
            (structure_.protocol.regrow level anchor table fresh) ∧
            structure_.protocol.stage point ≤ level + span
        · simp only [queried, true_and, and_self, ite_true]
          exact lawful.bad_mass level span point candidate checkpoint
            (structure_.protocol.cut_stage _ _ _ inside) queried.2
        · have none : ∀ reply : Coord → Slot, ¬ (point ∈ structure_.protocol.asked prover
              (structure_.protocol.regrow level anchor table fresh) ∧
              structure_.protocol.stage point ≤ level + span ∧
              reply ∈ structure_.bad level point candidate) :=
            fun reply holds => queried ⟨holds.1, holds.2.1⟩
          simp [none, queried]
      · simp [inside]

omit [Nonempty Slot] in
/-- Claim 2 of Appendix E.2, second part: at most `Q + 1` stage inputs can be the selected
checkpoint input after the table below them is replaced. -/
theorem selected_count_le {queries : ℕ} (bounded : prover.QueryBound queries) {level : ℕ}
    (within : level ≤ structure_.protocol.stages) (table : Point → Coord → Slot) :
    ∑ anchor, (if ∃ fresh, structure_.protocol.input prover
        (structure_.protocol.regrow level anchor table fresh) level = anchor
      then (1 : ℝ) else 0) ≤ queries + 1 := by
  have inside : (Finset.univ.filter fun anchor => ∃ fresh, structure_.protocol.input prover
      (structure_.protocol.regrow level anchor table fresh) level = anchor) ⊆
        insert (structure_.protocol.input prover table level)
          ((prover.queries table).toFinset.image fun point =>
            (structure_.protocol.cut level point).getD point) := by
    intro anchor member
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at member
    obtain ⟨fresh, selected⟩ := member
    rw [Finset.mem_insert, Finset.mem_image]
    by_cases touched : ∃ point ∈ prover.queries table,
        structure_.protocol.cut level point = some anchor
    · obtain ⟨point, queried, cutEq⟩ := touched
      exact Or.inr ⟨point, List.mem_toFinset.mpr queried, by rw [cutEq]; rfl⟩
    · left
      push Not at touched
      refine selected.symm.trans (structure_.protocol.input_congr prover table _ level
        (fun point queried => ?_) fun earlier less => ?_)
      · simp [Protocol.regrow, touched point queried]
      · have outside : structure_.protocol.cut level
            (structure_.protocol.input prover table earlier) ≠ some anchor := fun cutEq => by
          have := structure_.protocol.cut_stage _ _ _ cutEq
          rw [structure_.protocol.stage_input prover table (by omega)] at this
          omega
        simp [Protocol.regrow, outside]
  rw [Finset.sum_boole]
  have count := (Finset.card_le_card inside).trans (Finset.card_insert_le _ _)
  have short := (Finset.card_image_le (s := (prover.queries table).toFinset)
    (f := fun point => (structure_.protocol.cut level point).getD point)).trans
      ((List.toFinset_card_le (prover.queries table)).trans (bounded.queries_length_le table))
  exact_mod_cast count.trans (Nat.add_le_add_right short 1)

private noncomputable def failChance (level budget : ℕ)
    (child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)) (anchor : Point)
    (table : Point → Coord → Slot) : ℝ :=
  𝔼 own : Inner, 𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
    if structure_.protocol.input prover table level = anchor ∧
      checkpointFails structure_ prover level budget child table anchor own attempts
    then (1 : ℝ) else 0

private noncomputable def selectable (level budget : ℕ) (anchor : Point)
    (table : Point → Coord → Slot) : ℝ :=
  if ∃ fresh, structure_.protocol.input prover
    (structure_.protocol.regrow level anchor table fresh) level = anchor then 1 / (budget : ℝ) else 0

omit [Fintype Inner] [Nonempty Inner] [Nonempty Slot] in
private theorem segmentCharge_nonnegative (lawful : structure_.Lawful) (level span : ℕ)
    (anchor : Point) (table : Point → Coord → Slot) :
    0 ≤ segmentCharge structure_ prover level span anchor table :=
  Finset.sum_nonneg fun _ _ => by split <;> simp [lawful.error_nonnegative]

private theorem checkpoint_anchor_le (lawful : structure_.Lawful) {level span frontier : ℕ}
    (checkpoint : structure_.kind level = .checkpoint span) (deep : level + span + 1 ≤ frontier)
    {budget : ℕ} (positive : 0 < budget)
    {child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    (sound : Sound structure_ prover child (level + 1) frontier) (anchor : Point) :
    𝔼 table, failChance structure_ prover level budget child anchor table ≤
      𝔼 table, (segmentCharge structure_ prover level span anchor table +
        selectable structure_ prover level budget anchor table) := by
  refine (le_of_eq (structure_.protocol.expect_regrow level anchor _)).trans
    (le_trans ?_ (le_of_eq (structure_.protocol.expect_regrow level anchor _).symm))
  refine Finset.expect_le_expect fun table _ => ?_
  have selectSame (fresh : Point → Coord → Slot) :
      selectable structure_ prover level budget anchor
        (structure_.protocol.regrow level anchor table fresh) =
      selectable structure_ prover level budget anchor table := by
    unfold selectable
    simp only [Protocol.regrow_regrow]
  have charged : 0 ≤ 𝔼 fresh : Point → Coord → Slot, segmentCharge structure_ prover level span
      anchor (structure_.protocol.regrow level anchor table fresh) :=
    Finset.expect_nonneg fun _ _ => segmentCharge_nonnegative structure_ prover lawful _ _ _ _
  simp only [selectSame, Finset.expect_add_distrib]
  rw [Finset.expect_const Finset.univ_nonempty]
  unfold failChance
  calc
    _ = 𝔼 base : (Point → Coord → Slot) × Inner,
          𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
          (if structure_.protocol.input prover
              (structure_.protocol.regrow level anchor table base.1) level = anchor ∧
            checkpointFails structure_ prover level budget child
              (structure_.protocol.regrow level anchor table base.1) anchor base.2 attempts
            then (1 : ℝ) else 0) :=
      (expect_pair fun base : (Point → Coord → Slot) × Inner =>
        𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
          (if structure_.protocol.input prover
              (structure_.protocol.regrow level anchor table base.1) level = anchor ∧
            checkpointFails structure_ prover level budget child
              (structure_.protocol.regrow level anchor table base.1) anchor base.2 attempts
            then (1 : ℝ) else 0)).symm
    _ ≤ 𝔼 base : (Point → Coord → Slot) × Inner,
          𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
          (if (checkpointTry structure_ prover level child table anchor base).isSome = true ∧
            (firstPass (fun sample =>
                (checkpointTry structure_ prover level child table anchor sample).isSome = true)
                budget attempts = none ∨
              (∃ found, checkpointTry structure_ prover level child table anchor base =
                  some (.inl found) ∧ ¬ structure_.good level found) ∧
                ∃ sample, firstPass (fun sample =>
                    (checkpointTry structure_ prover level child table anchor sample).isSome = true)
                    budget attempts = some sample ∧
                  candidateKey structure_ level
                      (checkpointTry structure_ prover level child table anchor sample) =
                    candidateKey structure_ level
                      (checkpointTry structure_ prover level child table anchor base))
            then (1 : ℝ) else 0) := by
      refine Finset.expect_le_expect fun base _ => Finset.expect_le_expect fun attempts _ => ?_
      by_cases event : structure_.protocol.input prover
          (structure_.protocol.regrow level anchor table base.1) level = anchor ∧
          checkpointFails structure_ prover level budget child
            (structure_.protocol.regrow level anchor table base.1) anchor base.2 attempts
      · rw [ite_eq_left event, ite_eq_left (event_regrow structure_ prover level budget child table
          anchor base attempts event)]
      · rw [ite_eq_right event]
        split <;> norm_num
    _ ≤ if ∃ sample : (Point → Coord → Slot) × Inner,
          (checkpointTry structure_ prover level child table anchor sample).isSome = true
        then (𝔼 fresh : Point → Coord → Slot, segmentCharge structure_ prover level span anchor
          (structure_.protocol.regrow level anchor table fresh)) + 1 / budget else 0 := by
      refine (le_of_eq ?_).trans ((fork_le
        (fun sample : (Point → Coord → Slot) × Inner =>
          (checkpointTry structure_ prover level child table anchor sample).isSome = true)
        (fun base => ∃ found, checkpointTry structure_ prover level child table anchor base =
          some (.inl found) ∧ ¬ structure_.good level found)
        (fun sample => candidateKey structure_ level
          (checkpointTry structure_ prover level child table anchor sample))
        (𝔼 fresh : Point → Coord → Slot, segmentCharge structure_ prover level span anchor
          (structure_.protocol.regrow level anchor table fresh))
        charged positive fun base _ gated => ?_).trans (le_of_eq ?_))
      · congr!
      rotate_left
      · congr!
      obtain ⟨found, tried, notGood⟩ := gated
      obtain ⟨same, ran⟩ := (guarded_eq_some structure_ prover).mp tried
      obtain ⟨_, held, agrees⟩ := sound _ _ _ ran
      have start := lawful.refuted_start level span found anchor checkpoint held notGood
        (by rw [agrees level (by omega), same])
      refine le_trans (le_of_eq ?_) (segment_bad_le structure_ prover lawful checkpoint deep sound
        table anchor (structure_.candidate level found) start)
      rw [tried]
      congr!
    _ ≤ _ := by
      by_cases possible : ∃ sample : (Point → Coord → Slot) × Inner,
          (checkpointTry structure_ prover level child table anchor sample).isSome = true
      · obtain ⟨sample, passes⟩ := possible
        obtain ⟨out, ran⟩ := Option.isSome_iff_exists.mp passes
        obtain ⟨same, _⟩ := (guarded_eq_some structure_ prover).mp ran
        rw [ite_eq_left ⟨sample, passes⟩]
        unfold selectable
        rw [ite_eq_left ⟨sample.1, same⟩]
      · rw [ite_eq_right possible]
        unfold selectable
        split <;> positivity

/-- The success recurrence of a checkpoint (Appendix E.2, "Success"):
`s_ℓ ≥ s_{ℓ+1} − Σ_{j ∈ seg(ℓ)} ε_j·E[N_j(J)]`, with the loss of the limit of the loop. -/
theorem checkpoint_miss_le (lawful : structure_.Lawful) {level span frontier : ℕ}
    (checkpoint : structure_.kind level = .checkpoint span) (deep : level + span + 1 ≤ frontier)
    {queries : ℕ} (bounded : prover.QueryBound queries) {budget : ℕ} (positive : 0 < budget)
    {child : (Point → Coord → Slot) → Inner → Option (Cert ⊕ Break)}
    (sound : Sound structure_ prover child (level + 1) frontier) :
    𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Point Coord Slot budget Inner,
        (if (child table own).isSome ∧
          checkpointNode structure_ prover level budget child table own loops = none
          then (1 : ℝ) else 0) ≤
      (𝔼 table, ∑ point, if point ∈ structure_.protocol.asked prover table ∧
          level ≤ structure_.protocol.stage point ∧ structure_.protocol.stage point ≤ level + span
        then structure_.error (structure_.protocol.stage point) else 0) +
        (queries + 1) / budget := by
  calc
    _ ≤ 𝔼 table, 𝔼 own : Inner, 𝔼 loops : Loops Point Coord Slot budget Inner,
          (if checkpointFails structure_ prover level budget child table
            (structure_.protocol.input prover table level) own (loops none) then (1 : ℝ) else 0) := by
      refine Finset.expect_le_expect fun table _ => Finset.expect_le_expect fun own _ =>
        Finset.expect_le_expect fun loops _ => ?_
      by_cases event : (child table own).isSome ∧
          checkpointNode structure_ prover level budget child table own loops = none
      · rw [ite_eq_left event,
          ite_eq_left (checkpointNode_eq_none structure_ prover event.1 event.2)]
      · rw [ite_eq_right event]
        split <;> norm_num
    _ = 𝔼 table, 𝔼 own : Inner, 𝔼 attempts : Fin budget → (Point → Coord → Slot) × Inner,
          (if checkpointFails structure_ prover level budget child table
            (structure_.protocol.input prover table level) own attempts then (1 : ℝ) else 0) :=
      Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
        RandomOracle.expect_eval (none : Option Coord) fun attempts =>
          if checkpointFails structure_ prover level budget child table
            (structure_.protocol.input prover table level) own attempts then (1 : ℝ) else 0
    _ = ∑ anchor, 𝔼 table, failChance structure_ prover level budget child anchor table := by
      unfold failChance
      simp only [← Finset.expect_sum_comm]
      refine Finset.expect_congr rfl fun table _ => Finset.expect_congr rfl fun own _ =>
        Finset.expect_congr rfl fun attempts _ => ?_
      rw [Finset.sum_eq_single (structure_.protocol.input prover table level)]
      · simp
      · intro anchor _ different
        simp [Ne.symm different]
      · simp
    _ ≤ ∑ anchor, 𝔼 table, (segmentCharge structure_ prover level span anchor table +
          selectable structure_ prover level budget anchor table) :=
      Finset.sum_le_sum fun anchor _ => checkpoint_anchor_le structure_ prover lawful checkpoint
        deep positive sound anchor
    _ = 𝔼 table, ((∑ anchor, segmentCharge structure_ prover level span anchor table) +
          ∑ anchor, selectable structure_ prover level budget anchor table) := by
      rw [← Finset.expect_sum_comm]
      exact Finset.expect_congr rfl fun table _ => Finset.sum_add_distrib
    _ ≤ 𝔼 table, ((∑ point, if point ∈ structure_.protocol.asked prover table ∧
          level ≤ structure_.protocol.stage point ∧ structure_.protocol.stage point ≤ level + span
          then structure_.error (structure_.protocol.stage point) else 0) +
          (queries + 1) / budget) := by
      refine Finset.expect_le_expect fun table _ => add_le_add ?_ ?_
      · unfold segmentCharge
        rw [Finset.sum_comm]
        refine Finset.sum_le_sum fun point _ => ?_
        have errorNonnegative := lawful.error_nonnegative (structure_.protocol.stage point)
        rcases cutEq : structure_.protocol.cut level point with _ | anchor
        · simp only [reduceCtorEq, false_and, and_false, ite_false, Finset.sum_const_zero]
          split <;> linarith
        · have low := structure_.protocol.cut_stage _ _ _ cutEq
          rw [Finset.sum_eq_single anchor]
          · simp only [true_and]
            by_cases queried : point ∈ structure_.protocol.asked prover table ∧
                structure_.protocol.stage point ≤ level + span
            · rw [ite_eq_left queried, ite_eq_left ⟨queried.1, low, queried.2⟩]
            · rw [ite_eq_right queried]
              split <;> linarith
          · intro other _ different
            simp [Ne.symm different]
          · simp
      · unfold selectable
        have counted := selected_count_le structure_ prover bounded
          (Nat.le_of_lt (Nat.lt_of_le_of_lt (Nat.le_add_right level span)
            (lawful.segment level span checkpoint).1)) table
        have scaled : ∑ anchor, (if ∃ fresh, structure_.protocol.input prover
            (structure_.protocol.regrow level anchor table fresh) level = anchor
            then 1 / (budget : ℝ) else 0) =
            (∑ anchor, (if ∃ fresh, structure_.protocol.input prover
              (structure_.protocol.regrow level anchor table fresh) level = anchor
              then (1 : ℝ) else 0)) / budget := by
          have nonzero : (budget : ℝ) ≠ 0 := by positivity
          rw [eq_div_iff nonzero, Finset.sum_mul]
          exact Finset.sum_congr rfl fun _ _ => by split <;> simp [nonzero]
        rw [scaled]
        exact div_le_div_of_nonneg_right counted (by positivity)
    _ = _ := by
      rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty]

end NightstreamFPrime.Spec.Checkpoint
