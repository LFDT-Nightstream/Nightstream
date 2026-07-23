import Init.Data.List.Nat.Sum
import Init.Omega
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Rejection

/-!
Finite truncated rejection sampling over explicit Cartesian seed tapes.

Owns: duplicate-free Cartesian powers of a finite support, an executable
first-success scan with a finite cutoff, the induced uniform experiment, and
a cutoff-uniform expected-call bound by the inverse one-call success
probability (and hence by the inverse of any positive success floor).

Does not own: an infinite or Las Vegas sampler, almost-sure termination, a
limit probability space, asymptotic polynomial time, a protocol, Fiat--Shamir,
Rust, R1CS, or constraints.

Every probability remains the existing `FiniteUniform.Experiment`
probability.  Cartesian tapes preserve multiplicity and are not deduplicated
after mapping to outcomes.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.InteractiveReduction.FiniteUniform

universe uSeed uOutcome

variable {Seed : Type uSeed}
variable {Outcome : Type uOutcome}

/-- All length-`attemptLimit` tapes over `values`, in lexicographic block
order.  The head of a tape is the first rejection-sampling call. -/
def cartesianTapes (values : List Seed) : Nat -> List (List Seed)
  | 0 => [[]]
  | attemptLimit + 1 =>
      values.flatMap fun head =>
        (cartesianTapes values attemptLimit).map fun tail => head :: tail

private theorem sum_map_constant_nat
    {Element : Type uSeed}
    (values : List Element)
    (constant : Nat) :
    (values.map (fun _ => constant)).sum = values.length * constant := by
  rw [List.map_const', List.sum_replicate_nat]

theorem cartesianTapes_length
    (values : List Seed)
    (attemptLimit : Nat) :
    (cartesianTapes values attemptLimit).length =
      values.length ^ attemptLimit := by
  induction attemptLimit with
  | zero => rfl
  | succ smaller inductionHypothesis =>
      unfold cartesianTapes
      rw [List.length_flatMap]
      simp only [List.length_map, Function.comp_apply]
      rw [sum_map_constant_nat, inductionHypothesis, Nat.pow_succ,
        Nat.mul_comm]

theorem cartesianTapes_tape_length
    (values : List Seed)
    (attemptLimit : Nat) :
    forall tape, tape ∈ cartesianTapes values attemptLimit ->
      tape.length = attemptLimit := by
  induction attemptLimit with
  | zero =>
      intro tape member
      simp only [cartesianTapes, List.mem_singleton] at member
      subst member
      rfl
  | succ smaller inductionHypothesis =>
      intro tape member
      simp only [cartesianTapes, List.mem_flatMap, List.mem_map] at member
      rcases member with ⟨head, _, tail, tailMember, rfl⟩
      simp only [List.length_cons]
      rw [inductionHypothesis tail tailMember]

theorem cartesianTapes_nodup
    (values : List Seed)
    (valuesNodup : values.Nodup)
    (attemptLimit : Nat) :
    (cartesianTapes values attemptLimit).Nodup := by
  induction attemptLimit with
  | zero => simp [cartesianTapes]
  | succ smaller inductionHypothesis =>
      unfold cartesianTapes
      apply List.pairwise_flatMap.mpr
      constructor
      · intro head _
        rw [List.pairwise_map]
        exact inductionHypothesis.imp (by
          intro left right different equalCons
          exact different (List.cons.inj equalCons).2)
      · exact valuesNodup.imp (by
          intro leftHead rightHead differentHeads
          intro leftTape leftMember rightTape rightMember equalCons
          rcases List.mem_map.mp leftMember with
            ⟨leftTail, _, rfl⟩
          rcases List.mem_map.mp rightMember with
            ⟨rightTail, _, rfl⟩
          exact differentHeads (List.cons.inj equalCons).1)

/-- Cartesian power of a nonempty duplicate-free support. -/
def Support.cartesianPower
    (support : Support Seed)
    (attemptLimit : Nat) : Support (List Seed) where
  values := cartesianTapes support.values attemptLimit
  nodup := cartesianTapes_nodup support.values support.nodup attemptLimit
  nonempty := by
    apply List.ne_nil_of_length_pos
    rw [cartesianTapes_length]
    exact Nat.pow_pos support.cardinality_pos

theorem Support.cartesianPower_cardinality
    (support : Support Seed)
    (attemptLimit : Nat) :
    (support.cartesianPower attemptLimit).cardinality =
      support.cardinality ^ attemptLimit := by
  exact cartesianTapes_length support.values attemptLimit

theorem Support.cartesianPower_tape_length
    (support : Support Seed)
    (attemptLimit : Nat) :
    forall tape, tape ∈ (support.cartesianPower attemptLimit).values ->
      tape.length = attemptLimit :=
  cartesianTapes_tape_length support.values attemptLimit

/-- Number of verifier calls made by a first-success scan. -/
def firstSuccessCalls
    (success : Outcome -> Bool) : List Outcome -> Nat
  | [] => 0
  | head :: tail =>
      if success head then 1 else 1 + firstSuccessCalls success tail

@[simp]
theorem firstSuccessCalls_nil
    (success : Outcome -> Bool) :
    firstSuccessCalls success [] = 0 := rfl

theorem firstSuccessCalls_cons_of_failure
    (success : Outcome -> Bool)
    (head : Outcome)
    (tail : List Outcome)
    (headFailure : success head = false) :
    firstSuccessCalls success (head :: tail) =
      1 + firstSuccessCalls success tail := by
  simp [firstSuccessCalls, headFailure]

theorem firstSuccessCalls_cons_of_success
    (success : Outcome -> Bool)
    (head : Outcome)
    (tail : List Outcome)
    (headSuccess : success head = true) :
    firstSuccessCalls success (head :: tail) = 1 := by
  simp [firstSuccessCalls, headSuccess]

/-- First successful output, if the finite tape contains one. -/
def firstSuccessfulOutput
    (success : Outcome -> Bool) : List Outcome -> Option Outcome
  | [] => none
  | head :: tail =>
      if success head then some head else firstSuccessfulOutput success tail

/-- Executable result of finite truncated rejection sampling. -/
structure TruncatedFirstSuccessResult (Outcome : Type uOutcome) where
  attemptsUsed : Nat
  accepted : Option Outcome

/-- Scan a finite output tape until its first successful output. -/
def runTruncatedFirstSuccess
    (success : Outcome -> Bool)
    (tape : List Outcome) : TruncatedFirstSuccessResult Outcome where
  attemptsUsed := firstSuccessCalls success tape
  accepted := firstSuccessfulOutput success tape

theorem firstSuccessCalls_le_length
    (success : Outcome -> Bool)
    (tape : List Outcome) :
    firstSuccessCalls success tape <= tape.length := by
  induction tape with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases headSuccess : success head with
      | false =>
          rw [firstSuccessCalls_cons_of_failure success head tail headSuccess]
          simpa [Nat.add_comm] using
            Nat.add_le_add_left inductionHypothesis 1
      | true =>
          rw [firstSuccessCalls_cons_of_success success head tail headSuccess]
          simpa [Nat.add_comm] using
            Nat.succ_le_succ (Nat.zero_le tail.length)

theorem firstSuccessCalls_map
    {Mapped : Type uSeed}
    (success : Outcome -> Bool)
    (mapping : Mapped -> Outcome)
    (tape : List Mapped) :
    firstSuccessCalls success (tape.map mapping) =
      firstSuccessCalls (fun seed => success (mapping seed)) tape := by
  induction tape with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, firstSuccessCalls]
      rw [inductionHypothesis]

/-- Uniform finite experiment obtained by sampling an iid Cartesian seed tape
and scanning the mapped outputs to the first success. -/
def Experiment.truncatedFirstSuccess
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (attemptLimit : Nat) : Experiment (TruncatedFirstSuccessResult Outcome) where
  Seed := List experiment.Seed
  support := experiment.support.cartesianPower attemptLimit
  outcome := fun tape =>
    runTruncatedFirstSuccess success (tape.map experiment.outcome)

/-- Query cost charged to one Cartesian seed tape. -/
def Experiment.truncatedQueryCost
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (attemptLimit : Nat)
    (tape : (experiment.truncatedFirstSuccess success attemptLimit).Seed) : Nat :=
  firstSuccessCalls
    (fun seed => success (experiment.outcome seed)) tape

theorem Experiment.truncatedQueryCost_eq_attemptsUsed
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (attemptLimit : Nat)
    (tape : (experiment.truncatedFirstSuccess success attemptLimit).Seed) :
    experiment.truncatedQueryCost success attemptLimit tape =
      TruncatedFirstSuccessResult.attemptsUsed
        ((experiment.truncatedFirstSuccess success attemptLimit).outcome tape) := by
  unfold Experiment.truncatedQueryCost Experiment.truncatedFirstSuccess
    runTruncatedFirstSuccess
  exact (firstSuccessCalls_map success experiment.outcome tape).symm

private def totalFirstSuccessCalls
    (values : List Seed)
    (success : Seed -> Bool)
    (attemptLimit : Nat) : Nat :=
  ((cartesianTapes values attemptLimit).map
    (firstSuccessCalls success)).sum

private theorem Experiment.truncatedFirstSuccess_totalCost_eq
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (attemptLimit : Nat) :
    (experiment.truncatedFirstSuccess success attemptLimit).totalCost
        (experiment.truncatedQueryCost success attemptLimit) =
      totalFirstSuccessCalls experiment.support.values
        (fun seed => success (experiment.outcome seed)) attemptLimit := rfl

private theorem sum_map_flatMap_nat
    {Element : Type uSeed}
    {Mapped : Type uOutcome}
    (values : List Element)
    (mapping : Element -> List Mapped)
    (cost : Mapped -> Nat) :
    ((values.flatMap mapping).map cost).sum =
      (values.map fun value => ((mapping value).map cost).sum).sum := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.map_append, List.sum_append_nat,
        List.map_cons, List.sum_cons, inductionHypothesis]

private theorem sum_map_one_add_nat
    {Element : Type uSeed}
    (values : List Element)
    (cost : Element -> Nat) :
    (values.map (fun value => 1 + cost value)).sum =
      values.length + (values.map cost).sum := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis]
      omega

private theorem sum_firstSuccessCalls_cons
    (tails : List (List Seed))
    (success : Seed -> Bool)
    (head : Seed) :
    (tails.map fun tail => firstSuccessCalls success (head :: tail)).sum =
      tails.length +
        if success head then 0
        else (tails.map (firstSuccessCalls success)).sum := by
  cases headSuccess : success head with
  | false =>
      have mappedCalls :
          (tails.map fun tail =>
            firstSuccessCalls success (head :: tail)) =
          tails.map (fun tail => 1 + firstSuccessCalls success tail) := by
        apply List.map_congr_left
        intro tail _
        exact firstSuccessCalls_cons_of_failure
          success head tail headSuccess
      rw [mappedCalls, sum_map_one_add_nat]
      simp [headSuccess]
  | true =>
      have mappedCalls :
          (tails.map fun tail =>
            firstSuccessCalls success (head :: tail)) =
          tails.map (fun _ => 1) := by
        apply List.map_congr_left
        intro tail _
        exact firstSuccessCalls_cons_of_success
          success head tail headSuccess
      rw [mappedCalls, sum_map_constant_nat]
      simp [headSuccess]

private theorem sum_head_call_blocks
    (values : List Seed)
    (success : Seed -> Bool)
    (tailCardinality tailCalls : Nat) :
    (values.map fun head =>
      tailCardinality + if success head then 0 else tailCalls).sum =
      values.length * tailCardinality +
        values.countP (fun head => !success head) * tailCalls := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      cases headSuccess : success head with
      | false =>
          simp [List.countP_cons, headSuccess, inductionHypothesis,
            Nat.add_mul]
          omega
      | true =>
          simp [List.countP_cons, headSuccess, inductionHypothesis,
            Nat.add_mul]
          omega

private theorem length_eq_success_add_failure
    (values : List Seed)
    (success : Seed -> Bool) :
    values.length = values.countP success +
      values.countP (fun seed => !success seed) := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases headSuccess : success head with
      | false =>
          simp [List.countP_cons, headSuccess, inductionHypothesis]
          omega
      | true =>
          simp [List.countP_cons, headSuccess, inductionHypothesis]
          omega

private theorem totalFirstSuccessCalls_succ
    (values : List Seed)
    (success : Seed -> Bool)
    (attemptLimit : Nat) :
    totalFirstSuccessCalls values success (attemptLimit + 1) =
      values.length ^ (attemptLimit + 1) +
        values.countP (fun seed => !success seed) *
          totalFirstSuccessCalls values success attemptLimit := by
  change
    ((cartesianTapes values (attemptLimit + 1)).map
      (firstSuccessCalls success)).sum =
      values.length ^ (attemptLimit + 1) +
        values.countP (fun seed => !success seed) *
          totalFirstSuccessCalls values success attemptLimit
  rw [show cartesianTapes values (attemptLimit + 1) =
    values.flatMap (fun head =>
      (cartesianTapes values attemptLimit).map fun tail => head :: tail) from rfl]
  rw [sum_map_flatMap_nat]
  have blockMap :
      (values.map fun head =>
        (((cartesianTapes values attemptLimit).map fun tail => head :: tail).map
          (firstSuccessCalls success)).sum) =
      (values.map fun head =>
        (cartesianTapes values attemptLimit).length +
          if success head then 0
          else ((cartesianTapes values attemptLimit).map
            (firstSuccessCalls success)).sum) := by
    apply List.map_congr_left
    intro head _
    simpa [List.map_map] using
      sum_firstSuccessCalls_cons
        (cartesianTapes values attemptLimit) success head
  rw [blockMap, sum_head_call_blocks, cartesianTapes_length]
  change
    values.length * (values.length ^ attemptLimit) +
        values.countP (fun seed => !success seed) *
          totalFirstSuccessCalls values success attemptLimit =
      values.length ^ (attemptLimit + 1) +
        values.countP (fun seed => !success seed) *
          totalFirstSuccessCalls values success attemptLimit
  have powerStep :
      values.length * (values.length ^ attemptLimit) =
        values.length ^ (attemptLimit + 1) := by
    rw [Nat.pow_succ, Nat.mul_comm]
  rw [powerStep]

private theorem totalFirstSuccessCalls_mul_success_le
    (values : List Seed)
    (success : Seed -> Bool)
    (attemptLimit : Nat) :
    totalFirstSuccessCalls values success attemptLimit *
        values.countP success <=
      values.length ^ (attemptLimit + 1) := by
  induction attemptLimit with
  | zero =>
      simp [totalFirstSuccessCalls, cartesianTapes]
  | succ smaller inductionHypothesis =>
      have partition :
          values.countP success +
              values.countP (fun seed => !success seed) = values.length := by
        exact (length_eq_success_add_failure values success).symm
      have scaled :
          values.countP (fun seed => !success seed) *
              (totalFirstSuccessCalls values success smaller *
                values.countP success) <=
            values.countP (fun seed => !success seed) *
              values.length ^ (smaller + 1) :=
        Nat.mul_le_mul_left
          (values.countP (fun seed => !success seed)) inductionHypothesis
      change
        totalFirstSuccessCalls values success (smaller + 1) *
            values.countP success <=
          values.length ^ (smaller + 1 + 1)
      rw [totalFirstSuccessCalls_succ]
      calc
        (values.length ^ (smaller + 1) +
              values.countP (fun seed => !success seed) *
                totalFirstSuccessCalls values success smaller) *
            values.countP success =
            values.length ^ (smaller + 1) * values.countP success +
              values.countP (fun seed => !success seed) *
                (totalFirstSuccessCalls values success smaller *
                  values.countP success) := by
                rw [Nat.add_mul, Nat.mul_assoc]
        _ <= values.length ^ (smaller + 1) * values.countP success +
              values.countP (fun seed => !success seed) *
                values.length ^ (smaller + 1) :=
                Nat.add_le_add_left scaled _
        _ = values.length ^ (smaller + 1) *
              (values.countP success +
                values.countP (fun seed => !success seed)) := by
                rw [Nat.mul_add,
                  Nat.mul_comm
                    (values.countP (fun seed => !success seed))]
        _ = values.length ^ (smaller + 1 + 1) := by
          rw [partition]
          exact (Nat.pow_succ values.length (smaller + 1)).symm

private theorem one_div_probabilityBool_eq_count_ratio
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool) :
    1 / experiment.probabilityBool success =
      ratio experiment.support.cardinality
        (experiment.countBool success) := by
  unfold Experiment.probabilityBool ratio
  rw [Rat.div_def, Rat.div_def, Rat.div_def, Rat.inv_mul_rev,
    Rat.inv_inv, Rat.one_mul]

private theorem div_mul_eq_mul_div
    (left denominator right : Rat) :
    (left / denominator) * right = (left * right) / denominator := by
  rw [Rat.div_def, Rat.div_def]
  calc
    (left * denominator⁻¹) * right =
        left * (denominator⁻¹ * right) := by
          rw [Rat.mul_assoc]
    _ = left * (right * denominator⁻¹) := by
      rw [Rat.mul_comm denominator⁻¹ right]
    _ = (left * right) * denominator⁻¹ := by
      rw [Rat.mul_assoc]

/-- Exact count-ratio bound for the finite truncated sampler.  The bound is
uniform in the cutoff. -/
theorem Experiment.truncatedFirstSuccess_expectedQueries_le_countRatio
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (attemptLimit : Nat) :
    (experiment.truncatedFirstSuccess success attemptLimit).expectedCost
        (experiment.truncatedQueryCost success attemptLimit) <=
      ratio experiment.support.cardinality
        (experiment.countBool success) := by
  let seedSuccess : experiment.Seed -> Bool :=
    fun seed => success (experiment.outcome seed)
  have successCountPos : 0 < experiment.countBool success := by
    unfold Experiment.countBool
    rw [List.countP_eq_length_filter]
    exact List.length_pos_iff.mpr nonempty
  have tapeCardinalityPos :
      0 < experiment.support.cardinality ^ attemptLimit :=
    Nat.pow_pos experiment.support.cardinality_pos
  have naturalBound := totalFirstSuccessCalls_mul_success_le
    experiment.support.values seedSuccess attemptLimit
  unfold Experiment.expectedCost
  rw [Experiment.truncatedFirstSuccess_totalCost_eq]
  change
    (totalFirstSuccessCalls experiment.support.values seedSuccess attemptLimit :
        Rat) /
        ((experiment.support.cartesianPower attemptLimit).cardinality : Nat) <=
      (experiment.support.cardinality : Rat) /
        (experiment.countBool success : Rat)
  rw [Support.cartesianPower_cardinality]
  apply (le_div_iff_of_pos (Rat.natCast_pos.mpr successCountPos)).mpr
  have quotientRearrange :
      (totalFirstSuccessCalls experiment.support.values seedSuccess attemptLimit :
          Rat) /
          (experiment.support.cardinality ^ attemptLimit : Nat) *
          (experiment.countBool success : Rat) =
        ((totalFirstSuccessCalls experiment.support.values seedSuccess
            attemptLimit : Rat) *
          (experiment.countBool success : Rat)) /
            (experiment.support.cardinality ^ attemptLimit : Nat) := by
    exact div_mul_eq_mul_div
      (totalFirstSuccessCalls experiment.support.values seedSuccess
        attemptLimit : Rat)
      (experiment.support.cardinality ^ attemptLimit : Nat)
      (experiment.countBool success : Rat)
  rw [quotientRearrange]
  apply (div_le_iff_of_pos (Rat.natCast_pos.mpr tapeCardinalityPos)).mpr
  rw [← Rat.natCast_mul, ← Rat.natCast_mul]
  exact Rat.natCast_le_natCast.mpr (by
    simpa [seedSuccess, Experiment.countBool, Support.cardinality,
      Nat.pow_succ, Nat.mul_comm] using naturalBound)

/-- The finite truncated sampler uses at most the inverse exact success
probability in expectation, uniformly in the cutoff. -/
theorem Experiment.truncatedFirstSuccess_expectedQueries_le_inverseSuccess
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (attemptLimit : Nat) :
    (experiment.truncatedFirstSuccess success attemptLimit).expectedCost
        (experiment.truncatedQueryCost success attemptLimit) <=
      1 / experiment.probabilityBool success := by
  have countBound :=
    Experiment.truncatedFirstSuccess_expectedQueries_le_countRatio
      experiment success nonempty attemptLimit
  rw [one_div_probabilityBool_eq_count_ratio experiment success]
  exact countBound

/-- A positive lower bound on one-call success yields the floor-facing query
bound used by a later polynomial-time argument. -/
theorem Experiment.truncatedFirstSuccess_expectedQueries_le_inverseFloor
    (experiment : Experiment Outcome)
    (success : Outcome -> Bool)
    (nonempty : experiment.support.values.filter
      (fun seed => success (experiment.outcome seed)) ≠ [])
    (attemptLimit : Nat)
    (successFloor : Rat)
    (floorPos : 0 < successFloor)
    (floorBound : successFloor <= experiment.probabilityBool success) :
    (experiment.truncatedFirstSuccess success attemptLimit).expectedCost
        (experiment.truncatedQueryCost success attemptLimit) <=
      1 / successFloor := by
  exact Rat.le_trans
    (experiment.truncatedFirstSuccess_expectedQueries_le_inverseSuccess
      success nonempty attemptLimit)
    (div_le_div_of_nonneg_of_le_of_pos_le
      Rat.natCast_nonneg Rat.le_refl floorPos floorBound)

end Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
