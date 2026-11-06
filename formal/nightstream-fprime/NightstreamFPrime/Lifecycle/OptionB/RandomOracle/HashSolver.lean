import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshKnowledge
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.LightSolver
import NightstreamFPrime.Lifecycle.OptionB.RandomOracle.RefreshSolver
import NightstreamFPrime.Lifecycle.RandomOracle.LineFinder

/-! The Poseidon2 collision finder of option B in the random-oracle model: the prior-state
collision chances of the fork step (`ForkLink.collisionChance`) and the uniqueness step
(`UniquenessLink.stateChance`) as the success of an algorithm that outputs a Poseidon2
collision. This is the option-B form of `RandomOracleHashSolver`.

A `StateCollision` of two states names the colliding Poseidon2 inputs: the two state preimages,
or the two slot preimages of one slot. `collisionInputs` reads that input pair from the two
states (`PoseidonCollision`: two different inputs of one length with one Poseidon2 hash), and it
gives a pair
for every state collision (`collisionInputs_isSome`).

* Uniqueness step. `Reduction.solve` with the kernel `stateKernel` (the input pair of the two
  base outputs' prior states) runs the adversary, extracts, resamples and extracts again as the
  MSIS solver does. `stateChance_le_finder`: `budget · (stateChance − finder) ≤ Solver.work`.
* Fork step. `LineFinder.lineSolve` at coordinate `index` with the test `priorCollision`
  retries one `ρ` point. `collisionChance_le_finder`:
  `budget · (E[collision chance at index] − finder) ≤ Q + rhoCount + 1`.
* `hashSolve` picks one of the `rhoCount + 1` finders by its target. With a uniform target
  (`finderChance`), `hashCollisions_le_finder` bounds the sum of both chances by
  `(rhoCount + 1) · finderChance + (rhoCount · (Q + rhoCount + 1) + Solver.work) / budget`:
  `2 · finderChance + (4Q + 63)/budget` for the light fold
  (`light_hashCollisions_le_finder`) and `18 · finderChance + (52Q + 959)/budget` for the
  refresh fold (`refresh_hashCollisions_le_finder`), the bound of
  `RandomOracleHashSolver.hashCollisions_le_solver`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.OptionB.RandomOracle.HashSolver

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.OptionBFolds (freshToLight)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.OptionB.RandomOracle.ForkLink (Attempt Folds)

attribute [local instance low] Classical.propDecidable

/-- A Poseidon2 collision: two different inputs of one length with one hash. `Poseidon2.hash`
pads with zeros and does not encode the length, so `[x]` and `[x, 0]` share a hash with no
collision of the permutation. -/
def PoseidonCollision : Type :=
  {pair : List F × List F // pair.1 ≠ pair.2 ∧ pair.1.length = pair.2.length ∧
    Poseidon2.hash pair.1 = Poseidon2.hash pair.2}

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The slot preimages of one slot of two states. -/
def slotInputs (left right : State logicalWidth publicFits) (slot : Fin 2) : List F × List F :=
  (slotDomainChunk ++ serializeSlot (left.slots slot),
    slotDomainChunk ++ serializeSlot (right.slots slot))

/-- Different slot serializations with one slot digest are a Poseidon2 collision on the slot
preimages. -/
theorem slotInputs_collide {left right : State logicalWidth publicFits} {slot : Fin 2}
    (collides : serializeSlot (left.slots slot) ≠ serializeSlot (right.slots slot) ∧
      slotDigest (left.slots slot) = slotDigest (right.slots slot)) :
    (slotInputs left right slot).1 ≠ (slotInputs left right slot).2 ∧
      (slotInputs left right slot).1.length = (slotInputs left right slot).2.length ∧
      Poseidon2.hash (slotInputs left right slot).1 =
        Poseidon2.hash (slotInputs left right slot).2 :=
  ⟨fun same => collides.1 (List.append_cancel_left same),
    by simp only [slotInputs, List.length_append, serializeSlot_length_eq], collides.2⟩

/-- The Poseidon2 input pair that a collision of two states names: the state preimages, else
the slot preimages of slot 0, else those of slot 1. -/
noncomputable def collisionInputs (left right : State logicalWidth publicFits) :
    Option PoseidonCollision :=
  if states : statePreimage left ≠ statePreimage right ∧
      (statePreimage left).length = (statePreimage right).length ∧
      stateHash left = stateHash right then
    some ⟨(statePreimage left, statePreimage right), states⟩
  else if first : serializeSlot (left.slots 0) ≠ serializeSlot (right.slots 0) ∧
      slotDigest (left.slots 0) = slotDigest (right.slots 0) then
    some ⟨slotInputs left right 0, slotInputs_collide first⟩
  else if second : serializeSlot (left.slots 1) ≠ serializeSlot (right.slots 1) ∧
      slotDigest (left.slots 1) = slotDigest (right.slots 1) then
    some ⟨slotInputs left right 1, slotInputs_collide second⟩
  else none

/-- Every collision of two states gives a Poseidon2 input pair. -/
theorem collisionInputs_isSome {left right : State logicalWidth publicFits}
    (collision : StateCollision left right) : (collisionInputs left right).isSome := by
  unfold collisionInputs
  by_cases states : statePreimage left ≠ statePreimage right ∧
      (statePreimage left).length = (statePreimage right).length ∧
      stateHash left = stateHash right
  · rw [dite_eq_left states]
    rfl
  · rw [dite_eq_right states]
    rcases collision with direct | ⟨slot, collides⟩
    · exact absurd direct states
    · by_cases first : serializeSlot (left.slots 0) ≠ serializeSlot (right.slots 0) ∧
          slotDigest (left.slots 0) = slotDigest (right.slots 0)
      · rw [dite_eq_left first]
        rfl
      · rw [dite_eq_right first]
        have second : serializeSlot (left.slots 1) ≠ serializeSlot (right.slots 1) ∧
            slotDigest (left.slots 1) = slotDigest (right.slots 1) := by
          fin_cases slot
          · exact absurd collides first
          · exact collides
        rw [dite_eq_left second]
        rfl

/-- The prior-state collision of two attempts as a Poseidon2 input pair. -/
noncomputable def priorCollision (attempt attempt' : Attempt logicalWidth publicFits) :
    Option PoseidonCollision :=
  collisionInputs (toState attempt.link.prior) (toState attempt'.link.prior)

section Game

variable {schedule : Schedule}
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (key : CircuitKey logicalWidth publicFits)
  (folds : (Point schedule 8 → Answer) → Folds logicalWidth publicFits)
  (statement : Attempt logicalWidth publicFits → schedule.Statement)
  (messages : Attempt logicalWidth publicFits → schedule.Messages 8)
  (adversary : OracleComp (Point schedule 8) Answer (Attempt logicalWidth publicFits))

/-- The uniqueness-step kernel: the prior-state collision of the two base outputs. -/
noncomputable def stateKernel
    (view other : Reduction.View schedule 8 (Attempt logicalWidth publicFits)) :
    Option PoseidonCollision :=
  priorCollision view.base other.base

/-- The uniqueness-step state collision chance is at most the chance that the reduction with
the kernel `stateKernel` outputs a Poseidon2 collision, plus `Solver.work / budget`. -/
theorem stateChance_le_finder {queries : Nat} (bounded : adversary.QueryBound queries)
    (budget : ℕ) :
    budget * (UniquenessLink.stateChance ajtai key folds statement messages adversary -
        Solver.solverChance (ForkLink.game ajtai key folds statement messages) adversary
          stateKernel budget) ≤
      Solver.work schedule queries := by
  have bound := Solver.reductionChance_le_solver (ForkLink.game ajtai key folds statement messages)
    adversary stateKernel bounded budget
  have mono : UniquenessLink.stateChance ajtai key folds statement messages adversary ≤
      Uniqueness.reductionChance (ForkLink.game ajtai key folds statement messages) adversary
        fun oracle retries other otherRetries =>
          (stateKernel (Reduction.viewOf (ForkLink.game ajtai key folds statement messages)
              adversary oracle retries)
            (Reduction.viewOf (ForkLink.game ajtai key folds statement messages) adversary other
              otherRetries)).isSome :=
    Uniqueness.reductionChance_mono _ adversary
      fun _ _ _ _ _ _ _ collision => collisionInputs_isSome collision
  calc
    _ ≤ (budget : ℝ) * (Uniqueness.reductionChance
          (ForkLink.game ajtai key folds statement messages) adversary
            (fun oracle retries other otherRetries =>
              (stateKernel (Reduction.viewOf (ForkLink.game ajtai key folds statement messages)
                  adversary oracle retries)
                (Reduction.viewOf (ForkLink.game ajtai key folds statement messages) adversary
                  other otherRetries)).isSome) -
          Solver.solverChance (ForkLink.game ajtai key folds statement messages) adversary
            stateKernel budget) :=
      mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg budget)
    _ ≤ _ := bound

/-- The fork-step collision chance at `index` is at most the chance that the line finder with
the test `priorCollision` outputs a Poseidon2 collision, plus `(Q + rhoCount + 1) / budget`. -/
theorem collisionChance_le_finder {queries : Nat} (bounded : adversary.QueryBound queries)
    (index : Fin schedule.rhoCount) (budget : ℕ) :
    budget * (𝔼 oracle, (if Fork.Succeeds (ForkLink.game ajtai key folds statement messages)
          adversary oracle then
          ForkLink.collisionChance ajtai key folds statement messages adversary index oracle
        else 0) -
        LineFinder.lineChance (ForkLink.game ajtai key folds statement messages) index
          priorCollision adversary budget) ≤
      queries + schedule.rhoCount + 1 := by
  have bound := LineFinder.line_le (ForkLink.game ajtai key folds statement messages) index
    priorCollision adversary bounded budget
  have each (oracle : Point schedule 8 → Answer) :
      ForkLink.collisionChance ajtai key folds statement messages adversary index oracle ≤
        LineFinder.retryCollision (ForkLink.game ajtai key folds statement messages) index
          priorCollision adversary oracle :=
    Fork.retryChance_mono _ adversary index oracle fun _ _ collision =>
      collisionInputs_isSome collision
  have mono : 𝔼 oracle, (if Fork.Succeeds (ForkLink.game ajtai key folds statement messages)
        adversary oracle then
        ForkLink.collisionChance ajtai key folds statement messages adversary index oracle
      else 0) ≤
      𝔼 oracle, (if Fork.Succeeds (ForkLink.game ajtai key folds statement messages)
          adversary oracle then
          LineFinder.retryCollision (ForkLink.game ajtai key folds statement messages) index
            priorCollision adversary oracle else 0) :=
    Finset.expect_le_expect fun oracle _ => by
      split
      · exact each oracle
      · exact le_rfl
  calc
    _ ≤ (budget : ℝ) * (𝔼 oracle, (if Fork.Succeeds (ForkLink.game ajtai key folds statement
            messages) adversary oracle then
            LineFinder.retryCollision (ForkLink.game ajtai key folds statement messages) index
              priorCollision adversary oracle else 0) -
          LineFinder.lineChance (ForkLink.game ajtai key folds statement messages) index
            priorCollision adversary budget) :=
      mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg budget)
    _ ≤ _ := bound

/-- Named: instance search exceeds its size limit on this type. -/
noncomputable local instance (budget : ℕ) : Fintype (Fin budget → Point schedule 8 → Answer) :=
  Pi.instFintype

/-- The Poseidon2 collision finder with target `target`: the line finder at a `ρ` coordinate,
or the reduction with the state kernel. -/
noncomputable def hashSolve (target : Option (Fin schedule.rhoCount)) (budget : ℕ)
    (tape : Fin budget → Point schedule 8 → Answer) : Option PoseidonCollision :=
  match target with
  | some index => LineFinder.lineSolve (ForkLink.game ajtai key folds statement messages) index
      priorCollision adversary budget tape
  | none => Reduction.solve (ForkLink.game ajtai key folds statement messages) stateKernel
      adversary budget tape

/-- The chance that the finder with target `target` outputs a Poseidon2 collision. -/
noncomputable def targetChance (target : Option (Fin schedule.rhoCount)) (budget : ℕ) : ℝ :=
  𝔼 tape : Fin budget → Point schedule 8 → Answer,
    if (hashSolve ajtai key folds statement messages adversary target budget tape).isSome
    then 1 else 0

/-- The finder's chance with a uniform target. -/
noncomputable def finderChance (budget : ℕ) : ℝ :=
  𝔼 target : Option (Fin schedule.rhoCount),
    targetChance ajtai key folds statement messages adversary target budget

/-- The fork-step and uniqueness-step collision chances are at most `rhoCount + 1` times the
finder's chance plus `(rhoCount · (Q + rhoCount + 1) + Solver.work) / budget`. -/
theorem hashCollisions_le_finder {queries : Nat} (bounded : adversary.QueryBound queries)
    {budget : ℕ} (positive : 0 < budget) :
    𝔼 oracle, (if Fork.Succeeds (ForkLink.game ajtai key folds statement messages)
          adversary oracle then
        ∑ index, ForkLink.collisionChance ajtai key folds statement messages adversary index oracle
      else 0) +
        UniquenessLink.stateChance ajtai key folds statement messages adversary ≤
      ((schedule.rhoCount : ℝ) + 1) *
          finderChance ajtai key folds statement messages adversary budget +
        ((schedule.rhoCount : ℝ) * ((queries : ℝ) + schedule.rhoCount + 1) +
          (Solver.work schedule queries : ℝ)) / budget := by
  have positiveReal : (0 : ℝ) < budget := by exact_mod_cast positive
  have lines (index : Fin schedule.rhoCount) :=
    collisionChance_le_finder ajtai key folds statement messages adversary bounded index budget
  have state := stateChance_le_finder ajtai key folds statement messages adversary bounded budget
  have split : 𝔼 oracle, (if Fork.Succeeds (ForkLink.game ajtai key folds statement messages)
        adversary oracle then
        ∑ index, ForkLink.collisionChance ajtai key folds statement messages adversary index oracle
      else 0) =
      ∑ index, 𝔼 oracle, (if Fork.Succeeds (ForkLink.game ajtai key folds statement messages)
          adversary oracle then
          ForkLink.collisionChance ajtai key folds statement messages adversary index oracle
        else 0) := by
    rw [← Finset.expect_sum_comm]
    refine Finset.expect_congr rfl fun oracle _ => ?_
    split <;> simp
  have summed := Finset.sum_le_sum fun index (_ : index ∈ Finset.univ) => lines index
  rw [← Finset.mul_sum, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ,
    Fintype.card_fin, nsmul_eq_mul] at summed
  have total : ∑ target, targetChance ajtai key folds statement messages adversary target budget =
      ((schedule.rhoCount : ℝ) + 1) *
        finderChance ajtai key folds statement messages adversary budget := by
    unfold finderChance
    rw [Finset.expect_eq_sum_div_card, Finset.card_univ, Fintype.card_option, Fintype.card_fin]
    push_cast
    field_simp
  have targets : ∑ target, targetChance ajtai key folds statement messages adversary target budget =
      targetChance ajtai key folds statement messages adversary none budget +
        ∑ index, targetChance ajtai key folds statement messages adversary (some index) budget :=
    Fintype.sum_option _
  have stateTarget : targetChance ajtai key folds statement messages adversary none budget =
      Solver.solverChance (ForkLink.game ajtai key folds statement messages) adversary
        stateKernel budget := rfl
  have lineTarget : ∑ index, targetChance ajtai key folds statement messages adversary
        (some index) budget =
      ∑ index, LineFinder.lineChance (ForkLink.game ajtai key folds statement messages) index
        priorCollision adversary budget := rfl
  rw [split, ← total, targets, stateTarget, lineTarget, ← sub_le_iff_le_add',
    le_div_iff₀ positiveReal]
  linarith

end Game

section Light

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- The light Poseidon2 collision finder's chance with a uniform target. -/
noncomputable def lightFinderChance (budget : ℕ) : ℝ :=
  finderChance ajtai key (LightFork.folds ajtai) (fun attempt => freshToLight attempt.link.fold.fresh)
    (fun attempt => LightSchedule.messages (LightFork.proofOf attempt.link.fold.proof)) adversary
    budget

/-- The light fold's collision chances are at most `2 · finderChance + (4Q + 63) / budget`. -/
theorem light_hashCollisions_le_finder {queries : Nat} (bounded : adversary.QueryBound queries)
    {budget : ℕ} (positive : 0 < budget) :
    𝔼 oracle, (if Fork.Succeeds (LightFork.game ajtai key) adversary oracle then
        ∑ index, LightFork.collisionChance ajtai key adversary index oracle else 0) +
        LightKnowledge.stateChance ajtai key adversary ≤
      2 * lightFinderChance ajtai key adversary budget + (4 * queries + 63) / budget := by
  have bound := hashCollisions_le_finder ajtai key (LightFork.folds ajtai)
    (fun attempt => freshToLight attempt.link.fold.fresh)
    (fun attempt => LightSchedule.messages (LightFork.proofOf attempt.link.fold.proof))
    adversary bounded positive
  have count : ((LightSchedule.schedule logicalWidth publicFits).rhoCount : ℝ) = 1 := by
    show ((1 : Nat) : ℝ) = 1
    norm_num
  rw [count, LightSolver.work_eq] at bound
  calc
    _ ≤ _ := bound
    _ = _ := by
      unfold lightFinderChance
      push_cast
      ring

end Light

section Refresh

variable (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (key : CircuitKey logicalWidth publicFits)
  (adversary : OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
    (Attempt logicalWidth publicFits))

/-- The refresh Poseidon2 collision finder's chance with a uniform target. -/
noncomputable def refreshFinderChance (budget : ℕ) : ℝ :=
  finderChance ajtai key (RefreshFork.folds ajtai)
    (fun attempt => (attempt.link.fold.fresh, (RefreshFork.proofOf attempt.link.fold.proof).split))
    (fun attempt => RefreshSchedule.messages (RefreshFork.proofOf attempt.link.fold.proof).nifs)
    adversary budget

/-- The refresh fold's collision chances are at most
`18 · finderChance + (52Q + 959) / budget`. -/
theorem refresh_hashCollisions_le_finder {queries : Nat} (bounded : adversary.QueryBound queries)
    {budget : ℕ} (positive : 0 < budget) :
    𝔼 oracle, (if Fork.Succeeds (RefreshFork.game ajtai key) adversary oracle then
        ∑ index, RefreshFork.collisionChance ajtai key adversary index oracle else 0) +
        RefreshKnowledge.stateChance ajtai key adversary ≤
      18 * refreshFinderChance ajtai key adversary budget + (52 * queries + 959) / budget := by
  have bound := hashCollisions_le_finder ajtai key (RefreshFork.folds ajtai)
    (fun attempt => (attempt.link.fold.fresh, (RefreshFork.proofOf attempt.link.fold.proof).split))
    (fun attempt => RefreshSchedule.messages (RefreshFork.proofOf attempt.link.fold.proof).nifs)
    adversary bounded positive
  have count : ((RefreshSchedule.schedule logicalWidth publicFits).rhoCount : ℝ) = 17 := by
    show ((17 : Nat) : ℝ) = 17
    norm_num
  rw [count, RefreshSolver.work_eq] at bound
  calc
    _ ≤ _ := bound
    _ = _ := by
      unfold refreshFinderChance
      push_cast
      ring

end Refresh

end NightstreamFPrime.Lifecycle.OptionB.RandomOracle.HashSolver
