import NightstreamFPrime.Gadgets.Sampling.WideReduction.HintValues

/-! The integer meaning of every temporary helper slot. This is a reference
environment for the execution proof, not a second witness implementation. -/

namespace NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperValues

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

def quotient (draw : Draw) (round position : Nat) : Nat :=
  HintValues.limb ((drawIndex draw).val / 5 ^ (round + 1)) (4 - position)

def remainder (draw : Draw) (round position : Nat) : Nat :=
  (drawIndex draw).val / 5 ^ round / WitnessArithmetic.radix ^ (4 - position) % 5

def value (draw : Draw) (index : Nat) : Nat :=
  if index < 256 then
    (draw ⟨index / 64 % 4, Nat.mod_lt _ (by decide)⟩).val / 2 ^ (index % 64) % 2
  else if index < 864 then
    let relative := index - 256
    let accumulator := LimbArithmetic.accumulator draw (relative / 38)
    if relative % 38 = 0 then accumulator else accumulator / 2 ^ (relative % 38 - 1) % 2
  else
    let relative := index - 864
    if relative % 2 = 0 then quotient draw (relative / 10) (relative % 10 / 2)
    else remainder draw (relative / 10) (relative % 10 / 2)

def environment (base : Env) (start : Nat) (draw : Draw) : Env :=
  fun index => if start ≤ index ∧ index < start + HintProgram.helperCount then
    fieldOfNat (value draw (index - start)) else base index

def Present (env : Env) (start : Nat) (draw : Draw) : Prop :=
  ∀ index, start ≤ index → index < start + HintProgram.helperCount →
    env index = environment env start draw index

theorem present (base : Env) (start : Nat) (draw : Draw) :
    Present (environment base start draw) start draw := by
  intro index lower upper
  simp only [environment, if_pos (And.intro lower upper)]

theorem present_of_agree (left right : Env) (start : Nat) (draw : Draw)
    (present : Present left start draw)
    (same : ∀ index, start ≤ index → index < start + HintProgram.helperCount →
      right index = left index) : Present right start draw := by
  intro index lower upper
  rw [same index lower upper, present index lower upper]
  simp only [environment, if_pos (And.intro lower upper)]

theorem below (base : Env) (start : Nat) (draw : Draw) (index : Nat) (bound : index < start) :
    environment base start draw index = base index := by
  unfold environment
  rw [if_neg (by omega)]

theorem outside (base : Env) (start : Nat) (draw : Draw) :
    AgreesOutside base (environment base start draw) start HintProgram.helperCount := by
  intro index outside
  unfold environment
  rw [if_neg (by omega)]

private theorem assigned (base : Env) (start : Nat) (draw : Draw) (position : Nat)
    (bound : position < HintProgram.helperCount) :
    environment base start draw (start + position) = fieldOfNat (value draw position) := by
  unfold environment
  rw [if_pos (by omega), Nat.add_sub_cancel_left]

theorem source_bit (base : Env) (start : Nat) (draw : Draw) (lane : Fin 4) (bit : Nat)
    (bound : bit < 64) :
    environment base start draw (start + lane.val * 64 + bit) =
      fieldOfNat ((draw lane).val / 2 ^ bit % 2) := by
  rw [Nat.add_assoc, assigned base start draw _ (by have := lane.isLt; change _ < 1404; omega)]
  unfold value
  rw [if_pos (by have := lane.isLt; omega)]
  have divide : (lane.val * 64 + bit) / 64 = lane.val := by omega
  have modulo : (lane.val * 64 + bit) % 64 = bit := by omega
  simp only [divide, modulo, Nat.mod_eq_of_lt lane.isLt, Fin.eta]

theorem accumulator (base : Env) (start : Nat) (draw : Draw) (position : Nat)
    (bound : position < 16) :
    environment base start draw (HintProgram.limbStart start position) =
      fieldOfNat (LimbArithmetic.accumulator draw position) := by
  unfold HintProgram.limbStart HintProgram.sourceBitCount HintProgram.limbStride
    HintProgram.accumulatorBits
  rw [Nat.add_assoc, assigned base start draw _ (by change _ < 1404; omega)]
  unfold value
  rw [if_neg (by omega), if_pos (by omega)]
  simp

theorem accumulator_bit (base : Env) (start : Nat) (draw : Draw) (position bit : Nat)
    (positionBound : position < 16) (bitBound : bit < 37) :
    environment base start draw (HintProgram.limbStart start position + 1 + bit) =
      fieldOfNat (LimbArithmetic.accumulator draw position / 2 ^ bit % 2) := by
  have coordinate : HintProgram.limbStart start position + 1 + bit =
      start + (256 + position * 38 + 1 + bit) := by
    unfold HintProgram.limbStart HintProgram.sourceBitCount HintProgram.limbStride
      HintProgram.accumulatorBits
    omega
  rw [coordinate, assigned base start draw _ (by change _ < 1404; omega)]
  unfold value
  rw [if_neg (by omega), if_pos (by omega)]
  have subtract : 256 + position * 38 + 1 + bit - 256 = position * 38 + 1 + bit := by omega
  have divide : (position * 38 + 1 + bit) / 38 = position := by omega
  have modulo : (position * 38 + 1 + bit) % 38 = 1 + bit := by omega
  rw [subtract]
  dsimp only
  rw [divide, modulo, if_neg (by omega)]
  rw [show 1 + bit - 1 = bit by omega]

theorem division (base : Env) (start : Nat) (draw : Draw) (round position : Nat)
    (roundBound : round < 54) (positionBound : position < 5) :
    environment base start draw (HintProgram.divisionColumn start round position) =
        fieldOfNat (quotient draw round position) ∧
      environment base start draw (HintProgram.divisionColumn start round position + 1) =
        fieldOfNat (remainder draw round position) := by
  have coordinate : HintProgram.divisionColumn start round position =
      start + (864 + round * 10 + 2 * position) := by
    unfold HintProgram.divisionColumn HintProgram.divisionStart HintProgram.sourceBitCount
      HintProgram.limbCount HintProgram.limbStride HintProgram.accumulatorBits
      HintProgram.divisionStride HintProgram.divisionLimbs
    omega
  rw [coordinate]
  constructor
  · rw [assigned base start draw _ (by change _ < 1404; omega)]
    unfold value
    rw [if_neg (by omega), if_neg (by omega)]
    have hsub : 864 + round * 10 + 2 * position - 864 = round * 10 + 2 * position := by omega
    rw [hsub]
    dsimp only
    rw [if_pos (by omega)]
    have hround : (round * 10 + 2 * position) / 10 = round := by omega
    have hposition : (round * 10 + 2 * position) % 10 / 2 = position := by omega
    rw [hround, hposition]
  · rw [show start + (864 + round * 10 + 2 * position) + 1 =
        start + (864 + round * 10 + 2 * position + 1) by omega,
        assigned base start draw _ (by change _ < 1404; omega)]
    unfold value
    rw [if_neg (by omega), if_neg (by omega)]
    have hsub : 864 + round * 10 + 2 * position + 1 - 864 =
        round * 10 + 2 * position + 1 := by omega
    rw [hsub]
    dsimp only
    rw [if_neg (by omega)]
    have hround : (round * 10 + 2 * position + 1) / 10 = round := by omega
    have hposition : (round * 10 + 2 * position + 1) % 10 / 2 = position := by omega
    rw [hround, hposition]

end NightstreamFPrime.Gadgets.Sampling.WideReduction.HelperValues
