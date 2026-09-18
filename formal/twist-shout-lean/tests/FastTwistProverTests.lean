import TwistShout.FastTwistProver

open TwistShout

namespace tests.fasttwistprover

def zeroDigit : DigitCube 1 := fun _ => false

def oneDigit : DigitCube 1 := fun _ => true

def addr0 : Address 1 1 := fun _ => zeroDigit

def addr1 : Address 1 1 := fun _ => oneDigit

def cycleFalse : CycleCube 1 := fun _ => false

def cycleTrue : CycleCube 1 := fun _ => true

def sampleWriteAddress : CycleCube 1 → Address 1 1 :=
  fun j => if j 0 then addr0 else addr1

def sampleWa : AddressColumns (K := Rat) 1 1 1 :=
  fun i k j => TwistShout.cubeOneHot (K := Rat) (sampleWriteAddress j i) k

def sampleVal : TimeTable (K := Rat) 1 1 1 :=
  fun k j =>
    if j 0 then
      if k = addr0 then 13 else 17
    else
      if k = addr0 then 5 else 9

def sampleWv : CycleCube 1 → Rat :=
  fun j => if j 0 then 23 else 19

def sampleInc : TimeTable (K := Rat) 1 1 1 :=
  fun k j =>
    addressSelector (K := Rat) sampleWa k j * (sampleWv j - sampleVal k j)

def sampleVirtualVal : TimeTable (K := Rat) 1 1 1 :=
  reconstructedTimeTable (K := Rat) sampleInc

theorem sampleValidWa :
    ValidAddressColumns (K := Rat) sampleWa sampleWriteAddress := by
  intro j i k
  rfl

example :
    writeCheckExpression (K := Rat) (bitAddress (K := Rat) addr1)
      (bitVec (K := Rat) cycleFalse) sampleWa sampleWv sampleVirtualVal =
      writeWvExpression (K := Rat) (bitAddress (K := Rat) addr1)
        (bitVec (K := Rat) cycleFalse) sampleWa sampleWv -
        writeValueExpression (K := Rat) (bitAddress (K := Rat) addr1)
          (bitVec (K := Rat) cycleFalse) sampleWa sampleVirtualVal := by
  exact writeCheckExpression_eq_writeWvExpression_sub_writeValueExpression
    (K := Rat) (bitAddress (K := Rat) addr1) (bitVec (K := Rat) cycleFalse)
    sampleWa sampleWv sampleVirtualVal

example :
    writeWvAtCycle (K := Rat) (bitAddress (K := Rat) addr1) sampleWa sampleWv cycleFalse = 19 := by
  rw [sampleValidWa.writeWvAtCycle
    (queryAddress := bitAddress (K := Rat) addr1) (wv := sampleWv) (j := cycleFalse)]
  native_decide

example :
    writeValueAtCycle (K := Rat) (bitAddress (K := Rat) addr1) sampleWa sampleVirtualVal cycleFalse = 0 := by
  rw [sampleValidWa.writeValueAtCycle
    (queryAddress := bitAddress (K := Rat) addr1) (val := sampleVirtualVal) (j := cycleFalse)]
  native_decide

example :
    writeWvExpression (K := Rat) (bitAddress (K := Rat) addr1)
      (bitVec (K := Rat) cycleFalse) sampleWa sampleWv = 19 := by
  rw [sampleValidWa.writeWvExpression
    (queryAddress := bitAddress (K := Rat) addr1)
    (queryCycle := bitVec (K := Rat) cycleFalse)
    (wv := sampleWv)]
  native_decide

example :
    writeValueExpression (K := Rat) (bitAddress (K := Rat) addr1)
      (bitVec (K := Rat) cycleFalse) sampleWa sampleVirtualVal = 0 := by
  rw [sampleValidWa.writeValueExpression
    (queryAddress := bitAddress (K := Rat) addr1)
    (queryCycle := bitVec (K := Rat) cycleFalse)
    (val := sampleVirtualVal)]
  native_decide

example :
    writeCheckExpression (K := Rat) (bitAddress (K := Rat) addr1)
      (bitVec (K := Rat) cycleFalse) sampleWa sampleWv sampleVirtualVal = 19 := by
  rw [sampleValidWa.writeCheckExpression_eq_mle_sub_mle
    (queryAddress := bitAddress (K := Rat) addr1)
    (queryCycle := bitVec (K := Rat) cycleFalse)
    (wv := sampleWv)
    (val := sampleVirtualVal)]
  native_decide

example :
    localD1TwistLeadingCost 5 4 =
      valEvaluationOptimizedLeadingCost 4 +
        localD1ReadCheckLeadingCost 5 4 +
        localD1WriteCheckLeadingCost 5 4 := by
  exact localD1TwistLeadingCost_eq_sum 5 4

example :
    localD1TwistTotalCost 5 4 =
      valEvaluationIncTableCost 1 5 +
        valEvaluationOptimizedLeadingCost 4 +
        localD1ReadCheckLeadingCost 5 4 +
        localD1WriteCheckLeadingCost 5 4 := by
  exact localD1TwistTotalCost_eq_sum 5 4

example :
    localWriteAccessCost 3 ≤ localWorstWriteAccessCost 5 := by
  exact localWriteAccessCost_le_worstCase (i := 3) (m := 5) (by omega)

example :
    localReadAccessCost 3 ≤ localWorstReadAccessCost 5 := by
  exact localReadAccessCost_le_worstCase (i := 3) (m := 5) (by omega)

example :
    valEvaluationOptimizedLeadingCost 4 +
        alternativeReadCheckLeadingCost 1 5 4 +
        alternativeWriteCheckLeadingCost 1 5 4 =
      alternativeTwistLeadingCost 1 5 4 + 3 * twistCycleSpaceSize 4 := by
  exact alternativeTwistLeadingComponentSum_eq_paperPlusGap 1 5 4

example :
    alternativeTwistComponentLeadingCost 1 5 4 =
      valEvaluationOptimizedLeadingCost 4 +
        alternativeReadCheckLeadingCost 1 5 4 +
        alternativeWriteCheckLeadingCost 1 5 4 := by
  exact alternativeTwistComponentLeadingCost_eq_sum 1 5 4

example :
    alternativeTwistComponentLeadingCost 1 5 4 =
      alternativeTwistLeadingCost 1 5 4 + 3 * twistCycleSpaceSize 4 := by
  exact alternativeTwistComponentLeadingCost_eq_paperPlusGap 1 5 4

example :
    alternativeTwistTotalCost 1 5 4 =
      valEvaluationIncTableCost 1 5 + alternativeTwistComponentLeadingCost 1 5 4 := by
  exact alternativeTwistTotalCost_eq_inc_plus_componentLeading 1 5 4

example :
    alternativeTwistLeadingCost 1 5 4 = (5 * 5 + 10) * twistCycleSpaceSize 4 := by
  exact alternativeTwistLeadingCost_d1 5 4

#guard valEvaluationOptimizedTotalCost 1 5 4 = 2 * 2 ^ 5 + 4 * 2 ^ 4
#guard localD1TwistLeadingCost 5 4 = (7 * 5 + 15) * 2 ^ 4
#guard localD1TwistTotalCost 5 4 = 2 * 2 ^ 5 + (7 * 5 + 15) * 2 ^ 4
#guard alternativeTwistLeadingCost 1 5 4 = (5 * 5 + 10) * 2 ^ 4
#guard alternativeTwistComponentLeadingCost 1 5 4 = (5 * 5 + 13) * 2 ^ 4
#guard alternativeTwistTotalCost 1 5 4 = 2 * 2 ^ 5 + (5 * 5 + 13) * 2 ^ 4

end tests.fasttwistprover
