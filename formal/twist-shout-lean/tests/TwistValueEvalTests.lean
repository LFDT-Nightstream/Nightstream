import TwistShout.TwistValueEval

open TwistShout

namespace tests.twistvalueeval

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

example :
    virtualValue (K := Rat) sampleInc addr0 (bitVec (K := Rat) cycleFalse) = 0 := by
  rw [virtualValue_at_bitCycle (K := Rat) (inc := sampleInc) (k := addr0) (j := cycleFalse)]
  native_decide

example :
    virtualValue (K := Rat) sampleInc addr1 (bitVec (K := Rat) cycleTrue) = 10 := by
  rw [virtualValue_at_bitCycle (K := Rat) (inc := sampleInc) (k := addr1) (j := cycleTrue)]
  native_decide

example :
    valEvaluationExpression (K := Rat) sampleInc (bitAddress (K := Rat) addr1)
      (bitVec (K := Rat) cycleTrue) = 10 := by
  rw [valEvaluationExpression_at_bitPoint
    (K := Rat) (inc := sampleInc) (a := addr1) (j := cycleTrue)]
  native_decide

example :
    timeTableMLE (K := Rat) (reconstructedTimeTable (K := Rat) sampleInc)
      (bitAddress (K := Rat) addr1) (bitVec (K := Rat) cycleTrue) = 10 := by
  rw [timeTableMLE_reconstructedTimeTable_at_bitAddress
    (K := Rat) (inc := sampleInc) (a := addr1) (rCycle := bitVec (K := Rat) cycleTrue)]
  rw [virtualValue_at_bitCycle (K := Rat) (inc := sampleInc) (k := addr1) (j := cycleTrue)]
  native_decide

example :
    valEvaluationFinalRoundTarget (K := Rat) sampleInc
      (bitAddress (K := Rat) addr1)
      (bitVec (K := Rat) cycleTrue)
      (bitVec (K := Rat) cycleFalse) = 10 := by
  rw [valEvaluationFinalRoundTarget_at_bitPoint
    (K := Rat)
    (inc := sampleInc)
    (a := addr1)
    (queryCycle := bitVec (K := Rat) cycleTrue)
    (j := cycleFalse)]
  native_decide

#guard reconstructedTimeTable (K := Rat) sampleInc addr0 cycleFalse = 0
#guard reconstructedTimeTable (K := Rat) sampleInc addr1 cycleFalse = 0
#guard reconstructedTimeTable (K := Rat) sampleInc addr1 cycleTrue = 10

end tests.twistvalueeval
