import TwistShout.TwistCore

open TwistShout

namespace tests.twistcore

def zeroDigit : DigitCube 1 := fun _ => false

def oneDigit : DigitCube 1 := fun _ => true

def addr0 : Address 1 1 := fun _ => zeroDigit

def addr1 : Address 1 1 := fun _ => oneDigit

def cycleFalse : CycleCube 1 := fun _ => false

def cycleTrue : CycleCube 1 := fun _ => true

def sampleReadAddress : CycleCube 1 → Address 1 1 :=
  fun j => if j 0 then addr1 else addr0

def sampleWriteAddress : CycleCube 1 → Address 1 1 :=
  fun j => if j 0 then addr0 else addr1

def sampleRa : AddressColumns (K := Rat) 1 1 1 :=
  fun i k j => TwistShout.cubeOneHot (K := Rat) (sampleReadAddress j i) k

def sampleWa : AddressColumns (K := Rat) 1 1 1 :=
  fun i k j => TwistShout.cubeOneHot (K := Rat) (sampleWriteAddress j i) k

def sampleVal : TimeTable (K := Rat) 1 1 1 :=
  fun k j =>
    if j 0 then
      if k = addr0 then 13 else 17
    else
      if k = addr0 then 5 else 9

def sampleRv : CycleCube 1 → Rat :=
  readWriteOracleTable (K := Rat) sampleVal sampleReadAddress

def sampleWv : CycleCube 1 → Rat :=
  fun j => if j 0 then 23 else 19

def sampleInc : TimeTable (K := Rat) 1 1 1 :=
  fun k j =>
    addressSelector (K := Rat) sampleWa k j * (sampleWv j - sampleVal k j)

theorem sampleValidRa :
    ValidAddressColumns (K := Rat) sampleRa sampleReadAddress := by
  intro j i k
  rfl

theorem sampleValidWa :
    ValidAddressColumns (K := Rat) sampleWa sampleWriteAddress := by
  intro j i k
  rfl

theorem sampleReadRelation :
    ReadWriteMemoryRelation (K := Rat) sampleVal sampleReadAddress sampleRv := by
  intro j
  rfl

theorem sampleIncRelation :
    IncrementRelation (K := Rat) sampleVal sampleWa sampleWv sampleInc := by
  intro k j
  rfl

example :
    timeTableMLE (K := Rat) sampleVal (bitAddress (K := Rat) addr0) (bitVec (K := Rat) cycleFalse) = 5 := by
  rw [timeTableMLE_at_bitPoint (K := Rat) (val := sampleVal) (a := addr0) (j := cycleFalse)]
  native_decide

example :
    timeTableMLE (K := Rat) sampleVal (bitAddress (K := Rat) addr1) (bitVec (K := Rat) cycleTrue) = 17 := by
  rw [timeTableMLE_at_bitPoint (K := Rat) (val := sampleVal) (a := addr1) (j := cycleTrue)]
  native_decide

example (rCycle : Point (K := Rat) 1) :
    mle (K := Rat) sampleRv rCycle =
      rwReadCheckExpression (K := Rat) sampleRa sampleVal rCycle := by
  exact sampleReadRelation.readCheckIdentity sampleValidRa rCycle

example :
    rwReadCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleFalse) = 5 := by
  rw [← sampleReadRelation.readCheckAtBitCycle sampleValidRa (j := cycleFalse)]
  native_decide

example :
    rwReadCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleTrue) = 17 := by
  rw [← sampleReadRelation.readCheckAtBitCycle sampleValidRa (j := cycleTrue)]
  native_decide

example (queryAddress : Fin 1 → Point (K := Rat) 1) (queryCycle : Point (K := Rat) 1) :
    timeTableMLE (K := Rat) sampleInc queryAddress queryCycle =
      writeCheckExpression (K := Rat) queryAddress queryCycle sampleWa sampleWv sampleVal := by
  exact sampleIncRelation.writeCheckIdentity queryAddress queryCycle

example :
    writeCheckExpression (K := Rat) (bitAddress (K := Rat) addr1) (bitVec (K := Rat) cycleFalse)
      sampleWa sampleWv sampleVal = 10 := by
  rw [← sampleIncRelation.writeCheckAtBitPoint (a := addr1) (j := cycleFalse)]
  native_decide

example :
    sampleInc (sampleWriteAddress cycleFalse) cycleFalse = 10 := by
  rw [sampleValidWa.incrementAtWrittenAddress sampleIncRelation (j := cycleFalse)]
  native_decide

example :
    sampleInc addr0 cycleFalse = 0 := by
  have hneq : addr0 ≠ sampleWriteAddress cycleFalse := by
    native_decide
  rw [sampleValidWa.incrementAtOtherAddress sampleIncRelation
    (k := addr0) (j := cycleFalse) hneq]

example :
    twistReadCheckFinalRoundTarget (K := Rat) (bitVec (K := Rat) cycleFalse) sampleRa sampleVal
      (bitAddress (K := Rat) (sampleReadAddress cycleFalse)) (bitVec (K := Rat) cycleFalse) = 5 := by
  rw [sampleValidRa.twistReadCheckFinalRoundTarget_atBooleanPoint
    (val := sampleVal) (queryCycle := bitVec (K := Rat) cycleFalse) (j := cycleFalse)]
  native_decide

example :
    writeCheckFinalRoundTarget (K := Rat) (bitAddress (K := Rat) (sampleWriteAddress cycleFalse))
      (bitVec (K := Rat) cycleFalse) sampleWa sampleWv sampleVal
      (bitAddress (K := Rat) (sampleWriteAddress cycleFalse)) (bitVec (K := Rat) cycleFalse) = 10 := by
  rw [sampleValidWa.writeCheckFinalRoundTarget_atBooleanPoint
    (wv := sampleWv)
    (val := sampleVal)
    (queryAddress := bitAddress (K := Rat) (sampleWriteAddress cycleFalse))
    (queryCycle := bitVec (K := Rat) cycleFalse)
    (j := cycleFalse)]
  native_decide

#guard sampleRv cycleFalse = 5
#guard sampleRv cycleTrue = 17
#guard sampleWv cycleFalse = 19
#guard sampleWv cycleTrue = 23

end tests.twistcore
