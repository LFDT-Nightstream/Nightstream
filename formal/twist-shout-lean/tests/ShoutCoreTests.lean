import TwistShout.ShoutCore
import TwistShout.MLE

open scoped BigOperators
open TwistShout
open TwistShout

namespace tests.shoutcore

def zeroDigit : DigitCube 1 := fun _ => false

def oneDigit : DigitCube 1 := fun _ => true

def addr01 : Address 2 1 :=
  Fin.cases oneDigit (fun _ => zeroDigit)

def addr10 : Address 2 1 :=
  Fin.cases zeroDigit (fun _ => oneDigit)

def cycleFalse : CycleCube 1 := fun _ => false

def cycleTrue : CycleCube 1 := fun _ => true

def sampleAddress : CycleCube 1 → Address 2 1 :=
  fun j => if j 0 then addr10 else addr01

def sampleRa : AddressColumns (K := Rat) 2 1 1 :=
  fun i k j => TwistShout.cubeOneHot (K := Rat) (sampleAddress j i) k

def sampleVal : PublicTable (K := Rat) 2 1 :=
  fun k => if k = addr01 then 7 else if k = addr10 then 11 else 0

def sampleRv : CycleCube 1 → Rat :=
  readOracleTable (K := Rat) sampleVal sampleAddress

theorem sampleValid :
    ValidAddressColumns (K := Rat) sampleRa sampleAddress := by
  intro j i k
  rfl

theorem sampleRelation :
    ReadOnlyMemoryRelation (K := Rat) sampleVal sampleAddress sampleRv := by
  intro j
  rfl

example (rCycle : Point (K := Rat) 1) :
    mle (K := Rat) sampleRv rCycle =
      readCheckExpression (K := Rat) sampleRa sampleVal rCycle := by
  exact sampleRelation.readCheckIdentity sampleValid rCycle

example :
    tableMLE (K := Rat) sampleVal (bitAddress (K := Rat) addr01) = 7 := by
  rw [tableMLE_at_bitAddress (K := Rat) (val := sampleVal) (a := addr01)]
  native_decide

example :
    tableMLE (K := Rat) sampleVal (bitAddress (K := Rat) addr10) = 11 := by
  rw [tableMLE_at_bitAddress (K := Rat) (val := sampleVal) (a := addr10)]
  native_decide

example :
    readCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleFalse) = 7 := by
  rw [← sampleRelation.readCheckAtBitCycle sampleValid (j := cycleFalse)]
  native_decide

example :
    readCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleTrue) = 11 := by
  rw [← sampleRelation.readCheckAtBitCycle sampleValid (j := cycleTrue)]
  native_decide

example :
    readCheckFinalRoundTarget (K := Rat) (bitVec (K := Rat) cycleFalse) sampleRa sampleVal
      (bitAddress (K := Rat) (sampleAddress cycleFalse)) (bitVec (K := Rat) cycleFalse) = 7 := by
  rw [sampleValid.readCheckFinalRoundTarget_atBooleanPoint
    (val := sampleVal) (queryCycle := bitVec (K := Rat) cycleFalse) (j := cycleFalse)]
  native_decide

example :
    readCheckFinalRoundTarget (K := Rat) (bitVec (K := Rat) cycleFalse) sampleRa sampleVal
      (bitAddress (K := Rat) (sampleAddress cycleTrue)) (bitVec (K := Rat) cycleTrue) = 0 := by
  rw [sampleValid.readCheckFinalRoundTarget_atBooleanPoint
    (val := sampleVal) (queryCycle := bitVec (K := Rat) cycleFalse) (j := cycleTrue)]
  native_decide

#guard readOracleTable (K := Rat) sampleVal sampleAddress cycleFalse = 7
#guard readOracleTable (K := Rat) sampleVal sampleAddress cycleTrue = 11

end tests.shoutcore
