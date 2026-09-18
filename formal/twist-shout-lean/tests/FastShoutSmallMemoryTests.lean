import TwistShout.FastShoutSmallMemory
import TwistShout.MLE

open scoped BigOperators
open TwistShout
open TwistShout

namespace tests.fastshoutsmall

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
    aggregatedReadCheck (K := Rat) sampleVal sampleAddress rCycle =
      readCheckExpression (K := Rat) sampleRa sampleVal rCycle := by
  exact sampleValid.aggregatedReadCheck_eq_readCheckExpression sampleVal rCycle

example (rCycle : Point (K := Rat) 1) :
    mle (K := Rat) sampleRv rCycle =
      aggregatedReadCheck (K := Rat) sampleVal sampleAddress rCycle := by
  exact sampleRelation.aggregatedReadCheckIdentity rCycle

example :
    aggregatedReadCheck (K := Rat) sampleVal sampleAddress (bitVec (K := Rat) cycleFalse) = 7 := by
  rw [aggregatedReadCheck_eq_mle_readOracleTable
    (K := Rat) (val := sampleVal) (addr := sampleAddress) (rCycle := bitVec (K := Rat) cycleFalse)]
  calc
    mle (K := Rat) (readOracleTable (K := Rat) sampleVal sampleAddress) (bitVec (K := Rat) cycleFalse)
      = readOracleTable (K := Rat) sampleVal sampleAddress cycleFalse := by
          exact mle_at_bitVec (K := Rat) (readOracleTable (K := Rat) sampleVal sampleAddress) cycleFalse
    _ = 7 := by
          native_decide

example :
    readOracleTable (K := Rat) (batchedTable (K := Rat) 3 sampleVal) sampleAddress cycleFalse = 10 := by
  rw [readOracleTable_batchedTable (K := Rat) (z := 3) (val := sampleVal) (addr := sampleAddress)]
  native_decide

example :
    aggregatedReadCheck (K := Rat) (batchedTable (K := Rat) 3 sampleVal) sampleAddress
      (bitVec (K := Rat) cycleTrue) = 17 := by
  rw [aggregatedReadCheck_eq_mle_readOracleTable
    (K := Rat)
    (val := batchedTable (K := Rat) 3 sampleVal)
    (addr := sampleAddress)
    (rCycle := bitVec (K := Rat) cycleTrue)]
  calc
    mle (K := Rat) (readOracleTable (K := Rat) (batchedTable (K := Rat) 3 sampleVal) sampleAddress)
        (bitVec (K := Rat) cycleTrue)
      = readOracleTable (K := Rat) (batchedTable (K := Rat) 3 sampleVal) sampleAddress cycleTrue := by
          exact mle_at_bitVec (K := Rat)
            (readOracleTable (K := Rat) (batchedTable (K := Rat) 3 sampleVal) sampleAddress)
            cycleTrue
    _ = 17 := by
          native_decide

example :
    readCheckExpression (K := Rat) sampleRa (batchedTable (K := Rat) 3 sampleVal)
      (bitVec (K := Rat) cycleFalse) = 10 := by
  have hRead :
      readCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleFalse) = 7 := by
    rw [← sampleValid.aggregatedReadCheck_eq_readCheckExpression
      (val := sampleVal) (rCycle := bitVec (K := Rat) cycleFalse)]
    rw [aggregatedReadCheck_eq_mle_readOracleTable
      (K := Rat) (val := sampleVal) (addr := sampleAddress) (rCycle := bitVec (K := Rat) cycleFalse)]
    calc
      mle (K := Rat) (readOracleTable (K := Rat) sampleVal sampleAddress) (bitVec (K := Rat) cycleFalse)
        = readOracleTable (K := Rat) sampleVal sampleAddress cycleFalse := by
            exact mle_at_bitVec (K := Rat) (readOracleTable (K := Rat) sampleVal sampleAddress) cycleFalse
      _ = 7 := by
            native_decide
  have hAddr :
      addressValueExpression (K := Rat) sampleRa (bitVec (K := Rat) cycleFalse) = 1 := by
    native_decide
  rw [sampleValid.readCheckExpression_batchedTable
    (z := 3) (val := sampleVal) (rCycle := bitVec (K := Rat) cycleFalse)]
  rw [hRead, hAddr]
  native_decide

example :
    readCheckExpression (K := Rat) sampleRa (batchedTable (K := Rat) 3 sampleVal)
      (bitVec (K := Rat) cycleTrue) = 17 := by
  have hRead :
      readCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleTrue) = 11 := by
    rw [← sampleValid.aggregatedReadCheck_eq_readCheckExpression
      (val := sampleVal) (rCycle := bitVec (K := Rat) cycleTrue)]
    rw [aggregatedReadCheck_eq_mle_readOracleTable
      (K := Rat) (val := sampleVal) (addr := sampleAddress) (rCycle := bitVec (K := Rat) cycleTrue)]
    calc
      mle (K := Rat) (readOracleTable (K := Rat) sampleVal sampleAddress) (bitVec (K := Rat) cycleTrue)
        = readOracleTable (K := Rat) sampleVal sampleAddress cycleTrue := by
            exact mle_at_bitVec (K := Rat) (readOracleTable (K := Rat) sampleVal sampleAddress) cycleTrue
      _ = 11 := by
            native_decide
  have hAddr :
      addressValueExpression (K := Rat) sampleRa (bitVec (K := Rat) cycleTrue) = 2 := by
    native_decide
  rw [sampleValid.readCheckExpression_batchedTable
    (z := 3) (val := sampleVal) (rCycle := bitVec (K := Rat) cycleTrue)]
  rw [hRead, hAddr]
  native_decide

example :
    combinedShoutLeadingCost 2 4 =
      coreShoutLeadingCost 2 4 + booleanityOptimizedLeadingCost 2 4 := by
  exact combinedShoutLeadingCost_eq_sum 2 4

#guard addressSpaceSize 2 3 = 64
#guard digitSpaceSize 3 = 8
#guard cycleSpaceSize 4 = 16
#guard coreShoutD1Cost 3 4 = 40
#guard coreShoutGeneralCost 2 3 4 = 432
#guard coreShoutGeneralImprovedCost 2 3 4 = 400
#guard booleanityFirstRoundsCost 2 3 4 = 144
#guard booleanityUnoptimizedLeadingCost 2 4 = 192
#guard booleanityOptimizedLeadingCost 2 4 = 96
#guard batchedRafAdditionalCost 2 3 = 64
#guard combinedShoutLeadingCost 2 4 = 192

end tests.fastshoutsmall
