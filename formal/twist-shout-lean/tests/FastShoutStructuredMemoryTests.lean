import TwistShout.FastShoutStructuredMemory
import TwistShout.MLE

open scoped BigOperators
open TwistShout

namespace tests.fastshoutstructured

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

def sampleOracle : StructuredTableOracle (K := Rat) sampleVal :=
  StructuredTableOracle.ofTableMLE (K := Rat) sampleVal

theorem sampleValid :
    ValidAddressColumns (K := Rat) sampleRa sampleAddress := by
  intro j i k
  rfl

example (rAddress : Fin 2 → Point (K := Rat) 1) :
    sampleOracle.eval rAddress = TwistShout.tableMLE (K := Rat) sampleVal rAddress := by
  exact sampleOracle.eval_eq rAddress

example :
    sampleOracle.eval (TwistShout.bitAddress (K := Rat) addr01) = 7 := by
  rw [sampleOracle.eval_at_bitAddress]
  native_decide

example :
    sampleOracle.eval (TwistShout.bitAddress (K := Rat) addr10) = 11 := by
  rw [sampleOracle.eval_at_bitAddress]
  native_decide

example (queryCycle : Point (K := Rat) 1) (rAddress : Fin 2 → Point (K := Rat) 1)
    (boundCycle : Point (K := Rat) 1) :
    structuredReadCheckFinalRoundTarget (K := Rat) queryCycle sampleRa sampleOracle rAddress boundCycle =
      TwistShout.readCheckFinalRoundTarget (K := Rat) queryCycle sampleRa sampleVal rAddress boundCycle := by
  exact sampleOracle.structuredReadCheckFinalRoundTarget_eq queryCycle sampleRa rAddress boundCycle

example :
    structuredReadCheckFinalRoundTarget (K := Rat) (bitVec (K := Rat) cycleFalse) sampleRa sampleOracle
      (TwistShout.bitAddress (K := Rat) (sampleAddress cycleFalse)) (bitVec (K := Rat) cycleFalse) = 7 := by
  rw [sampleOracle.readCheckFinalRoundTarget_atBooleanPoint
    sampleValid (queryCycle := bitVec (K := Rat) cycleFalse) (j := cycleFalse)]
  native_decide

example :
    (sampleOracle.batched 3).eval (TwistShout.bitAddress (K := Rat) addr10) = 17 := by
  rw [sampleOracle.batched_eval_at_bitAddress (z := 3)]
  native_decide

example :
    structuredReadCheckFinalRoundTarget (K := Rat) (bitVec (K := Rat) cycleTrue) sampleRa
      (sampleOracle.batched 3) (TwistShout.bitAddress (K := Rat) (sampleAddress cycleTrue))
      (bitVec (K := Rat) cycleTrue) = 17 := by
  rw [sampleOracle.batched_readCheckFinalRoundTarget_atBooleanPoint
    sampleValid (z := 3) (queryCycle := bitVec (K := Rat) cycleTrue) (j := cycleTrue)]
  native_decide

example :
    structuredShoutLeadingCost 6 3 2 4 =
      structuredReadValueEvalLeadingCost 4 +
        structuredReadCheckLeadingCost 6 2 4 +
        structuredBooleanityLeadingCost 6 2 4 +
        structuredRafLeadingCost 6 4 +
        structuredHammingLeadingCost 3 4 := by
  exact structuredShoutLeadingCost_eq_sum 6 3 2 4

example :
    structuredBooleanityLeadingCost 6 2 4 = (4 * 3 * 2 + 3 * 2) * TwistShout.cycleSpaceSize 4 := by
  exact structuredBooleanityLeadingCost_eq_chunked 3 2 4

example :
    structuredReadCheckLeadingCost 6 2 4 = (2 * 3 * 2 + 2 * 2) * TwistShout.cycleSpaceSize 4 := by
  exact structuredReadCheckLeadingCost_eq_chunked 3 2 4

example :
    structuredShoutLeadingCost 6 3 2 4 =
      (7 * 3 * 2 + 2 * 2 + 3 * 2 + 3 + 2) * TwistShout.cycleSpaceSize 4 := by
  exact structuredShoutLeadingCost_eq_chunked 3 2 4

#guard structuredReadValueEvalLeadingCost 4 = 32
#guard structuredReadCheckLeadingCost 6 2 4 = 256
#guard structuredBooleanityLeadingCost 6 2 4 = 480
#guard structuredRafLeadingCost 6 4 = 96
#guard structuredHammingLeadingCost 3 4 = 48
#guard structuredShoutLeadingCost 6 3 2 4 = 912

end tests.fastshoutstructured
