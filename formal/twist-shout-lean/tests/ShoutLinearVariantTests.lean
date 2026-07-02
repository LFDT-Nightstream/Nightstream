import TwistShout.ShoutLinearVariant
import TwistShout.MLE

open TwistShout
open TwistShout

namespace tests.shoutlinear

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
  fun i k j => cubeOneHot (K := Rat) (sampleAddress j i) k

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
    linearReadCheckExpression (K := Rat) sampleRa sampleVal rCycle =
      readCheckExpression (K := Rat) sampleRa sampleVal rCycle := by
  exact linearReadCheckExpression_eq_readCheckExpression (K := Rat) sampleRa sampleVal rCycle

example (rCycle : Point (K := Rat) 1) :
    mle (K := Rat) sampleRv rCycle =
      linearReadCheckExpression (K := Rat) sampleRa sampleVal rCycle := by
  exact sampleRelation.linearReadCheckIdentity sampleValid rCycle

example :
    linearReadCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleFalse) = 7 := by
  rw [← sampleRelation.linearReadCheckAtBitCycle sampleValid (j := cycleFalse)]
  native_decide

example :
    linearReadCheckExpression (K := Rat) sampleRa sampleVal (bitVec (K := Rat) cycleTrue) = 11 := by
  rw [← sampleRelation.linearReadCheckAtBitCycle sampleValid (j := cycleTrue)]
  native_decide

example :
    diagonalEqWeight (K := Rat) (bitVec (K := Rat) cycleFalse)
      (diagonalCycleTuple (d := 2) cycleFalse) = 1 := by
  rw [diagonalEqWeight_at_diagonalCycleTuple
    (K := Rat) (d := 2) (rCycle := bitVec (K := Rat) cycleFalse) (j := cycleFalse)]
  native_decide

example :
    diagonalEqPointWeight (K := Rat) (bitVec (K := Rat) cycleTrue)
      (fun _ : Fin 2 => bitVec (K := Rat) cycleTrue) = 1 := by
  rw [diagonalEqPointWeight_at_diagonalBitVec
    (K := Rat) (d := 2) (queryCycle := bitVec (K := Rat) cycleTrue) (j := cycleTrue)]
  native_decide

example :
    linearReadCheckFinalRoundTarget (K := Rat) (bitVec (K := Rat) cycleFalse) sampleRa sampleVal
      (bitAddress (K := Rat) (sampleAddress cycleFalse))
      (fun _ : Fin 2 => bitVec (K := Rat) cycleFalse) = 7 := by
  rw [sampleValid.linearReadCheckFinalRoundTarget_atDiagonalBooleanPoint
    (val := sampleVal) (queryCycle := bitVec (K := Rat) cycleFalse) (j := cycleFalse)]
  native_decide

example :
    linearVariantRoundCount 2 3 4 = 2 * (3 + 4) := by
  exact linearVariantRoundCount_eq_mul 2 3 4

example :
    linearVariantRoundCount 2 3 4 =
      standardShoutRoundCount 2 3 4 + (2 - 1) * 4 := by
  exact linearVariantRoundCount_eq_standardPlus (d := 2) (m := 3) (t := 4)

example :
    linearVariantBaseCost 9 4 =
      linearVariantEqArrayCost 4 +
        linearVariantProductCost 9 4 +
        linearVariantPrefixRoundCost 9 4 +
        linearVariantLastRoundCost 4 := by
  exact linearVariantBaseCost_eq_sum 9 4

example :
    linearVariantGruenCost 9 4 =
      linearVariantBaseCost 9 4 - linearVariantGruenSaving 4 := by
  exact linearVariantGruenCost_eq_base_minus_saving 9 4

example :
    linearVariantGruenCost 9 4 ≤ standardShoutFinalRoundsQuadraticCost 9 4 := by
  exact linearVariantGruenCost_le_standardQuadratic (d := 9) (t := 4) (by omega)

#guard linearVariantRoundCount 2 3 4 = 14
#guard standardShoutRoundCount 2 3 4 = 10
#guard linearVariantEqArrayCost 4 = 8

end tests.shoutlinear
