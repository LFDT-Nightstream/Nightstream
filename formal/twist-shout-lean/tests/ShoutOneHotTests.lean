import TwistShout.ShoutOneHot
import TwistShout.MLE

open scoped BigOperators
open TwistShout
open TwistShout

namespace tests

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

theorem sampleValid :
    ValidAddressColumns (K := Rat) sampleRa sampleAddress := by
  intro j i k
  rfl

example (i : Fin 2) (j : CycleCube 1) :
    hammingWeightAtCycle (K := Rat) sampleRa i j = 1 := by
  exact sampleValid.hammingWeightAtCycle i j

example (i : Fin 2) (rAddress : Point (K := Rat) 1) (rCycle : Point (K := Rat) 1) :
    booleanityExpression (K := Rat) sampleRa i rAddress rCycle = 0 := by
  exact sampleValid.booleanityExpression i rAddress rCycle

example (i : Fin 2) (rCycle : Point (K := Rat) 1) :
    hammingWeightExpression (K := Rat) sampleRa i rCycle = 1 := by
  exact sampleValid.hammingWeightExpression i rCycle

example (rCycle : Point (K := Rat) 1) :
    mle (K := Rat) (addressOracleTable (K := Rat) sampleAddress) rCycle =
      addressValueExpression (K := Rat) sampleRa rCycle := by
  exact sampleValid.addressValueExpression rCycle

example :
    addressValue (K := Rat) addr01 = 1 := by
  native_decide

example :
    addressValue (K := Rat) addr10 = 2 := by
  native_decide

example :
    addressValueExpression (K := Rat) sampleRa (bitVec (K := Rat) cycleFalse) = 1 := by
  rw [← sampleValid.addressValueExpression (rCycle := bitVec (K := Rat) cycleFalse)]
  calc
    mle (K := Rat) (addressOracleTable (K := Rat) sampleAddress) (bitVec (K := Rat) cycleFalse)
      = addressOracleTable (K := Rat) sampleAddress cycleFalse := by
          exact mle_at_bitVec (K := Rat) (addressOracleTable (K := Rat) sampleAddress) cycleFalse
    _ = 1 := by
          native_decide

example :
    addressValueExpression (K := Rat) sampleRa (bitVec (K := Rat) cycleTrue) = 2 := by
  rw [← sampleValid.addressValueExpression (rCycle := bitVec (K := Rat) cycleTrue)]
  calc
    mle (K := Rat) (addressOracleTable (K := Rat) sampleAddress) (bitVec (K := Rat) cycleTrue)
      = addressOracleTable (K := Rat) sampleAddress cycleTrue := by
          exact mle_at_bitVec (K := Rat) (addressOracleTable (K := Rat) sampleAddress) cycleTrue
    _ = 2 := by
          native_decide

end tests
