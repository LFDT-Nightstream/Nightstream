import TwistShout.LessThanPoly

open TwistShout

namespace tests.lessthanpoly

def bits2 (b0 b1 : Bool) : Cube 2 :=
  Fin.cases b0 (fun _ => b1)

def b00 : Cube 2 := bits2 false false

def b01 : Cube 2 := bits2 true false

def b10 : Cube 2 := bits2 false true

def b11 : Cube 2 := bits2 true true

def sampleInc : Cube 2 → Rat
  | x =>
      if x = b00 then 3
      else if x = b01 then 5
      else if x = b10 then 7
      else 11

example (r : Point (K := Rat) 2) :
    mle (K := Rat) (prefixTable (K := Rat) sampleInc) r =
      prefixExpression (K := Rat) sampleInc r := by
  exact mle_prefixTable (K := Rat) sampleInc r

example :
    ltPoly (K := Rat) b01 (bitVec (K := Rat) b10) = 1 := by
  rw [ltPoly_at_bitVec (K := Rat) b01 b10]
  native_decide

example :
    ltPoly (K := Rat) b10 (bitVec (K := Rat) b01) = 0 := by
  rw [ltPoly_at_bitVec (K := Rat) b10 b01]
  native_decide

example :
    prefixTable (K := Rat) sampleInc b10 = 8 := by
  native_decide

example :
    prefixExpression (K := Rat) sampleInc (bitVec (K := Rat) b10) = 8 := by
  rw [prefixExpression_at_bitVec (K := Rat) sampleInc b10]
  native_decide

#guard cubeValue b00 = 0
#guard cubeValue b01 = 1
#guard cubeValue b10 = 2
#guard cubeValue b11 = 3
#guard prefixTable (K := Rat) sampleInc b00 = 0
#guard prefixTable (K := Rat) sampleInc b01 = 3
#guard prefixTable (K := Rat) sampleInc b10 = 8
#guard prefixTable (K := Rat) sampleInc b11 = 15

end tests.lessthanpoly
