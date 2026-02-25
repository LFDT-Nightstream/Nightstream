import SuperNeo.Field

namespace SuperNeo

open F

def oneMinus (x : F) : F := (1 : F) - x

def eqTerm (x y : F) : F :=
  x * y + oneMinus x * oneMinus y

/-- eq(x,y) = Π_i (x_i y_i + (1-x_i)(1-y_i)). -/
def eqPoly (x y : Array F) : F :=
  if x.size != y.size then
    0
  else
    Id.run do
      let mut acc : F := 1
      for i in [0:x.size] do
        acc := acc * eqTerm x[i]! y[i]!
      return acc

def bitsToFArray (width mask : Nat) : Array F :=
  Id.run do
    let mut out := Array.replicate width (0 : F)
    for i in [0:width] do
      let bit := (mask / (2 ^ i)) % 2
      out := out.set! i (F.ofNat bit)
    return out

def isBoolF (x : F) : Bool :=
  decide (x = 0 ∨ x = 1)

/-- Indicator behavior on Boolean points: eq(x,y)=1 iff x=y, else 0. -/
def eqHypercubeIndicator (x y : Array F) : Bool :=
  if x.size != y.size then
    false
  else if !(x.all isBoolF && y.all isBoolF) then
    false
  else
    let e := eqPoly x y
    if decide (x = y) then
      decide (e = 1)
    else
      decide (e = 0)

def eqPolySanity : Bool :=
  let x := #[0, 1, 0, 1]
  let y := #[0, 1, 0, 1]
  let z := #[1, 0, 1, 0]
  decide (eqPoly x y = 1 ∧ eqPoly x z = 0)

end SuperNeo
