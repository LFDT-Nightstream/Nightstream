import CompressionLean.Ports.Circuit.Ir

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word Ext)

abbrev ExtVar := Fin 3 → Var

namespace Algebra

def extConstant (value : Ext) : Build ExtVar := Build.extConstant ![value.c0, value.c1, value.c2]

def extZero : Build ExtVar := extConstant 0

def extOne : Build ExtVar := extConstant 1

def privateExt : Build ExtVar := do
  let x ← Build.priv
  let y ← Build.priv
  let z ← Build.priv
  pure ![x, y, z]

def lift (value : Var) : Build ExtVar := do
  let zeroWord ← Build.const 0
  pure ![value, zeroWord, zeroWord]

def extScaleConstant (a : ExtVar) (by_ : Word) : Build ExtVar := do
  let by_ ← Build.const by_
  Build.extScale a by_

def dot (x y : Array ExtVar) : Build ExtVar := do
  let mut total ← extZero
  for (x, y) in x.zip y do
    let term ← Build.extMul x y
    total ← Build.extAdd total term
  pure total

def oneMinus (x : ExtVar) : Build ExtVar := do
  let one ← extOne
  Build.extSub one x

def eq1 (x y : ExtVar) : Build ExtVar := do
  let both ← Build.extMul x y
  let twice ← Build.extAdd both both
  let one ← extOne
  let shifted ← Build.extAdd twice one
  let sum ← Build.extAdd x y
  Build.extSub shifted sum

def eqEval (x y : Array ExtVar) : Build ExtVar := do
  let mut total ← extOne
  for (x, y) in x.zip y do
    let factor ← eq1 x y
    total ← Build.extMul total factor
  pure total

def eqTable (point : Array ExtVar) : Build (Array ExtVar) := do
  let mut table := #[← extOne]
  for x in point do
    let low ← oneMinus x
    let mut next := #[]
    for value in table do
      next := next.push (← Build.extMul value low)
    for value in table do
      next := next.push (← Build.extMul value x)
    table := next
  pure table

def eqAt (point : Array ExtVar) (index : ℕ) : Build ExtVar := do
  let mut total ← extOne
  for (x, t) in point.zipIdx do
    let factor ← if index.testBit t then pure x else oneMinus x
    total ← Build.extMul total factor
  pure total

def interpolate (values : Array ExtVar) (x : ExtVar) : Build ExtVar := do
  let n := values.size
  let mut total ← extZero
  for (value, i) in values.zipIdx do
    let mut numerator ← extOne
    let mut denominator : Word := 1
    for j in [0:n] do
      if i ≠ j then
        let node ← extConstant (Ext.ofGl (Word.ofNat j))
        let factor ← Build.extSub x node
        numerator ← Build.extMul numerator factor
        denominator := denominator * (Word.ofNat i - Word.ofNat j)
    let weight ← extScaleConstant numerator denominator.inv
    let term ← Build.extMul value weight
    total ← Build.extAdd total term
  pure total

def horner (coefficients : Array ExtVar) (x : ExtVar) : Build ExtVar := do
  let mut total ← extZero
  for c in coefficients.reverse do
    let scaled ← Build.extMul total x
    total ← Build.extAdd scaled c
  pure total

def powerFromBits (generator : Word) (bits : Array Var) : Build Var := do
  let mut total ← Build.const 1
  let mut base := generator
  for bit in bits do
    let one ← Build.const 1
    let factorMinus ← Build.scale bit (base - 1)
    let factor ← Build.add one factorMinus
    total ← Build.mul total factor
    base := base * base
  pure total

end Algebra

abbrev KVar := Var × Var

structure KxVar where
  re : ExtVar
  im : ExtVar

namespace KxVar

def zero : Build KxVar := do
  let zero ← Algebra.extZero
  pure ⟨zero, zero⟩

def one : Build KxVar := do
  let one ← Algebra.extOne
  let zero ← Algebra.extZero
  pure ⟨one, zero⟩

def fromK (value : KVar) : Build KxVar := do
  let re ← Algebra.lift value.1
  let im ← Algebra.lift value.2
  pure ⟨re, im⟩

def add (a b : KxVar) : Build KxVar := do
  let re ← Build.extAdd a.re b.re
  let im ← Build.extAdd a.im b.im
  pure ⟨re, im⟩

def mul (a b : KxVar) : Build KxVar := do
  let rr ← Build.extMul a.re b.re
  let ii ← Build.extMul a.im b.im
  let seven ← Algebra.extScaleConstant ii 7
  let ri ← Build.extMul a.re b.im
  let ir ← Build.extMul a.im b.re
  let re ← Build.extAdd rr seven
  let im ← Build.extAdd ri ir
  pure ⟨re, im⟩

def scale (a : KxVar) (by_ : ExtVar) : Build KxVar := do
  let re ← Build.extMul a.re by_
  let im ← Build.extMul a.im by_
  pure ⟨re, im⟩

def oneMinus (a : KxVar) : Build KxVar := do
  let re ← Algebra.oneMinus a.re
  let zero ← Algebra.extZero
  let im ← Build.extSub zero a.im
  pure ⟨re, im⟩

end KxVar

end CompressionLean.Circuit
