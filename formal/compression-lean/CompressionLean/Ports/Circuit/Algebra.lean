import CompressionLean.Ports.Circuit.Ir

/-!
Owns the field helpers of `crates/neo-spartan/src/circuit/algebra.rs` as
builder programs: extension constants and proof values, lifts, dot products,
eq factors and tables, interpolation, Horner, the `K ⊗ Ext` values (`KxVar`)
and `power_from_bits`. Each one calls `Build` in the order of its Rust
counterpart, so it writes the instructions that `tests/internal/lowering.rs`
logs for it. Points are low bit first.

`algebra::constant`, `algebra::private` and `algebra::scale` only turn a
field value into a `u64`; here they are `Build.const`, `Build.priv` and
`Build.scale`.
-/

namespace CompressionLean.Circuit

open CompressionLean.Replay (Word Ext)

/-- An extension value: the handles of its three coordinates (`Backend::E`). -/
abbrev ExtVar := Fin 3 → Var

namespace Algebra

/-- `ext_constant` (`algebra.rs:26-28`). -/
def extConstant (value : Ext) : Build ExtVar := Build.extConstant ![value.c0, value.c1, value.c2]

/-- `ext_zero` (`algebra.rs:30-32`). -/
def extZero : Build ExtVar := extConstant 0

/-- `ext_one` (`algebra.rs:34-36`). -/
def extOne : Build ExtVar := extConstant 1

/-- `private_ext`: three proof words (`algebra.rs:39-42`). -/
def privateExt : Build ExtVar := do
  let x ← Build.priv
  let y ← Build.priv
  let z ← Build.priv
  pure ![x, y, z]

/-- `lift`: a base word as an extension value (`algebra.rs:45-48`). -/
def lift (value : Var) : Build ExtVar := do
  let zeroWord ← Build.const 0
  pure ![value, zeroWord, zeroWord]

/-- `ext_scale_constant` (`algebra.rs:50-53`). -/
def extScaleConstant (a : ExtVar) (by_ : Word) : Build ExtVar := do
  let by_ ← Build.const by_
  Build.extScale a by_

/-- `dot`: `Σ_i x_i·y_i` (`algebra.rs:56-64`). -/
def dot (x y : Array ExtVar) : Build ExtVar := do
  let mut total ← extZero
  for (x, y) in x.zip y do
    let term ← Build.extMul x y
    total ← Build.extAdd total term
  pure total

/-- `one_minus` (`algebra.rs:67-70`). -/
def oneMinus (x : ExtVar) : Build ExtVar := do
  let one ← extOne
  Build.extSub one x

/-- `eq1`: `x·y + (1 - x)(1 - y)` as `2xy + 1 - (x + y)` (`algebra.rs:73-80`). -/
def eq1 (x y : ExtVar) : Build ExtVar := do
  let both ← Build.extMul x y
  let twice ← Build.extAdd both both
  let one ← extOne
  let shifted ← Build.extAdd twice one
  let sum ← Build.extAdd x y
  Build.extSub shifted sum

/-- `eq_eval`: `Π_t eq(x_t, y_t)` (`algebra.rs:83-91`). -/
def eqEval (x y : Array ExtVar) : Build ExtVar := do
  let mut total ← extOne
  for (x, y) in x.zip y do
    let factor ← eq1 x y
    total ← Build.extMul total factor
  pure total

/-- `eq_table`: `eq(point, i)` for every `i < 2^|point|` (`algebra.rs:94-108`). -/
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

/-- `eq_at`: `eq(point, index)` for a known `index` (`algebra.rs:111-118`). -/
def eqAt (point : Array ExtVar) (index : ℕ) : Build ExtVar := do
  let mut total ← extOne
  for (x, t) in point.zipIdx do
    let factor ← if index.testBit t then pure x else oneMinus x
    total ← Build.extMul total factor
  pure total

/-- `interpolate`: the polynomial through `(i, values[i])` at `x`
(`algebra.rs:121-140`). The denominator is a known field value. -/
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

/-- `horner`: `Σ_i c_i·x^i` (`algebra.rs:143-150`). -/
def horner (coefficients : Array ExtVar) (x : ExtVar) : Build ExtVar := do
  let mut total ← extZero
  for c in coefficients.reverse do
    let scaled ← Build.extMul total x
    total ← Build.extAdd scaled c
  pure total

/-- `power_from_bits`: `generator^index` from the little-endian bits of
`index` (`algebra.rs:225-236`). -/
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

/-- A `K = F[u]/(u² - 7)` value: `(re, im)` (`algebra::K`). -/
abbrev KVar := Var × Var

/-- A `K ⊗ Ext = Ext[u]/(u² - 7)` value (`algebra::Kx`, `algebra.rs:156-222`). -/
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

/-- `1 - a`. -/
def oneMinus (a : KxVar) : Build KxVar := do
  let re ← Algebra.oneMinus a.re
  let zero ← Algebra.extZero
  let im ← Build.extSub zero a.im
  pure ⟨re, im⟩

end KxVar

end CompressionLean.Circuit
