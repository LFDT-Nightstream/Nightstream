import CompressionLean.Ports.Circuit.Views

namespace CompressionLean.Circuit

open CompressionLean.Replay (Relation D)

structure MixingVar where
  powers : Array ExtVar
  kappa : ℕ
  matrices : ℕ

namespace MixingVar

def new (lambda : ExtVar) (rel : Relation) : Build MixingVar := do
  let mut powers := #[← Algebra.extOne]
  for _ in [1:rel.rowCount] do
    powers := powers.push (← Build.extMul powers.back! lambda)
  pure { powers, kappa := rel.kappa, matrices := rel.matrices }

def commitment (m : MixingVar) (row : ℕ) : ExtVar := m.powers[row]!

def evalK (m : MixingVar) : ExtVar × ExtVar := (m.powers[m.kappa]!, m.powers[m.kappa + 1]!)

def evalA (m : MixingVar) (j : ℕ) : ExtVar × ExtVar :=
  (m.powers[m.kappa + 2 + 2 * j]!, m.powers[m.kappa + 3 + 2 * j]!)

def «public» (m : MixingVar) (block : ℕ) : ExtVar := m.powers[m.kappa + 2 + 2 * m.matrices + block]!

end MixingVar

namespace Ring

def project (value : KVar) (weights : ExtVar × ExtVar) : Build ExtVar := do
  let re ← Build.extScale weights.1 value.1
  let im ← Build.extScale weights.2 value.2
  Build.extAdd re im

def targets (statement : ClaimView) (m : MixingVar) : Build (Array ExtVar) :=
  (Array.range D).mapM fun lane => do
    let mut total ← Algebra.extZero
    for (coefficients, row) in statement.commitment.zipIdx do
      let term ← Build.extScale (m.commitment row) coefficients[lane]!
      total ← Build.extAdd total term
    let term ← project statement.evalK[lane]! m.evalK
    total ← Build.extAdd total term
    for (values, j) in statement.evalA.zipIdx do
      let term ← project values[lane]! (m.evalA j)
      total ← Build.extAdd total term
    for (values, block) in statement.publicInput.zipIdx do
      let term ← Build.extScale (m.public block) values[lane]!
      total ← Build.extAdd total term
    pure total

def liftedTarget (target quotient : Array ExtVar) (zeta : ExtVar) : Build ExtVar := do
  let mut middle := zeta
  for _ in [1:27] do
    middle ← Build.extMul middle zeta
  let top ← Build.extMul middle middle
  let one ← Algebra.extOne
  let sum ← Build.extAdd top middle
  let phi ← Build.extAdd sum one
  let target ← Algebra.horner target zeta
  let quotient ← Algebra.horner quotient zeta
  let lifted ← Build.extMul quotient phi
  Build.extAdd target lifted

def tau (zeta : ExtVar) : Build (Array ExtVar) := do
  let mut tau := Array.replicate D (← Algebra.extZero)
  let mut power ← Algebra.extOne
  for p in [0:D] do
    for column in [0:D] do
      let sign := Replay.bar p column
      unless sign == 0 do
        let term ← Algebra.extScaleConstant power sign
        tau := tau.set! column (← Build.extAdd tau[column]! term)
    power ← Build.extMul power zeta
  pure tau

end Ring

end CompressionLean.Circuit
