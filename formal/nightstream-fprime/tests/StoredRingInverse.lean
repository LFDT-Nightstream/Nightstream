import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverse

/-! Execute the stored extended-GCD candidate and check its result with the
protocol RingF multiplication. Cases cover field normalization, reduction
across the Phi81 boundary, and a dense Bezout cofactor. -/

namespace NightstreamFPrime.Tests.StoredRingInverse

open NightstreamFPrime.Spec
open Phi81Relation.EvaluationHomomorphism.StoredRingInverse

private def check (name : String) (value : StoredRing) : IO Unit := do
  let inverse := candidate value
  let product := ringFMul value.get inverse.get
  for degree in [:ringDegree] do
    if bound : degree < ringDegree then
      let index : Fin ringDegree := ⟨degree, bound⟩
      unless product index == ringFOne index do
        throw (IO.userError s!"{name}: inverse failed at coefficient {degree}")
  IO.println s!"{name}: all {ringDegree} product coefficients passed"

private def regression (_ : Unit) : IO Unit := do
  check "constant 2" (Vector.ofFn fun i => if i.val = 0 then 2 else 0)
  check "X" (Vector.ofFn fun i => if i.val = 1 then 1 else 0)
  check "1 + X" (Vector.ofFn fun i => if i.val ≤ 1 then 1 else 0)

#eval regression ()

end NightstreamFPrime.Tests.StoredRingInverse
