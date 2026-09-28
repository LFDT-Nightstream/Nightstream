import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingPowerInverse
import NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingInverse

/-! Compare the two executable inverse candidates on the existing regression
inputs and a dense binomial input. Check every returned coefficient and
every product coefficient; record execution times separately from build time. -/

namespace NightstreamFPrime.Tests.StoredRingPowerInverseComparison

open NightstreamFPrime.Spec
open Phi81Relation.EvaluationHomomorphism

private def check (name : String) (value : StoredRingArithmetic.StoredRing) : IO Unit := do
  -- The IO stores force evaluation inside each timed region. Pure lets can
  -- be moved past the timer reads by the compiler.
  let oldResult ← IO.mkRef value
  let newResult ← IO.mkRef
    (⟨value, 0⟩ : Folding.PiRLC.PaperForkExtractionWork.Result StoredRingArithmetic.StoredRing)
  let oldStart ← IO.monoNanosNow
  oldResult.set (StoredRingInverse.candidate value)
  let oldEnd ← IO.monoNanosNow
  let newStart ← IO.monoNanosNow
  newResult.set (StoredRingPowerInverse.inverse value)
  let newEnd ← IO.monoNanosNow
  let old ← oldResult.get
  let result ← newResult.get
  unless old == result.value do
    throw (IO.userError s!"{name}: inverse candidates differ")
  let product := StoredRingArithmetic.multiply result.value value
  for degree in [:ringDegree] do
    if bound : degree < ringDegree then
      let index : Fin ringDegree := ⟨degree, bound⟩
      unless product.value.get index == ringFOne index do
        throw (IO.userError s!"{name}: inverse product failed at coefficient {degree}")
  unless result.work ≤ StoredRingPowerInverse.inverseWork do
    throw (IO.userError s!"{name}: inverse clock exceeds proved bound")
  IO.println s!"{name}: all {ringDegree} inverse and product coefficients passed; xgcd_ns={oldEnd-oldStart}; power_ns={newEnd-newStart}; power_work={result.work}; bound={StoredRingPowerInverse.inverseWork}"

private def regression (_ : Unit) : IO Unit := do
  check "constant 2" (Vector.ofFn fun i => if i.val = 0 then 2 else 0)
  check "X" (Vector.ofFn fun i => if i.val = 1 then 1 else 0)
  check "1 + X" (Vector.ofFn fun i => if i.val ≤ 1 then 1 else 0)
  check "dense (1 + X)^53" (Vector.ofFn fun i =>
    ⟨Nat.choose (ringDegree - 1) i.val % goldilocksModulus, Nat.mod_lt _ (by decide)⟩)

#eval regression ()

end NightstreamFPrime.Tests.StoredRingPowerInverseComparison
