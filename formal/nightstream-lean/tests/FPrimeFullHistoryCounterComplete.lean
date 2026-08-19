import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCounterSound

namespace NightstreamTests.FPrimeFullHistoryCounterComplete

open Nightstream.Implementation.R1CS

namespace Local

open FPrimeFullHistoryCounterLocalSound.Compiler

example : compilerRows.length = FPrimeFullHistoryCounter.rowCount := by
  native_decide

example : FPrimeFullHistoryCounter.rows = compilerRows := exactRows_eq

example (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (execution : ExecutionWitness field assignment) :
    Satisfies FPrimeFullHistoryCounter.rows assignment :=
  complete prime execution

end Local

example (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse} {assignment : Nat → Nat}
    (execution : FPrimeFullHistoryCounterSound.Compiler.ExecutionWitness
      field assignment) :
    Satisfies FPrimeFullHistoryCounterSound.globalRows assignment :=
  FPrimeFullHistoryCounterSound.Compiler.complete prime execution

end NightstreamTests.FPrimeFullHistoryCounterComplete
