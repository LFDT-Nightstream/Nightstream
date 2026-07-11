import Nightstream.Assurance.FPrimeFullHistoryCircuitComplete

namespace Nightstream.Tests.FPrimeFullHistoryCircuit

open Nightstream.Implementation.R1CS
open Nightstream.Assurance.FPrimeFullHistoryCircuit

example : FPrimeFullHistoryRows.fullRows.length = 4076614 := by
  calc
    FPrimeFullHistoryRows.fullRows.length =
        FPrimeFullHistoryManifest.totalRows :=
      FPrimeFullHistoryRows.fullRows_length
    _ = 4076614 := rfl

example
    (prime : EuclidPrime goldilocksP)
    {field : CanonicalU64Complete.FieldInverse}
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (witness : CompilerWitness field assignment) :
    Satisfies FPrimeFullHistoryRows.fullRows assignment :=
  fPrimeCircuit_complete prime canonical one witness

example
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (rows : Satisfies FPrimeFullHistoryRows.fullRows assignment) :
    Nightstream.Assurance.ValidExecution Edge
        (TerminalValid assignment canonical)
        initialState (finalState assignment canonical) 2 ∨
      BadEvent assignment :=
  fPrimeCircuit_sound_or_bad prime canonical one rows

end Nightstream.Tests.FPrimeFullHistoryCircuit
