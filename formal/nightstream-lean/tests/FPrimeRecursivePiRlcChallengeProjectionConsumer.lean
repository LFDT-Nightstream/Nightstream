import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.ProjectionConsumer

/-!
Public theorem-shape regressions for the active PiRLC challenge handoff.

| Theorem | Explicit boundary | Result |
|---|---|---|
| physical alias | fixed generated trace and projection owners | all 810 producer/consumer columns agree |
| decoded alias | arbitrary canonical assignment carrier | both phases decode the same 15 `RingF` values |
| field-derived handoff | explicit embedded-row satisfaction | sampler output equals independent first-accepted semantics |
| authority handoff | separate sampler authority premise | projection challenges equal verifier-derived challenges |
-/

namespace NightstreamTests.FPrimeRecursivePiRlcChallengeProjectionConsumer

open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.ProjectionConsumer
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcProjection.YZcolNormalForm
open Nightstream.SuperNeo.Concrete

#check fieldDerivedChallenges
#check sampler_output_column_eq_layout
#check sampler_columns_eq_layout
#check samplerChallengesBound_fieldDerived

example :
    ∀ index : Fin sourceCount,
      samplerColumns index = (limb0Pair index).rhoColumns :=
  sampler_columns_eq_projection_columns

example (assignment : Nat → Nat) :
    decodedSamplerChallenges assignment = decodedChallenges assignment :=
  decodedSamplerChallenges_eq_decodedChallenges assignment

example
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment) :
    SamplerChallengesBound assignment
      (fieldDerivedChallenges assignment canonical) :=
  samplerChallengesBound_fieldDerived prime canonical one accepted

example
    {assignment : Nat → Nat}
    {challenges : Fin sourceCount → RingF}
    (bound : SamplerChallengesBound assignment challenges) :
    decodedChallenges assignment = challenges :=
  decodedChallenges_eq_of_samplerBound bound

end NightstreamTests.FPrimeRecursivePiRlcChallengeProjectionConsumer
