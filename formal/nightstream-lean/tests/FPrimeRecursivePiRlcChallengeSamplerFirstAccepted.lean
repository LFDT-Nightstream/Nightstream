import Nightstream.Implementation.R1CS.Correspondence.FPrimeRecursive.PiRlcChallenge.Sampler.FirstAccepted

/-!
Public theorem-shape regressions for the active PiRLC sampler closure.

Assurance tier: implementation/R1CS correspondence conditional on explicit
normalized-row embeddings, whole-row satisfaction, canonical field values,
and the constant-one wire. Poseidon2 transcript provenance remains open.
-/

namespace NightstreamTests.FPrimeRecursivePiRlcChallengeSamplerFirstAccepted

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.SamplerLayout
open Nightstream.Implementation.R1CS.FPrimeRecursivePiRlcChallenge.Sampler

#check ScalarSemantics.counterChain
#check TailSources.acceptColumnMap
#check TailSources.symbolColumnMap
#check TailSources.cumulativeColumnMap
#check TailSources.nonzeroPriorColumnMap
#check TailSources.outputColumnMap
#check TailSources.accepted_sourceBindings
#check FirstAccepted.enoughAccepted
#check FirstAccepted.outputAt_refines
#check FirstAccepted.accepted_refines
#check FirstAccepted.embeddedRows_refine_firstAccepted

example
    (prime : EuclidPrime goldilocksP)
    {fullRows : List Row} {assignment : Nat → Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted : Rows.EmbeddedRowsSatisfied fullRows assignment)
    (rho : Fin scalarCount) :
    FirstAccepted.FieldFirstAcceptedRefines assignment canonical rho :=
  FirstAccepted.embeddedRows_refine_firstAccepted
    prime canonical one accepted rho

end NightstreamTests.FPrimeRecursivePiRlcChallengeSamplerFirstAccepted
