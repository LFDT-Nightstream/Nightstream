import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.Terminal.CandidateRefinement
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.Terminal.FirstAccepted

/-!
Terminal `Pi_RLC` selection output over verifier-owned machine candidates.

Assurance tier: implementation/R1CS correspondence. This module composes the
independent transcript-candidate provenance theorem with the independently
proved rejection and first-accepted semantics. It does not copy the Rust
sampler or trust the generated output wires.

Owns: the exact 64-candidate machine prefix for each `rho : Fin 15`; equality
with the sampler's canonical field-candidate prefix; the first 54 accepted
machine coefficients; and their centered encoding in every production output
wire.

Does not own: the post-`Pi_CCS` initial transcript state, embedding the 54
coefficients into the production quotient ring, `Pi_RLC` algebraic identities,
Rust trace conformance, row removal, or cost totals.

Emits constraints: no.

Authority boundary: selection witnesses only route candidates already proved
to come from the verifier-owned Poseidon2 schedule. The 54 outputs are derived
by the independent `firstAccepted` function; a prover-supplied coefficient or
digest is never accepted as authority.

| Protocol | Phase | Constraint family | Indexed object | Proven obligation |
|---|---|---|---|---|
| `Pi_RLC` | bounded sampler | candidate prefix | 64 machine chunks per scalar | field candidate list equals the connected machine stream prefix |
| `Pi_RLC` | rejection | acceptance decision | each machine chunk | independent verifier decides acceptance and symbol |
| `Pi_RLC` | bounded sampler | success premise | one 64-candidate machine prefix | accepted rows prove at least 54 accepted chunks |
| `Pi_RLC` | first accepted | 54-of-64 selection | 54 coefficient positions | semantic output is exactly the first 54 accepted machine symbols |
| `Pi_RLC` | output encoding | centered field wire | 54 production outputs | each wire encodes the corresponding machine-derived coefficient |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Fixed four-digest prefix of the connected transcript-machine stream for
one terminal scalar coordinate. -/
def machineCandidates
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : List ProductionAlphabet.Chunk :=
  List.ofFn fun candidate : Fin SelectionRows.candidateCount =>
    ProductionSchedule.candidateStream TranscriptMachine.machine
      (Transcript.Terminal.ScheduleRefinement.afterEnterState
        assignment canonical rho)
      rho.val candidate.val

/-- The sampler's 64 canonical field chunks are exactly the connected
transcript-machine prefix, point by point and in independent source order. -/
theorem fieldCandidates_eq_machineCandidates
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    TailPrefixCounts.candidates (TailSources.layout rho) assignment canonical =
      machineCandidates assignment canonical rho := by
  unfold TailPrefixCounts.candidates machineCandidates
  apply congrArg List.ofFn
  funext candidate
  exact Transcript.Terminal.CandidateRefinement.accepted_refines_candidateStream
    canonical one accepted rho candidate

/-- Independent first-accepted output over the machine-derived 64-candidate
prefix. -/
def semanticOutput
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : List ProductionAlphabet.Coefficient :=
  Nightstream.SuperNeo.Sampling.FirstAccepted.firstAccepted
    ProductionAlphabet.verifier ProductionAlphabet.coefficientCount
    (machineCandidates assignment canonical rho)

/-- Field-derived first-accepted semantics and machine-derived semantics are
identical after transcript provenance closes. -/
theorem fieldSemanticOutput_eq_machineSemanticOutput
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    FirstAccepted.semanticOutput assignment canonical rho =
      semanticOutput assignment canonical rho := by
  unfold FirstAccepted.semanticOutput TailFirstAccepted.semanticOutput
    semanticOutput
  rw [fieldCandidates_eq_machineCandidates canonical one accepted rho]

/-- Accepted terminal rows prove bounded-sampler success for the exact
machine-derived candidate prefix. This is the machine-facing transport of the
row-owned success theorem; it introduces no new acceptance condition. -/
theorem enoughAccepted
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    Nightstream.SuperNeo.Sampling.FirstAccepted.Enough
      ProductionAlphabet.verifier ProductionAlphabet.coefficientCount
      (machineCandidates assignment canonical rho) := by
  rw [← fieldCandidates_eq_machineCandidates canonical one accepted rho]
  exact FirstAccepted.enoughAccepted prime canonical one accepted rho

/-- Accepted selection rows guarantee that the machine-derived output has the
full production coefficient count. This is the explicit bounded-sampler
success condition; no claim is made for a prefix with fewer accepted chunks. -/
theorem semanticOutput_length
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    (semanticOutput assignment canonical rho).length =
      ProductionAlphabet.coefficientCount := by
  unfold semanticOutput
  exact
    Nightstream.SuperNeo.Sampling.FirstAccepted.firstAccepted_length_of_enough
      (enoughAccepted prime canonical one accepted rho)

/-- Total 54-coordinate scalar view. Its default coefficient is unreachable
under `semanticOutput_length`, but keeps the definition total independently of
accepted rows. -/
def scalar
    (assignment : Nat -> Nat)
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (rho : Fin ScalarRows.scalarCount) : ProductionStrongSet.Scalar :=
  fun position =>
    (semanticOutput assignment canonical rho).getD position.val
      TailFirstAccepted.defaultCoefficient

/-- Every production output wire is the centered Goldilocks representation of
the corresponding coefficient in the machine-derived scalar. -/
theorem outputAt_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount)
    (position : Fin ProductionAlphabet.coefficientCount) :
    TailRows.localAssignment assignment rho
        (SelectionRows.outputCol position.val) =
      CandidateOrder.centeredField
        (scalar assignment canonical rho position) := by
  have fieldRefined :=
    FirstAccepted.outputAt_refines prime canonical one accepted rho position
  rw [fieldSemanticOutput_eq_machineSemanticOutput
    canonical one accepted rho] at fieldRefined
  exact fieldRefined

/-- Complete output-vector form of `outputAt_refines`. Ring assembly remains a
separate obligation with its own mathematical owner. -/
theorem productionOutput_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryTerminalPiRlcTranscriptRhos.Accepted assignment)
    (rho : Fin ScalarRows.scalarCount) :
    FirstAccepted.productionOutput assignment rho =
      List.ofFn fun position : Fin ProductionAlphabet.coefficientCount =>
        CandidateOrder.centeredField
          (scalar assignment canonical rho position) := by
  apply congrArg List.ofFn
  funext position
  exact outputAt_refines prime canonical one accepted rho position

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.Terminal.MachineOutput
