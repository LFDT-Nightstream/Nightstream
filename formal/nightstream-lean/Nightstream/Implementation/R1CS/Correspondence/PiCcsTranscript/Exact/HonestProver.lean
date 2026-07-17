import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.CompleteSchedule
import Nightstream.Implementation.R1CS.Correspondence.PiCcsTranscript.Exact.Refinement
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver

/-!
Honest prover for the complete exact minimal mixed-width `Pi_CCS` candidate
transcript.

Assurance tier: semantic-to-executable completeness refinement.

Owns: derivation of pre-SumCheck FE/NC coins from the binding successor;
computation of the verifier-owned FE initial claim; construction of honest
message-before-challenge FE and NC certificates; exact carrier packaging; and
accepted, source-bound execution of the complete concrete schedule.

Does not own: authority of the outer initial state or binding fields,
the paper single-`Q` transcript, Fiat--Shamir probability, Poseidon2 row
soundness, Rust/R1CS conformance, costs, or row removal.

Emits constraints: no.

Authority boundary: the theorem receives semantic source data and paper
obligations. It computes coins and the FE initial claim from verifier replay,
constructs prover messages sequentially, and then packages exactly those
messages into `Exact.Carrier`. It never accepts caller-supplied challenges,
round points, phase-boundary states, output claims, or a loose shape proof.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.exact.prover.coins` | derive FE/NC coins after the complete binding prefix | computed | `completeInput`, `complete_of_paperObligations` |
| `nifs.pi_ccs.exact.prover.initial` | compute FE initial from public input and derived coins | computed | `expectedFeInitial` |
| `nifs.pi_ccs.exact.prover.messages` | construct FE then NC messages before their challenges | derived | semantic `Protocol.HonestProver.complete_of_paperObligations` |
| `nifs.pi_ccs.exact.prover.carrier` | package those typed messages losslessly into the exact physical carrier | direct dataflow | `Carrier.ofProtocolCertificate` |
| `nifs.pi_ccs.exact.prover.acceptance` | the concrete transcript machines accept the exact certificate | derived | `complete_of_paperObligations` |
| `nifs.pi_ccs.exact.prover.output` | output claims are canonical and source-bound at replay-derived points | derived | `complete_of_paperObligations` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.HonestProver

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

/-- Verifier-owned FE initial claim after the binding and concrete challenge
phases. -/
def expectedFeInitial
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (initialState : State)
    (binding : Binding.Input)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape) : Nightstream.SuperNeo.Concrete.K :=
  let publicInput := PublicInput.ofSources data
  let afterBinding := Binding.run initialState binding
  let coins := Coins.feCoins afterBinding shape domain
    (CompleteSchedule.degreeBound publicInput)
  Polynomial.Fe.initial profile publicInput coins

/-- Canonical complete exact transcript input underlying one semantic protocol
certificate. -/
def completeInput
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (initialState : State)
    (binding : Binding.Input)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (certificate :
      Protocol.Certificate (PublicInput.ofSources data) domain) :
    CompleteSchedule.Input (PublicInput.ofSources data) domain where
  initialState := initialState
  binding := binding
  expectedFeInitial :=
    expectedFeInitial initialState binding profile data
  carrier := Carrier.ofProtocolCertificate certificate

/-- Every source satisfying the independent paper obligation set has an exact
physical certificate accepted by the concrete Poseidon2 transcript machines.

The result also records source binding for the output claims and derives the
legacy loose transcript shape from the exact carrier. The complete schedule
and `Exact.Refinement` then expose the same FE/NC challenges and successor
states as the typed semantic verifier. -/
theorem complete_of_paperObligations
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (initialState : State)
    (binding : Binding.Input)
    (profile : Polynomial.Fe.SupportedProfile shape domain)
    (data : Data shape)
    (obligations : Semantics.Paper.Holds data) :
    ∃ certificate :
        Protocol.Certificate (PublicInput.ofSources data) domain,
      let input :=
        completeInput initialState binding profile data certificate
      Protocol.Accepted
          (FeRefinement.machine input.expectedFeInitial)
          NcRefinement.machine
          (CompleteSchedule.challengeOutput input).state
          profile
          (PublicInput.ofSources data)
          (CompleteSchedule.feCoins input)
          (CompleteSchedule.ncCoins input)
          certificate /\
        BoundToSources covers data
          (Protocol.derive
            (FeRefinement.machine input.expectedFeInitial)
            NcRefinement.machine
            (CompleteSchedule.challengeOutput input).state
            certificate).outputPoints
          certificate.output /\
        Nightstream.Implementation.R1CS.PiCcsTranscript.Schedule.WellShaped
          (CompleteSchedule.scheduleInput input) := by
  let publicInput := PublicInput.ofSources data
  let degree := CompleteSchedule.degreeBound publicInput
  let bindingState := Binding.run initialState binding
  let challengeOutput := Coins.run bindingState shape domain degree
  let feCoins := Coins.feCoins bindingState shape domain degree
  let ncCoins := Coins.ncCoins bindingState shape domain degree
  let initialClaim := Polynomial.Fe.initial profile publicInput feCoins
  rcases
      Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.HonestProver.complete_of_paperObligations
        covers
        (FeRefinement.machine initialClaim)
        NcRefinement.machine
        challengeOutput.state
        profile data feCoins ncCoins obligations with
    ⟨certificate, accepted, outputBound⟩
  refine ⟨certificate, ?_, ?_, ?_⟩
  · simpa [completeInput, expectedFeInitial,
      CompleteSchedule.challengeOutput, CompleteSchedule.afterBinding,
      CompleteSchedule.feCoins, CompleteSchedule.ncCoins,
      publicInput, degree, bindingState, challengeOutput,
      feCoins, ncCoins, initialClaim]
      using accepted
  · simpa [completeInput, expectedFeInitial,
      CompleteSchedule.challengeOutput, CompleteSchedule.afterBinding,
      publicInput, degree, bindingState, challengeOutput,
      feCoins, initialClaim]
      using outputBound
  · exact CompleteSchedule.scheduleInput_wellShaped
      (completeInput initialState binding profile data certificate)

end Nightstream.Implementation.R1CS.PiCcsTranscript.Exact.HonestProver
