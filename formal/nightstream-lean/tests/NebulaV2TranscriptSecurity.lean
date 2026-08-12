import Nightstream.Assurance.NebulaV2.TranscriptSecurity

set_option autoImplicit false

namespace Nightstream.Tests.NebulaV2TranscriptSecurity

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Assurance.NebulaV2.FingerprintSecurity
open Nightstream.Assurance.NebulaV2.IdealTranscript
open Nightstream.Assurance.NebulaV2.TranscriptSecurity

example (polynomial : MvPolynomial (Fin 2) ChallengeField) :
    uniformTableProbability (AcceptsPolynomial polynomial) =
      repeatedProbability polynomial :=
  uniformTableProbability_accepts_eq_repeatedProbability polynomial

example
    {Outcome : Type}
    {model : ProbabilityModel Outcome}
    {actualTable idealTable : Outcome → ChallengeTable ChallengeField}
    {events : FailureEvents Outcome}
    {budget : Budget}
    (contract :
      CouplingContract model actualTable idealTable events budget)
    (event : ChallengeTable ChallengeField → Prop) :
    model.probability (fun outcome => event (actualTable outcome)) ≤
      uniformTableProbability event + budget.total :=
  actual_event_probability_le_uniform_add_failure contract event

end Nightstream.Tests.NebulaV2TranscriptSecurity
