import Nightstream.Assurance.Nebula.TranscriptSecurity

set_option autoImplicit false

namespace Nightstream.Tests.NebulaTranscriptSecurity

open Nightstream.Implementation.Nebula.ConcreteField
open Nightstream.Assurance.Nebula.FingerprintSecurity
open Nightstream.Assurance.Nebula.IdealTranscript
open Nightstream.Assurance.Nebula.TranscriptSecurity

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

end Nightstream.Tests.NebulaTranscriptSecurity
