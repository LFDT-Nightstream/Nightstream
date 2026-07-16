import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-!
Replay-to-sampler bridge for the noninteractive `Pi_RLC` scalar vector.

Protocol: SuperNeo `Pi_RLC` inside the candidate NIFS schedule.
Phase: the boundary after the represented `Pi_CCS` messages and before the
first `Pi_RLC` coordinate response.
Constraint family: transcript-state authority and sampled-challenge provenance.

Owns: the exact reached-state specialization of `ResponseRefinesAt`, extraction
of the accepted attempt's complete `Pi_RLC` vector from canonical replay, and
transport of a separately proved strong-set law to those carried challenges.

Does not own: completeness of the partial transcript carrier, alpha/gamma,
PiCCS terminal/output authority, concrete serialization, the production
candidate stream or bound, Poseidon2, probability analysis, Rust, R1CS, or
counts.

Emits constraints: no.

Authority boundary: the state is computed by replaying
`piCcsPrefixDirectives` from the typed context seed. It is never supplied by a
caller. Sampler refinement is still an explicit implementation obligation;
canonical replay alone cannot establish it, and sampler refinement alone does
not bind the attempt's carried challenges.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| NIFS | pre-PiRLC prefix | `Replay.postPiCcsState` | compute the state from the exact represented prefix |
| PiRLC | replay suffix | `acceptsCanonical_challenges_eq_oracle` | every carried coordinate equals the oracle response at that one reached state |
| PiRLC | sampler refinement | `ReplayResponseRefines` | the oracle response equals one transcript-chained batch of coefficient-sampled scalars at the reached state |
| PiRLC | provenance | `acceptsCanonical_challenges_eq_sampled` | every carried scalar is assembled from its own first-accepted coefficient vector |
| PiRLC | strong set | `acceptsCanonical_challenges_valid` | coefficient validity plus assembly validity holds for every carried scalar |
| PiRLC | failure | `shortfall_excludes_replayResponseRefines` | shortfall at any scalar coordinate fail-closes the reached-state batch refinement |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge

open Nightstream.SuperNeo.Sampling

universe uPublicParameters uVerifierKey uStructure uPublicInput uPoint
  uEvaluation uCommitment uScalar uChallenge uValue uDigest uCandidate
  uCoefficient

/-- Canonical replay reaches the PiRLC suffix from exactly the state computed
by the represented PiCCS prefix. -/
theorem acceptsCanonical_piRlcSuffix
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (accepted : Replay.AcceptsCanonical oracle context attempt) :
    Replay.AcceptsFrom oracle (Replay.postPiCcsState oracle context attempt)
      (piRlcEvents attempt) := by
  unfold Replay.AcceptsCanonical Replay.Accepts at accepted
  rw [canonicalEvents] at accepted
  have suffix := Replay.acceptsFrom_suffix oracle (oracle.seed context)
    (piCcsPrefixEvents attempt) (piRlcEvents attempt) accepted
  rw [eraseResponses_piCcsPrefixEvents] at suffix
  simpa [Replay.postPiCcsState] using suffix

/-- Canonical replay binds every carried PiRLC coordinate to the oracle
response at the one actually reached post-PiCCS state. -/
theorem acceptsCanonical_challenges_eq_oracle
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {params : GlobalParams}
    {arity : BatchArity params}
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (accepted : Replay.AcceptsCanonical oracle context attempt)
    (coordinate : Fin arity.total) :
    attempt.piRlc.challenges coordinate =
      oracle.piRlcResponse (Replay.postPiCcsState oracle context attempt)
        coordinate := by
  have suffix := acceptsCanonical_piRlcSuffix oracle context attempt accepted
  have allCoordinates := Replay.piRlc_list_eq oracle
    (Replay.postPiCcsState oracle context attempt)
    attempt.piRlc.challenges (List.finRange arity.total) (by
      simpa [piRlcEvents] using suffix)
  exact allCoordinates coordinate (by simp)

/-- The exact implementation obligation connecting the oracle response to one
bounded first-accepted execution at the reached state. -/
def ReplayResponseRefines
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {params : GlobalParams}
    {arity : BatchArity params}
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (specification : Specification
      oracle.Digest Candidate Coefficient Scalar)
    (bound : Nat)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) : Prop :=
  ResponseRefinesAt oracle.piRlcResponse specification bound
    (Replay.postPiCcsState oracle context attempt)

/-- Canonical replay plus reached-state batch refinement proves that every
challenge consumed by PiRLC is exactly the scalar assembled from that
coordinate's first-accepted coefficient vector. -/
theorem acceptsCanonical_challenges_eq_sampled
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {params : GlobalParams}
    {arity : BatchArity params}
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (specification : Specification
      oracle.Digest Candidate Coefficient Scalar)
    (bound : Nat)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (accepted : Replay.AcceptsCanonical oracle context attempt)
    (refinement : ReplayResponseRefines oracle specification bound context attempt) :
    exists batch : BatchExecution specification arity.total bound
        (Replay.postPiCcsState oracle context attempt),
      forall coordinate,
        attempt.piRlc.challenges coordinate = challenge batch coordinate := by
  unfold ReplayResponseRefines at refinement
  rcases refinement with ⟨batch, responseEq⟩
  exact ⟨batch, fun coordinate =>
    (acceptsCanonical_challenges_eq_oracle oracle context attempt accepted
      coordinate).trans (responseEq coordinate)⟩

/-- A concrete strong-set law therefore applies to the challenges actually
consumed by the accepted PiRLC attempt, not merely to an abstract output. -/
theorem acceptsCanonical_challenges_valid
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {params : GlobalParams}
    {arity : BatchArity params}
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (specification : Specification
      oracle.Digest Candidate Coefficient Scalar)
    (valid : Scalar -> Prop)
    (strongSet : StrongSetLaw specification valid)
    (bound : Nat)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (accepted : Replay.AcceptsCanonical oracle context attempt)
    (refinement : ReplayResponseRefines oracle specification bound context attempt)
    (coordinate : Fin arity.total) :
    valid (attempt.piRlc.challenges coordinate) := by
  rw [acceptsCanonical_challenges_eq_oracle oracle context attempt accepted]
  unfold ReplayResponseRefines at refinement
  exact responseRefinesAt_valid strongSet refinement coordinate

/-- Shortfall in the reached state's fixed prefix rules out the implementation
refinement; it cannot be repaired by replay acceptance or fallback values. -/
theorem shortfall_excludes_replayResponseRefines
    {PublicParameters : Type uPublicParameters}
    {VerifierKey : Type uVerifierKey}
    {Structure : Type uStructure}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Scalar : Type uScalar}
    {Challenge : Type uChallenge}
    {Value : Type uValue}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {params : GlobalParams}
    {arity : BatchArity params}
    (oracle : Replay.Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (specification : Specification
      oracle.Digest Candidate Coefficient Scalar)
    (bound : Nat)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (coordinate : Fin arity.total)
    (shortfall : ShortfallAt specification bound
      (Replay.postPiCcsState oracle context attempt) coordinate.val) :
    ¬ ReplayResponseRefines oracle specification bound context attempt := by
  unfold ReplayResponseRefines
  exact shortfall_excludes_responseRefinesAt coordinate shortfall

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ReplayBridge
