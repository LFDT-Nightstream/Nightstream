import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Events

/-!
Deterministic replay of the typed non-interactive NIFS schedule.

Owns: a hash-chain-shaped random-oracle boundary, deterministic event
materialization, the exact state reached by a directive prefix, and the proof
that every verifier challenge is a projection of the digest obtained from the
exact ordered prefix.

Does not own: a concrete hash or encoding, collision resistance,
unpredictability, rejection sampling, challenge-set membership, paper phase
acceptance, backend refinement, or cost accounting.

Emits backend obligations: no.

Authority boundary: the oracle interface cannot inspect a semantic
`SumCheck.Round`; it receives only `SumCheckMessage`. Every absorb replaces the
state with `chain previous message`. Challenge events do not accept a caller
supplied challenge relation: their value must equal a deterministic,
domain-separated projection of the current chained state.

| Protocol | Phase | Constraint family | Mathematical obligation |
|---|---|---|---|
| NIFS | seed | context digest | derive one initial state from the typed public context |
| NIFS | every prover message | prefix chain | derive the next state only from the previous state and exact typed message |
| PiCCS | FE/NC response | SumCheck projection | derive the carried round challenge from the post-message state and typed phase/index |
| PiCCS -> PiRLC | reached state | `postPiCcsState` | fold the exact canonical pre-PiRLC directive prefix from the typed context seed |
| PiRLC | scalar response | strong-set projection boundary | derive each carried scalar from one response batch rooted at the common post-PiCCS state; sampler refinement owns the transcript-chained coefficient executions inside that batch |
| all | materialization | honest replay | a materialized schedule replays by construction |
| all | exactness | replay inversion | every accepted event list is exactly the materialization of its erased typed skeleton |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay

universe uPublicParameters uVerifierKey uStructure uPublicInput uPoint
  uEvaluation uCommitment uScalar uChallenge uValue uDigest

/--
Abstract random-oracle boundary with an enforced chained-prefix shape.

The four operations are deterministic functions. Security of any concrete
instantiation, including resistance to a constant or colliding implementation,
is deliberately outside this model-level interface.
-/
structure Oracle
    (PublicParameters : Type uPublicParameters)
    (VerifierKey : Type uVerifierKey)
    (Structure : Type uStructure)
    (PublicInput : Type uPublicInput)
    (Point : Type uPoint)
    (Evaluation : Type uEvaluation)
    (Commitment : Type uCommitment)
    (Scalar : Type uScalar)
    (Challenge : Type uChallenge)
    (Value : Type uValue)
    (params : GlobalParams)
    (arity : BatchArity params) where
  Digest : Type uDigest
  seed : Context
    PublicParameters VerifierKey Structure PublicInput Point Evaluation
      Commitment params arity -> Digest
  chain : Digest -> ProverMessage
    Structure PublicInput Point Evaluation Commitment Challenge Value
      params arity -> Digest
  sumCheckResponse : Digest -> SumCheckPhase -> Nat -> Challenge
  piRlcResponse : Digest -> Fin arity.total -> Scalar

/-- The digest reached after executing only the state transitions represented
by a directive list. Response directives leave the state unchanged. -/
def stateAfter
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity) :
    oracle.Digest ->
    List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity) ->
    oracle.Digest
  | state, [] => state
  | state, .absorb message :: rest =>
      stateAfter oracle (oracle.chain state message) rest
  | state, .sumCheckChallenge _ _ :: rest =>
      stateAfter oracle state rest
  | state, .piRlcChallenge _ :: rest =>
      stateAfter oracle state rest

/-- The exact common state reached after the represented PiCCS messages and
before the first PiRLC coordinate response. -/
def postPiCcsState
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) : oracle.Digest :=
  stateAfter oracle (oracle.seed context) (piCcsPrefixDirectives attempt)

/-- Materialize every verifier response from one evolving digest. -/
def materializeFrom
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity) :
    oracle.Digest ->
    List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity) ->
    List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
  | _, [] => []
  | state, .absorb message :: rest =>
      .absorb message :: materializeFrom oracle (oracle.chain state message) rest
  | state, .sumCheckChallenge phase roundIndex :: rest =>
      .sumCheckChallenge phase roundIndex
          (oracle.sumCheckResponse state phase roundIndex) ::
        materializeFrom oracle state rest
  | state, .piRlcChallenge coordinate :: rest =>
      .piRlcChallenge coordinate (oracle.piRlcResponse state coordinate) ::
        materializeFrom oracle state rest

/-- Materializing an appended schedule reaches the suffix from exactly the
state computed by the prefix. -/
theorem materializeFrom_append
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (state : oracle.Digest)
    (before after : List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity)) :
    materializeFrom oracle state (before ++ after) =
      materializeFrom oracle state before ++
        materializeFrom oracle (stateAfter oracle state before) after := by
  induction before generalizing state with
  | nil => rfl
  | cons directive rest inductionHypothesis =>
      cases directive <;>
        simp only [List.cons_append, materializeFrom, stateAfter,
          List.cons.injEq, true_and] <;>
        exact inductionHypothesis _

/-- Relational replay from a fixed digest over a fully materialized event list. -/
inductive AcceptsFrom
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity) :
    oracle.Digest ->
    List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) -> Prop where
  | nil (state : oracle.Digest) : AcceptsFrom oracle state []
  | absorb
      (state : oracle.Digest)
      (message : ProverMessage
        Structure PublicInput Point Evaluation Commitment Challenge Value
          params arity)
      (rest)
      (accepted : AcceptsFrom oracle (oracle.chain state message) rest) :
      AcceptsFrom oracle state (.absorb message :: rest)
  | sumCheckChallenge
      (state : oracle.Digest)
      (phase : SumCheckPhase)
      (roundIndex : Nat)
      (challenge : Challenge)
      (rest)
      (derived : challenge = oracle.sumCheckResponse state phase roundIndex)
      (accepted : AcceptsFrom oracle state rest) :
      AcceptsFrom oracle state
        (.sumCheckChallenge phase roundIndex challenge :: rest)
  | piRlcChallenge
      (state : oracle.Digest)
      (coordinate : Fin arity.total)
      (challenge : Scalar)
      (rest)
      (derived : challenge = oracle.piRlcResponse state coordinate)
      (accepted : AcceptsFrom oracle state rest) :
      AcceptsFrom oracle state (.piRlcChallenge coordinate challenge :: rest)

/-- Accepted replay of an appended event list reaches its suffix from the
state determined solely by the erased prefix. This theorem exposes the exact
state boundary without trusting the carried response values. -/
theorem acceptsFrom_suffix
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (state : oracle.Digest)
    (before after : List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity))
    (accepted : AcceptsFrom oracle state (before ++ after)) :
    AcceptsFrom oracle
      (stateAfter oracle state (eraseResponses before)) after := by
  induction before generalizing state with
  | nil =>
      simpa [eraseResponses, stateAfter] using accepted
  | cons event rest inductionHypothesis =>
      cases event with
      | absorb message =>
          cases accepted with
          | absorb _ _ _ acceptedTail =>
              simpa [eraseResponses, eraseResponse, stateAfter] using
                inductionHypothesis (oracle.chain state message) acceptedTail
      | sumCheckChallenge phase roundIndex challenge =>
          cases accepted with
          | sumCheckChallenge _ _ _ _ _ _ acceptedTail =>
              simpa [eraseResponses, eraseResponse, stateAfter] using
                inductionHypothesis state acceptedTail
      | piRlcChallenge coordinate challenge =>
          cases accepted with
          | piRlcChallenge _ _ _ _ _ acceptedTail =>
              simpa [eraseResponses, eraseResponse, stateAfter] using
                inductionHypothesis state acceptedTail

/-- Replay begins only from the oracle's typed context seed. -/
def Accepts
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (events : List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)) : Prop :=
  AcceptsFrom oracle (oracle.seed context) events

/-- Every response materialized by the oracle replays against the same oracle. -/
theorem accepts_materializeFrom
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (state : oracle.Digest)
    (directives : List (Directive
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity)) :
    AcceptsFrom oracle state (materializeFrom oracle state directives) := by
  induction directives generalizing state with
  | nil =>
      exact .nil state
  | cons directive rest inductionHypothesis =>
      cases directive with
      | absorb message =>
          exact .absorb state message _
            (inductionHypothesis (oracle.chain state message))
      | sumCheckChallenge phase roundIndex =>
          exact .sumCheckChallenge state phase roundIndex _ _ rfl
            (inductionHypothesis state)
      | piRlcChallenge coordinate =>
          exact .piRlcChallenge state coordinate _ _ rfl
            (inductionHypothesis state)

/-- Accepted replay is exactly deterministic materialization of its skeleton. -/
theorem materializeFrom_eraseResponses_eq_of_acceptsFrom
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (state : oracle.Digest)
    (events : List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity))
    (accepted : AcceptsFrom oracle state events) :
    materializeFrom oracle state (eraseResponses events) = events := by
  induction accepted with
  | nil => rfl
  | absorb state message rest accepted inductionHypothesis =>
      simp only [eraseResponses, List.map_cons, eraseResponse, materializeFrom,
        List.cons.injEq, true_and]
      simpa only [eraseResponses] using inductionHypothesis
  | sumCheckChallenge state phase roundIndex challenge rest derived accepted
      inductionHypothesis =>
      rw [derived]
      simp only [eraseResponses, List.map_cons, eraseResponse, materializeFrom,
        List.cons.injEq, true_and]
      simpa only [eraseResponses] using inductionHypothesis
  | piRlcChallenge state coordinate challenge rest derived accepted
      inductionHypothesis =>
      rw [derived]
      simp only [eraseResponses, List.map_cons, eraseResponse, materializeFrom,
        List.cons.injEq, true_and]
      simpa only [eraseResponses] using inductionHypothesis

/-- Oracle-populated events for one exact NIFS attempt. -/
def materializedSchedule
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    List (Event
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :=
  materializeFrom oracle (oracle.seed context) (schedule attempt)

/-- Canonical replay surface for one attempt. Unlike raw `AcceptsFrom`, this
predicate fixes the exact protocol skeleton supplied by `canonicalEvents`. -/
def AcceptsCanonical
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) : Prop :=
  Accepts oracle context (canonicalEvents attempt)

/-- The oracle-populated canonical schedule always replays. -/
theorem accepts_materializedSchedule
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    Accepts oracle context (materializedSchedule oracle context attempt) := by
  exact accepts_materializeFrom oracle (oracle.seed context) (schedule attempt)

/-- If carried challenges equal replay materialization, canonical replay holds. -/
theorem accepts_of_carrierAgreement
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity)
    (agreement : canonicalEvents attempt =
      materializedSchedule oracle context attempt) :
    AcceptsCanonical oracle context attempt := by
  unfold AcceptsCanonical
  rw [agreement]
  exact accepts_materializedSchedule oracle context attempt

/-- Canonical replay is equivalent to exact oracle materialization. -/
theorem acceptsCanonical_iff_carrierAgreement
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (context : Context
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment params arity)
    (attempt : Nifs.Attempt
      Structure PublicInput Point Evaluation Commitment Scalar Challenge Value
        params arity) :
    AcceptsCanonical oracle context attempt ↔
      canonicalEvents attempt = materializedSchedule oracle context attempt := by
  constructor
  · intro accepted
    unfold AcceptsCanonical at accepted
    have exactMaterialization :=
      materializeFrom_eraseResponses_eq_of_acceptsFrom oracle
        (oracle.seed context) (canonicalEvents attempt) accepted
    rw [eraseResponses_canonicalEvents] at exactMaterialization
    simpa [materializedSchedule] using exactMaterialization.symm
  · intro agreement
    exact accepts_of_carrierAgreement oracle context attempt agreement

/-- A SumCheck response immediately after a message uses its post-message state. -/
theorem sumCheck_after_message_eq
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (state : oracle.Digest)
    (message : ProverMessage
      Structure PublicInput Point Evaluation Commitment Challenge Value
        params arity)
    (phase : SumCheckPhase)
    (roundIndex : Nat)
    (challenge : Challenge)
    (rest)
    (accepted : AcceptsFrom oracle state
      (.absorb message :: .sumCheckChallenge phase roundIndex challenge :: rest)) :
    challenge = oracle.sumCheckResponse
      (oracle.chain state message) phase roundIndex := by
  cases accepted with
  | absorb _ _ _ acceptedTail =>
      cases acceptedTail with
      | sumCheckChallenge _ _ _ _ _ derived _ =>
          exact derived

/-- Every PiRLC coordinate is projected from the exact current prefix. -/
theorem piRlc_head_eq
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (state : oracle.Digest)
    (coordinate : Fin arity.total)
    (challenge : Scalar)
    (rest)
    (accepted : AcceptsFrom oracle state
      (.piRlcChallenge coordinate challenge :: rest)) :
    challenge = oracle.piRlcResponse state coordinate := by
  cases accepted with
  | piRlcChallenge _ _ _ _ derived _ =>
      exact derived

/-- A suffix consisting only of PiRLC responses uses one unchanged state for
every listed coordinate. The list may be any coordinate order; the canonical
schedule later instantiates it with `List.finRange`. -/
theorem piRlc_list_eq
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
    (oracle : Oracle
      PublicParameters VerifierKey Structure PublicInput Point Evaluation
        Commitment Scalar Challenge Value params arity)
    (state : oracle.Digest)
    (response : Fin arity.total -> Scalar)
    (coordinates : List (Fin arity.total))
    (accepted : AcceptsFrom oracle state
      (coordinates.map fun coordinate =>
        Event.piRlcChallenge coordinate (response coordinate))) :
    forall coordinate,
      coordinate ∈ coordinates ->
        response coordinate = oracle.piRlcResponse state coordinate := by
  induction coordinates with
  | nil =>
      simp
  | cons head tail inductionHypothesis =>
      cases accepted with
      | piRlcChallenge _ _ _ _ derived acceptedTail =>
          intro coordinate member
          rcases List.mem_cons.mp member with rfl | member
          · exact derived
          · exact inductionHypothesis acceptedTail coordinate member

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay
