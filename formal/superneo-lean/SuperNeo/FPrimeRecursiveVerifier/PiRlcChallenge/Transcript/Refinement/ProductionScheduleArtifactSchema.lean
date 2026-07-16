import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Schedule

/-!
Owns: the typed schema and executable audit predicates for the generated fixed
PiRLC transcript schedule artifact.

Does not own: generated values, transcript absorb contents, counter values,
Poseidon2 functional semantics, sampler correctness, or row removal.

Emits constraints: no. This file classifies and checks external evidence.

Authority boundary: `materializedSource*` describes the satisfied source R1CS;
`estimatedLowNorm*` is explicitly estimator-only; nonlinear counts are
diagnostic trace events. No field in this schema is cryptographic authority.

| Predicate/type | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `EvidenceTier` | artifact metadata | Separates source, estimator, and trace evidence | Honest generator | No |
| `FixedScheduleOrder` | `challenge.transcript` | Ordered 15-by-4 stage geometry and four lane decompositions per round | Generated sample list | No |
| `TranscriptTreeReconciles` | `challenge` | Transcript children and challenge children sum exactly | Generated cost census | No |
| `NonlinearCensusConsistent` | Poseidon trace | Every attributed permutation owns 86 S-boxes | Diagnostic trace | No |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact

/-- Evidence tiers must remain visible in every consumer of generated costs. -/
inductive EvidenceTier where
  | materializedSourceR1cs
  | traceReconciledEstimate
  | diagnosticTrace
deriving DecidableEq, Repr

/-- One disjoint stage cost with deliberately non-ambiguous field names. -/
structure Cost where
  materializedSourceRows : Nat
  materializedSourceColumns : Nat
  estimatedLowNormRows : Nat
  estimatedLowNormColumns : Nat
  tracedPoseidonPermutations : Nat
  tracedSboxes : Nat
deriving DecidableEq, Repr

def Cost.zero : Cost :=
  { materializedSourceRows := 0
    materializedSourceColumns := 0
    estimatedLowNormRows := 0
    estimatedLowNormColumns := 0
    tracedPoseidonPermutations := 0
    tracedSboxes := 0 }

/-- Componentwise sum for immediate-child ownership reconciliation. -/
def Cost.add (left right : Cost) : Cost :=
  { materializedSourceRows := left.materializedSourceRows + right.materializedSourceRows
    materializedSourceColumns := left.materializedSourceColumns + right.materializedSourceColumns
    estimatedLowNormRows := left.estimatedLowNormRows + right.estimatedLowNormRows
    estimatedLowNormColumns := left.estimatedLowNormColumns + right.estimatedLowNormColumns
    tracedPoseidonPermutations :=
      left.tracedPoseidonPermutations + right.tracedPoseidonPermutations
    tracedSboxes := left.tracedSboxes + right.tracedSboxes }

/-- Add a list of disjoint stage costs. -/
def sumCosts (costs : List Cost) : Cost :=
  costs.foldl Cost.add Cost.zero

/-- One production digest checkpoint followed by its lane decompositions. -/
structure DigestRound where
  rhoIndex : Nat
  roundIndex : Nat
  laneDecompositionOccurrences : Nat
  digest : Cost
  lanes : Cost
deriving DecidableEq, Repr

/-- One separator, sampler initialization, and four digest rounds. -/
structure RhoSample where
  rhoIndex : Nat
  separator : Cost
  samplerInitialization : Cost
  rounds : List DigestRound
deriving DecidableEq, Repr

/-- Exact typed order expected from the fixed production schedule. -/
def FixedScheduleOrder (samples : List RhoSample) : Prop :=
  samples.map (fun sample => sample.rhoIndex) = List.range rhoCount ∧
    samples.Forall (fun sample =>
      sample.rounds.map (fun round => round.rhoIndex) =
          List.replicate fixedDigestRounds sample.rhoIndex ∧
        sample.rounds.map (fun round => round.roundIndex) =
          List.range fixedDigestRounds ∧
        sample.rounds.Forall (fun round =>
          round.laneDecompositionOccurrences = digestLanes))

/-- Flatten all generated separator costs in schedule order. -/
def separatorCosts (samples : List RhoSample) : List Cost :=
  samples.map (fun sample => sample.separator)

/-- Flatten all generated sampler-initialization costs in schedule order. -/
def samplerInitializationCosts (samples : List RhoSample) : List Cost :=
  samples.map (fun sample => sample.samplerInitialization)

/-- Flatten all generated digest costs in schedule order. -/
def digestCosts (samples : List RhoSample) : List Cost :=
  samples.flatMap (fun sample => sample.rounds.map (fun round => round.digest))

/-- Flatten all generated lane-decomposition costs in schedule order. -/
def laneCosts (samples : List RhoSample) : List Cost :=
  samples.flatMap (fun sample => sample.rounds.map (fun round => round.lanes))

/-- Per-event lists must sum to their generated stage-family totals. -/
def EventFamiliesReconcile
    (samples : List RhoSample)
    (rhoDomainSeparators samplerInitializations digestRounds
      laneDecompositions : Cost) : Prop :=
  sumCosts (separatorCosts samples) = rhoDomainSeparators ∧
    sumCosts (samplerInitializationCosts samples) = samplerInitializations ∧
    sumCosts (digestCosts samples) = digestRounds ∧
    sumCosts (laneCosts samples) = laneDecompositions

/-- Exact protocol -> phase -> constraint-family ownership equations. -/
def TranscriptTreeReconciles
    (bindOutputsDigest rhoDomainSeparators digestRounds laneDecompositions
      transcript sampler challenge : Cost) : Prop :=
  Cost.add
      (Cost.add (Cost.add bindOutputsDigest rhoDomainSeparators) digestRounds)
      laneDecompositions = transcript ∧
    Cost.add transcript sampler = challenge

/-- Production Poseidon2 uses 86 `x^7` S-boxes per permutation. -/
def NonlinearCensusConsistent (cost : Cost) : Prop :=
  cost.tracedSboxes = 86 * cost.tracedPoseidonPermutations

/-- Every transcript nonlinear event and its family totals have the same census. -/
def AllNonlinearCensusesConsistent
    (samples : List RhoSample)
    (bindOutputsDigest rhoDomainSeparators digestRounds laneDecompositions
      transcript : Cost) : Prop :=
  NonlinearCensusConsistent bindOutputsDigest ∧
    NonlinearCensusConsistent rhoDomainSeparators ∧
    NonlinearCensusConsistent digestRounds ∧
    NonlinearCensusConsistent laneDecompositions ∧
    NonlinearCensusConsistent transcript ∧
    samples.Forall (fun sample =>
      NonlinearCensusConsistent sample.separator ∧
        sample.rounds.Forall (fun round =>
          NonlinearCensusConsistent round.digest ∧
            NonlinearCensusConsistent round.lanes))

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.ProductionScheduleArtifact
