import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.Replay
import Nightstream.SuperNeo.Sampling.FirstAccepted

/-!
Transcript-chained bounded sampling semantics for noninteractive `Pi_RLC`.

Protocol: SuperNeo `Pi_RLC` plus HyperNova Construction 3.
Phase: the verifier responses after the final represented `Pi_CCS` prover
message.
Constraint family: per-challenge candidate generation, rejection, ordered
coefficient selection, scalar assembly, and strong-set membership.

Owns: one verifier-owned candidate stream and successor transcript state per
`Pi_RLC` challenge index; exact first-accepted selection of a fixed coefficient
vector for each challenge; assembly of that vector into the scalar consumed by
`Pi_RLC`; transcript-state threading across the complete challenge batch; and
explicit bounded-shortfall failure at every coordinate.

Does not own: the concrete candidate encoding, chunk width, rejection bucket,
coefficient count, number of Poseidon2 squeezes, concrete scalar assembly,
strong-set theorem, probability distribution, transcript serialization,
Poseidon2, Rust, R1CS, or counts.

Emits constraints: no.

Authority boundary: each `Source` is produced from the previous verifier-owned
state and the challenge index. It jointly names that challenge's candidate
stream and the successor state; neither is supplied by the prover. Concrete
refinement must still prove that both fields arise from the same fixed
Poseidon2 execution. A challenge is valid only after proving both that every
selected coefficient is valid and that assembling those coefficients yields a
valid `Pi_RLC` scalar.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_RLC` | challenge schedule | `stateAt` / `sourceAt` | thread one verifier-owned state through every `K+k` challenge index |
| `Pi_RLC` | one challenge | `CoefficientExecution` | select the first fixed number of accepted coefficients from that challenge's stream |
| `Pi_RLC` | scalar assembly | `Specification.assemble` | build one `Pi_RLC` scalar from the complete selected coefficient vector |
| `Pi_RLC` | batch | `BatchExecution` | retain one bounded coefficient execution for every carried scalar |
| `Pi_RLC` | strong set | `StrongSetLaw` | accepted coefficients are valid and their assembly is a valid scalar |
| `Pi_RLC` | fixed bound | `ShortfallAt` | any coordinate shortfall rules out batch refinement |
| `Pi_RLC` | completeness split | `available_or_exists_shortfall` | a finite batch either exists or one exact coordinate shortfalls |
| `Pi_RLC` | replay bridge | `ResponseRefinesAt` | every replay scalar equals its transcript-chained sampled scalar |
-/

namespace Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

open Nightstream.SuperNeo.Sampling

universe uDigest uCandidate uCoefficient uScalar

/-- One verifier-owned challenge source. The candidate stream and successor
state must later be refined to the same concrete transcript execution. -/
structure Source
    (Digest : Type uDigest)
    (Candidate : Type uCandidate) where
  stream : FirstAccepted.CandidateStream Candidate
  nextState : Digest

/-- Abstract production-shaped sampler contract. One scalar is assembled from
one complete first-accepted coefficient vector at every challenge index. -/
structure Specification
    (Digest : Type uDigest)
    (Candidate : Type uCandidate)
    (Coefficient : Type uCoefficient)
    (Scalar : Type uScalar) where
  coefficientCount : Nat
  source : Digest -> Nat -> Source Digest Candidate
  verifier : FirstAccepted.Verifier Candidate Coefficient
  assemble : (Fin coefficientCount -> Coefficient) -> Scalar

/-- State before challenge `index`, computed from the initial post-PiCCS state
and every prior verifier-owned successor state. -/
def stateAt
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (initial : Digest) : Nat -> Digest
  | 0 => initial
  | index + 1 =>
      (specification.source
        (stateAt specification initial index) index).nextState

/-- Candidate source used for one challenge index in the threaded schedule. -/
def sourceAt
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (initial : Digest)
    (index : Nat) : Source Digest Candidate :=
  specification.source (stateAt specification initial index) index

@[simp] theorem stateAt_zero
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (initial : Digest) :
    stateAt specification initial 0 = initial := by
  rfl

@[simp] theorem stateAt_succ
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (initial : Digest)
    (index : Nat) :
    stateAt specification initial (index + 1) =
      (sourceAt specification initial index).nextState := by
  rfl

/-- Successful bounded coefficient execution for one challenge index. -/
abbrev CoefficientExecution
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (bound : Nat)
    (initial : Digest)
    (index : Nat) :=
  FirstAccepted.BoundedExecution specification.verifier
    specification.coefficientCount
    (sourceAt specification initial index).stream bound

/-- One selected coefficient of an exact-length bounded execution. -/
def coefficient
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {bound : Nat}
    {initial : Digest}
    {index : Nat}
    (execution : CoefficientExecution specification bound initial index)
    (position : Fin specification.coefficientCount) : Coefficient :=
  execution.output.get ⟨position.val, by
    rw [execution.output_length]
    exact position.isLt⟩

theorem coefficient_mem_output
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {bound : Nat}
    {initial : Digest}
    {index : Nat}
    (execution : CoefficientExecution specification bound initial index)
    (position : Fin specification.coefficientCount) :
    coefficient execution position ∈ execution.output := by
  exact List.get_mem execution.output _

/-- One bounded coefficient execution for every transcript-chained challenge. -/
structure BatchExecution
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (challengeCount bound : Nat)
    (initial : Digest) where
  execution : (coordinate : Fin challengeCount) ->
    CoefficientExecution specification bound initial coordinate.val

/-- The scalar challenge assembled from one coordinate's selected coefficients. -/
def challenge
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {challengeCount bound : Nat}
    {initial : Digest}
    (batch : BatchExecution specification challengeCount bound initial)
    (coordinate : Fin challengeCount) : Scalar :=
  specification.assemble fun position =>
    coefficient (batch.execution coordinate) position

/-- Every selected coefficient has an accepted candidate preimage in the
consumed prefix of its own transcript-chained stream. -/
theorem coefficient_has_accepted_preimage
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {challengeCount bound : Nat}
    {initial : Digest}
    (batch : BatchExecution specification challengeCount bound initial)
    (coordinate : Fin challengeCount)
    (position : Fin specification.coefficientCount) :
    exists candidate,
      candidate ∈ FirstAccepted.streamPrefix
          (sourceAt specification initial coordinate.val).stream
          (batch.execution coordinate).consumed /\
        specification.verifier.accepts candidate = true /\
        specification.verifier.symbol candidate =
          coefficient (batch.execution coordinate) position := by
  have member := coefficient_mem_output (batch.execution coordinate) position
  rw [(batch.execution coordinate).reference.output_eq] at member
  exact FirstAccepted.mem_firstAccepted member

/-- The two-part strong-set bridge required by the actual protocol shape:
accepted chunks must decode to valid coefficients, and a complete valid
coefficient vector must assemble to a valid scalar challenge. -/
structure StrongSetLaw
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (validScalar : Scalar -> Prop) where
  validCoefficient : Coefficient -> Prop
  accepted_coefficient_valid : forall candidate,
    specification.verifier.accepts candidate = true ->
      validCoefficient (specification.verifier.symbol candidate)
  assembled_valid : forall coefficients,
    (forall position, validCoefficient (coefficients position)) ->
      validScalar (specification.assemble coefficients)

theorem coefficient_valid
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {validScalar : Scalar -> Prop}
    (strongSet : StrongSetLaw specification validScalar)
    {challengeCount bound : Nat}
    {initial : Digest}
    (batch : BatchExecution specification challengeCount bound initial)
    (coordinate : Fin challengeCount)
    (position : Fin specification.coefficientCount) :
    strongSet.validCoefficient
      (coefficient (batch.execution coordinate) position) := by
  rcases coefficient_has_accepted_preimage batch coordinate position with
    ⟨candidate, _, accepted, symbolEq⟩
  rw [← symbolEq]
  exact strongSet.accepted_coefficient_valid candidate accepted

/-- A complete accepted coefficient vector assembles to a valid PiRLC scalar. -/
theorem challenge_valid
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {validScalar : Scalar -> Prop}
    (strongSet : StrongSetLaw specification validScalar)
    {challengeCount bound : Nat}
    {initial : Digest}
    (batch : BatchExecution specification challengeCount bound initial)
    (coordinate : Fin challengeCount) :
    validScalar (challenge batch coordinate) := by
  exact strongSet.assembled_valid _ fun position =>
    coefficient_valid strongSet batch coordinate position

/-- Named bounded failure for one challenge coordinate. -/
def ShortfallAt
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (bound : Nat)
    (initial : Digest)
    (index : Nat) : Prop :=
  FirstAccepted.Shortfall specification.verifier
    specification.coefficientCount
    (FirstAccepted.streamPrefix
      (sourceAt specification initial index).stream bound)

/-- A fixed bound is available only when every challenge coordinate has a
bounded coefficient execution. -/
def Available
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (challengeCount bound : Nat)
    (initial : Digest) : Prop :=
  exists _batch : BatchExecution specification challengeCount bound initial,
    True

/-- Finite bounded sampling is total as an outcome: either every challenge
coordinate has its canonical bounded execution, or one exact coordinate has
too few accepted candidates. This is a logical completeness split, not a
claim that a hash-derived stream always succeeds. -/
theorem available_or_exists_shortfall
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    (specification : Specification Digest Candidate Coefficient Scalar)
    (challengeCount bound : Nat)
    (initial : Digest) :
    Available specification challengeCount bound initial \/
      Exists fun coordinate : Fin challengeCount =>
        ShortfallAt specification bound initial coordinate.val := by
  classical
  by_cases shortfall :
      Exists fun coordinate : Fin challengeCount =>
        ShortfallAt specification bound initial coordinate.val
  · exact Or.inr shortfall
  · apply Or.inl
    refine ⟨{ execution := fun coordinate => ?_ }, trivial⟩
    have noShortfall :
        ¬ ShortfallAt specification bound initial coordinate.val := by
      intro coordinateShortfall
      exact shortfall ⟨coordinate, coordinateShortfall⟩
    have enough :
        FirstAccepted.Enough specification.verifier
          specification.coefficientCount
          (FirstAccepted.streamPrefix
            (sourceAt specification initial coordinate.val).stream bound) := by
      unfold ShortfallAt FirstAccepted.Shortfall at noShortfall
      unfold FirstAccepted.Enough
      omega
    let output :=
      FirstAccepted.firstAccepted specification.verifier
        specification.coefficientCount
        (FirstAccepted.streamPrefix
          (sourceAt specification initial coordinate.val).stream bound)
    have success :
        FirstAccepted.boundedSample specification.verifier
            specification.coefficientCount
            (FirstAccepted.streamPrefix
              (sourceAt specification initial coordinate.val).stream bound) =
          some output := by
      exact FirstAccepted.boundedSample_eq_some_iff.mpr ⟨enough, rfl⟩
    exact Classical.choose
      (FirstAccepted.BoundedExecution.exists_of_bounded_success success)

/-- Availability excludes shortfall at each coordinate independently. -/
theorem available_excludes_shortfall
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {challengeCount bound : Nat}
    {initial : Digest}
    (available : Available specification challengeCount bound initial)
    (coordinate : Fin challengeCount) :
    ¬ ShortfallAt specification bound initial coordinate.val := by
  intro shortfall
  rcases available with ⟨batch, _⟩
  let execution := batch.execution coordinate
  have success :
      FirstAccepted.boundedSample specification.verifier
          specification.coefficientCount
          (FirstAccepted.streamPrefix
            (sourceAt specification initial coordinate.val).stream bound) =
        some execution.output :=
    FirstAccepted.boundedSample_eq_some_iff_boundedExecution.mpr
      ⟨execution, rfl⟩
  have failed :
      FirstAccepted.boundedSample specification.verifier
          specification.coefficientCount
          (FirstAccepted.streamPrefix
            (sourceAt specification initial coordinate.val).stream bound) =
        none :=
    FirstAccepted.boundedSample_eq_none_iff_shortfall.mpr shortfall
  rw [failed] at success
  contradiction

/-- Exact bridge from the replay response to the transcript-chained batch.
No scalar may be selected directly or sampled from an independent state. -/
def ResponseRefinesAt
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {challengeCount : Nat}
    (response : Digest -> Fin challengeCount -> Scalar)
    (specification : Specification Digest Candidate Coefficient Scalar)
    (bound : Nat)
    (initial : Digest) : Prop :=
  exists batch : BatchExecution specification challengeCount bound initial,
    forall coordinate,
      response initial coordinate = challenge batch coordinate

/-- Response refinement exposes every per-coordinate reference execution and
its least cursor inside the fixed bound. -/
theorem responseRefinesAt_implies_reference_within
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {challengeCount bound : Nat}
    {response : Digest -> Fin challengeCount -> Scalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {initial : Digest}
    (refinement : ResponseRefinesAt response specification bound initial) :
    exists batch : BatchExecution specification challengeCount bound initial,
      forall coordinate,
        (batch.execution coordinate).consumed <= bound /\
          FirstAccepted.ReferenceExecution specification.verifier
            specification.coefficientCount
            (sourceAt specification initial coordinate.val).stream
            (batch.execution coordinate).output
            (batch.execution coordinate).consumed /\
          response initial coordinate = challenge batch coordinate := by
  rcases refinement with ⟨batch, responseEq⟩
  exact ⟨batch, fun coordinate =>
    ⟨(batch.execution coordinate).within,
      (batch.execution coordinate).reference,
      responseEq coordinate⟩⟩

/-- Every refined replay response is a strong-set scalar, after both
coefficient validity and assembly validity have been proved. -/
theorem responseRefinesAt_valid
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {challengeCount bound : Nat}
    {response : Digest -> Fin challengeCount -> Scalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {validScalar : Scalar -> Prop}
    (strongSet : StrongSetLaw specification validScalar)
    {initial : Digest}
    (refinement : ResponseRefinesAt response specification bound initial)
    (coordinate : Fin challengeCount) :
    validScalar (response initial coordinate) := by
  rcases refinement with ⟨batch, responseEq⟩
  rw [responseEq coordinate]
  exact challenge_valid strongSet batch coordinate

/-- Shortfall in any one coordinate fail-closes the whole batch refinement. -/
theorem shortfall_excludes_responseRefinesAt
    {Digest : Type uDigest}
    {Candidate : Type uCandidate}
    {Coefficient : Type uCoefficient}
    {Scalar : Type uScalar}
    {challengeCount bound : Nat}
    {response : Digest -> Fin challengeCount -> Scalar}
    {specification : Specification Digest Candidate Coefficient Scalar}
    {initial : Digest}
    (coordinate : Fin challengeCount)
    (shortfall : ShortfallAt specification bound initial coordinate.val) :
    ¬ ResponseRefinesAt response specification bound initial := by
  intro refinement
  rcases refinement with ⟨batch, _⟩
  have available : Available specification challengeCount bound initial :=
    ⟨batch, trivial⟩
  exact (available_excludes_shortfall available coordinate) shortfall

end Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
