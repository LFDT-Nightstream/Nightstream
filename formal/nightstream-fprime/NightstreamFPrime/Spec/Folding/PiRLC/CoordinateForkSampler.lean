import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtraction
import Mathlib.Logic.Function.Basic

/-!
Typed output of one interactive PiRLC coordinate-retry extractor. Candidates
carry a valid challenge and an actual oracle response assignment. The first
accepted response ends the search; a repeated base challenge causes failure.
Successful per-coordinate searches construct the existing `CompleteFork`.

`check` must decide the exact verifier-computed CE(B) response relation. It is
not a prover-supplied success flag. Distribution and expected-call bounds are
owned by `CoordinateRetry` and `CoordinateForkProbability`.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkSampler

open NightstreamFPrime.Spec
open PaperForkExtraction

variable {Structure Assignment PublicInput Point Evaluation Commitment Scalar : Type*}
  {semantics : RelationSemantics
    Structure Assignment PublicInput Point Evaluation Commitment}
  {params : GlobalParams} {arity : BatchArity params}
  (algebra : Algebra Structure Assignment PublicInput Point Evaluation Commitment
    Scalar semantics params)
  (batch : InputBatch Structure PublicInput Point Evaluation Commitment params arity)

/-- One fresh oracle call on the changed coordinate. -/
abbrev Candidate := {challenge : Scalar // algebra.challengeValid challenge} × Assignment

/-- All unchanged coordinates are taken from the base response. -/
def response (base : Fin arity.total → Scalar) (coordinate : Fin arity.total)
    (candidate : Candidate algebra) : Response Assignment Scalar params arity where
  challenges := Function.update base coordinate candidate.1.val
  assignment := candidate.2

/-- Observe a finite prefix of the retry stream. This has no fixed production
sampler window. A prefix with no accepted response returns `none`. -/
def firstResponse (check : Response Assignment Scalar params arity → Bool)
    (base : Fin arity.total → Scalar) (coordinate : Fin arity.total)
    (candidates : List (Candidate algebra)) : Option (Response Assignment Scalar params arity) :=
  (candidates.find? fun candidate => check (response algebra base coordinate candidate)).map
    (response algebra base coordinate)

/-- Stop at first acceptance and reject a repeated base challenge. -/
def forkResponse [DecidableEq Scalar]
    (check : Response Assignment Scalar params arity → Bool)
    (base : Fin arity.total → Scalar) (coordinate : Fin arity.total)
    (candidates : List (Candidate algebra)) : Option (Response Assignment Scalar params arity) :=
  (firstResponse algebra check base coordinate candidates).bind fun found =>
    if base coordinate ≠ found.challenges coordinate then some found else none

private theorem firstResponse_some
    (check : Response Assignment Scalar params arity → Bool)
    (base : Fin arity.total → Scalar) (coordinate : Fin arity.total)
    (candidates : List (Candidate algebra))
    (found : Response Assignment Scalar params arity)
    (sampled : firstResponse algebra check base coordinate candidates = some found) :
    ∃ candidate, found = response algebra base coordinate candidate ∧ check found = true := by
  unfold firstResponse at sampled
  cases searched : candidates.find?
      (fun candidate => check (response algebra base coordinate candidate)) with
  | none => simp [searched] at sampled
  | some candidate =>
      simp only [searched, Option.map_some, Option.some.injEq] at sampled
      refine ⟨candidate, sampled.symm, ?_⟩
      rw [← sampled]
      exact List.find?_some
        (p := fun value => check (response algebra base coordinate value)) searched

/-- Every returned fork opens the actual combined output, has valid challenges,
changes the named coordinate, and agrees with the base at all other coordinates. -/
theorem forkResponse_sound [DecidableEq Scalar]
    (check : Response Assignment Scalar params arity → Bool)
    (check_spec : ∀ found, check found = true ↔
      found.Success semantics params algebra batch)
    (base : Fin arity.total → Scalar)
    (baseStrong : ∀ index, algebra.challengeValid (base index))
    (coordinate : Fin arity.total) (candidates : List (Candidate algebra))
    (found : Response Assignment Scalar params arity)
    (sampled : forkResponse algebra check base coordinate candidates = some found) :
    found.Success semantics params algebra batch ∧
      (∀ index, algebra.challengeValid (found.challenges index)) ∧
      PaperForkAlgebra.AgreeExcept coordinate base found.challenges ∧
      base coordinate ≠ found.challenges coordinate := by
  unfold forkResponse at sampled
  cases first : firstResponse algebra check base coordinate candidates with
  | none => simp [first] at sampled
  | some firstFound =>
      simp only [first, Option.bind_some] at sampled
      split at sampled
      next changed =>
        have same : firstFound = found := Option.some.inj sampled
        subst firstFound
        rcases firstResponse_some algebra check base coordinate candidates found first with
          ⟨candidate, equal, accepted⟩
        refine ⟨(check_spec found).mp accepted, ?_, ?_, changed⟩
        · intro index
          rw [equal]
          change algebra.challengeValid (Function.update base coordinate candidate.1.val index)
          by_cases atCoordinate : index = coordinate
          · subst index
            simpa using candidate.1.property
          · simpa [Function.update_of_ne atCoordinate] using baseStrong index
        · intro index different
          rw [equal]
          exact (Function.update_of_ne different candidate.1.val base).symm
      next unchanged => simp at sampled

/-- The successful extractor output is the existing operational fork consumed
by the inverse-difference extractor. No source witness is supplied here. -/
def completeForkOfSamples [DecidableEq Scalar]
    (check : Response Assignment Scalar params arity → Bool)
    (check_spec : ∀ found, check found = true ↔
      found.Success semantics params algebra batch)
    (base : Response Assignment Scalar params arity)
    (baseSuccess : base.Success semantics params algebra batch)
    (baseStrong : ∀ index, algebra.challengeValid (base.challenges index))
    (candidates : Fin arity.total → List (Candidate algebra))
    (forks : Fin arity.total → Response Assignment Scalar params arity)
    (sampled : ∀ coordinate,
      forkResponse algebra check base.challenges coordinate (candidates coordinate) =
        some (forks coordinate)) : CompleteFork semantics params algebra batch where
  base := base
  forks := forks
  baseSuccess := baseSuccess
  baseStrong := baseStrong
  forkSuccess coordinate :=
    (forkResponse_sound algebra batch check check_spec base.challenges baseStrong
      coordinate (candidates coordinate) (forks coordinate) (sampled coordinate)).1
  forkStrong coordinate :=
    (forkResponse_sound algebra batch check check_spec base.challenges baseStrong
      coordinate (candidates coordinate) (forks coordinate) (sampled coordinate)).2.1
  agreeExcept coordinate :=
    (forkResponse_sound algebra batch check check_spec base.challenges baseStrong
      coordinate (candidates coordinate) (forks coordinate) (sampled coordinate)).2.2.1
  changed coordinate :=
    (forkResponse_sound algebra batch check check_spec base.challenges baseStrong
      coordinate (candidates coordinate) (forks coordinate) (sampled coordinate)).2.2.2

end NightstreamFPrime.Spec.Folding.PiRLC.CoordinateForkSampler
