import Nightstream.SuperNeo.InteractiveReduction.Paper

/-!
Generic coordinate-wise forking boundary from SuperNeo Appendix C, Theorem 10.

Owns: actual scalar challenge vectors, vector-at-once oracle answers, one base
query plus one coordinate fork, the corrected Appendix-D.5 loss shape, and the
generic probability/EPT contract still required from a finite-uniform forking
theorem.

Does not own: `Pi_RLC`, CE semantics, source validity, ambient extraction,
relaxed binding, a proof of the finite-uniform theorem, Fiat--Shamir, Rust,
R1CS, or costs.

Emits constraints: no.

The contract below is deliberately generic.  A theorem conditional on it is
not an unconditional proof of the `Pi_RLC` weak reduction until its
finite-uniform lower bound and expected-query clauses are instantiated.  The
loss uses Appendix D.5's `(ell + 1) / |C|`, not the conflicting rendered
`ell / |C^ell|` in the local statement of Theorem 10.
-/

namespace Nightstream.SuperNeo.InteractiveReduction.CoordinateForking

open Nightstream.SuperNeo.InteractiveReduction.Paper

universe uWeight uScalar uAnswer uAdversary uExtractor

/-- The verifier's complete challenge vector. -/
abbrev ChallengeVector (Scalar : Type uScalar) (coordinates : Nat) :=
  Fin coordinates -> Scalar

/-- One vector-at-once oracle.  There is no coordinate-curried answer path. -/
abbrev Oracle
    (Scalar : Type uScalar)
    (Answer : Type uAnswer)
    (coordinates : Nat) :=
  ChallengeVector Scalar coordinates -> Answer

/-- One base query and one query forked at each challenge coordinate. -/
structure ForkSample (Scalar : Type uScalar) (coordinates : Nat) where
  base : ChallengeVector Scalar coordinates
  forks : Fin coordinates -> ChallengeVector Scalar coordinates

/-- The exact special-set success predicate from Theorem 10.  Oracle answers
are obtained by applying the same vector-at-once oracle to every queried
vector. -/
structure AcceptedCoordinateFork
    {Scalar : Type uScalar}
    {Answer : Type uAnswer}
    {coordinates : Nat}
    (valid : Scalar -> Prop)
    (verify : ChallengeVector Scalar coordinates -> Answer -> Prop)
    (oracle : Oracle Scalar Answer coordinates)
    (sample : ForkSample Scalar coordinates) : Prop where
  baseAccepted : verify sample.base (oracle sample.base)
  forkAccepted : forall coordinate,
    verify (sample.forks coordinate) (oracle (sample.forks coordinate))
  baseValid : forall index, valid (sample.base index)
  forkValid : forall coordinate index,
    valid (sample.forks coordinate index)
  agreeExcept : forall coordinate index, index ≠ coordinate ->
    sample.base index = sample.forks coordinate index
  changed : forall coordinate,
    sample.base coordinate ≠ sample.forks coordinate coordinate

/-- Appendix D.5's selected coordinate-fork loss `(ell + 1) / |C|`.
`ratio` is supplied by the concrete probability scale. -/
def correctedLoss
    {Weight : Type uWeight}
    (ratio : Nat -> Nat -> Weight)
    (coordinates challengeSetCardinality : Nat) : Weight :=
  ratio (coordinates + 1) challengeSetCardinality

/-- The still-open generic finite-uniform coordinate-forking theorem.

The challenge and fork experiments are concrete members of the contract, so
the lower bound cannot be silently moved to a different distribution.  The
only verifier predicate in the lower bound is the supplied generic `verify`;
there is no `Pi_RLC` result, source witness, or relaxed-binding event here.

`challengeSetCardinality_pos` keeps division-by-zero outside the contract.
The deployment's `ratio` operation owns its remaining arithmetic totality
laws. -/
structure Theorem10Contract
    {Weight : Type uWeight}
    {Scalar : Type uScalar}
    {Answer : Type uAnswer}
    {Adversary : Type uAdversary}
    {Extractor : Type uExtractor}
    (scale : ProbabilityScale Weight)
    (ratio : Nat -> Nat -> Weight)
    (coordinates : Nat)
    (valid : Scalar -> Prop)
    (verify : Adversary -> ChallengeVector Scalar coordinates -> Answer -> Prop)
    (oracle : Adversary -> Oracle Scalar Answer coordinates) where
  challengeSetCardinality : Nat
  challengeSetCardinality_pos : 0 < challengeSetCardinality
  uniformChallenges : Adversary ->
    ProbabilityExperiment scale (ChallengeVector Scalar coordinates)
  forkSamples : Adversary -> Extractor ->
    ProbabilityExperiment scale (ForkSample Scalar coordinates)
  coordinateExtractor : Adversary -> Extractor
  adversaryExpectedPolynomialTime : Adversary -> Prop
  extractorExpectedPolynomialTime : Adversary -> Extractor -> Prop
  expectedQueriesAtMost : Adversary -> Extractor -> Nat -> Prop
  extractor_ept : forall adversary,
    adversaryExpectedPolynomialTime adversary ->
    extractorExpectedPolynomialTime adversary (coordinateExtractor adversary)
  expectedQueries_le : forall adversary,
    adversaryExpectedPolynomialTime adversary ->
    expectedQueriesAtMost adversary (coordinateExtractor adversary)
      (coordinates + 1)
  lowerBound : forall adversary,
    adversaryExpectedPolynomialTime adversary ->
    scale.le
      (scale.subtract
        ((uniformChallenges adversary).probability fun challenges =>
          verify adversary challenges (oracle adversary challenges))
        (correctedLoss ratio coordinates challengeSetCardinality))
      ((forkSamples adversary (coordinateExtractor adversary)).probability
        fun sample =>
          AcceptedCoordinateFork valid (verify adversary)
            (oracle adversary) sample)

end Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
