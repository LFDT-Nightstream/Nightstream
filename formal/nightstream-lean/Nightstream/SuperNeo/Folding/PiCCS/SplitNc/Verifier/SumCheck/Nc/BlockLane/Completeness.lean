import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane.Semantics

/-!
Algebraic completeness for canonical block×lane NC SumCheck.

Assurance tier: model-level.

Owns: construction of an accepted five-slot certificate from independent NC
truth after a typed verifier challenge point is fixed, and executable/logical
checker parity for that path.

Does not own: Fiat–Shamir challenge derivation, exact fixed-profile domain
selection, packed-output terminal binding, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: the certificate is constructed from semantic expected
rounds of the independent polynomial. The theorem is post-challenge algebraic
completeness, not a transcript prover and not a production conformance claim.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.sumcheck.complete.logical` | independent NC truth has an accepted five-slot certificate | derived | `complete_of_truth` |
| `nifs.pi_ccs.nc.block_lane.sumcheck.complete.executable` | the executable checker accepts the same certificate | derived | `check_complete_of_truth` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev ops := ConcreteCarrier.extensionOps

/-- Honest full-carrier NC truth has a complete five-slot certificate at
every typed verifier point. -/
theorem complete_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data)
    (point : Point domain) :
    ∃ certificate : Certificate,
      Accepted InitialSum.claimedInitial point.coordinates
        (InitialSum.sumcheckPolynomial covers data coins point.coordinates)
        certificate := by
  let polynomial := InitialSum.sumcheckPolynomial covers data coins
  have representable :
      FixedPhase.ExpectedRoundsRepresentable ops.toOps polynomial
        Polynomial.Nc.Degree.ncSumcheckDegreeBound point.coordinates := by
    simpa [polynomial] using expectedRoundsRepresentable
      covers data coins point
  rcases FixedPhase.exists_honest_certificate ops.toOps polynomial
      Polynomial.Nc.Degree.ncSumcheckDegreeBound point.coordinates
      representable with ⟨certificate, honest⟩
  have initialIsTrue :
      InitialSum.claimedInitial =
        FixedPhase.semanticInitial ops.toOps polynomial
          point.coordinates.length := by
    rw [InitialSum.claimedInitial_eq_sumcheckHypercubeSum_of_truth
      covers data coins truth]
    unfold InitialSum.sumcheckHypercubeSum FixedPhase.semanticInitial
    rw [point.coordinates_length]
  refine ⟨certificate, ?_⟩
  unfold Accepted
  exact FixedPhase.complete ops.toOps polynomial InitialSum.claimedInitial
    point.coordinates certificate initialIsTrue honest

/-- The executable fixed-width checker is perfectly complete for the same
post-challenge algebraic path. -/
theorem check_complete_of_truth
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (truth : Semantics.Nc.Truth data)
    (point : Point domain) :
    ∃ certificate : Certificate,
      check InitialSum.claimedInitial point.coordinates
        (InitialSum.sumcheckPolynomial covers data coins point.coordinates)
        certificate = true := by
  rcases complete_of_truth covers data coins truth point with
    ⟨certificate, accepted⟩
  exact ⟨certificate,
    (check_eq_true_iff_accepted InitialSum.claimedInitial point.coordinates
      (InitialSum.sumcheckPolynomial covers data coins point.coordinates)
      certificate).2 accepted⟩

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.SumCheck.Nc.BlockLane
