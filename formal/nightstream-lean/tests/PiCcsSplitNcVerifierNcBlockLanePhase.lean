import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane

/-!
Focused regressions for the canonical block×lane NC phase evaluator.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.verify.chain` | executable and logical phase acceptance coincide | divergent verifier path |
| `nifs.pi_ccs.nc.block_lane.verify.soundness` | acceptance yields truth, explicit output failure, or a named algebraic event | digest/scalar terminal promoted to authority |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

universe uState

/-- One verifier implementation owns both executable and logical replay. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : Claims shape)
    (certificate : Transcript.Nc.BlockLane.Certificate domain) :
    check machine initialState coins message certificate = true ↔
      Accepted machine initialState coins message certificate :=
  check_eq_true_iff_accepted machine initialState coins message certificate

/-- Packed output remains an explicit full-lane authority boundary in the
deterministic soundness conclusion. -/
example
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    {State : Type uState}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : Claims shape)
    (certificate : Transcript.Nc.BlockLane.Certificate domain)
    (challengeSetSize : Nat)
    (accepted : Accepted machine initialState coins message certificate) :
    Semantics.Nc.Truth data ∨
      ¬ Terminal.PackedYZcolBoundAtBlock covers data
        (derive machine initialState certificate).point.block message ∨
      SumCheck.Nc.BlockLane.BadEvent covers data coins
        (derive machine initialState certificate).point
        certificate.toSumCheck challengeSetSize :=
  accepted_implies_truth_or_unbound_or_badEvent noZeroDivisors covers data
    machine initialState coins message certificate challengeSetSize accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc.BlockLane.Tests
