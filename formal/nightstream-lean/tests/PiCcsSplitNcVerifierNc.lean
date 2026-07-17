import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc

/-!
Focused type regressions for the canonical verifier-visible NC phase.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.verify.point` | typed point coordinates are exactly transcript-derived challenges | point reordering or caller override |
| `nifs.pi_ccs.nc.verify.chain` | executable/logical phase acceptance coincide | composition drift |
| `nifs.pi_ccs.nc.verify.soundness` | missing `yZcol` authority remains an explicit outcome | output terminal promoted to authority |
-/

namespace NightstreamTests.PiCcsSplitNcVerifierNc

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Nc

example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : OutputMessage shape)
    (certificate : Transcript.Nc.Certificate domain) :
    check machine initialState coins message certificate = true ↔
      Accepted machine initialState coins message certificate :=
  check_eq_true_iff_accepted machine initialState coins message certificate

example
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    {State : Type}
    (covers : domain.Covers shape)
    (data : Data shape)
    (machine : Transcript.Nc.Machine State)
    (initialState : State)
    (coins : Mixing.Coins domain)
    (message : OutputMessage shape)
    (certificate : Transcript.Nc.Certificate domain)
    (challengeSetSize : Nat)
    (accepted : Accepted machine initialState coins message certificate) :
    Semantics.Nc.Truth data ∨
      ¬ YZcolBoundToSources covers data
        ({ rPrime := data.priorPoint,
           sPrime := (derive machine initialState certificate).point.column } :
          VerifierPoints shape domain)
        message ∨
      SumCheck.Nc.BadEvent covers data coins
        (derive machine initialState certificate).point.coordinates
        certificate.toSumCheck challengeSetSize :=
  accepted_implies_truth_or_unbound_or_badEvent noZeroDivisors covers data
    machine initialState coins message certificate challengeSetSize accepted

end NightstreamTests.PiCcsSplitNcVerifierNc
