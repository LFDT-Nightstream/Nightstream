import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc

/-!
Focused type regression for the Split-NC output-authority composition.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.output_terminal.soundness` | raw-message acceptance names missing `yZcol` source binding separately from algebraic bad events | output digest or terminal scalar promoted to semantic authority |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc.Tests

open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc

example
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : OutputMessage shape)
    (certificate : SumCheck.Nc.Certificate)
    (challengeSetSize : Nat)
    (accepted :
      SumCheck.Nc.Accepted InitialSum.claimedInitial point.coordinates
        (Terminal.terminalFromMessage .paperNc message coins point)
        certificate) :
    Semantics.Nc.Truth data ∨
      ¬ YZcolBoundToSources covers data
        ({ rPrime := data.priorPoint, sPrime := point.column } :
          VerifierPoints shape domain)
        message ∨
      SumCheck.Nc.BadEvent covers data coins point.coordinates certificate
        challengeSetSize :=
  acceptedFromMessage_implies_truth_or_unbound_or_badEvent
    noZeroDivisors covers data coins point message certificate
    challengeSetSize accepted

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.OutputAuthority.Nc.Tests
