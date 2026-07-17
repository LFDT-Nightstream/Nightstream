import Nightstream.SuperNeo.Folding.PiCCS.SplitNc

/-!
Focused regressions for the independent Split-NC NC terminal bridge.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.terminal.output.lane.active` | an active lane is read without reindexing | active-lane permutation |
| `nifs.pi_ccs.nc.terminal.output.binding` | source-bound `yZcol` evaluates to the independent nested MLE | legacy-output/source-projection drift |
| `nifs.pi_ccs.nc.terminal.equality` | all three named gamma conventions share the exact terminal theorem | implicit production convention |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal

/-- Active output lanes retain their exact source/lane coordinate. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (message : OutputMessage shape)
    (source : Fin shape.sourceCount)
    (lane : Fin ringDegree) :
    paddedYZcol (domain := domain) message source
        (domain.phi81Lane covers lane) = message.yZcol source lane :=
  paddedYZcol_live covers message source lane

/-- The paper-relative terminal consumes the same source-bound output bridge. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : OutputMessage shape)
    (bound : YZcolBoundToSources covers data
      ({ rPrime := data.priorPoint, sPrime := point.column } :
        VerifierPoints shape domain)
      message) :
    terminalFromMessage .paperNc message coins point =
      Mixing.qAtPoint .paperNc covers data coins point :=
  terminal_eq_qAtPoint_of_yZcolBoundToSources
    .paperNc covers data coins point message bound

/-- The paper-joint shifted convention remains explicit. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : OutputMessage shape)
    (bound : YZcolBoundToSources covers data
      ({ rPrime := data.priorPoint, sPrime := point.column } :
        VerifierPoints shape domain)
      message) :
    terminalFromMessage .paperJointQ message coins point =
      Mixing.qAtPoint .paperJointQ covers data coins point :=
  terminal_eq_qAtPoint_of_yZcolBoundToSources
    .paperJointQ covers data coins point message bound

/-- The diagnostic Split-V1 convention uses the same binding theorem without
being silently promoted to paper authority. -/
example
    {shape : SemanticShape}
    {domain : FlatNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : OutputMessage shape)
    (bound : YZcolBoundToSources covers data
      ({ rPrime := data.priorPoint, sPrime := point.column } :
        VerifierPoints shape domain)
      message) :
    terminalFromMessage .splitV1 message coins point =
      Mixing.qAtPoint .splitV1 covers data coins point :=
  terminal_eq_qAtPoint_of_yZcolBoundToSources
    .splitV1 covers data coins point message bound

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Terminal.Tests
