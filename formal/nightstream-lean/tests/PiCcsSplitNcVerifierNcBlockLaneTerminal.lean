import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal

/-!
Focused regressions for the canonical block×lane NC packed terminal.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.terminal.output.padding` | inactive lanes are computed zero | prover-controlled padding |
| `nifs.pi_ccs.nc.block_lane.terminal.output.mle` | bound packed output equals the source projection at the final block point | binding at `betaBlock` or lane-order drift |
| `nifs.pi_ccs.nc.block_lane.terminal.equality` | full lane binding implies the independent terminal | treating scalar equality or a digest as authority |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

/-- Padded output lanes never come from the certificate. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (message : Claims shape)
    (source : Fin shape.sourceCount)
    (lane : Fin domain.laneCount)
    (padding : ringDegree ≤ lane.val) :
    paddedYZcol message source lane = K.zero :=
  paddedYZcol_padding message source lane padding

/-- Full active-lane binding recovers the nested authoritative source MLE at
the same typed final point. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (point : Point domain)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data point.block message)
    (source : Fin shape.sourceCount) :
    valueAt (domain := domain) message source point.lane =
      SourceProjection.sourceValueAt covers data source point :=
  valueAt_eq_sourceValueAt_of_bound
    covers data point message bound source

/-- Authority flows from complete binding to terminal equality, never in the
opposite direction. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (point : Point domain)
    (message : Claims shape)
    (bound : PackedYZcolBoundAtBlock covers data point.block message) :
    terminalFromMessage message coins point =
      Mixing.qAtPoint covers data coins point :=
  terminal_eq_qAtPoint_of_bound covers data coins point message bound

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Terminal.Tests
