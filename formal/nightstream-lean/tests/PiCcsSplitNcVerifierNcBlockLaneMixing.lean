import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing

/-!
Focused regressions for canonical block×lane NC source mixing.

| Stage path | Property under test | Failure caught |
|---|---|---|
| `nifs.pi_ccs.nc.block_lane.mixing.source` | source `i` has exactly weight `gamma^i` | legacy exponent convention entering the canonical verifier |
| `nifs.pi_ccs.nc.block_lane.mixing.decode` | exact points round-trip and malformed arity rejects | coordinate-order or implicit-padding drift |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing.Tests

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

private abbrev ops := ConcreteCarrier.extensionOps

/-- The canonical verifier has one gamma schedule: relative paper NC. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    mixedRangeAt covers data coins point =
      FiniteSumAlgebra.sumMap ops
        (canonicalFinIndices shape.sourceCount) (fun source =>
          SignedJointIdentity.gammaTerm ops coins.gamma
            (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.sourceExponent
              shape .paperNc source)
            (SourceProjection.rangeValueAt covers data source point)) := by
  rfl

/-- Exact typed serialization evaluates the same equality-gated source mix. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    polynomial covers data coins point.coordinates =
      some (qAtPoint covers data coins point) :=
  polynomial_coordinates_eq_qAtPoint covers data coins point

/-- Wrong-arity block×lane points reject. -/
example
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K)
    (different : coordinates.length ≠
      domain.blockVariables + domain.laneVariables) :
    polynomial covers data coins coordinates = none :=
  polynomial_eq_none_of_length_ne
    covers data coins coordinates different

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing.Tests
