import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.SourceProjection
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.Gamma
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath

/-!
Challenge mixing for the canonical Split-NC block×lane polynomial.

Assurance tier: model-level.

Owns: typed block/lane/gamma coins, the paper-relative gamma compression of
source cubics, block and lane equality gating, and fail-closed polynomial
evaluation in the canonical block-then-lane coordinate order.

Does not own: alternative gamma conventions, source projection, the initial
claim, SumCheck messages, transcript derivation, packed `yZcol` binding, Rust,
R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: the polynomial consumes source MLEs derived from
authoritative assignments. Its coins are explicit semantic inputs; a later
Poseidon2 transcript refinement must derive them. This module does not treat
a prover-supplied digest or polynomial value as authority.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.mixing.coins` | one block point, lane point, and gamma | security boundary | `Coins` |
| `nifs.pi_ccs.nc.block_lane.mixing.source` | source `i` has exactly the paper-relative weight `gamma^i` | computed | `mixedRangeAt` |
| `nifs.pi_ccs.nc.block_lane.mixing.selector.block` | bind the verifier block point | computed | `qAtPoint` |
| `nifs.pi_ccs.nc.block_lane.mixing.selector.lane` | bind the verifier lane point | computed | `qAtPoint` |
| `nifs.pi_ccs.nc.block_lane.mixing.decode` | reject every wrong coordinate arity | checked | `polynomial` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane

private abbrev ops := ConcreteCarrier.extensionOps

/-- Verifier challenges consumed by the canonical NC polynomial. -/
structure Coins (domain : BlockNcDomain) where
  betaBlock : CubePoint K domain.blockVariables
  betaA : CubePoint K domain.laneVariables
  gamma : K

/-- Gamma compression of all source cubics at one canonical block×lane
point. The cubic is evaluated after source interpolation at this point. -/
def mixedRangeAt
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) : K :=
  FiniteSumAlgebra.sumMap ops
    (canonicalFinIndices shape.sourceCount) fun source =>
      SignedJointIdentity.gammaTerm ops coins.gamma
        (Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Mixing.sourceExponent
          shape .paperNc source)
        (SourceProjection.rangeValueAt covers data source point)

/-- Equality-gated NC polynomial over the block×lane product domain. -/
def qAtPoint
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) : K :=
  K.mul
    (K.mul
      (SumCheckTruthPath.pointEquality ops point.block coins.betaBlock)
      (SumCheckTruthPath.pointEquality ops point.lane coins.betaA))
    (mixedRangeAt covers data coins point)

/-- Fail-closed evaluator in the fixed block-then-lane order. -/
def polynomial
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K) : Option K :=
  match Point.decode (domain := domain) coordinates with
  | some point => some (qAtPoint covers data coins point)
  | none => none

/-- Exact typed serialization evaluates the same NC polynomial. -/
theorem polynomial_coordinates_eq_qAtPoint
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (point : Point domain) :
    polynomial covers data coins point.coordinates =
      some (qAtPoint covers data coins point) := by
  rw [polynomial, Point.decode_coordinates]

/-- Malformed coordinate arity rejects rather than truncating or padding. -/
theorem polynomial_eq_none_of_length_ne
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Coins domain)
    (coordinates : List K)
    (different : coordinates.length ≠
      domain.blockVariables + domain.laneVariables) :
    polynomial covers data coins coordinates = none := by
  rw [polynomial, Point.decode_eq_none_of_length_ne coordinates different]

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Mixing
