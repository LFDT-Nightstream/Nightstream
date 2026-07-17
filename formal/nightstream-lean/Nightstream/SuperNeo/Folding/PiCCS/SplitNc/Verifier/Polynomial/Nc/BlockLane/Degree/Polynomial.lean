import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Source

/-!
Quartic bound for the canonical Split-NC block×lane polynomial.

Assurance tier: model-level.

Owns: affinity of each equality-selector coordinate, multiplication of that
selector by the cubic source mix, the resulting individual-degree-four
bound, and projection to exactly five constant-first message coefficients.

Does not own: Boolean suffix summation, recursive SumCheck rounds, transcript
encoding, canonical wire padding, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: each coefficient representation is constructed from the
source-derived polynomial and verifier-owned coins. Five slots are a derived
width (`degree + 1`), never a prover-supplied degree claim.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.degree.selector.block` | one block equality coordinate is affine | derived | `pointEquality_block_affine` |
| `nifs.pi_ccs.nc.block_lane.degree.selector.lane` | one lane equality coordinate is affine | derived | `pointEquality_lane_affine` |
| `nifs.pi_ccs.nc.block_lane.degree.polynomial.block` | block slices are degree at most four | derived | `qAtPoint_block_quartic` |
| `nifs.pi_ccs.nc.block_lane.degree.polynomial.lane` | lane slices are degree at most four | derived | `qAtPoint_lane_quartic` |
| `nifs.pi_ccs.nc.block_lane.degree.message.block` | a typed block slice has five constant-first slots | derived | `qAtPoint_block_has_five_coefficients` |
| `nifs.pi_ccs.nc.block_lane.degree.message.lane` | a typed lane slice has five constant-first slots | derived | `qAtPoint_lane_has_five_coefficients` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Source

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev laws := ConcreteCarrier.extensionLaws
private abbrev Polynomial := SumCheck.Finite.FixedPolynomial K

/-- The block equality selector is affine in each block coordinate. -/
theorem pointEquality_block_affine
    {domain : BlockNcDomain}
    (coins : Mixing.Coins domain)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.betaBlock := by
  unfold SumCheckTruthPath.pointEquality
  apply pointEqualityCoordinates_affine
  rw [coins.betaBlock.dimension]
  exact length

/-- The lane equality selector is affine in each lane coordinate. -/
theorem pointEquality_lane_affine
    {domain : BlockNcDomain}
    (coins : Mixing.Coins domain)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial 1, ∀ point,
      polynomial.evaluate ops.toOps point =
        SumCheckTruthPath.pointEquality ops
          (cubeSlice before after length point) coins.betaA := by
  unfold SumCheckTruthPath.pointEquality
  apply pointEqualityCoordinates_affine
  rw [coins.betaA.dimension]
  exact length

/-- Each block-coordinate slice of the complete equality-gated polynomial
has individual degree at most four. -/
theorem qAtPoint_block_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ polynomial : Polynomial ncSumcheckDegreeBound, ∀ point,
      polynomial.evaluate ops.toOps point =
        Mixing.qAtPoint covers data coins {
          block := cubeSlice before after length point
          lane := lane } := by
  rcases pointEquality_block_affine coins before after length with
    ⟨selectorPolynomial, selectorRepresents⟩
  rcases mixedRangeAt_block_cubic covers data coins lane
    before after length with ⟨rangePolynomial, rangeRepresents⟩
  let laneSelector :=
    SumCheckTruthPath.pointEquality ops lane coins.betaA
  refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps laneSelector
    (SumCheck.Finite.FixedPolynomial.mul ops.toOps
      selectorPolynomial rangePolynomial), ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_scale ops.toOps polynomialLaws]
  calc
    ops.mul laneSelector
        ((SumCheck.Finite.FixedPolynomial.mul ops.toOps
          selectorPolynomial rangePolynomial).evaluate ops.toOps point) =
      ops.mul laneSelector
        (ops.mul (selectorPolynomial.evaluate ops.toOps point)
          (rangePolynomial.evaluate ops.toOps point)) :=
      congrArg (ops.mul laneSelector)
        (SumCheck.Finite.FixedPolynomial.evaluate_mul ops.toOps polynomialLaws
          selectorPolynomial rangePolynomial point)
    _ = Mixing.qAtPoint covers data coins {
          block := cubeSlice before after length point
          lane := lane } := by
      rw [selectorRepresents, rangeRepresents]
      unfold Mixing.qAtPoint
      dsimp only [laneSelector]
      let blockSelector := SumCheckTruthPath.pointEquality ops
        (cubeSlice before after length point) coins.betaBlock
      let rangeValue := Mixing.mixedRangeAt covers data coins {
        block := cubeSlice before after length point
        lane := lane }
      change ops.mul laneSelector (ops.mul blockSelector rangeValue) =
        ops.mul (ops.mul blockSelector laneSelector) rangeValue
      calc
        ops.mul laneSelector (ops.mul blockSelector rangeValue) =
            ops.mul (ops.mul laneSelector blockSelector) rangeValue :=
          (laws.mul_assoc _ _ _).symm
        _ = ops.mul (ops.mul blockSelector laneSelector) rangeValue := by
          rw [laws.mul_comm laneSelector blockSelector]

/-- Each lane-coordinate slice of the complete equality-gated polynomial
has individual degree at most four. -/
theorem qAtPoint_lane_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ polynomial : Polynomial ncSumcheckDegreeBound, ∀ point,
      polynomial.evaluate ops.toOps point =
        Mixing.qAtPoint covers data coins {
          block := block
          lane := cubeSlice before after length point } := by
  rcases pointEquality_lane_affine coins before after length with
    ⟨selectorPolynomial, selectorRepresents⟩
  rcases mixedRangeAt_lane_cubic covers data coins block
    before after length with ⟨rangePolynomial, rangeRepresents⟩
  let blockSelector :=
    SumCheckTruthPath.pointEquality ops block coins.betaBlock
  refine ⟨SumCheck.Finite.FixedPolynomial.scale ops.toOps blockSelector
    (SumCheck.Finite.FixedPolynomial.mul ops.toOps
      selectorPolynomial rangePolynomial), ?_⟩
  intro point
  rw [SumCheck.Finite.FixedPolynomial.evaluate_scale ops.toOps polynomialLaws]
  calc
    ops.mul blockSelector
        ((SumCheck.Finite.FixedPolynomial.mul ops.toOps
          selectorPolynomial rangePolynomial).evaluate ops.toOps point) =
      ops.mul blockSelector
        (ops.mul (selectorPolynomial.evaluate ops.toOps point)
          (rangePolynomial.evaluate ops.toOps point)) :=
      congrArg (ops.mul blockSelector)
        (SumCheck.Finite.FixedPolynomial.evaluate_mul ops.toOps polynomialLaws
          selectorPolynomial rangePolynomial point)
    _ = Mixing.qAtPoint covers data coins {
          block := block
          lane := cubeSlice before after length point } := by
      rw [selectorRepresents, rangeRepresents]
      unfold Mixing.qAtPoint
      dsimp only [blockSelector]
      exact (laws.mul_assoc _ _ _).symm

/-- A block-coordinate slice projects to five constant-first raw message
slots. The degree-four slot may evaluate to zero. -/
theorem qAtPoint_block_has_five_coefficients
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (lane : CubePoint K domain.laneVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.blockVariables) :
    ∃ message : SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point =
        Mixing.qAtPoint covers data coins {
          block := cubeSlice before after length point
          lane := lane } := by
  exact Represents.message_shape
    (qAtPoint_block_quartic covers data coins lane before after length)

/-- A lane-coordinate slice projects to the same five-slot shape. The
degree-four slot may evaluate to zero. -/
theorem qAtPoint_lane_has_five_coefficients
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (block : CubePoint K domain.blockVariables)
    (before after : List K)
    (length : before.length + 1 + after.length = domain.laneVariables) :
    ∃ message : SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point =
        Mixing.qAtPoint covers data coins {
          block := block
          lane := cubeSlice before after length point } := by
  exact Represents.message_shape
    (qAtPoint_lane_quartic covers data coins block before after length)

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial
