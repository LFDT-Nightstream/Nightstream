import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.InitialSum

/-!
Recursive-round degree bridge for canonical Split-NC block×lane SumCheck.

Assurance tier: model-level.

Owns: splitting an exact flattened coordinate slice at the block/lane
boundary, lifting typed quartic bounds to the total SumCheck evaluator,
preserving the bound under Boolean suffix sums, and deriving five slots for
every honest expected round.

Does not own: certificate round count, transcript replay, message parsing,
terminal binding, Rust, R1CS, costs, or row removal.

Emits constraints: no.

Authority boundary: exact arity determines whether a flattened coordinate is
a block or lane slice. The totalized evaluator's malformed branch is never
used in these theorems. Round coefficients are derived from the semantic
polynomial; no prover-supplied degree is trusted.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.block_lane.degree.sumcheck.decode.block` | a prefix inside block coordinates decodes to one block slice | derived | `ofCoordinates_eq_blockSlice` |
| `nifs.pi_ccs.nc.block_lane.degree.sumcheck.decode.lane` | a later prefix decodes to one lane slice | derived | `ofCoordinates_eq_laneSlice` |
| `nifs.pi_ccs.nc.block_lane.degree.sumcheck.total` | every exact total-polynomial coordinate slice is quartic | derived | `sumcheckPolynomial_slice_quartic` |
| `nifs.pi_ccs.nc.block_lane.degree.sumcheck.suffix` | Boolean suffix summation preserves the quartic ceiling | derived | `expectedRound_quartic` |
| `nifs.pi_ccs.nc.block_lane.degree.sumcheck.message` | every honest expected round has five slots | derived | `expectedRound_has_five_coefficients` |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.DegreeSupport
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.Polynomial

private abbrev ops := ConcreteCarrier.extensionOps
private abbrev Polynomial := Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial K

/-- A scalar function has a degree-four, five-slot representation. The high
slot may be zero. -/
abbrev RepresentsAtMostFour (function : K → K) : Prop :=
  DegreeSupport.Represents ncSumcheckDegreeBound function

/-- Every degree-four representation projects to five verifier-visible
constant-first slots. -/
theorem representsAtMostFour_message_shape
    {function : K → K}
    (represented : RepresentsAtMostFour function) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point = function point := by
  exact DegreeSupport.Represents.message_shape represented

private theorem cubePoint_eq_of_coordinates_eq
    {variables : Nat}
    (left right : CubePoint K variables)
    (coordinates : left.coordinates = right.coordinates) :
    left = right := by
  cases left
  cases right
  simp_all

private theorem ofCoordinates_eq_blockSlice
    {domain : BlockNcDomain}
    (before after : List K)
    (beforeBlock : before.length < domain.blockVariables)
    (totalLength : before.length + 1 + after.length =
      domain.blockVariables + domain.laneVariables)
    (point : K) :
    let blockAfter :=
      after.take (domain.blockVariables - before.length - 1)
    let laneCoordinates :=
      after.drop (domain.blockVariables - before.length - 1)
    let blockLength : before.length + 1 + blockAfter.length =
        domain.blockVariables := by
      dsimp only [blockAfter]
      rw [List.length_take]
      omega
    let laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      block := cubeSlice before blockAfter blockLength point
      lane := { coordinates := laneCoordinates, dimension := laneLength } } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    rw [List.take_append]
    rw [List.take_of_length_le (Nat.le_of_lt beforeBlock)]
    have remainingSucc :
        domain.blockVariables - before.length =
          (domain.blockVariables - before.length - 1) + 1 := by
      omega
    rw [remainingSucc]
    rfl
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    rw [List.drop_append]
    rw [List.drop_eq_nil_of_le (Nat.le_of_lt beforeBlock)]
    have remainingSucc :
        domain.blockVariables - before.length =
          (domain.blockVariables - before.length - 1) + 1 := by
      omega
    rw [remainingSucc]
    rfl

private theorem ofCoordinates_eq_laneSlice
    {domain : BlockNcDomain}
    (before after : List K)
    (blockBefore : domain.blockVariables ≤ before.length)
    (totalLength : before.length + 1 + after.length =
      domain.blockVariables + domain.laneVariables)
    (point : K) :
    let blockCoordinates := before.take domain.blockVariables
    let laneBefore := before.drop domain.blockVariables
    let blockLength : blockCoordinates.length = domain.blockVariables := by
      dsimp only [blockCoordinates]
      rw [List.length_take]
      omega
    let laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    Point.ofCoordinates (before ++ point :: after) (by simp; omega) = {
      block := { coordinates := blockCoordinates, dimension := blockLength }
      lane := cubeSlice laneBefore after laneLength point } := by
  dsimp only
  apply Point.ext
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates
    simp only
    exact List.take_append_of_le_length blockBefore
  · apply cubePoint_eq_of_coordinates_eq
    unfold Point.ofCoordinates cubeSlice
    simp only
    exact List.drop_append_of_le_length blockBefore

private theorem sumcheckPolynomial_eq_qAtPoint_of_length
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (coordinates : List K)
    (length : coordinates.length =
      domain.blockVariables + domain.laneVariables) :
    InitialSum.sumcheckPolynomial covers data coins coordinates =
      Mixing.qAtPoint covers data coins
        (Point.ofCoordinates coordinates length) := by
  unfold InitialSum.sumcheckPolynomial Mixing.polynomial Point.decode
  rw [dif_pos length]
  rfl

/-- Every exact flattened coordinate slice of the total SumCheck polynomial
has degree at most four. -/
theorem sumcheckPolynomial_slice_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (before after : List K)
    (length : before.length + 1 + after.length =
      domain.blockVariables + domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      InitialSum.sumcheckPolynomial covers data coins
        (before ++ point :: after) := by
  by_cases beforeBlock : before.length < domain.blockVariables
  · let blockAfter :=
      after.take (domain.blockVariables - before.length - 1)
    let laneCoordinates :=
      after.drop (domain.blockVariables - before.length - 1)
    have blockLength : before.length + 1 + blockAfter.length =
        domain.blockVariables := by
      dsimp only [blockAfter]
      rw [List.length_take]
      omega
    have laneLength : laneCoordinates.length = domain.laneVariables := by
      dsimp only [laneCoordinates]
      rw [List.length_drop]
      omega
    let lane : CubePoint K domain.laneVariables := {
      coordinates := laneCoordinates
      dimension := laneLength }
    rcases qAtPoint_block_quartic covers data coins lane
      before blockAfter blockLength with ⟨slice, sliceRepresents⟩
    refine ⟨slice, ?_⟩
    intro point
    change slice.evaluate ops.toOps point =
      InitialSum.sumcheckPolynomial covers data coins
        (before ++ point :: after)
    rw [sumcheckPolynomial_eq_qAtPoint_of_length
      covers data coins (before ++ point :: after) (by
        simp only [List.length_append, List.length_cons]
        omega)]
    rw [ofCoordinates_eq_blockSlice before after beforeBlock length]
    exact sliceRepresents point
  · have blockBefore : domain.blockVariables ≤ before.length :=
      Nat.le_of_not_gt beforeBlock
    let blockCoordinates := before.take domain.blockVariables
    let laneBefore := before.drop domain.blockVariables
    have blockLength : blockCoordinates.length = domain.blockVariables := by
      dsimp only [blockCoordinates]
      rw [List.length_take]
      omega
    have laneLength : laneBefore.length + 1 + after.length =
        domain.laneVariables := by
      dsimp only [laneBefore]
      rw [List.length_drop]
      omega
    let block : CubePoint K domain.blockVariables := {
      coordinates := blockCoordinates
      dimension := blockLength }
    rcases qAtPoint_lane_quartic covers data coins block
      laneBefore after laneLength with ⟨slice, sliceRepresents⟩
    refine ⟨slice, ?_⟩
    intro point
    change slice.evaluate ops.toOps point =
      InitialSum.sumcheckPolynomial covers data coins
        (before ++ point :: after)
    rw [sumcheckPolynomial_eq_qAtPoint_of_length
      covers data coins (before ++ point :: after) (by
        simp only [List.length_append, List.length_cons]
        omega)]
    rw [ofCoordinates_eq_laneSlice before after blockBefore length]
    exact sliceRepresents point

/-- Boolean suffix summation preserves a degree-four representation. -/
theorem sumCompletions_quartic
    (polynomial : List K → K)
    (fixed : List K)
    (remaining : Nat)
    (represented : ∀ vertex : BooleanVertex remaining,
      ∃ slice : Polynomial ncSumcheckDegreeBound, ∀ point,
        slice.evaluate ops.toOps point =
          polynomial
            ((fixed ++ [point]) ++
              SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex)) :
    RepresentsAtMostFour fun point =>
      Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
        ops.toOps polynomial (fixed ++ [point]) remaining := by
  exact DegreeSupport.sumCompletions_represents
    polynomial fixed remaining represented

/-- Every honest expected block×lane SumCheck round has degree at most four.
`fixed` is the verifier challenge prefix and `remaining` is the Boolean suffix
after the exposed round variable. -/
theorem expectedRound_quartic
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (fixed : List K)
    (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.blockVariables + domain.laneVariables) :
    RepresentsAtMostFour fun point =>
      Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
        ops.toOps (InitialSum.sumcheckPolynomial covers data coins)
        (fixed ++ [point]) remaining := by
  apply sumCompletions_quartic
  intro vertex
  have suffixLength :
      (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex).length =
        remaining :=
    SumCheckTruthPath.VertexEncoding.fieldCoordinates_length ops vertex
  rcases sumcheckPolynomial_slice_quartic covers data coins fixed
    (SumCheckTruthPath.VertexEncoding.fieldCoordinates ops vertex) (by
      rw [suffixLength]
      exact length) with ⟨slice, sliceRepresents⟩
  refine ⟨slice, ?_⟩
  intro point
  simpa only [List.append_assoc, List.singleton_append] using
    sliceRepresents point

/-- Every honest expected round therefore has five constant-first slots.
This does not enforce the certificate's number of rounds. -/
theorem expectedRound_has_five_coefficients
    {shape : SemanticShape}
    {domain : BlockNcDomain}
    (covers : domain.Covers shape)
    (data : Data shape)
    (coins : Mixing.Coins domain)
    (fixed : List K)
    (remaining : Nat)
    (length : fixed.length + 1 + remaining =
      domain.blockVariables + domain.laneVariables) :
    ∃ message : Nightstream.SuperNeo.SumCheck.Finite.Message K,
      message.coefficients.length = ncMessageWidth ∧
      message.degreeUpperBound = ncSumcheckDegreeBound ∧
      ∀ point, message.evaluate ops.toOps point =
        Nightstream.SuperNeo.SumCheck.Finite.HypercubeTruth.sumCompletions
          ops.toOps (InitialSum.sumcheckPolynomial covers data coins)
          (fixed ++ [point]) remaining := by
  apply representsAtMostFour_message_shape
  exact expectedRound_quartic covers data coins fixed remaining length

end Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.Degree.SumCheck
