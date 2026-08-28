import NightstreamFPrime.Layout.Stage1.RunningTransitionData

/-! Owns the closed-form PiCCS round-point wire bound. -/

namespace NightstreamFPrime.Layout.Stage1.RunningTransitionInputs

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Circuit.Quadratic
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Gadgets.Poseidon2.Duplex
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.PiCCS.v1_1
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Layout.Polynomial.Horner
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

def roundStride : Nat := 5328
def roundSampleC0Offset : Nat := 4136
def roundSampleC1Offset : Nat := 4728

def directRoundPoint (start : Nat)
    (coordinate : Fin productionShape.cubeVariables) : KExpr :=
  ⟨Expr.var (start + coordinate.val * roundStride + roundSampleC0Offset),
    Expr.var (start + coordinate.val * roundStride + roundSampleC1Offset)⟩

private def encodeKExpr (value : KExpr) : Expr × Expr :=
  (value.c0, value.c1)

private theorem encodeKExpr_injective : Function.Injective encodeKExpr := by
  intro left right equal
  cases left
  cases right
  simp [encodeKExpr] at equal
  simp_all

private theorem listMap_injective {Alpha Beta : Type}
    {function : Alpha → Beta} (injective : Function.Injective function) :
    Function.Injective (List.map function) := by
  intro left right equal
  induction left generalizing right with
  | nil =>
      cases right <;> simp_all
  | cons head tail inductionHypothesis =>
      cases right with
      | nil => simp at equal
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at equal
          rw [injective equal.1, inductionHypothesis equal.2]

private def fixedLogicalWidth : Nat :=
  phaseOffset + RunningTransition.exactPrivateCount

private def fixedPublicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth fixedLogicalWidth := by
  apply Nat.le_trans (m := fixedLogicalWidth)
  · norm_num [fixedLogicalWidth, phaseOffset,
      RunningTransition.exactPrivateCount, ringDegree, publicRingColumns]
  · exact Phi81CarrierLayout.logicalWidth_le_carrierWidth fixedLogicalWidth

private theorem fixedLayoutSamplesConcrete :
    let parent := PiCCSInputs.interface
      fixedLogicalWidth fixedPublicFits
    let frozen := Formal.atOffset parent PiCCSInputs.phaseOffset
    let roundInterface := Formal.roundTranscriptInterface frozen
    let start := Formal.roundTranscriptStart frozen
    (RoundTranscript.layoutWiring roundInterface start).samples =
      List.ofFn (directRoundPoint start) := by
  apply listMap_injective encodeKExpr_injective
  decide

private theorem fixedLayoutSamples
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    let parent := PiCCSInputs.interface logicalWidth publicFits
    let frozen := Formal.atOffset parent PiCCSInputs.phaseOffset
    let roundInterface := Formal.roundTranscriptInterface frozen
    let start := Formal.roundTranscriptStart frozen
    (RoundTranscript.layoutWiring roundInterface start).samples =
      List.ofFn (directRoundPoint start) := by
  exact fixedLayoutSamplesConcrete

theorem recursivePoint_eq_direct
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (coordinate : Fin productionShape.cubeVariables) :
    (recursiveRunningExpr logicalWidth publicFits).point coordinate =
      directRoundPoint PiCCSStarts.roundTranscriptWitnessStart coordinate := by
  let parent := PiCCSInputs.interface logicalWidth publicFits
  let frozen := Formal.atOffset parent PiCCSInputs.phaseOffset
  let roundInterface := Formal.roundTranscriptInterface frozen
  let start := Formal.roundTranscriptStart frozen
  change RoundTranscript.challenge roundInterface start coordinate = _
  rw [RoundTranscript.challenge_eq_challengeFast_pointwise]
  unfold RoundTranscript.challengeFast
  rw [fixedLayoutSamples]
  rw [List.getD_eq_get (List.ofFn (directRoundPoint start)) KExpr.zero
    ⟨coordinate.val, by
      simp only [List.length_ofFn]
      exact coordinate.isLt⟩]
  rw [List.get_ofFn]
  change directRoundPoint start coordinate = _
  have startEq : start = PiCCSStarts.roundTranscriptWitnessStart := by
    dsimp [start, roundInterface, frozen, parent]
    rw [Formal.roundTranscriptStart_atOffset,
      Formal.roundTranscriptOffset_eq, Formal.challengeOffset_eq]
    rfl
  rw [startEq]

theorem recursivePointBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (_relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ coordinate,
      ((recursiveRunningExpr logicalWidth publicFits).point coordinate
        ).VarsBelow phaseOffset := by
  intro coordinate
  rw [recursivePoint_eq_direct coordinate]
  rw [PiCCSStarts.roundTranscriptWitnessStart_eq]
  simp only [directRoundPoint, KExpr.VarsBelow, Expr.VarsBelow]
  have coordinateBound := coordinate.isLt
  norm_num [phaseOffset, roundStride, roundSampleC0Offset,
    roundSampleC1Offset, productionShape, cubeVariables,
    Phi81MatrixSource.phi81Shape] at coordinateBound ⊢
  omega

theorem recursivePointLinear
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (_relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ coordinate,
      KExprLinear
        ((recursiveRunningExpr logicalWidth publicFits).point coordinate) := by
  intro coordinate
  rw [recursivePoint_eq_direct coordinate]
  refine ⟨rfl, rfl, ?_, ?_⟩ <;>
    simp [directRoundPoint, Nonconstant]

end NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
