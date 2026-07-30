import Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalHonest

/-!
Contract: constructive completeness for the four terminal FE products.

Owns the four canonical multiplication witnesses following the FE point
equalities.  It does not assert the final authoritative terminal equality;
that boundary remains for the selected frozen FE theorem.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalProductsHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointSupport
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

def freshSelectorRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  KMul.rows
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.freshLaneEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.freshRowEqualityInput input))
    (KSplitNcFeTerminal.freshSelectorFrame input)

def freshContributionRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  KMul.rows (KSplitNcFeTerminal.freshSelector input)
    (KSplitNcFeTerminal.freshOutput input)
    (KSplitNcFeTerminal.freshContributionFrame input)

def carriedSelectorRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  KMul.rows
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.carriedLaneEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.carriedRowEqualityInput input))
    (KSplitNcFeTerminal.carriedSelectorFrame input)

def carriedContributionRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  KMul.rows (KSplitNcFeTerminal.carriedSelector input)
    (KSplitNcFeTerminal.carriedTarget input)
    (KSplitNcFeTerminal.carriedContributionFrame input)

def afterFreshSelector
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KMulHonest.witness
    (KSplitNcFeTerminalHonest.afterCarriedRow input assignment)
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.freshLaneEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.freshRowEqualityInput input))
    (KSplitNcFeTerminal.freshSelectorFrame input)

def afterFreshContribution
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KMulHonest.witness (afterFreshSelector input assignment)
    (KSplitNcFeTerminal.freshSelector input)
    (KSplitNcFeTerminal.freshOutput input)
    (KSplitNcFeTerminal.freshContributionFrame input)

def afterCarriedSelector
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KMulHonest.witness (afterFreshContribution input assignment)
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.carriedLaneEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcFeTerminal.carriedRowEqualityInput input))
    (KSplitNcFeTerminal.carriedSelectorFrame input)

def witness
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KMulHonest.witness (afterCarriedSelector input assignment)
    (KSplitNcFeTerminal.carriedSelector input)
    (KSplitNcFeTerminal.carriedTarget input)
    (KSplitNcFeTerminal.carriedContributionFrame input)

def freshSelectorPrefix
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  KSplitNcFeTerminalHonest.pointPrefix input ++ freshSelectorRows input

def freshContributionPrefix
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  freshSelectorPrefix input ++ freshContributionRows input

def carriedSelectorPrefix
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  freshContributionPrefix input ++ carriedSelectorRows input

/-- Every FE-terminal row except the final equality to the authoritative
terminal input. -/
def computedRows
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : List Row :=
  carriedSelectorPrefix input ++ carriedContributionRows input

private theorem rowsBelow_append
    {left right : List Row} {boundary : Nat}
    (leftBelow : RowsBelow left boundary)
    (rightBelow : RowsBelow right boundary) :
    RowsBelow (left ++ right) boundary := by
  intro row member column mentioned
  exact (List.mem_append.1 member).elim
    (fun inLeft => leftBelow row inLeft column mentioned)
    (fun inRight => rightBelow row inRight column mentioned)

private theorem rowsBelow_mono
    {rows : List Row} {lower upper : Nat}
    (below : RowsBelow rows lower) (ordered : lower ≤ upper) :
    RowsBelow rows upper := by
  intro row member column mentioned
  exact Nat.lt_of_lt_of_le
    (below row member column mentioned) ordered

private theorem satisfies_after_append
    {oldRows newRows : List Row}
    {oldAssignment newAssignment : Nat → Nat}
    {boundary : Nat}
    (oldSatisfied : Satisfies oldRows oldAssignment)
    (oldBelow : RowsBelow oldRows boundary)
    (off :
      ∀ column, column < boundary →
        newAssignment column = oldAssignment column)
    (newSatisfied : Satisfies newRows newAssignment) :
    Satisfies (oldRows ++ newRows) newAssignment := by
  have preserved :
      Satisfies oldRows newAssignment := by
    apply KHornerSupport.satisfies_extend _
      oldAssignment newAssignment
    · intro row member column mentioned
      exact (off column (oldBelow row member column mentioned)).symm
    · exact oldSatisfied
  intro row member
  exact (List.mem_append.1 member).elim
    (preserved row) (newSatisfied row)

private theorem appendMul_honest
    (prefixRows : List Row) (oldAssignment : Nat → Nat)
    (left right : Carried) (base step : Nat)
    (prefixSatisfied : Satisfies prefixRows oldAssignment)
    (prefixBelow : RowsBelow prefixRows (base + 3 * step))
    (leftBelow : CarriedBelow left (base + 3 * step))
    (rightBelow : CarriedBelow right (base + 3 * step)) :
    Satisfies
      (prefixRows ++ KMul.rows left right (KFrames.frameAt base step))
      (KMulHonest.witness oldAssignment left right
        (KFrames.frameAt base step)) := by
  apply satisfies_after_append prefixSatisfied prefixBelow
  · intro column below
    exact KMulHonest.witness_off_before oldAssignment left right
      base step column below
  · exact KMulHonest.witness_satisfies oldAssignment left right
      (KFrames.frameAt base step)
      (KMulHonest.canonical_distinct base step)
      (KMulHonest.fresh_of_before left.low base step leftBelow.1)
      (KMulHonest.fresh_of_before left.high base step leftBelow.2)
      (KMulHonest.fresh_of_before right.low base step rightBelow.1)
      (KMulHonest.fresh_of_before right.high base step rightBelow.2)

private theorem appendMul_below_next
    (prefixRows : List Row) (left right : Carried) (base step : Nat)
    (prefixBelow : RowsBelow prefixRows (base + 3 * step))
    (leftBelow : CarriedBelow left (base + 3 * step))
    (rightBelow : CarriedBelow right (base + 3 * step)) :
    RowsBelow
      (prefixRows ++ KMul.rows left right (KFrames.frameAt base step))
      (base + 3 * (step + 1)) := by
  apply rowsBelow_append
  · exact rowsBelow_mono prefixBelow (by omega)
  · exact mul_rows_below left right base step
      (base + 3 * (step + 1))
      (carried_mono leftBelow (by omega))
      (carried_mono rightBelow (by omega))
      (by omega)

theorem frameBase_le_productBase
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    input.frameBase ≤ KSplitNcFeTerminal.productBase input := by
  unfold KSplitNcFeTerminal.productBase
    KSplitNcFeTerminal.equalityBase
    KSplitNcFeTerminal.carriedTargetBase
    KSplitNcFeTerminal.carriedBase
    KSplitNcFeTerminal.freshHornerBase
  omega

private theorem pointOutput_below
    {variables boundary : Nat}
    (input : KPointEquality.Input variables)
    (positive : 0 < boundary)
    (rightBelow :
      ∀ index, CarriedBelow (input.right index) boundary)
    (endBound :
      input.frameBase + KSplitNcFeTerminal.pointEqualityRows variables ≤
        boundary) :
    CarriedBelow (KPointEquality.equalityCarried input) boundary := by
  apply pointEquality_output_below input positive rightBelow
  unfold KSplitNcFeTerminal.pointEqualityRows at endBound
  simpa [Nat.add_assoc] using endBound

structure SourceBounds
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) : Prop where
  gamma : CarriedBelow input.gamma input.frameBase
  pointLane :
    ∀ coordinate,
      CarriedBelow (input.pointLane coordinate) input.frameBase
  pointRow :
    ∀ coordinate,
      CarriedBelow (input.pointRow coordinate) input.frameBase
  betaA :
    ∀ coordinate,
      CarriedBelow (input.betaA coordinate) input.frameBase
  betaR :
    ∀ coordinate,
      CarriedBelow (input.betaR coordinate) input.frameBase
  alpha :
    ∀ coordinate,
      CarriedBelow (input.alpha coordinate) input.frameBase
  priorPoint :
    ∀ coordinate,
      CarriedBelow (input.priorPoint coordinate) input.frameBase
  message :
    ∀ source matrix lane,
      CarriedBelow
        (input.messageYRing source matrix lane) input.frameBase

private theorem freshLaneOutput_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    CarriedBelow
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.freshLaneEqualityInput input))
      (KSplitNcFeTerminal.productBase input) := by
  apply pointOutput_below
  · exact Nat.lt_of_lt_of_le positive (frameBase_le_productBase input)
  · intro coordinate
    exact carried_mono (sources.betaA coordinate)
      (frameBase_le_productBase input)
  · change
      KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
        KSplitNcFeTerminal.productBase input
    unfold KSplitNcFeTerminal.productBase
    omega

private theorem freshRowOutput_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    CarriedBelow
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.freshRowEqualityInput input))
      (KSplitNcFeTerminal.productBase input) := by
  apply pointOutput_below
  · exact Nat.lt_of_lt_of_le positive (frameBase_le_productBase input)
  · intro coordinate
    exact carried_mono (sources.betaR coordinate)
      (frameBase_le_productBase input)
  · change
      KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
        KSplitNcFeTerminal.productBase input
    unfold KSplitNcFeTerminal.productBase
    omega

private theorem carriedLaneOutput_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    CarriedBelow
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.carriedLaneEqualityInput input))
      (KSplitNcFeTerminal.productBase input) := by
  apply pointOutput_below
  · exact Nat.lt_of_lt_of_le positive (frameBase_le_productBase input)
  · intro coordinate
    exact carried_mono (sources.alpha coordinate)
      (frameBase_le_productBase input)
  · change
      KSplitNcFeTerminal.equalityBase input +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables +
            KSplitNcFeTerminal.pointEqualityRows domain.laneVariables ≤
        KSplitNcFeTerminal.productBase input
    unfold KSplitNcFeTerminal.productBase
    omega

private theorem carriedRowOutput_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    CarriedBelow
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.carriedRowEqualityInput input))
      (KSplitNcFeTerminal.productBase input) := by
  apply pointOutput_below
  · exact Nat.lt_of_lt_of_le positive (frameBase_le_productBase input)
  · intro coordinate
    exact carried_mono (sources.priorPoint coordinate)
      (frameBase_le_productBase input)
  · change
      KSplitNcFeTerminal.equalityBase input +
            2 * KSplitNcFeTerminal.pointEqualityRows domain.laneVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables +
            KSplitNcFeTerminal.pointEqualityRows shape.rowVariables ≤
        KSplitNcFeTerminal.productBase input
    unfold KSplitNcFeTerminal.productBase
    omega

private theorem freshOutput_below_stepOne
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    CarriedBelow (KSplitNcFeTerminal.freshOutput input)
      (KSplitNcFeTerminal.productBase input + 3) := by
  unfold KSplitNcFeTerminal.freshOutput
  apply horner_output_below
  · intro output member
    exact carried_mono
      (KSplitNcFeTerminalHonest.freshOutputs_below_hornerBase
        input positive sources.message output member)
      (by
        unfold KSplitNcFeTerminal.productBase
          KSplitNcFeTerminal.equalityBase
          KSplitNcFeTerminal.carriedTargetBase
          KSplitNcFeTerminal.carriedBase
        omega)
  · rw [KSplitNcFeTerminal.freshOutputs_length]
    unfold KSplitNcFeTerminal.productBase
      KSplitNcFeTerminal.equalityBase
      KSplitNcFeTerminal.carriedTargetBase
      KSplitNcFeTerminal.carriedBase
    omega

private theorem carriedTarget_below_stepThree
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    CarriedBelow (KSplitNcFeTerminal.carriedTarget input)
      (KSplitNcFeTerminal.productBase input + 9) :=
  carried_mono
    (KSplitNcFeTerminalHonest.target_below_equalityBase input)
    (by
      unfold KSplitNcFeTerminal.productBase
      omega)

theorem freshSelectorPrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (sources : SourceBounds input) :
    Satisfies (freshSelectorPrefix input)
      (afterFreshSelector input assignment) := by
  have satisfied :=
    appendMul_honest
      (KSplitNcFeTerminalHonest.pointPrefix input)
      (KSplitNcFeTerminalHonest.afterCarriedRow input assignment)
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.freshLaneEqualityInput input))
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.freshRowEqualityInput input))
      (KSplitNcFeTerminal.productBase input) 0
      (KSplitNcFeTerminalHonest.pointPrefix_honest input assignment
        positive constantWire sources.gamma sources.pointLane
        sources.pointRow sources.betaA sources.betaR sources.alpha
        sources.priorPoint sources.message)
      (KSplitNcFeTerminalHonest.pointPrefix_below_productBase input
        positive sources.gamma sources.pointLane sources.pointRow
        sources.betaA sources.betaR sources.alpha sources.priorPoint
        sources.message)
      (freshLaneOutput_below input positive sources)
      (freshRowOutput_below input positive sources)
  simpa [freshSelectorPrefix, freshSelectorRows, afterFreshSelector,
    KSplitNcFeTerminal.freshSelectorFrame] using satisfied

theorem freshSelectorPrefix_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (freshSelectorPrefix input)
      (KSplitNcFeTerminal.productBase input + 3) := by
  have bounded :=
    appendMul_below_next
      (KSplitNcFeTerminalHonest.pointPrefix input)
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.freshLaneEqualityInput input))
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.freshRowEqualityInput input))
      (KSplitNcFeTerminal.productBase input) 0
      (KSplitNcFeTerminalHonest.pointPrefix_below_productBase input
        positive sources.gamma sources.pointLane sources.pointRow
        sources.betaA sources.betaR sources.alpha sources.priorPoint
        sources.message)
      (freshLaneOutput_below input positive sources)
      (freshRowOutput_below input positive sources)
  simpa [freshSelectorPrefix, freshSelectorRows,
    KSplitNcFeTerminal.freshSelectorFrame] using bounded

theorem freshContributionPrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (sources : SourceBounds input) :
    Satisfies (freshContributionPrefix input)
      (afterFreshContribution input assignment) := by
  have selectorBelow :
      CarriedBelow (KSplitNcFeTerminal.freshSelector input)
        (KSplitNcFeTerminal.productBase input + 3) :=
    frame_output_below (KSplitNcFeTerminal.productBase input) 0
      (KSplitNcFeTerminal.productBase input + 3) (by omega)
  have satisfied :=
    appendMul_honest
      (freshSelectorPrefix input)
      (afterFreshSelector input assignment)
      (KSplitNcFeTerminal.freshSelector input)
      (KSplitNcFeTerminal.freshOutput input)
      (KSplitNcFeTerminal.productBase input) 1
      (freshSelectorPrefix_honest input assignment positive
        constantWire sources)
      (by
        simpa [Nat.mul_one] using
          freshSelectorPrefix_below input positive sources)
      (by simpa [Nat.mul_one] using selectorBelow)
      (by simpa [Nat.mul_one] using
        freshOutput_below_stepOne input positive sources)
  simpa [freshContributionPrefix, freshContributionRows,
    afterFreshContribution,
    KSplitNcFeTerminal.freshContributionFrame] using satisfied

theorem freshContributionPrefix_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (freshContributionPrefix input)
      (KSplitNcFeTerminal.productBase input + 6) := by
  have selectorBelow :
      CarriedBelow (KSplitNcFeTerminal.freshSelector input)
        (KSplitNcFeTerminal.productBase input + 3) :=
    frame_output_below (KSplitNcFeTerminal.productBase input) 0
      (KSplitNcFeTerminal.productBase input + 3) (by omega)
  have bounded :=
    appendMul_below_next
      (freshSelectorPrefix input)
      (KSplitNcFeTerminal.freshSelector input)
      (KSplitNcFeTerminal.freshOutput input)
      (KSplitNcFeTerminal.productBase input) 1
      (by
        simpa [Nat.mul_one] using
          freshSelectorPrefix_below input positive sources)
      (by simpa [Nat.mul_one] using selectorBelow)
      (by simpa [Nat.mul_one] using
        freshOutput_below_stepOne input positive sources)
  simpa [freshContributionPrefix, freshContributionRows,
    KSplitNcFeTerminal.freshContributionFrame] using bounded

theorem carriedSelectorPrefix_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (sources : SourceBounds input) :
    Satisfies (carriedSelectorPrefix input)
      (afterCarriedSelector input assignment) := by
  have laneBelow :=
    carried_mono (carriedLaneOutput_below input positive sources)
      (show KSplitNcFeTerminal.productBase input ≤
          KSplitNcFeTerminal.productBase input + 6 by omega)
  have rowBelow :=
    carried_mono (carriedRowOutput_below input positive sources)
      (show KSplitNcFeTerminal.productBase input ≤
          KSplitNcFeTerminal.productBase input + 6 by omega)
  have satisfied :=
    appendMul_honest
      (freshContributionPrefix input)
      (afterFreshContribution input assignment)
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.carriedLaneEqualityInput input))
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.carriedRowEqualityInput input))
      (KSplitNcFeTerminal.productBase input) 2
      (freshContributionPrefix_honest input assignment positive
        constantWire sources)
      (by
        simpa using freshContributionPrefix_below input positive sources)
      (by simpa using laneBelow)
      (by simpa using rowBelow)
  simpa [carriedSelectorPrefix, carriedSelectorRows,
    afterCarriedSelector,
    KSplitNcFeTerminal.carriedSelectorFrame] using satisfied

theorem carriedSelectorPrefix_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (carriedSelectorPrefix input)
      (KSplitNcFeTerminal.productBase input + 9) := by
  have laneBelow :=
    carried_mono (carriedLaneOutput_below input positive sources)
      (show KSplitNcFeTerminal.productBase input ≤
          KSplitNcFeTerminal.productBase input + 6 by omega)
  have rowBelow :=
    carried_mono (carriedRowOutput_below input positive sources)
      (show KSplitNcFeTerminal.productBase input ≤
          KSplitNcFeTerminal.productBase input + 6 by omega)
  have bounded :=
    appendMul_below_next
      (freshContributionPrefix input)
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.carriedLaneEqualityInput input))
      (KPointEquality.equalityCarried
        (KSplitNcFeTerminal.carriedRowEqualityInput input))
      (KSplitNcFeTerminal.productBase input) 2
      (by
        simpa using freshContributionPrefix_below input positive sources)
      (by simpa using laneBelow)
      (by simpa using rowBelow)
  simpa [carriedSelectorPrefix, carriedSelectorRows,
    KSplitNcFeTerminal.carriedSelectorFrame] using bounded

theorem computedRows_honest
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (sources : SourceBounds input) :
    Satisfies (computedRows input) (witness input assignment) := by
  have selectorBelow :
      CarriedBelow (KSplitNcFeTerminal.carriedSelector input)
        (KSplitNcFeTerminal.productBase input + 9) :=
    frame_output_below (KSplitNcFeTerminal.productBase input) 2
      (KSplitNcFeTerminal.productBase input + 9) (by omega)
  have satisfied :=
    appendMul_honest
      (carriedSelectorPrefix input)
      (afterCarriedSelector input assignment)
      (KSplitNcFeTerminal.carriedSelector input)
      (KSplitNcFeTerminal.carriedTarget input)
      (KSplitNcFeTerminal.productBase input) 3
      (carriedSelectorPrefix_honest input assignment positive
        constantWire sources)
      (by
        simpa using carriedSelectorPrefix_below input positive sources)
      (by simpa using selectorBelow)
      (by simpa using carriedTarget_below_stepThree input)
  simpa [computedRows, carriedContributionRows, witness,
    KSplitNcFeTerminal.carriedContributionFrame] using satisfied

theorem computedRows_below
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (computedRows input)
      (KSplitNcFeTerminal.productBase input + 12) := by
  have selectorBelow :
      CarriedBelow (KSplitNcFeTerminal.carriedSelector input)
        (KSplitNcFeTerminal.productBase input + 9) :=
    frame_output_below (KSplitNcFeTerminal.productBase input) 2
      (KSplitNcFeTerminal.productBase input + 9) (by omega)
  have bounded :=
    appendMul_below_next
      (carriedSelectorPrefix input)
      (KSplitNcFeTerminal.carriedSelector input)
      (KSplitNcFeTerminal.carriedTarget input)
      (KSplitNcFeTerminal.productBase input) 3
      (by
        simpa using carriedSelectorPrefix_below input positive sources)
      (by simpa using selectorBelow)
      (by simpa using carriedTarget_below_stepThree input)
  simpa [computedRows, carriedContributionRows,
    KSplitNcFeTerminal.carriedContributionFrame] using bounded

private theorem source_below_productStep
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (column step : Nat) (below : column < input.frameBase) :
    column < KSplitNcFeTerminal.productBase input + 3 * step :=
  Nat.lt_of_lt_of_le below
    (Nat.le_trans (frameBase_le_productBase input) (by omega))

theorem witness_off_source
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) (column : Nat)
    (below : column < input.frameBase) :
    witness input assignment column = assignment column := by
  calc
    witness input assignment column =
        afterCarriedSelector input assignment column :=
      KMulHonest.witness_off_before
        (afterCarriedSelector input assignment)
        (KSplitNcFeTerminal.carriedSelector input)
        (KSplitNcFeTerminal.carriedTarget input)
        (KSplitNcFeTerminal.productBase input) 3 column
        (source_below_productStep input column 3 below)
    _ = afterFreshContribution input assignment column :=
      KMulHonest.witness_off_before
        (afterFreshContribution input assignment)
        (KPointEquality.equalityCarried
          (KSplitNcFeTerminal.carriedLaneEqualityInput input))
        (KPointEquality.equalityCarried
          (KSplitNcFeTerminal.carriedRowEqualityInput input))
        (KSplitNcFeTerminal.productBase input) 2 column
        (source_below_productStep input column 2 below)
    _ = afterFreshSelector input assignment column :=
      KMulHonest.witness_off_before
        (afterFreshSelector input assignment)
        (KSplitNcFeTerminal.freshSelector input)
        (KSplitNcFeTerminal.freshOutput input)
        (KSplitNcFeTerminal.productBase input) 1 column
        (source_below_productStep input column 1 below)
    _ = KSplitNcFeTerminalHonest.afterCarriedRow input assignment column :=
      KMulHonest.witness_off_before
        (KSplitNcFeTerminalHonest.afterCarriedRow input assignment)
        (KPointEquality.equalityCarried
          (KSplitNcFeTerminal.freshLaneEqualityInput input))
        (KPointEquality.equalityCarried
          (KSplitNcFeTerminal.freshRowEqualityInput input))
        (KSplitNcFeTerminal.productBase input) 0 column
        (source_below_productStep input column 0 below)
    _ = assignment column :=
      KSplitNcFeTerminalHonest.afterCarriedRow_off_source
        input assignment column below

theorem witness_constantWire
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1) :
    witness input assignment 0 = 1 := by
  rw [witness_off_source input assignment 0 positive]
  exact constantWire

/-- The sole semantic boundary left by constructive FE-terminal completeness.
The selected frozen FE theorem must derive this equality; callers may not use
it as a replacement for that theorem. -/
structure TerminalBinding
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat) : Prop where
  low :
    lcEval (witness input assignment)
        (KSplitNcFeTerminal.terminalExpression input).low =
      lcEval (witness input assignment) input.terminal.low
  high :
    lcEval (witness input assignment)
        (KSplitNcFeTerminal.terminalExpression input).high =
      lcEval (witness input assignment) input.terminal.high

theorem rows_eq_computedRows_append_terminal
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    KSplitNcFeTerminal.rows input =
      computedRows input ++
        KEquality.rows
          (KSplitNcFeTerminal.terminalExpression input) input.terminal := by
  simp [KSplitNcFeTerminal.rows, KSplitNcFeTerminal.rowGroups,
    computedRows, carriedSelectorPrefix, freshContributionPrefix,
    freshSelectorPrefix, KSplitNcFeTerminalHonest.pointPrefix,
    KSplitNcFeTerminalHonest.carriedLanePrefix,
    KSplitNcFeTerminalHonest.freshRowPrefix,
    KSplitNcFeTerminalHonest.freshLanePrefix,
    KSplitNcFeTerminalHonest.arithmeticRows,
    freshSelectorRows, freshContributionRows, carriedSelectorRows,
    carriedContributionRows, List.append_assoc]

theorem rows_honest_of_binding
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (sources : SourceBounds input)
    (binding : TerminalBinding input assignment) :
    Satisfies (KSplitNcFeTerminal.rows input) (witness input assignment) := by
  rw [rows_eq_computedRows_append_terminal]
  have computed :=
    computedRows_honest input assignment positive constantWire sources
  have terminal :=
    KEquality.rows_complete (witness input assignment)
      (KSplitNcFeTerminal.terminalExpression input) input.terminal
      (witness_constantWire input assignment positive constantWire)
      binding.low binding.high
  intro row member
  exact (List.mem_append.1 member).elim
    (computed row) (terminal row)

end Nightstream.Implementation.R1CS.Canonical.KSplitNcFeTerminalProductsHonest
