import Nightstream.Implementation.R1CS.Canonical.KPointEqualityHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcNcArithmeticHonest

/-!
Contract: constructive completeness for the selected block×lane NC endpoint.

Every auxiliary value is constructed here.  The only semantic boundary is
the pair of equalities to the authoritative initial and terminal columns;
the selected verifier completeness theorem must derive those equalities.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpointHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

def afterBlock
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KPointEqualityHonest.witness
    (KSplitNcNcArithmeticHonest.witness input assignment)
    (KSplitNcNcEndpoint.blockEqualityInput input)

def afterLane
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KPointEqualityHonest.witness (afterBlock input assignment)
    (KSplitNcNcEndpoint.laneEqualityInput input)

def afterSelector
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KMulHonest.witness (afterLane input assignment)
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.blockEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.laneEqualityInput input))
    (KSplitNcNcEndpoint.selectorFrame input)

def witness
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Nat → Nat :=
  KMulHonest.witness (afterSelector input assignment)
    (KSplitNcNcEndpoint.selector input)
    (KSplitNcNcEndpoint.mixedOutput input)
    (KSplitNcNcEndpoint.terminalFrame input)

structure SourceBounds
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) : Prop where
  arithmetic : KSplitNcNcArithmeticHonest.SourceBounds input
  betaBlock :
    ∀ coordinate,
      CarriedBelow (input.betaBlock coordinate) input.frameBase
  betaA :
    ∀ coordinate, CarriedBelow (input.betaA coordinate) input.frameBase
  pointBlock :
    ∀ coordinate,
      CarriedBelow (input.pointBlock coordinate) input.frameBase
  initial : CarriedBelow input.initial input.frameBase
  terminal : CarriedBelow input.terminal input.frameBase

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

private theorem block_left_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input) :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcNcEndpoint.blockEqualityInput input).left coordinate)
        (KSplitNcNcEndpoint.equalityBase input) := by
  intro coordinate
  exact carried_mono (sources.pointBlock coordinate) (by
    unfold KSplitNcNcEndpoint.equalityBase
      KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
    omega)

private theorem block_right_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input) :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcNcEndpoint.blockEqualityInput input).right coordinate)
        (KSplitNcNcEndpoint.equalityBase input) := by
  intro coordinate
  exact carried_mono (sources.betaBlock coordinate) (by
    unfold KSplitNcNcEndpoint.equalityBase
      KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
    omega)

theorem blockPrefix_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies
      (KSplitNcNcArithmeticHonest.rows input ++
        KPointEquality.rows
          (KSplitNcNcEndpoint.blockEqualityInput input))
      (afterBlock input assignment) := by
  have arithmeticSatisfied :=
    KSplitNcNcArithmeticHonest.rows_honest
      input assignment positive sources.arithmetic
  have arithmeticPreserved :
      Satisfies (KSplitNcNcArithmeticHonest.rows input)
        (afterBlock input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (KSplitNcNcArithmeticHonest.witness input assignment)
      (afterBlock input assignment)
    · intro row member column mentioned
      symm
      exact KPointEqualityHonest.witness_off_block
        (KSplitNcNcArithmeticHonest.witness input assignment)
        (KSplitNcNcEndpoint.blockEqualityInput input)
        column
        (KSplitNcNcArithmeticHonest.rows_below_equalityBase
          input positive sources.arithmetic row member column mentioned)
    · exact arithmeticSatisfied
  have blockSatisfied :
      Satisfies
        (KPointEquality.rows
          (KSplitNcNcEndpoint.blockEqualityInput input))
        (afterBlock input assignment) := by
    apply KPointEqualityHonest.rows_honest
    · change 0 < KSplitNcNcEndpoint.equalityBase input
      unfold KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega
    · exact block_left_below input sources
    · exact block_right_below input sources
  intro row member
  exact (List.mem_append.1 member).elim
    (arithmeticPreserved row) (blockSatisfied row)

private theorem blockPrefix_below_laneBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow
      (KSplitNcNcArithmeticHonest.rows input ++
        KPointEquality.rows
          (KSplitNcNcEndpoint.blockEqualityInput input))
      (KSplitNcNcEndpoint.laneEqualityInput input).frameBase := by
  apply rowsBelow_append
  · exact rowsBelow_mono
      (KSplitNcNcArithmeticHonest.rows_below_equalityBase
        input positive sources.arithmetic)
      (by
        change KSplitNcNcEndpoint.equalityBase input ≤
          KSplitNcNcEndpoint.equalityBase input +
            KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
        omega)
  · apply pointEquality_rows_below
    · change 0 <
        KSplitNcNcEndpoint.equalityBase input +
          KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
      unfold KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega
    · intro coordinate
      exact carried_mono (block_left_below input sources coordinate)
        (by
          change KSplitNcNcEndpoint.equalityBase input ≤
            KSplitNcNcEndpoint.equalityBase input +
              KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
          omega)
    · intro coordinate
      exact carried_mono (block_right_below input sources coordinate)
        (by
          change KSplitNcNcEndpoint.equalityBase input ≤
            KSplitNcNcEndpoint.equalityBase input +
              KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
          omega)
    · change
        KSplitNcNcEndpoint.equalityBase input +
            3 * domain.blockVariables +
            3 * (domain.blockVariables - 1) ≤
          KSplitNcNcEndpoint.equalityBase input +
            KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
      unfold KSplitNcNcEndpoint.pointEqualityRows
      omega

private theorem lane_left_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input) :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcNcEndpoint.laneEqualityInput input).left coordinate)
        (KSplitNcNcEndpoint.laneEqualityInput input).frameBase := by
  intro coordinate
  exact carried_mono (sources.arithmetic.pointLane coordinate) (by
    change input.frameBase ≤
      KSplitNcNcEndpoint.equalityBase input +
        KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
    unfold KSplitNcNcEndpoint.equalityBase
      KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
    omega)

private theorem lane_right_below
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (sources : SourceBounds input) :
    ∀ coordinate,
      CarriedBelow
        ((KSplitNcNcEndpoint.laneEqualityInput input).right coordinate)
        (KSplitNcNcEndpoint.laneEqualityInput input).frameBase := by
  intro coordinate
  exact carried_mono (sources.betaA coordinate) (by
    change input.frameBase ≤
      KSplitNcNcEndpoint.equalityBase input +
        KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
    unfold KSplitNcNcEndpoint.equalityBase
      KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
    omega)

theorem pointPrefix_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies
      (KSplitNcNcArithmeticHonest.rows input ++
        KPointEquality.rows
          (KSplitNcNcEndpoint.blockEqualityInput input) ++
        KPointEquality.rows
          (KSplitNcNcEndpoint.laneEqualityInput input))
      (afterLane input assignment) := by
  have prefixSatisfied :=
    blockPrefix_honest input assignment positive sources
  have prefixPreserved :
      Satisfies
        (KSplitNcNcArithmeticHonest.rows input ++
          KPointEquality.rows
            (KSplitNcNcEndpoint.blockEqualityInput input))
        (afterLane input assignment) := by
    apply KHornerSupport.satisfies_extend _
      (afterBlock input assignment) (afterLane input assignment)
    · intro row member column mentioned
      symm
      exact KPointEqualityHonest.witness_off_block
        (afterBlock input assignment)
        (KSplitNcNcEndpoint.laneEqualityInput input) column
        (blockPrefix_below_laneBase input positive sources
          row member column mentioned)
    · exact prefixSatisfied
  have laneSatisfied :
      Satisfies
        (KPointEquality.rows
          (KSplitNcNcEndpoint.laneEqualityInput input))
        (afterLane input assignment) := by
    apply KPointEqualityHonest.rows_honest
    · change 0 <
        KSplitNcNcEndpoint.equalityBase input +
          KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
      unfold KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega
    · exact lane_left_below input sources
    · exact lane_right_below input sources
  intro row member
  rcases List.mem_append.1 member with inPrefix | inLane
  · exact prefixPreserved row inPrefix
  · exact laneSatisfied row inLane

private theorem pointPrefix_below_productBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow
      (KSplitNcNcArithmeticHonest.rows input ++
        KPointEquality.rows
          (KSplitNcNcEndpoint.blockEqualityInput input) ++
        KPointEquality.rows
          (KSplitNcNcEndpoint.laneEqualityInput input))
      (KSplitNcNcEndpoint.productBase input) := by
  apply rowsBelow_append
  · exact rowsBelow_mono
      (blockPrefix_below_laneBase input positive sources)
      (by
        change
          KSplitNcNcEndpoint.equalityBase input +
              KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables ≤
            KSplitNcNcEndpoint.productBase input
        unfold KSplitNcNcEndpoint.productBase
        omega)
  · apply pointEquality_rows_below
    · change 0 < KSplitNcNcEndpoint.productBase input
      unfold KSplitNcNcEndpoint.productBase
        KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega
    · intro coordinate
      exact carried_mono (lane_left_below input sources coordinate) (by
        change
          KSplitNcNcEndpoint.equalityBase input +
              KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables ≤
            KSplitNcNcEndpoint.productBase input
        unfold KSplitNcNcEndpoint.productBase
        omega)
    · intro coordinate
      exact carried_mono (lane_right_below input sources coordinate) (by
        change
          KSplitNcNcEndpoint.equalityBase input +
              KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables ≤
            KSplitNcNcEndpoint.productBase input
        unfold KSplitNcNcEndpoint.productBase
        omega)
    · change
        KSplitNcNcEndpoint.equalityBase input +
            KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables +
            3 * domain.laneVariables +
            3 * (domain.laneVariables - 1) ≤
          KSplitNcNcEndpoint.productBase input
      unfold KSplitNcNcEndpoint.productBase
        KSplitNcNcEndpoint.pointEqualityRows
      omega

private theorem blockOutput_below_productBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    CarriedBelow
      (KPointEquality.equalityCarried
        (KSplitNcNcEndpoint.blockEqualityInput input))
      (KSplitNcNcEndpoint.productBase input) := by
  apply pointEquality_output_below
  · change 0 < KSplitNcNcEndpoint.productBase input
    unfold KSplitNcNcEndpoint.productBase
      KSplitNcNcEndpoint.equalityBase
      KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
    omega
  · intro coordinate
    exact carried_mono (block_right_below input sources coordinate) (by
      unfold KSplitNcNcEndpoint.productBase
      omega)
  · change
      KSplitNcNcEndpoint.equalityBase input +
          3 * domain.blockVariables +
          3 * (domain.blockVariables - 1) ≤
        KSplitNcNcEndpoint.productBase input
    unfold KSplitNcNcEndpoint.productBase
      KSplitNcNcEndpoint.pointEqualityRows
    omega

private theorem laneOutput_below_productBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    CarriedBelow
      (KPointEquality.equalityCarried
        (KSplitNcNcEndpoint.laneEqualityInput input))
      (KSplitNcNcEndpoint.productBase input) := by
  apply pointEquality_output_below
  · change 0 < KSplitNcNcEndpoint.productBase input
    unfold KSplitNcNcEndpoint.productBase
      KSplitNcNcEndpoint.equalityBase
      KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
    omega
  · intro coordinate
    exact carried_mono (lane_right_below input sources coordinate) (by
      change
        KSplitNcNcEndpoint.equalityBase input +
            KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables ≤
          KSplitNcNcEndpoint.productBase input
      unfold KSplitNcNcEndpoint.productBase
      omega)
  · change
      KSplitNcNcEndpoint.equalityBase input +
          KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables +
          3 * domain.laneVariables +
          3 * (domain.laneVariables - 1) ≤
        KSplitNcNcEndpoint.productBase input
    unfold KSplitNcNcEndpoint.productBase
      KSplitNcNcEndpoint.pointEqualityRows
    omega

def pointRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) : List Row :=
  KSplitNcNcArithmeticHonest.rows input ++
    KPointEquality.rows (KSplitNcNcEndpoint.blockEqualityInput input) ++
      KPointEquality.rows (KSplitNcNcEndpoint.laneEqualityInput input)

def selectorRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) : List Row :=
  KMul.rows
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.blockEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.laneEqualityInput input))
    (KSplitNcNcEndpoint.selectorFrame input)

def terminalProductRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) : List Row :=
  KMul.rows (KSplitNcNcEndpoint.selector input)
    (KSplitNcNcEndpoint.mixedOutput input)
    (KSplitNcNcEndpoint.terminalFrame input)

def computedRows
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) : List Row :=
  pointRows input ++ selectorRows input ++ terminalProductRows input

theorem selectorPrefix_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies (pointRows input ++ selectorRows input)
      (afterSelector input assignment) := by
  exact appendMul_honest (pointRows input) (afterLane input assignment)
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.blockEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.laneEqualityInput input))
    (KSplitNcNcEndpoint.productBase input) 0
    (pointPrefix_honest input assignment positive sources)
    (pointPrefix_below_productBase input positive sources)
    (blockOutput_below_productBase input positive sources)
    (laneOutput_below_productBase input positive sources)

private theorem selectorPrefix_below_terminalBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    RowsBelow (pointRows input ++ selectorRows input)
      (KSplitNcNcEndpoint.productBase input + 3) := by
  exact appendMul_below_next (pointRows input)
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.blockEqualityInput input))
    (KPointEquality.equalityCarried
      (KSplitNcNcEndpoint.laneEqualityInput input))
    (KSplitNcNcEndpoint.productBase input) 0
    (pointPrefix_below_productBase input positive sources)
    (blockOutput_below_productBase input positive sources)
    (laneOutput_below_productBase input positive sources)

private theorem selector_below_terminalBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    CarriedBelow (KSplitNcNcEndpoint.selector input)
      (KSplitNcNcEndpoint.productBase input + 3) := by
  unfold KSplitNcNcEndpoint.selector KSplitNcNcEndpoint.selectorFrame
  exact frame_output_below (KSplitNcNcEndpoint.productBase input) 0 _
    (by omega)

private theorem mixedOutput_below_terminalBase
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    CarriedBelow (KSplitNcNcEndpoint.mixedOutput input)
      (KSplitNcNcEndpoint.productBase input + 3) :=
  carried_mono
    (KSplitNcNcArithmeticHonest.mixedOutput_below_equalityBase input)
    (by unfold KSplitNcNcEndpoint.productBase; omega)

theorem computedRows_honest
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (sources : SourceBounds input) :
    Satisfies (computedRows input) (witness input assignment) := by
  exact appendMul_honest (pointRows input ++ selectorRows input)
    (afterSelector input assignment)
    (KSplitNcNcEndpoint.selector input)
    (KSplitNcNcEndpoint.mixedOutput input)
    (KSplitNcNcEndpoint.productBase input) 1
    (selectorPrefix_honest input assignment positive sources)
    (selectorPrefix_below_terminalBase input positive sources)
    (selector_below_terminalBase input)
    (mixedOutput_below_terminalBase input)

theorem witness_off_source
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (column : Nat) (below : column < input.frameBase) :
    witness input assignment column = assignment column := by
  unfold witness KSplitNcNcEndpoint.terminalFrame
  rw [KMulHonest.witness_off_before _ _ _
    (KSplitNcNcEndpoint.productBase input) 1 column (by
      unfold KSplitNcNcEndpoint.productBase
        KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega)]
  unfold afterSelector KSplitNcNcEndpoint.selectorFrame
  rw [KMulHonest.witness_off_before _ _ _
    (KSplitNcNcEndpoint.productBase input) 0 column (by
      unfold KSplitNcNcEndpoint.productBase
        KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega)]
  unfold afterLane
  rw [KPointEqualityHonest.witness_off_block _ _
    column (by
      change column <
        KSplitNcNcEndpoint.equalityBase input +
          KSplitNcNcEndpoint.pointEqualityRows domain.blockVariables
      unfold KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega)]
  unfold afterBlock
  rw [KPointEqualityHonest.witness_off_block _ _
    column (by
      change column < KSplitNcNcEndpoint.equalityBase input
      unfold KSplitNcNcEndpoint.equalityBase
        KSplitNcNcEndpoint.mixedBase KSplitNcNcEndpoint.normBase
      omega)]
  exact KSplitNcNcArithmeticHonest.witness_off_source
    input assignment column below

theorem witness_constantWire
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1) :
    witness input assignment 0 = 1 := by
  rw [witness_off_source input assignment 0 positive]
  exact constantWire

structure EndpointBinding
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat) : Prop where
  initialLow :
    lcEval (witness input assignment) KLinear.zeroCarried.low =
      lcEval (witness input assignment) input.initial.low
  initialHigh :
    lcEval (witness input assignment) KLinear.zeroCarried.high =
      lcEval (witness input assignment) input.initial.high
  terminalLow :
    lcEval (witness input assignment)
        (KSplitNcNcEndpoint.terminalExpression input).low =
      lcEval (witness input assignment) input.terminal.low
  terminalHigh :
    lcEval (witness input assignment)
        (KSplitNcNcEndpoint.terminalExpression input).high =
      lcEval (witness input assignment) input.terminal.high

theorem rows_eq_initial_append_computed_append_terminal
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    KSplitNcNcEndpoint.rows input =
      KSplitNcNcEndpoint.initialRows input ++ (computedRows input ++
        KEquality.rows
          (KSplitNcNcEndpoint.terminalExpression input) input.terminal) := by
  simp [KSplitNcNcEndpoint.rows, KSplitNcNcEndpoint.rowGroups,
    KSplitNcNcEndpoint.initialRows, computedRows, pointRows,
    KSplitNcNcArithmeticHonest.rows, selectorRows, terminalProductRows,
    List.append_assoc]

theorem rows_honest_of_binding
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain)
    (assignment : Nat → Nat)
    (positive : 0 < input.frameBase)
    (constantWire : assignment 0 = 1)
    (sources : SourceBounds input)
    (binding : EndpointBinding input assignment) :
    Satisfies (KSplitNcNcEndpoint.rows input)
      (witness input assignment) := by
  rw [rows_eq_initial_append_computed_append_terminal]
  have one := witness_constantWire input assignment positive constantWire
  have initial :=
    KEquality.rows_complete (witness input assignment)
      KLinear.zeroCarried input.initial one
      binding.initialLow binding.initialHigh
  have computed :=
    computedRows_honest input assignment positive sources
  have terminal :=
    KEquality.rows_complete (witness input assignment)
      (KSplitNcNcEndpoint.terminalExpression input) input.terminal one
      binding.terminalLow binding.terminalHigh
  intro row member
  rcases List.mem_append.1 member with inInitial | inRest
  · exact initial row inInitial
  rcases List.mem_append.1 inRest with inComputed | inTerminal
  · exact computed row inComputed
  · exact terminal row inTerminal

end Nightstream.Implementation.R1CS.Canonical.KSplitNcNcEndpointHonest
