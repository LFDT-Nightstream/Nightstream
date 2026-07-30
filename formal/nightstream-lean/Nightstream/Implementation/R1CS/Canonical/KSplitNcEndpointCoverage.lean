import Nightstream.Implementation.R1CS.Canonical.KCompositeAllocationCoverage
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpoints

/-!
Contract: exact converse-to-conservation coverage for the three verifier-owned
Split-NC endpoint programs.

The proofs follow each endpoint's explicit placement arithmetic.  They do not
infer coverage from row counts or dense-span declarations.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointCoverage

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.AllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KCompositeAllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KFrameAllocationCoverage
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

private theorem dense_mem_iff (base width column : Nat) :
    column ∈ (List.range width).map (fun offset => base + offset) ↔
      base ≤ column ∧ column < base + width := by
  constructor
  · intro member
    rcases List.mem_map.1 member with ⟨offset, inRange, rfl⟩
    exact ⟨by omega, by
      have bound := List.mem_range.1 inRange
      omega⟩
  · intro bounds
    exact List.mem_map.2
      ⟨column - base, List.mem_range.2 (by omega), by omega⟩

private theorem locateBlock
    (base width count column : Nat)
    (widthPositive : 0 < width)
    (bounds : base ≤ column ∧ column < base + width * count) :
    ∃ index, index < count ∧
      base + width * index ≤ column ∧
        column < base + width * (index + 1) := by
  let offset := column - base
  let index := offset / width
  have columnEq : column = base + offset := by
    simp only [offset]
    omega
  have split :
      offset = width * index + offset % width := by
    have raw := Nat.div_add_mod offset width
    simp only [index] at raw ⊢
    exact raw.symm
  have remainder : offset % width < width :=
    Nat.mod_lt offset widthPositive
  have offsetBound : offset < count * width := by
    simp only [offset]
    rw [Nat.mul_comm]
    omega
  have indexLt : index < count :=
    (Nat.div_lt_iff_lt_mul widthPositive).2 offsetBound
  have lowerOffset : width * index ≤ offset := by
    rw [Nat.mul_comm]
    exact Nat.div_mul_le_self offset width
  have upperOffset : offset < width * (index + 1) := by
    calc
      offset = width * index + offset % width := split
      _ < width * index + width := Nat.add_lt_add_left remainder _
      _ = width * (index + 1) := by rw [Nat.mul_succ]
  refine ⟨index, indexLt, ?_, ?_⟩ <;> omega

private theorem canonicalIndex_mem (count : Nat) (index : Fin count) :
    index ∈ canonicalFinIndices count := by
  unfold canonicalFinIndices
  rw [List.mem_ofFn]
  exact ⟨index, rfl⟩

private theorem liftGroup
    {groups : List (List Row)} {group : List Row} {columns : List Nat}
    (groupMember : group ∈ groups)
    (covered : RowsCover group columns) :
    RowsCover groups.flatten columns := by
  intro column member
  rcases covered column member with ⟨row, rowMember, mentioned⟩
  exact
    ⟨row, List.mem_flatten.2 ⟨group, groupMember, rowMember⟩, mentioned⟩

private theorem pointColumns_of_bounds
    {variables : Nat} (input : KPointEquality.Input variables)
    (column : Nat)
    (bounds :
      input.frameBase ≤ column ∧
        column <
          input.frameBase + 3 * variables + 3 * (variables - 1)) :
    column ∈ KPointEquality.columns input := by
  unfold KPointEquality.columns
  by_cases inFactors : column < input.frameBase + 3 * variables
  · apply List.mem_append_left
    rw [KFrames.frameColumns_mem_iff]
    omega
  · apply List.mem_append_right
    rw [KFrames.frameColumns_mem_iff]
    unfold KPointEquality.productBase
    omega

private theorem feIndex_of_ordinal
    (shape : SemanticShape) (ordinal : Nat)
    (bound : ordinal < shape.matrixCount * shape.runningCount) :
    ∃ index ∈ KSplitNcFeInitial.indices shape,
      KSplitNcFeInitial.ordinal index = ordinal := by
  have runningPositive : 0 < shape.runningCount := by
    rcases Nat.eq_zero_or_pos shape.runningCount with zero | positive
    · rw [zero, Nat.mul_zero] at bound
      omega
    · exact positive
  let matrix : Fin shape.matrixCount :=
    ⟨ordinal / shape.runningCount, by
      exact
        (Nat.div_lt_iff_lt_mul runningPositive).2
          (by simpa only [Nat.mul_comm] using bound)⟩
  let running : Fin shape.runningCount :=
    ⟨ordinal % shape.runningCount,
      Nat.mod_lt ordinal runningPositive⟩
  let index : KSplitNcFeInitial.Index shape := (matrix, running)
  have ordinalEq : KSplitNcFeInitial.ordinal index = ordinal := by
    change
      (ordinal / shape.runningCount) * shape.runningCount +
          ordinal % shape.runningCount =
        ordinal
    rw [Nat.mul_comm]
    exact Nat.div_add_mod ordinal shape.runningCount
  refine ⟨index, ?_, ordinalEq⟩
  unfold KSplitNcFeInitial.indices
  apply List.mem_flatMap.2
  refine ⟨matrix, canonicalIndex_mem _ matrix, ?_⟩
  apply List.mem_map.2
  exact ⟨running, canonicalIndex_mem _ running, rfl⟩

private theorem feInitialMle
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain) :
    RowsCover
      (KSplitNcFeInitial.mleRows input)
      ((List.range
          (KSplitNcFeInitial.rowsPerMle domain *
            (shape.matrixCount * shape.runningCount))).map
        (fun offset => input.frameBase + offset)) := by
  intro column member
  rw [dense_mem_iff] at member
  let width := KSplitNcFeInitial.rowsPerMle domain
  have widthPositive : 0 < width := by
    rcases Nat.eq_zero_or_pos width with zero | positive
    · rw [show KSplitNcFeInitial.rowsPerMle domain = width from rfl,
        zero, Nat.zero_mul, Nat.add_zero] at member
      omega
    · exact positive
  rcases locateBlock input.frameBase width
      (shape.matrixCount * shape.runningCount) column widthPositive
      (by simpa only [width] using member) with
    ⟨ordinal, ordinalLt, lower, upper⟩
  simp only [width] at lower upper
  unfold KSplitNcFeInitial.rowsPerMle at lower upper
  rcases feIndex_of_ordinal shape ordinal ordinalLt with
    ⟨index, indexMember, ordinalEq⟩
  have localColumn :
      column ∈
        KFrames.frameColumns (KSplitNcFeInitial.mleBase input index)
          (KBooleanMle.frameCount domain.laneVariables) := by
    rw [KFrames.frameColumns_mem_iff]
    unfold KSplitNcFeInitial.mleBase KSplitNcFeInitial.rowsPerMle
    rw [ordinalEq]
    constructor
    · exact lower
    · rw [Nat.mul_succ] at upper
      omega
  rcases booleanMle (KSplitNcFeInitial.mleBase input index)
      (KSplitNcFeInitial.table input index)
      (KSplitNcFeInitial.alphaCoordinates input) 0
      column (by simpa only [Nat.mul_zero, Nat.add_zero] using localColumn) with
    ⟨row, rowMember, mentioned⟩
  exact
    ⟨row, List.mem_flatMap.2 ⟨index, indexMember, rowMember⟩, mentioned⟩

/-- Every FE-initial auxiliary column is used by either its lane-MLE or dense
gamma Horner program. -/
theorem feInitial
    {shape : SemanticShape} {domain : FlatNcDomain}
    (input : KSplitNcFeInitial.Input shape domain) :
    RowsCover (KSplitNcFeInitial.rows input)
      (KSplitNcFeInitial.columns input) := by
  intro column member
  unfold KSplitNcFeInitial.columns at member
  rw [dense_mem_iff] at member
  let mleWidth :=
    KSplitNcFeInitial.rowsPerMle domain *
      (shape.matrixCount * shape.runningCount)
  by_cases inMle : column < input.frameBase + mleWidth
  · have mleColumn :
        column ∈
          (List.range mleWidth).map
            (fun offset => input.frameBase + offset) := by
      rw [dense_mem_iff]
      omega
    rcases feInitialMle input column mleColumn with
      ⟨row, rowMember, mentioned⟩
    exact
      ⟨row,
        List.mem_append_left _
          (List.mem_append_left _ rowMember),
        mentioned⟩
  · have hornerColumn :
        column ∈
          KFrames.frameColumns (KSplitNcFeInitial.hornerBase input)
            ((KSplitNcFeInitial.coefficients input).length - 1) := by
      rw [KFrames.frameColumns_mem_iff]
      unfold KSplitNcFeInitial.hornerBase
      rw [KSplitNcFeInitial.coefficients_length]
      unfold KSplitNcFeInitial.allocationWidth at member
      omega
    rcases horner input.gamma (KSplitNcFeInitial.hornerBase input)
        (KSplitNcFeInitial.coefficients input) 0 column
        (by simpa only [Nat.mul_zero, Nat.add_zero] using hornerColumn) with
      ⟨row, rowMember, mentioned⟩
    exact
      ⟨row,
        List.mem_append_left _
          (List.mem_append_right _ rowMember),
        mentioned⟩

private theorem ncMle
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    RowsCover
      (KSplitNcNcEndpoint.mleRows input)
      ((List.range
          (KSplitNcNcEndpoint.rowsPerMle domain *
            shape.sourceCount)).map
        (fun offset => input.frameBase + offset)) := by
  intro column member
  rw [dense_mem_iff] at member
  let width := KSplitNcNcEndpoint.rowsPerMle domain
  have widthPositive : 0 < width := by
    rcases Nat.eq_zero_or_pos width with zero | positive
    · rw [show KSplitNcNcEndpoint.rowsPerMle domain = width from rfl,
        zero, Nat.zero_mul, Nat.add_zero] at member
      omega
    · exact positive
  rcases locateBlock input.frameBase width shape.sourceCount
      column widthPositive (by simpa only [width] using member) with
    ⟨ordinal, ordinalLt, lower, upper⟩
  simp only [width] at lower upper
  unfold KSplitNcNcEndpoint.rowsPerMle at lower upper
  let source : Fin shape.sourceCount := ⟨ordinal, ordinalLt⟩
  have localColumn :
      column ∈
        KFrames.frameColumns (KSplitNcNcEndpoint.mleBase input source)
          (KBooleanMle.frameCount domain.laneVariables) := by
    rw [KFrames.frameColumns_mem_iff]
    unfold KSplitNcNcEndpoint.mleBase
    change
      input.frameBase +
          3 * KBooleanMle.frameCount domain.laneVariables * ordinal ≤
        column ∧
      column <
        input.frameBase +
          3 * KBooleanMle.frameCount domain.laneVariables * ordinal +
            3 * KBooleanMle.frameCount domain.laneVariables
    constructor
    · exact lower
    · rw [Nat.mul_succ] at upper
      omega
  rcases booleanMle (KSplitNcNcEndpoint.mleBase input source)
      (KSplitNcNcEndpoint.sourceTable input source)
      (KSplitNcNcEndpoint.laneCoordinates input) 0
      column (by simpa only [Nat.mul_zero, Nat.add_zero] using localColumn) with
    ⟨row, rowMember, mentioned⟩
  exact
    ⟨row,
      List.mem_flatMap.2
        ⟨source, canonicalIndex_mem _ source, rowMember⟩,
      mentioned⟩

private theorem ncNorm
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    RowsCover
      (KSplitNcNcEndpoint.normRows input)
      ((List.range (6 * shape.sourceCount)).map
        (fun offset => KSplitNcNcEndpoint.normBase input + offset)) := by
  intro column member
  rw [dense_mem_iff] at member
  rcases locateBlock (KSplitNcNcEndpoint.normBase input) 6
      shape.sourceCount column (by decide)
      (by simpa only using member) with
    ⟨ordinal, ordinalLt, lower, upper⟩
  let source : Fin shape.sourceCount := ⟨ordinal, ordinalLt⟩
  have localColumn :
      column ∈ KStrictNorm.columns
        (KSplitNcNcEndpoint.normInput input source) := by
    unfold KStrictNorm.columns KSplitNcNcEndpoint.normInput
    rw [KFrames.frameColumns_mem_iff]
    change
      KSplitNcNcEndpoint.normBase input + 6 * ordinal ≤ column ∧
        column <
          KSplitNcNcEndpoint.normBase input + 6 * ordinal + 6
    rw [Nat.mul_succ] at upper
    exact ⟨lower, by omega⟩
  rcases strictNorm (KSplitNcNcEndpoint.normInput input source)
      column localColumn with ⟨row, rowMember, mentioned⟩
  exact
    ⟨row,
      List.mem_flatMap.2
        ⟨source, canonicalIndex_mem _ source, rowMember⟩,
      mentioned⟩

/-- Every block×lane NC endpoint auxiliary is used by its MLE, norm,
gamma-fold, point-equality, or terminal-product program. -/
theorem ncEndpoint
    {shape : SemanticShape} {domain : BlockNcDomain}
    (input : KSplitNcNcEndpoint.Input shape domain) :
    RowsCover (KSplitNcNcEndpoint.rows input)
      (KSplitNcNcEndpoint.columns input) := by
  intro column member
  unfold KSplitNcNcEndpoint.columns at member
  rw [dense_mem_iff] at member
  by_cases inMle :
      column < KSplitNcNcEndpoint.normBase input
  · have localColumn :
        column ∈
          (List.range
            (KSplitNcNcEndpoint.rowsPerMle domain *
              shape.sourceCount)).map
            (fun offset => input.frameBase + offset) := by
      rw [dense_mem_iff]
      exact
        ⟨member.1,
          by
            simpa only [KSplitNcNcEndpoint.normBase] using inMle⟩
    unfold KSplitNcNcEndpoint.rows
    exact liftGroup
      (by simp [KSplitNcNcEndpoint.rowGroups])
      (ncMle input) column localColumn
  · by_cases inNorm :
      column < KSplitNcNcEndpoint.mixedBase input
    · have localColumn :
          column ∈
            (List.range (6 * shape.sourceCount)).map
              (fun offset =>
                KSplitNcNcEndpoint.normBase input + offset) := by
        rw [dense_mem_iff]
        exact
          ⟨Nat.le_of_not_gt inMle,
            by
              simpa only [KSplitNcNcEndpoint.mixedBase] using inNorm⟩
      unfold KSplitNcNcEndpoint.rows
      exact liftGroup
        (by simp [KSplitNcNcEndpoint.rowGroups])
        (ncNorm input) column localColumn
    · by_cases inMixed :
        column < KSplitNcNcEndpoint.equalityBase input
      · have localColumn :
            column ∈
              KFrames.frameColumns
                (KSplitNcNcEndpoint.mixedBase input)
                (shape.sourceCount - 1) := by
          rw [KFrames.frameColumns_mem_iff]
          exact
            ⟨Nat.le_of_not_gt inNorm,
              by
                simpa only [KSplitNcNcEndpoint.equalityBase] using
                  inMixed⟩
        have covered :=
          horner input.gamma (KSplitNcNcEndpoint.mixedBase input)
            (KSplitNcNcEndpoint.normOutputs input) 0
        rw [KSplitNcNcEndpoint.normOutputs_length] at covered
        rcases covered column
            (by simpa only [Nat.mul_zero, Nat.add_zero] using localColumn) with
          ⟨row, rowMember, mentioned⟩
        refine ⟨row, ?_, mentioned⟩
        unfold KSplitNcNcEndpoint.rows
        exact List.mem_flatten.2
          ⟨KSplitNcNcEndpoint.mixedRows input,
            by simp [KSplitNcNcEndpoint.rowGroups],
            rowMember⟩
      · by_cases inBlock :
          column <
            (KSplitNcNcEndpoint.laneEqualityInput input).frameBase
        · have localColumn :=
            pointColumns_of_bounds
              (KSplitNcNcEndpoint.blockEqualityInput input) column
              (by
                constructor
                · change KSplitNcNcEndpoint.equalityBase input ≤ column
                  exact Nat.le_of_not_gt inMixed
                · have upper := inBlock
                  change
                    column <
                      KSplitNcNcEndpoint.equalityBase input +
                        KSplitNcNcEndpoint.pointEqualityRows
                          domain.blockVariables at upper
                  unfold KSplitNcNcEndpoint.pointEqualityRows at upper
                  change
                    column <
                      KSplitNcNcEndpoint.equalityBase input +
                          3 * domain.blockVariables +
                        3 * (domain.blockVariables - 1)
                  omega)
          unfold KSplitNcNcEndpoint.rows
          exact liftGroup
            (by simp [KSplitNcNcEndpoint.rowGroups])
            (pointEquality
              (KSplitNcNcEndpoint.blockEqualityInput input))
            column localColumn
        · by_cases inLane :
            column < KSplitNcNcEndpoint.productBase input
          · have localColumn :=
              pointColumns_of_bounds
                (KSplitNcNcEndpoint.laneEqualityInput input) column
                (by
                  constructor
                  · exact Nat.le_of_not_gt inBlock
                  · have upper := inLane
                    change
                      column <
                        (KSplitNcNcEndpoint.laneEqualityInput input).frameBase +
                          KSplitNcNcEndpoint.pointEqualityRows
                            domain.laneVariables at upper
                    unfold KSplitNcNcEndpoint.pointEqualityRows at upper
                    omega)
            unfold KSplitNcNcEndpoint.rows
            exact liftGroup
              (by simp [KSplitNcNcEndpoint.rowGroups])
              (pointEquality
                (KSplitNcNcEndpoint.laneEqualityInput input))
              column localColumn
          · have productEnd :
                column <
                  KSplitNcNcEndpoint.productBase input + 6 := by
              have endEq :
                  input.frameBase +
                      KSplitNcNcEndpoint.allocationWidth input =
                    KSplitNcNcEndpoint.productBase input + 6 := by
                unfold KSplitNcNcEndpoint.allocationWidth
                  KSplitNcNcEndpoint.productBase
                  KSplitNcNcEndpoint.equalityBase
                  KSplitNcNcEndpoint.mixedBase
                  KSplitNcNcEndpoint.normBase
                omega
              rw [endEq] at member
              exact member.2
            by_cases inSelector :
                column <
                  KSplitNcNcEndpoint.productBase input + 3
            · have localColumn :
                  column ∈
                    KFrames.frameColumns
                      (KSplitNcNcEndpoint.productBase input) 1 := by
                rw [KFrames.frameColumns_mem_iff]
                exact ⟨by omega, inSelector⟩
              unfold KSplitNcNcEndpoint.rows
              exact liftGroup
                (by
                  simp [KSplitNcNcEndpoint.rowGroups,
                    KSplitNcNcEndpoint.selectorFrame])
                (mul
                  (KPointEquality.equalityCarried
                    (KSplitNcNcEndpoint.blockEqualityInput input))
                  (KPointEquality.equalityCarried
                    (KSplitNcNcEndpoint.laneEqualityInput input))
                  (KSplitNcNcEndpoint.productBase input) 0)
                column
                (by simpa only [Nat.mul_zero, Nat.add_zero] using localColumn)
            · have localColumn :
                  column ∈
                    KFrames.frameColumns
                      (KSplitNcNcEndpoint.productBase input + 3) 1 := by
                rw [KFrames.frameColumns_mem_iff]
                exact ⟨by omega, productEnd⟩
              unfold KSplitNcNcEndpoint.rows
              exact liftGroup
                (by
                  simp [KSplitNcNcEndpoint.rowGroups,
                    KSplitNcNcEndpoint.terminalFrame])
                (mul
                  (KSplitNcNcEndpoint.selector input)
                  (KSplitNcNcEndpoint.mixedOutput input)
                  (KSplitNcNcEndpoint.productBase input) 1)
                column
                (by simpa only [Nat.mul_one] using localColumn)

private theorem feTerminalFresh
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    RowsCover
      (KSplitNcFeTerminal.freshRows input)
      ((List.range
          (KSplitNcFeTerminal.sparseRowsPerFresh input *
            shape.freshCount)).map
        (fun offset => input.frameBase + offset)) := by
  intro column member
  rw [dense_mem_iff] at member
  let width := KSplitNcFeTerminal.sparseRowsPerFresh input
  have widthPositive : 0 < width := by
    rcases Nat.eq_zero_or_pos width with zero | positive
    · rw [show KSplitNcFeTerminal.sparseRowsPerFresh input = width from rfl,
        zero, Nat.zero_mul, Nat.add_zero] at member
      omega
    · exact positive
  rcases locateBlock input.frameBase width shape.freshCount
      column widthPositive (by simpa only [width] using member) with
    ⟨ordinal, ordinalLt, lower, upper⟩
  simp only [width] at lower upper
  let fresh : Fin shape.freshCount := ⟨ordinal, ordinalLt⟩
  have localColumn :
      column ∈
        KSparsePolynomial.columns
          (KSplitNcFeTerminal.freshPolynomialInput input fresh) := by
    unfold KSparsePolynomial.columns
    simp only [KSplitNcFeTerminal.freshPolynomialInput, fresh]
    rw [KFrames.frameColumns_mem_iff]
    unfold KSplitNcFeTerminal.sparseRowsPerFresh
      KSplitNcFeTerminal.polynomialDegreeSum at lower upper ⊢
    constructor
    · rw [Nat.mul_comm ordinal] 
      exact lower
    · rw [Nat.mul_succ] at upper
      rw [Nat.mul_comm ordinal]
      omega
  rcases sparsePolynomial
      (KSplitNcFeTerminal.freshPolynomialInput input fresh)
      column localColumn with ⟨row, rowMember, mentioned⟩
  exact
    ⟨row,
      List.mem_flatMap.2
        ⟨fresh, canonicalIndex_mem _ fresh, rowMember⟩,
      mentioned⟩

private theorem carriedTarget
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    RowsCover
      (KSplitNcFeTerminal.carriedRows input)
      [KSplitNcFeTerminal.carriedTargetBase input,
        KSplitNcFeTerminal.carriedTargetBase input + 1] := by
  intro column member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl
  · let row :=
      KEquality.equalityRow
        (KSplitNcFeInitial.evaluated
          (KSplitNcFeTerminal.carriedInput input)).low
        (KSplitNcFeTerminal.carriedTarget input).low
    refine ⟨row, ?_, Or.inr (Or.inr ?_)⟩
    · unfold row KSplitNcFeTerminal.carriedRows
        KSplitNcFeInitial.rows
      apply List.mem_append_right
      simp [KEquality.rows, KSplitNcFeTerminal.carriedInput]
    · simp [row, KEquality.equalityRow,
        KSplitNcFeTerminal.carriedTarget,
        LinCombNormal.Mentions]
  · let row :=
      KEquality.equalityRow
        (KSplitNcFeInitial.evaluated
          (KSplitNcFeTerminal.carriedInput input)).high
        (KSplitNcFeTerminal.carriedTarget input).high
    refine ⟨row, ?_, Or.inr (Or.inr ?_)⟩
    · unfold row KSplitNcFeTerminal.carriedRows
        KSplitNcFeInitial.rows
      apply List.mem_append_right
      simp [KEquality.rows, KSplitNcFeTerminal.carriedInput]
    · simp [row, KEquality.equalityRow,
        KSplitNcFeTerminal.carriedTarget,
        LinCombNormal.Mentions]

/-- Every FE-terminal auxiliary is used by the fresh polynomial, gamma fold,
carried branch, point-equality, or branch-product program. -/
theorem feTerminal
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domain : FlatNcDomain}
    (input : KSplitNcFeTerminal.Input polynomialInput domain) :
    RowsCover (KSplitNcFeTerminal.rows input)
      (KSplitNcFeTerminal.columns input) := by
  intro column member
  unfold KSplitNcFeTerminal.columns at member
  rw [dense_mem_iff] at member
  by_cases inFresh :
      column < KSplitNcFeTerminal.freshHornerBase input
  · have localColumn :
        column ∈
          (List.range
            (KSplitNcFeTerminal.sparseRowsPerFresh input *
              shape.freshCount)).map
            (fun offset => input.frameBase + offset) := by
      rw [dense_mem_iff]
      exact
        ⟨member.1,
          by
            simpa only [KSplitNcFeTerminal.freshHornerBase] using
              inFresh⟩
    unfold KSplitNcFeTerminal.rows
    exact liftGroup
      (by simp [KSplitNcFeTerminal.rowGroups])
      (feTerminalFresh input) column localColumn
  · by_cases inFreshHorner :
      column < KSplitNcFeTerminal.carriedBase input
    · have localColumn :
          column ∈
            KFrames.frameColumns
              (KSplitNcFeTerminal.freshHornerBase input)
              (shape.freshCount - 1) := by
        rw [KFrames.frameColumns_mem_iff]
        exact
          ⟨Nat.le_of_not_gt inFresh,
            by
              simpa only [KSplitNcFeTerminal.carriedBase] using
                inFreshHorner⟩
      have covered :=
        horner input.gamma (KSplitNcFeTerminal.freshHornerBase input)
          (KSplitNcFeTerminal.freshOutputs input) 0
      rw [KSplitNcFeTerminal.freshOutputs_length] at covered
      unfold KSplitNcFeTerminal.rows
      exact liftGroup
        (by
          unfold KSplitNcFeTerminal.rowGroups
          simp only [List.mem_cons]
          exact Or.inr (Or.inl rfl))
        covered column
        (by simpa only [Nat.mul_zero, Nat.add_zero] using localColumn)
    · by_cases inCarried :
        column < KSplitNcFeTerminal.carriedTargetBase input
      · have localColumn :
            column ∈
              KSplitNcFeInitial.columns
                (KSplitNcFeTerminal.carriedInput input) := by
          unfold KSplitNcFeInitial.columns
          rw [dense_mem_iff]
          constructor
          · change KSplitNcFeTerminal.carriedBase input ≤ column
            exact Nat.le_of_not_gt inFreshHorner
          · have upper := inCarried
            change
              column <
                KSplitNcFeTerminal.carriedBase input +
                  KSplitNcFeTerminal.carriedInternalWidth input at upper
            simpa only [KSplitNcFeTerminal.carriedInput,
              KSplitNcFeTerminal.carriedInternalWidth,
              KSplitNcFeInitial.allocationWidth] using upper
        unfold KSplitNcFeTerminal.rows
        exact liftGroup
          (by
            unfold KSplitNcFeTerminal.rowGroups
            simp only [List.mem_cons]
            exact Or.inr (Or.inr (Or.inl rfl)))
          (feInitial (KSplitNcFeTerminal.carriedInput input))
          column localColumn
      · by_cases inTarget :
          column < KSplitNcFeTerminal.equalityBase input
        · have localColumn :
              column ∈
                [KSplitNcFeTerminal.carriedTargetBase input,
                  KSplitNcFeTerminal.carriedTargetBase input + 1] := by
            have lower := Nat.le_of_not_gt inCarried
            have upper := inTarget
            change
              column <
                KSplitNcFeTerminal.carriedTargetBase input + 2 at upper
            have classified :
                column = KSplitNcFeTerminal.carriedTargetBase input ∨
                  column =
                    KSplitNcFeTerminal.carriedTargetBase input + 1 := by
              omega
            simpa only [List.mem_cons, List.not_mem_nil, or_false] using
              classified
          unfold KSplitNcFeTerminal.rows
          exact liftGroup
            (by simp [KSplitNcFeTerminal.rowGroups])
            (carriedTarget input) column localColumn
        · by_cases inFreshLane :
            column <
              (KSplitNcFeTerminal.freshRowEqualityInput input).frameBase
          · have localColumn :=
              pointColumns_of_bounds
                (KSplitNcFeTerminal.freshLaneEqualityInput input) column
                (by
                  constructor
                  · change KSplitNcFeTerminal.equalityBase input ≤ column
                    exact Nat.le_of_not_gt inTarget
                  · have upper := inFreshLane
                    change
                      column <
                        KSplitNcFeTerminal.equalityBase input +
                          KSplitNcFeTerminal.pointEqualityRows
                            domain.laneVariables at upper
                    unfold KSplitNcFeTerminal.pointEqualityRows at upper
                    change
                      column <
                        KSplitNcFeTerminal.equalityBase input +
                            3 * domain.laneVariables +
                          3 * (domain.laneVariables - 1)
                    omega)
            unfold KSplitNcFeTerminal.rows
            exact liftGroup
              (by simp [KSplitNcFeTerminal.rowGroups])
              (pointEquality
                (KSplitNcFeTerminal.freshLaneEqualityInput input))
              column localColumn
          · by_cases inFreshRow :
              column <
                (KSplitNcFeTerminal.carriedLaneEqualityInput input).frameBase
            · have localColumn :=
                pointColumns_of_bounds
                  (KSplitNcFeTerminal.freshRowEqualityInput input) column
                  (by
                    constructor
                    · exact Nat.le_of_not_gt inFreshLane
                    · have upper := inFreshRow
                      change
                        column <
                          (KSplitNcFeTerminal.freshRowEqualityInput input).frameBase +
                            KSplitNcFeTerminal.pointEqualityRows
                              shape.rowVariables at upper
                      unfold KSplitNcFeTerminal.pointEqualityRows at upper
                      omega)
              unfold KSplitNcFeTerminal.rows
              exact liftGroup
                (by simp [KSplitNcFeTerminal.rowGroups])
                (pointEquality
                  (KSplitNcFeTerminal.freshRowEqualityInput input))
                column localColumn
            · by_cases inCarriedLane :
                column <
                  (KSplitNcFeTerminal.carriedRowEqualityInput input).frameBase
              · have localColumn :=
                  pointColumns_of_bounds
                    (KSplitNcFeTerminal.carriedLaneEqualityInput input)
                    column
                    (by
                      constructor
                      · exact Nat.le_of_not_gt inFreshRow
                      · have upper := inCarriedLane
                        dsimp [
                          KSplitNcFeTerminal.carriedLaneEqualityInput,
                          KSplitNcFeTerminal.carriedRowEqualityInput] at upper ⊢
                        unfold KSplitNcFeTerminal.pointEqualityRows at upper ⊢
                        omega)
                unfold KSplitNcFeTerminal.rows
                exact liftGroup
                  (by simp [KSplitNcFeTerminal.rowGroups])
                  (pointEquality
                    (KSplitNcFeTerminal.carriedLaneEqualityInput input))
                  column localColumn
              · by_cases inCarriedRow :
                  column < KSplitNcFeTerminal.productBase input
                · have localColumn :=
                    pointColumns_of_bounds
                      (KSplitNcFeTerminal.carriedRowEqualityInput input)
                      column
                      (by
                        constructor
                        · exact Nat.le_of_not_gt inCarriedLane
                        · have upper := inCarriedRow
                          dsimp [
                            KSplitNcFeTerminal.carriedRowEqualityInput,
                            KSplitNcFeTerminal.productBase] at upper ⊢
                          unfold KSplitNcFeTerminal.pointEqualityRows at upper ⊢
                          omega)
                  unfold KSplitNcFeTerminal.rows
                  exact liftGroup
                    (by simp [KSplitNcFeTerminal.rowGroups])
                    (pointEquality
                      (KSplitNcFeTerminal.carriedRowEqualityInput input))
                    column localColumn
                · have productEnd :
                      column <
                        KSplitNcFeTerminal.productBase input + 12 := by
                    have endEq :
                        input.frameBase +
                            KSplitNcFeTerminal.allocationWidth input =
                          KSplitNcFeTerminal.productBase input + 12 := by
                      unfold KSplitNcFeTerminal.allocationWidth
                        KSplitNcFeTerminal.productBase
                        KSplitNcFeTerminal.equalityBase
                        KSplitNcFeTerminal.carriedTargetBase
                        KSplitNcFeTerminal.carriedBase
                        KSplitNcFeTerminal.freshHornerBase
                      omega
                    rw [endEq] at member
                    exact member.2
                  by_cases first :
                      column < KSplitNcFeTerminal.productBase input + 3
                  · have localColumn :
                        column ∈
                          KFrames.frameColumns
                            (KSplitNcFeTerminal.productBase input) 1 := by
                      rw [KFrames.frameColumns_mem_iff]
                      exact ⟨by omega, first⟩
                    unfold KSplitNcFeTerminal.rows
                    exact liftGroup
                      (by
                        simp [KSplitNcFeTerminal.rowGroups,
                          KSplitNcFeTerminal.freshSelectorFrame])
                      (mul
                        (KPointEquality.equalityCarried
                          (KSplitNcFeTerminal.freshLaneEqualityInput input))
                        (KPointEquality.equalityCarried
                          (KSplitNcFeTerminal.freshRowEqualityInput input))
                        (KSplitNcFeTerminal.productBase input) 0)
                      column
                      (by simpa only [Nat.mul_zero, Nat.add_zero] using
                        localColumn)
                  · by_cases second :
                      column < KSplitNcFeTerminal.productBase input + 6
                    · have localColumn :
                          column ∈
                            KFrames.frameColumns
                              (KSplitNcFeTerminal.productBase input + 3) 1 := by
                        rw [KFrames.frameColumns_mem_iff]
                        exact ⟨by omega, second⟩
                      unfold KSplitNcFeTerminal.rows
                      exact liftGroup
                        (by
                          simp [KSplitNcFeTerminal.rowGroups,
                            KSplitNcFeTerminal.freshContributionFrame])
                        (mul
                          (KSplitNcFeTerminal.freshSelector input)
                          (KSplitNcFeTerminal.freshOutput input)
                          (KSplitNcFeTerminal.productBase input) 1)
                        column
                        (by simpa only [Nat.mul_one] using localColumn)
                    · by_cases third :
                        column < KSplitNcFeTerminal.productBase input + 9
                      · have localColumn :
                            column ∈
                              KFrames.frameColumns
                                (KSplitNcFeTerminal.productBase input + 6) 1 := by
                          rw [KFrames.frameColumns_mem_iff]
                          exact ⟨by omega, third⟩
                        unfold KSplitNcFeTerminal.rows
                        exact liftGroup
                          (by
                            simp [KSplitNcFeTerminal.rowGroups,
                              KSplitNcFeTerminal.carriedSelectorFrame])
                          (mul
                            (KPointEquality.equalityCarried
                              (KSplitNcFeTerminal.carriedLaneEqualityInput
                                input))
                            (KPointEquality.equalityCarried
                              (KSplitNcFeTerminal.carriedRowEqualityInput
                                input))
                            (KSplitNcFeTerminal.productBase input) 2)
                          column
                          (by
                            change
                              column ∈
                                KFrames.frameColumns
                                  (KSplitNcFeTerminal.productBase input +
                                    3 * 2) 1
                            simpa only [Nat.reduceMul] using localColumn)
                      · have localColumn :
                            column ∈
                              KFrames.frameColumns
                                (KSplitNcFeTerminal.productBase input + 9) 1 := by
                          rw [KFrames.frameColumns_mem_iff]
                          exact ⟨by omega, productEnd⟩
                        unfold KSplitNcFeTerminal.rows
                        exact liftGroup
                          (by
                            simp [KSplitNcFeTerminal.rowGroups,
                              KSplitNcFeTerminal.carriedContributionFrame])
                          (mul
                            (KSplitNcFeTerminal.carriedSelector input)
                            (KSplitNcFeTerminal.carriedTarget input)
                            (KSplitNcFeTerminal.productBase input) 3)
                          column
                          (by
                            change
                              column ∈
                                KFrames.frameColumns
                                  (KSplitNcFeTerminal.productBase input +
                                    3 * 3) 1
                            simpa only [Nat.reduceMul] using localColumn)

/-- The declared dense endpoint allocation is exactly covered by one of the
three emitted endpoint programs. -/
theorem endpoints
    {shape : SemanticShape}
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (input : KSplitNcEndpoints.Input polynomialInput domains) :
    RowsCover (KSplitNcEndpoints.rows input)
      (KSplitNcEndpoints.columns input) := by
  intro column member
  unfold KSplitNcEndpoints.columns at member
  rw [dense_mem_iff] at member
  by_cases inInitial :
      column < KSplitNcEndpoints.feTerminalBase input
  · have localColumn :
        column ∈
          KSplitNcFeInitial.columns
            (KSplitNcEndpoints.feInitialInput input) := by
      unfold KSplitNcFeInitial.columns
      rw [dense_mem_iff]
      constructor
      · exact member.1
      · simpa only [KSplitNcEndpoints.feTerminalBase] using inInitial
    unfold KSplitNcEndpoints.rows
    exact liftGroup
      (by simp [KSplitNcEndpoints.rowGroups])
      (feInitial (KSplitNcEndpoints.feInitialInput input))
      column localColumn
  · by_cases inTerminal :
      column < KSplitNcEndpoints.ncBase input
    · have localColumn :
          column ∈
            KSplitNcFeTerminal.columns
              (KSplitNcEndpoints.feTerminalInput input) := by
        unfold KSplitNcFeTerminal.columns
        rw [dense_mem_iff]
        constructor
        · change KSplitNcEndpoints.feTerminalBase input ≤ column
          exact Nat.le_of_not_gt inInitial
        · simpa only [KSplitNcEndpoints.ncBase] using inTerminal
      unfold KSplitNcEndpoints.rows
      exact liftGroup
        (by simp [KSplitNcEndpoints.rowGroups])
        (feTerminal (KSplitNcEndpoints.feTerminalInput input))
        column localColumn
    · have localColumn :
          column ∈
            KSplitNcNcEndpoint.columns
              (KSplitNcEndpoints.ncInput input) := by
        unfold KSplitNcNcEndpoint.columns
        rw [dense_mem_iff]
        constructor
        · change KSplitNcEndpoints.ncBase input ≤ column
          exact Nat.le_of_not_gt inTerminal
        · have endEq :
              input.frameBase +
                  KSplitNcEndpoints.allocationWidth input =
                KSplitNcEndpoints.ncBase input +
                  KSplitNcNcEndpoint.allocationWidth
                    (KSplitNcEndpoints.ncInput input) := by
            unfold KSplitNcEndpoints.allocationWidth
              KSplitNcEndpoints.ncBase KSplitNcEndpoints.feTerminalBase
            omega
          rw [endEq] at member
          exact member.2
      unfold KSplitNcEndpoints.rows
      exact liftGroup
        (by simp [KSplitNcEndpoints.rowGroups])
        (ncEndpoint (KSplitNcEndpoints.ncInput input))
        column localColumn

end Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointCoverage
