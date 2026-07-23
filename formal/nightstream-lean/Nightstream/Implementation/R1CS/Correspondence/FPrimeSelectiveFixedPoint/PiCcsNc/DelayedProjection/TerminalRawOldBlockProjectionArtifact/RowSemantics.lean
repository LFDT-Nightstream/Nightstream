import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.PhysicalOwnership

/-!
Exact row semantics for the generated final-round-factorized production
raw-old-block projection.

This leaf proves the generated tensor, raw-child product, final-scale, and
terminal row coefficients equal the optimized compiler rows.  The terminal
family is parent-versus-scale-output; the deleted direct parent-versus-prefix-
sum equations do not occur in the statement or proof.

Owns: coefficient-level equality between every generated artifact row and the
independent optimized compiler equation for its unique four-family owner.

Does not own: runtime emitter columns, assignment values, row satisfaction,
terminal CE, semantic acceptance, security events, costs, or row-removal
authority beyond this exact generated program.

Emits constraints: no.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.projection_rows.tensor` | each generated prefix-tensor row equals its symbolic recurrence row | derived coefficient equality |
| `f_prime.pi_ccs_nc.delayed.projection_rows.raw_child` | each child/lane/block product row equals the symbolic raw-coordinate product | derived coefficient equality |
| `f_prime.pi_ccs_nc.delayed.projection_rows.final_scale` | each generated scale row multiplies the complete lane sum by verifier-owned `1 - oldBlock[18]` | derived coefficient equality |
| `f_prime.pi_ccs_nc.delayed.projection_rows.terminal` | each retained terminal row equates the parent limb to the generated scale output | derived coefficient equality |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionFinalScaleCompiler
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt

private theorem productionPrefixShape :
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.ShapeValid
      productionLayout := by
  refine
    { positiveLanes := productionPositiveLanes
      levelCount := productionLevelCount
      tensorSchedule := productionTensorSchedule
      finalTensorWidth := productionFinalTensorWidth
      tensorDefinitionCanonical := ?_
      rawCoefficientsCanonical := productionRawCoefficientsCanonical
      tensorTraceShape := ?_ }
  · intro level levelMember multiplication
    change level ∈ productionTensorLevels at levelMember
    rcases List.mem_map.mp levelMember with
      ⟨round, _roundMember, rfl⟩
    dsimp only
    intro definition definitionMember
    exact generatedTrace_definitions_canonical
      (tensorTrace round multiplication.val) definition
      (by simpa [productionTensorLevel] using definitionMember)
  · intro level levelMember multiplication
    change level ∈ productionTensorLevels at levelMember
    rcases List.mem_map.mp levelMember with
      ⟨round, _roundMember, rfl⟩
    simpa [productionTensorLevel] using
      generatedTensorTrace_sumLayout round multiplication.val

/-- The generated profile has exactly the optimized compiler shape, including
the explicit final-coordinate association and all 54 five-row scale traces. -/
theorem productionShape : ShapeValid productionFactoredLayout := by
  refine
    { baseShape := productionPrefixShape
      factorEnabled := productionFactorEnabled
      tensorVariables := ?_
      factoredVariable := ?_
      prefixPointColumns := productionPrefixPointColumn
      finalPointColumn := ?_
      blocksFitPrefix := productionBlocksFitPrefix
      scaleOperands := productionFinalScaleOperands
      scaleDefinitionCanonical := productionFinalScaleDefinitionsCanonical
      scaleTraceShape := productionFinalScaleTrace_sumLayout }
  · rw [productionTensorVariables, productionFactoredBase,
      productionBlockVariables]
  · rw [productionFactoredVariable, productionTensorVariables]
  · rw [productionFinalPointColumn, productionFactoredVariable,
      productionFullOldBlockColumn]

private theorem fin18_cases {predicate : Fin 18 -> Prop}
    (case0 : predicate 0) (case1 : predicate 1)
    (case2 : predicate 2) (case3 : predicate 3)
    (case4 : predicate 4) (case5 : predicate 5)
    (case6 : predicate 6) (case7 : predicate 7)
    (case8 : predicate 8) (case9 : predicate 9)
    (case10 : predicate 10) (case11 : predicate 11)
    (case12 : predicate 12) (case13 : predicate 13)
    (case14 : predicate 14) (case15 : predicate 15)
    (case16 : predicate 16) (case17 : predicate 17) :
    forall index, predicate index := by
  intro index
  refine Fin.cases case0 ?_ index
  intro index
  refine Fin.cases case1 ?_ index
  intro index
  refine Fin.cases case2 ?_ index
  intro index
  refine Fin.cases case3 ?_ index
  intro index
  refine Fin.cases case4 ?_ index
  intro index
  refine Fin.cases case5 ?_ index
  intro index
  refine Fin.cases case6 ?_ index
  intro index
  refine Fin.cases case7 ?_ index
  intro index
  refine Fin.cases case8 ?_ index
  intro index
  refine Fin.cases case9 ?_ index
  intro index
  refine Fin.cases case10 ?_ index
  intro index
  refine Fin.cases case11 ?_ index
  intro index
  refine Fin.cases case12 ?_ index
  intro index
  refine Fin.cases case13 ?_ index
  intro index
  refine Fin.cases case14 ?_ index
  intro index
  refine Fin.cases case15 ?_ index
  intro index
  refine Fin.cases case16 ?_ index
  intro index
  have valueZero : index.val = 0 :=
    Nat.eq_zero_of_le_zero (Nat.le_of_lt_succ index.isLt)
  have indexZero : index = (0 : Fin 1) := Fin.ext valueZero
  subst index
  exact case17

private theorem map_range_eq_ofFn
    {Item : Type} (count : Nat) (value : Nat -> Item) :
    (List.range count).map value =
      List.ofFn fun index : Fin count => value index.val := by
  apply List.ext_get
  · simp
  · intro index leftLt rightLt
    simp only [List.get_eq_getElem, List.getElem_map, List.getElem_range,
      List.getElem_ofFn]

private theorem productionRawTerms
    (coordinate : Fin productionLayout.logicalWidth) :
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.rawTerms
        (coordinate.val % 54) (coordinate.val / 54) =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawTerms
        productionLayout coordinate := by
  unfold
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.rawTerms
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawTerms
  rw [map_range_eq_ofFn]
  apply congrArg List.ofFn
  funext child
  apply Prod.ext
  · unfold
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.rawWitnessColumn
    rw [productionChildWitnessFirst, productionActiveLanes,
      productionBlockCount]
    simp [childWitnessColumn, childWitnessFirst, childWitnessFirstNat,
      witnessOffset,
      Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionPlan.blockCount,
      Nat.add_assoc]
  · unfold
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.radixCoefficient
    rw [productionRadix]
    rfl

private theorem productionChiTerms
    (coordinate : Fin productionLayout.logicalWidth) :
    chiTerms (coordinate.val / 54) =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateChiTerms
        productionLayout coordinate := by
  change
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt.tensorTermsAt
        18 (coordinate.val / 54) = _
  exact (productionCoordinateChiTerms coordinate).symm

private theorem productionProductRow
    (coordinate : Fin productionFactoredLayout.base.logicalWidth)
    (limb : Fin 2) :
    productRow (coordinate.val % 54) (coordinate.val / 54) limb.val =
      expectedRow (.coordinate coordinate limb) := by
  change
    productRow (coordinate.val % 54) (coordinate.val / 54) limb.val =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.expectedRow
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.RowIndex.coordinate
          coordinate limb)
  refine Fin.cases ?_ (fun tail => ?_) limb
  · unfold productRow
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.expectedRow
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
    simp only [if_pos]
    rw [productionRawTerms, productionChiTerms]
    rfl
  · have tailZero : tail = 0 := Fin.ext (by omega)
    subst tail
    unfold productRow
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.expectedRow
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.coordinateDefinitions
    simp only [Fin.val_succ, Fin.val_zero, Nat.zero_add, OfNat.ofNat,
      Nat.reduceEqDiff, if_false]
    rw [productionRawTerms, productionChiTerms]
    rfl

private theorem productionTensorRow
    (index : PrefixTensorRowIndex productionFactoredLayout.base) :
    tensorRow index.level.val index.multiplication.val
        index.definition.val = expectedRow (.tensor index) := by
  change
    tensorRow index.level.val index.multiplication.val
        index.definition.val =
      Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.expectedRow
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.RowIndex.tensor
          index)
  rcases index with ⟨level, multiplication, definition⟩
  unfold tensorRow
    Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.expectedRow
  have traceEq :
      tensorTrace level.val multiplication.val =
        Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
          (layout := productionLayout)
          ⟨level, multiplication, definition⟩ := by
    revert multiplication
    refine fin18_cases (predicate := fun current =>
        forall multiplication : Fin
            (productionLayout.tensorLevels.get current).multiplicationCount,
          tensorTrace current.val multiplication.val =
            Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
              (layout := productionLayout)
              ⟨current, multiplication, definition⟩)
      ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_ level <;>
      intro multiplication <;> rfl
  rw [traceEq]
  have definitionLt :
      definition.val <
        (Nightstream.Implementation.R1CS.TerminalRawOldBlockProjectionCompiler.tensorTrace
          (layout := productionLayout)
          ⟨level, multiplication, definition⟩).definitions.length := by
    simpa [KMulTrace.definitions] using definition.isLt
  rw [List.getElem?_eq_getElem definitionLt]
  rfl

private theorem productionScaleRow
    (lane : Fin productionFactoredLayout.base.activeLanes)
    (definition : Fin 5) :
    finalScaleRow lane.val definition.val =
      expectedRow (.scale lane definition) := by
  unfold finalScaleRow expectedRow
  have traceEq :
      finalScaleTrace lane.val = productionFactoredLayout.scale lane :=
    (productionFinalScaleTrace lane).symm
  rw [traceEq]
  simp [KMulTrace.definitions]

private theorem productionTerminalRow
    (lane : Fin productionFactoredLayout.base.activeLanes) (limb : Fin 2) :
    terminalRow lane.val limb.val =
      expectedRow (.terminal lane limb) := by
  refine Fin.cases ?_ (fun tail => ?_) limb
  · unfold terminalRow expectedRow terminalRowsFor terminalTerms
    rfl
  · have tailZero : tail = 0 := Fin.ext (by omega)
    subst tail
    unfold terminalRow expectedRow terminalRowsFor terminalTerms
    rfl

/-- Every generated physical row equals its independently compiled optimized
symbolic row. -/
theorem productionRowAt (index : RowIndex productionFactoredLayout) :
    artifactRow (productionPhysicalIndex index) = expectedRow index := by
  cases index with
  | tensor tensorIndex =>
      unfold artifactRow
      rw [productionOwner_tensor]
      exact productionTensorRow tensorIndex
  | coordinate coordinate limb =>
      unfold artifactRow
      rw [productionOwner_coordinate]
      exact productionProductRow coordinate limb
  | scale lane definition =>
      unfold artifactRow
      rw [productionOwner_scale]
      exact productionScaleRow lane definition
  | terminal lane limb =>
      unfold artifactRow
      rw [productionOwner_terminal]
      exact productionTerminalRow lane limb

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
