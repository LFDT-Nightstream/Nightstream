import NightstreamFPrime.Export.Stage1.ApplicationDirectPlan

/-!
Owns application soundness for arbitrary assignments to the final logical
columns. Decoding evaluates the existing source map. Accepted application
rows bind the actual pilot input and output preimages to the selected step.
-/

namespace NightstreamFPrime.Export.Stage1.ApplicationAssignmentSoundness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open ApplicationDirectPlan

abbrev Program := Lifecycle.Stage1.Application.Program

/-- The source environment is determined by the logical assignment and the
existing application resolver. No source-value agreement is assumed. -/
def decodedEnv {application : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) : Env :=
  SourceCompiler.sourceEnv fun column =>
    ((sourceMap geometry).form column).eval assignment

private theorem locationSource_injective {application : Program} :
    Function.Injective (Location.sourceColumn (application := application)) := by
  intro left right same
  cases left <;> cases right
  all_goals
    rename_i leftIndex rightIndex
    have leftBound := leftIndex.isLt
    have rightBound := rightIndex.isLt
    simp only [Location.sourceColumn,
      Layout.Stage1.ApplicationInputs.inputColumn_value,
      Layout.Stage1.ApplicationInputs.outputColumn_value,
      Layout.Stage1.ApplicationInputs.witnessColumn,
      Layout.Stage1.ApplicationInputs.localStart,
      Layout.Stage1.ApplicationInputs.currentWordStart,
      Layout.Stage1.ApplicationInputs.witnessStart,
      Layout.Stage1.Spartan.privateColumnCount] at same
    simp only [Lifecycle.Stage1.Application.stateWordCount] at leftBound rightBound
    first
    | exact congrArg Location.input (Fin.ext (by omega))
    | exact congrArg Location.witness (Fin.ext (by omega))
    | exact congrArg Location.output (Fin.ext (by omega))
    | exact congrArg Location.localValues (Fin.ext (by omega))
    | omega

private theorem locationSource_bound {application : Program}
    (location : Location application) :
    location.sourceColumn < ApplicationRetainedBlocks.sourceWidth application := by
  cases location with
  | input index => exact ((ApplicationRetainedBlocks.inputBlock application).source index).isLt
  | witness index => exact ((ApplicationRetainedBlocks.witnessBlock application).source index).isLt
  | output index => exact ((ApplicationRetainedBlocks.outputBlock application).source index).isLt
  | localValues index => exact ((ApplicationRetainedBlocks.localBlock application).source index).isLt

private theorem locationSource_allowed {application : Program}
    (location : Location application) :
    ApplicationDirectSource.SourceAllowed application location.sourceColumn := by
  cases location with
  | input index => exact Or.inl ⟨index, rfl⟩
  | witness index => exact Or.inr (Or.inl ⟨index, rfl⟩)
  | output index => exact Or.inr (Or.inr (Or.inl ⟨index, rfl⟩))
  | localValues index =>
      exact Or.inr (Or.inr (Or.inr ⟨by
        change Layout.Stage1.ApplicationInputs.localStart application ≤
          Layout.Stage1.ApplicationInputs.localStart application + index.val
        omega, locationSource_bound (.localValues index)⟩))

private theorem decodedEnv_location {application : Program} {logicalWidth : Nat}
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth) (location : Location application) :
    decodedEnv geometry assignment location.sourceColumn =
      (location.form geometry).eval assignment := by
  unfold decodedEnv
  rw [SourceCompiler.sourceEnv_at _
    ⟨location.sourceColumn, locationSource_bound location⟩]
  have complete := classifySource_complete application (locationSource_allowed location)
  cases found : classifySource application location.sourceColumn with
  | none => simp [found] at complete
  | some located =>
      have same : located.location = location := locationSource_injective located.owns
      simp only [ApplicationDirectPlan.sourceMap, found, same]

/-- The application plan is exactly its physical source rows evaluated in
the environment decoded from this arbitrary logical assignment. -/
theorem rowsZero_iff_rowsHold {application : Program} {logicalWidth : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ApplicationRetainedGeometry.oneColumn geometry) = 1) :
    (ApplicationDirectPlan.plan fits geometry).RowsZero assignment ↔
      R1CS.RowsHold (decodedEnv geometry assignment)
        (ApplicationDirectSource.sourceRows application) := by
  have preserves : (sourceMap geometry).Preserves assignment
      (decodedEnv geometry assignment) := by
    intro column
    exact (SourceCompiler.sourceEnv_at
      (fun sourceColumn => ((sourceMap geometry).form sourceColumn).eval assignment)
      column).symm
  have rowPreserves : ∀ index, OrdinarySourcePlan.SourceMap.PreservesRow
      ((inputs fits geometry).sourceMap index) assignment
      (decodedEnv geometry assignment)
      ((ApplicationDirectSource.program application fits).row index)
      ((ApplicationDirectSource.program application fits).bounded index) := by
    intro index
    refine ⟨?_, ?_, ?_⟩
    · intro term member
      exact preserves ⟨term.1,
        ((ApplicationDirectSource.program application fits).bounded index).1 term member⟩
    · intro term member
      exact preserves ⟨term.1,
        ((ApplicationDirectSource.program application fits).bounded index).2.1 term member⟩
    · intro term member
      exact preserves ⟨term.1,
        ((ApplicationDirectSource.program application fits).bounded index).2.2 term member⟩
  have bridge := OrdinarySourcePlan.Program.rowsZero_iff
    (ApplicationDirectSource.program application fits) (inputs fits geometry)
    assignment (decodedEnv geometry assignment) one rowPreserves
  exact bridge.trans (ApplicationDirectSource.program_holds_iff_rowsHold
    application fits (decodedEnv geometry assignment))

/-- Every accepted logical assignment makes the actual pilot output current
state equal to the selected application step on its actual pilot input current
state and its decoded witness. No canonical encoding premise is required. -/
theorem rowsZero_implies_step {application : Program} {logicalWidth : Nat}
    (fits : PerApplicationPackage.FitsTwoPow28 application)
    (geometry : ApplicationRetainedGeometry.Geometry application logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (ApplicationRetainedGeometry.oneColumn geometry) = 1)
    (rows : (ApplicationDirectPlan.plan fits geometry).RowsZero assignment) :
    (List.ofFn fun index : Lifecycle.Stage1.Application.StateIndex =>
      ((PiRLCPoseidonGeometry.outputInputBlock application).form
        (PiRLCPoseidonGeometry.outputInputStart application)
        (PiRLCPoseidonGeometry.outputInputFits
          (ApplicationRetainedGeometry.pilotGeometry geometry))
        (Location.preimageWord index)).eval assignment) =
      application.step
        (List.ofFn fun index : Lifecycle.Stage1.Application.StateIndex =>
          ((PiRLCPoseidonGeometry.priorInputBlock application).form
            (PiRLCPoseidonGeometry.priorInputStart application)
            (PiRLCPoseidonGeometry.priorInputFits
              (ApplicationRetainedGeometry.pilotGeometry geometry))
            (Location.preimageWord index)).eval assignment)
        (List.ofFn fun index : Fin application.witnessWordCount =>
          ((Location.witness index).form geometry).eval assignment) := by
  have holds := ApplicationDirectSource.rowsHold_implies_applicationHolds
    application (decodedEnv geometry assignment)
    ((rowsZero_iff_rowsHold fits geometry assignment one).mp rows)
  have inputEq :
      Lifecycle.Stage1.Application.inputState
        (Layout.Stage1.ApplicationInputs.interface application)
        (Layout.Stage1.ApplicationInputs.localStart application)
        (decodedEnv geometry assignment) =
      List.ofFn (fun index : Lifecycle.Stage1.Application.StateIndex =>
        ((PiRLCPoseidonGeometry.priorInputBlock application).form
          (PiRLCPoseidonGeometry.priorInputStart application)
          (PiRLCPoseidonGeometry.priorInputFits
            (ApplicationRetainedGeometry.pilotGeometry geometry))
          (Location.preimageWord index)).eval assignment) := by
    apply congrArg List.ofFn
    funext index
    change decodedEnv geometry assignment
      (Location.input (application := application) index).sourceColumn = _
    rw [decodedEnv_location, Location.input_form_eq_pilot]
  have outputEq :
      Lifecycle.Stage1.Application.outputState
        (Layout.Stage1.ApplicationInputs.interface application)
        (Layout.Stage1.ApplicationInputs.localStart application)
        (decodedEnv geometry assignment) =
      List.ofFn (fun index : Lifecycle.Stage1.Application.StateIndex =>
        ((PiRLCPoseidonGeometry.outputInputBlock application).form
          (PiRLCPoseidonGeometry.outputInputStart application)
          (PiRLCPoseidonGeometry.outputInputFits
            (ApplicationRetainedGeometry.pilotGeometry geometry))
          (Location.preimageWord index)).eval assignment) := by
    apply congrArg List.ofFn
    funext index
    change decodedEnv geometry assignment
      (Location.output (application := application) index).sourceColumn = _
    rw [decodedEnv_location, Location.output_form_eq_pilot]
  have witnessEq :
      Lifecycle.Stage1.Application.witnessValue
        (Layout.Stage1.ApplicationInputs.interface application)
        (Layout.Stage1.ApplicationInputs.localStart application)
        (decodedEnv geometry assignment) =
      List.ofFn (fun index : Fin application.witnessWordCount =>
        ((Location.witness index).form geometry).eval assignment) := by
    apply congrArg List.ofFn
    funext index
    exact decodedEnv_location geometry assignment (.witness index)
  unfold Lifecycle.Stage1.Application.Holds at holds
  rw [inputEq, outputEq, witnessEq] at holds
  exact holds

end NightstreamFPrime.Export.Stage1.ApplicationAssignmentSoundness
