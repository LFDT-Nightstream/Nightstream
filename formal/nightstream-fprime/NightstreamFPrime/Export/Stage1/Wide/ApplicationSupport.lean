import NightstreamFPrime.Export.Stage1.Wide.PilotPoseidonSupport

namespace NightstreamFPrime.Export.Stage1.Wide.ReadSupport

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation FormSupport

theorem application_block (program : Program) {sourceWidth : Nat} (retained : LowNormBlock.Block sourceWidth)
    (start : Nat) (fits : start + retained.coordinateCount ≤ PerApplicationFixedPoint.logicalWidth program)
    (slot : Fin retained.slotCount) (lower : RetainedLayout.applicationStart program ≤ start) :
    Form program (retained.form start fits slot) := by
  apply block
  intro column low high
  have width := RetainedLayout.referenceWidth_eq program
  have appStart := (RetainedLayout.boundaries program).2.2.2
  exact Or.inr (Or.inr (Or.inl ⟨by omega, by omega⟩))

theorem application_input (program : Program)
    (geometry : ApplicationRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (lane : Lifecycle.Stage1.Application.StateIndex) : Form program (ApplicationDirectPlan.inputForm geometry lane) := by
  rw [ApplicationDirectPlan.inputForm_eq_pilot]
  exact prior_word program (ApplicationRetainedGeometry.pilotGeometry geometry) _

theorem application_output (program : Program)
    (geometry : ApplicationRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (lane : Lifecycle.Stage1.Application.StateIndex) : Form program (ApplicationDirectPlan.outputForm geometry lane) := by
  rw [ApplicationDirectPlan.outputForm_eq_pilot]
  exact output_word program (ApplicationRetainedGeometry.pilotGeometry geometry) _

theorem application_location (program : Program)
    (geometry : ApplicationRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program))
    (selected : program.compactHashChain = none) (value : ApplicationOrdinaryPlan.Location program) :
    Form program (value.form (ApplicationRetainedGeometry.ordinaryGeometry geometry selected)) := by
  cases value <;> dsimp only [ApplicationOrdinaryPlan.Location.form]
  case input index => exact application_input program geometry index
  case output index => exact application_output program geometry index
  case witness index => exact application_block program _ _ _ _ (Nat.le_refl _)
  case localValues index =>
    apply application_block
    change RetainedLayout.applicationStart program ≤ RetainedLayout.applicationStart program + _
    omega

theorem compact_application {columns : Nat} {predicate : Fin columns → Prop}
    (interface : Layout.Stage1.Poseidon2HashChainCompact.Interface columns)
    (one : predicate interface.oneColumn)
    (prior : ∀ lane, Supported predicate (interface.priorState lane))
    (message : ∀ lane, Supported predicate (interface.message lane))
    (digest : ∀ lane, Supported predicate (interface.digest lane))
    (sboxes : ∀ invocation slot, Supported predicate (interface.sbox invocation slot)) :
    PlanSupported predicate (Layout.Stage1.Poseidon2HashChainCompact.plan interface) := by
  have output (invocation : Fin 3) (lane : Fin 8) :
      Supported predicate (Layout.Stage1.Poseidon2HashChainCompact.output interface invocation lane) :=
    external _ (fun _ => sboxes invocation _) lane
  have blockLane (forms : Fin 4 → SparseForm columns) (supported : ∀ lane, Supported predicate (forms lane))
      (lane : Fin 8) : Supported predicate (Layout.Stage1.Poseidon2HashChainCompact.blockLane forms lane) := by
    unfold Layout.Stage1.Poseidon2HashChainCompact.blockLane
    split
    · exact supported _
    · exact empty _
  apply append
  · apply poseidon_family
    · exact one
    · intro invocation lane
      dsimp only [Layout.Stage1.Poseidon2HashChainCompact.family]
      unfold Layout.Stage1.Poseidon2HashChainCompact.input
      split_ifs
      · exact add (FormSupport.singleton _ _ one) (blockLane _ prior lane)
      · exact add (output _ lane) (blockLane _ message lane)
      · exact add (output _ lane) (FormSupport.singleton _ _ one)
      · exact output _ lane
    · exact sboxes
  · apply pin_plan
    · exact one
    · intro lane; exact add (digest lane) (scale (-1) (output _ _))

theorem application (program : Program) (fits : PerApplicationPackage.FitsTwoPow28 program)
    (geometry : ApplicationRetainedGeometry.Geometry program (PerApplicationFixedPoint.logicalWidth program)) :
    Plans program (ApplicationDirectPlan.plan fits geometry) := by
  cases selected : program.compactHashChain with
  | none =>
    rw [ApplicationDirectPlan.plan_none fits geometry selected]
    apply source_plan
    · exact one program _ rfl
    · intro row column
      dsimp only [ApplicationOrdinaryPlan.inputs, ApplicationOrdinaryPlan.sourceMap]
      split
      · exact empty _
      · exact application_location program geometry selected _
  | some certificate =>
    rw [ApplicationDirectPlan.plan_some fits geometry certificate selected]
    apply compact_application
    · exact one program _ rfl
    · exact application_input program geometry
    · intro lane; exact application_block program _ _ _ _ (Nat.le_refl _)
    · exact application_output program geometry
    · intro invocation slot
      apply application_block
      change RetainedLayout.applicationStart program ≤ RetainedLayout.applicationStart program + _
      omega

end NightstreamFPrime.Export.Stage1.Wide.ReadSupport
