import NightstreamFPrime.Export.Stage1.Wide.PiDECSourceRenaming

/-! Transfer PiDEC row satisfaction from the wide physical witness to the
reference source view used by the retained encoders. No old sampler is run. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiDECSourceWitness

open NightstreamFPrime.Circuit NightstreamFPrime.Layout NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompactRows PiDECSourceRenaming

private theorem restore_injective : Function.Injective restore := by
  intro left right same
  unfold restore at same
  change (if left < 27496062 then left + 208165 else left + 925480) =
    (if right < 27496062 then right + 208165 else right + 925480) at same
  split_ifs at same <;> omega

theorem sourceEnv_restore_of_supported (env : Env) (column : Nat)
    (supported : Layout.Stage1.PiDECSourceSupport.Source (restore column)) :
    SourceAssignment.sourceEnv env (restore column) = env column := by
  let located := (PiDECDirectPlan.classifySource (restore column)).get
    (PiDECDirectPlan.classifySource_complete supported)
  have same : PiDECSource.column located.location = column :=
    restore_injective ((restore_location located.location).trans located.owns)
  rw [← same]
  exact sourceEnv_restore env located.location

variable {width : Nat}
  {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}

private theorem old_constraints_supported (relation : ProductionKey.LogicalRelation width fits) :
    ∀ expression ∈ PiDEC.v1_1.logicalConstraints relation
      (Layout.Stage1.PiDECInputs.interface width fits) Layout.Stage1.PiDECInputs.phaseOffset,
      expression.VarsSatisfy Layout.Stage1.PiDECSourceSupport.Source := by
  have same := PiDECArithmetic.constraints_eq_logical relation
  dsimp only [PiDECArithmetic.phaseInterface] at same
  rw [← same]
  intro expression member
  simp only [PiDECArithmetic.constraints, List.mem_append] at member
  rcases member with ((member | member) | member) | member
  · exact PiDECDirectSupport.publicConstraints_varsSatisfy.get expression member
  · exact PiDECDirectSupport.commitmentConstraints_varsSatisfy.get expression member
  · exact PiDECDirectSupport.evalKConstraints_varsSatisfy.get expression member
  · exact PiDECDirectSupport.evalAConstraints_varsSatisfy.get expression member

theorem old_rows_supported (relation : ProductionKey.LogicalRelation width fits) :
    ∀ row ∈ PiDEC.v1_1.physicalRows relation
      (Layout.Stage1.PiDECInputs.interface width fits) Layout.Stage1.PiDECInputs.phaseOffset,
      row.VarsSatisfy Layout.Stage1.PiDECSourceSupport.Source := by
  rw [PiDEC.v1_1.physicalRows_eq_lowerConstraints]
  intro row member
  have scope := R1CS.lowerConstraints_rows_varsSatisfy _ _ _
    (old_constraints_supported relation) row member
  apply scope.mono row
  intro index supported
  rcases supported with source | fresh
  · exact source
  · apply Layout.Stage1.PiDECSourceSupport.fresh_source
    have count := PiDEC.v1_1.totalFreshCount_eq relation
      (Layout.Stage1.PiDECInputs.interface width fits) Layout.Stage1.PiDECInputs.phaseOffset
      (Layout.Stage1.PiDECInputs.inputShapes relation)
    change _ ≤ index ∧ index < _ at fresh
    rw [count] at fresh
    exact fresh

private theorem combination_support (combination : R1CS.LinearCombination)
    (supported : (renameCombination restore combination).VarsSatisfy
      Layout.Stage1.PiDECSourceSupport.Source) :
    combination.VarsSatisfy (fun column => Layout.Stage1.PiDECSourceSupport.Source (restore column)) := by
  intro term member
  exact supported (restore term.1, term.2) (List.mem_map.mpr ⟨term, member, rfl⟩)

/-- Any accepted physical PiDEC witness also accepts the reference row view.
Only the existing PiDEC input assumptions and physical acceptance are required. -/
theorem rowsHold (relation : ProductionKey.LogicalRelation width fits)
    (ajtai : AjtaiKey (logicalWidth := width) (publicFits := fits)) (env : Env)
    (assumptions : PiDEC.v1_1.Formal.Assumptions relation
      (Layout.Stage1.Wide.PiDECInputs.interface width fits) Layout.Stage1.Wide.PiDECInputs.phaseOffset env)
    (accepted : PiDEC.v1_1.PhysicalHolds relation
      (Layout.Stage1.Wide.PiDECInputs.interface width fits) Layout.Stage1.Wide.PiDECInputs.phaseOffset env) :
    PiDEC.v1_1.PhysicalHolds relation (Layout.Stage1.PiDECInputs.interface width fits)
      Layout.Stage1.PiDECInputs.phaseOffset (SourceAssignment.sourceEnv env) := by
  have phase := PiDEC.v1_1.physical_implies_phaseHolds relation ajtai
    (Layout.Stage1.Wide.PiDECInputs.interface width fits) Layout.Stage1.Wide.PiDECInputs.phaseOffset
    env assumptions accepted
  have mapped := physicalRows_of_phase relation ajtai env assumptions phase
  intro row member
  rw [mapped] at member
  rcases List.mem_map.mp member with ⟨source, sourceMember, rfl⟩
  have oldSupport := old_rows_supported relation (renameRow restore source)
    (by rw [mapped]; exact List.mem_map.mpr ⟨source, sourceMember, rfl⟩)
  apply (renameRow_holds restore source (SourceAssignment.sourceEnv env)).mpr
  apply source.holds_of_agree
    (fun column => Layout.Stage1.PiDECSourceSupport.Source (restore column)) env
    (fun column => SourceAssignment.sourceEnv env (restore column))
  · exact ⟨combination_support _ oldSupport.1, combination_support _ oldSupport.2.1,
      combination_support _ oldSupport.2.2⟩
  · exact sourceEnv_restore_of_supported env
  · exact accepted source sourceMember

end NightstreamFPrime.Export.Stage1.Wide.PiDECSourceWitness
