import Lean
import NightstreamFPrime

/-! Axiom gate. Every exported theorem is audited here with an explicit
import and an explicit `#print axioms`. The gate fails closed: a theorem whose
axioms are not exactly the allowed set is a build error. -/

open Lean Elab Command in
/-- Fail unless `decl` depends only on `propext`, `Classical.choice`,
`Quot.sound` (or a subset). -/
elab "#audit_axioms " decl:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo decl
  let axioms ← liftCoreM <| (Lean.collectAxioms name)
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let bad := axioms.toList.filter (fun a => !allowed.contains a)
  if bad.isEmpty then
    logInfo m!"{name}: {axioms.toList}"
  else
    throwError m!"{name} depends on disallowed axioms: {bad}"

/-! ## Spec -/
#audit_axioms NightstreamFPrime.Spec.GlobalParams.rlc_bound_for
#audit_axioms NightstreamFPrime.Spec.production_parameter_values
#audit_axioms NightstreamFPrime.Spec.production_norm_stages
#audit_axioms NightstreamFPrime.Spec.production_msis_norm_bound
#audit_axioms NightstreamFPrime.Spec.production_rlc_bound_one_fresh
#audit_axioms NightstreamFPrime.Spec.Poseidon2.constant_table_shape
#audit_axioms NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.check_eq_true_iff_accepted
#audit_axioms NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.complete
#audit_axioms NightstreamFPrime.Spec.SumCheck.Finite.FixedPhase.false_acceptance_implies_bad_challenge
#audit_axioms NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier.complete
#audit_axioms NightstreamFPrime.Spec.Folding.PiDEC.PaperVerifier.reduce_knowledge
#audit_axioms NightstreamFPrime.Spec.Folding.PiRLC.combinedOutput_holds
#audit_axioms NightstreamFPrime.Spec.Folding.PiRLC.complete
#audit_axioms NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolVerifier.check_eq_true_iff_accepted
#audit_axioms NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolVerifier.check_complete_of_accepted
#audit_axioms NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ProtocolVerifier.check_implies_tableTruth_or_badEvent
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.Paper.holds_iff_base_or_recursive
#audit_axioms NightstreamFPrime.Spec.HyperNova.Construction2.Paper.holds_iff_transition
#audit_axioms NightstreamFPrime.Lifecycle.productionShape_sourceCount
#audit_axioms NightstreamFPrime.Circuit.holds_append
#audit_axioms NightstreamFPrime.Circuit.holdsFlat_implies_holds
#audit_axioms NightstreamFPrime.Circuit.executeRecipes_agreesOutside
#audit_axioms NightstreamFPrime.Circuit.executeRecipes_holds_recipeConstraints
#audit_axioms NightstreamFPrime.Circuit.recipesCausal_append
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Layer.externalF_eq_reference
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Layer.internalF_eq_reference
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Layer.fullF_eq_reference
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Layer.partialF_eq_reference
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile_schedule_sound
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile_schedule_causal
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Permutation.compile_schedule_recipe_count
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Hash.hashF_eq_reference
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Hash.compile_sound
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Hash.compile_causal
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Hash.compile_recipes_length
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Formal.soundness
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Formal.completeness
#audit_axioms NightstreamFPrime.Gadgets.Poseidon2.Formal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PriorStateHash.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PriorStateHash.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PriorStateHash.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PriorStateHash.builder_implies_recursive_slot
#audit_axioms NightstreamFPrime.Lifecycle.OutputHash.soundness
#audit_axioms NightstreamFPrime.Lifecycle.OutputHash.completeness
#audit_axioms NightstreamFPrime.Lifecycle.OutputHash.circuit
#audit_axioms NightstreamFPrime.Lifecycle.OutputHash.builder_implies_output_slot
#audit_axioms NightstreamFPrime.Lifecycle.Pilot.phase_soundness
#audit_axioms NightstreamFPrime.Lifecycle.Pilot.builders_imply_hash_slots
#audit_axioms NightstreamFPrime.Layout.R1CS.lowerExpression_sound
#audit_axioms NightstreamFPrime.Layout.R1CS.lowerConstraints_sound
#audit_axioms NightstreamFPrime.Layout.R1CS.lowerConstraints_rows_length
#audit_axioms NightstreamFPrime.Layout.R1CS.lowerConstraints_complete_of_noFresh
#audit_axioms NightstreamFPrime.Layout.Poseidon2.hashPhysical_complete
#audit_axioms NightstreamFPrime.Layout.Pilot.rowOwners_length
#audit_axioms NightstreamFPrime.Layout.Pilot.physicalColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.Pilot.physicalRowCount_eq
#audit_axioms NightstreamFPrime.Layout.Pilot.physical_implies_spec
#audit_axioms NightstreamFPrime.Layout.PilotProduction.physical_complete
#audit_axioms NightstreamFPrime.Layout.PilotProduction.physicalRowCount_eq
#audit_axioms NightstreamFPrime.Layout.PilotProduction.physicalColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PilotProduction.jointDomain_le_twoPow24
#audit_axioms NightstreamFPrime.Layout.PilotProduction.protocolEnv_represents
#audit_axioms NightstreamFPrime.Layout.PilotProduction.protocolEnv_represents_of_agreesBelow
#audit_axioms NightstreamFPrime.Layout.PilotProduction.physical_implies_recursive_hash_slots
#audit_axioms NightstreamFPrime.Layout.PilotSpartan.sourceBoundaries_eq
#audit_axioms NightstreamFPrime.Layout.PilotSpartan.spartanToSource_sourceToSpartan
#audit_axioms NightstreamFPrime.Layout.PilotSpartan.sourceToSpartan_ne_constant
#audit_axioms NightstreamFPrime.Layout.PilotSpartan.remapRows_hold
#audit_axioms NightstreamFPrime.Export.Package.decode_encode
#audit_axioms NightstreamFPrime.Export.Package.artifact_decode_encode
#audit_axioms NightstreamFPrime.Export.Pilot.canonicalRows_length
#audit_axioms NightstreamFPrime.Export.Pilot.circuitPackage_decode_encode
#audit_axioms NightstreamFPrime.Export.Pilot.circuitPackage_row_coverage
#audit_axioms NightstreamFPrime.Export.Pilot.circuitPackage_layout_matches
#audit_axioms NightstreamFPrime.Export.Pilot.artifact_identifier
#audit_axioms NightstreamFPrime.Lifecycle.Transcript.piRlcChallenges_member
#audit_axioms NightstreamFPrime.Lifecycle.ProductionKey.piRlcResponse_valid
#audit_axioms NightstreamFPrime.Lifecycle.PaperAlgebra.evaluations_eq_paper
#audit_axioms NightstreamFPrime.Spec.GoldilocksPrime.goldilocks_natPrime
#audit_axioms NightstreamFPrime.Lifecycle.ProductionKey.key
#audit_axioms NightstreamFPrime.Lifecycle.StepHolds
#audit_axioms NightstreamFPrime.Lifecycle.TerminalHolds
