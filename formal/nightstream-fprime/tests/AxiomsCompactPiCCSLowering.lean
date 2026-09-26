import NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness
import NightstreamFPrime.Layout.SumCheck.CompactChain
import NightstreamFPrime.Layout.PiCCS.v1_1.GammaPowers
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.Support.GammaPowers
import NightstreamFPrime.Lifecycle.PiCCS.v1_1.WitnessSupport
import tests.AxiomAudit

/-! Kernel and cost gates for the compact PiCCS lowering batch. -/

namespace NightstreamFPrime.Tests.CompactPiCCSLowering

theorem sumcheck_local_coordinates : (504 + 1764) * 41 = 92988 := by decide

theorem sumcheck_local_savings :
    424657 - 2324 = 422333 ∧ 424601 * 41 - 92988 = 17315653 := by decide

end NightstreamFPrime.Tests.CompactPiCCSLowering

#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.evaluate_zero
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.evaluate_one
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.booleanSum_eval
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.coefficientSum_varsBelow
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.booleanSum_varsBelow
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.compile_recipes_length
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.compile_checks_length
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.compile_scope
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.compile_chain_iff
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.flatConstraints_opsAt
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.soundness
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.build
#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.localLength_eq
#audit_axioms NightstreamFPrime.Layout.SumCheck.CompactChain.compile_output_linear
#audit_axioms NightstreamFPrime.Layout.SumCheck.CompactChain.compile_costs
#audit_axioms NightstreamFPrime.Layout.SumCheck.CompactChain.production_costs
#audit_axioms NightstreamFPrime.Tests.CompactPiCCSLowering.sumcheck_local_coordinates
#audit_axioms NightstreamFPrime.Tests.CompactPiCCSLowering.sumcheck_local_savings
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.schedule_valid
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.recipes_length
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.wire_varsBelow
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.recipe_varsBelow
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.recipes_causal
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.product_sound
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.wire_sound
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.build
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.GammaPowers.transcript_wire_costs
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.GammaPowers.local_coordinate_count
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.GammaPowers.local_savings

#audit_axioms NightstreamFPrime.Gadgets.SumCheck.CompactChain.compile_cons
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit_ops
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.semanticOutput_varsBelow
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.pointLength_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.GammaPowers.input_costs
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.localLength_eq
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.flatConstraints_eq
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.flatConstraints_length
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.flatConstraints_varsBelow
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.wire_varsSatisfy
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.flatConstraints_varsSatisfy
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.GammaPowers.witnessesFromConstraints


#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness.source_input
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness.source_rows
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness.output_eq_permute
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness.equations
#audit_axioms NightstreamFPrime.Layout.ProductionRelation.PoseidonCompactWitness.family_member
