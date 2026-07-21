import Nightstream.HyperNova
import Nightstream.Protocol
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies
import Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.SideAnchor
import Nightstream.Protocol.FPrime.XOut
import Nightstream.Protocol.FPrime.Paper.CertificateVerifier
import Nightstream.Protocol.FPrime.Paper.Soundness
import Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash
import Nightstream.Protocol.FPrime.Step
import tests.Axioms.Support

/-!
Fail-closed protocol axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-- info: 'Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision

/-- info: 'Nightstream.HyperNova.Construction2.Default.replicatedDefault_allPairs' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Default.replicatedDefault_allPairs

/-- info: 'Nightstream.HyperNova.Construction2.Default.emptyRunning_zeroArity' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Default.emptyRunning_zeroArity

/-- info: 'Nightstream.Protocol.FPrime.Paper.ProgramCounter.ofIndex_raw' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.ProgramCounter.ofIndex_raw

/-- info: 'Nightstream.Protocol.FPrime.Paper.ProgramCounter.index_ofIndex' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.ProgramCounter.index_ofIndex

/-- info: 'Nightstream.Protocol.FPrime.Paper.ProgramCounter.ofIndex_index' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.ProgramCounter.ofIndex_index

/-- info: 'Nightstream.Protocol.FPrime.Paper.holds_of_base' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.holds_of_base

/-- info: 'Nightstream.Protocol.FPrime.Paper.holds_of_recursive' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.holds_of_recursive

/-- info: 'Nightstream.Protocol.FPrime.Paper.NifsVerifier.EdgeWitness.transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.NifsVerifier.EdgeWitness.transition

/-- info: 'Nightstream.Protocol.FPrime.Paper.NifsVerifier.Transition.outputStructure' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.NifsVerifier.Transition.outputStructure

/-- info: 'Nightstream.Protocol.FPrime.Paper.NifsVerifier.EdgeWitness.outputStructure' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.NifsVerifier.EdgeWitness.outputStructure

/-- info: 'Nightstream.Protocol.FPrime.Paper.selected_nifs_transition' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.selected_nifs_transition

/-- info: 'Nightstream.Protocol.FPrime.Paper.RecursiveHolds.runningStructuresBound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.RecursiveHolds.runningStructuresBound

/-- info: 'Nightstream.Protocol.FPrime.Paper.inputIndexedStep_of_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.inputIndexedStep_of_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.paperFPrimeStep_of_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.paperFPrimeStep_of_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.updatedRunning_selected' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.updatedRunning_selected

/-- info: 'Nightstream.Protocol.FPrime.Paper.updatedRunning_other' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.updatedRunning_other

/-- info: 'Nightstream.Protocol.FPrime.Paper.derivedOutput_application' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.derivedOutput_application

/-- info: 'Nightstream.Protocol.FPrime.Paper.derivedOutput_outputHolds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.derivedOutput_outputHolds

/-! Independent fixed-active outer F-prime semantics. -/

/-! Production-shaped fixed-carrier model source baseline. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources.paperHolds' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Sources.paperHolds

/-! Model-level checked running-authority baseline. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority.accepted_of_combinedOpening' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.RunningAuthority.accepted_of_combinedOpening

/-! Complete model-level fixed-active honest baseline context. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.semanticInput' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.semanticInput

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.runningAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.runningAccepted

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.samplerBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.samplerBound

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.honestPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.honestPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.exists_resultTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestBaseline.Context.exists_resultTransition

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.priorPcValid' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.priorPcValid

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.selectedIndex_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.selectedIndex_eq

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.selectedStructures_eq_expected' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.selectedStructures_eq_expected

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.selectedInputAuthority' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Obligations.selectedInputAuthority

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.outerCheck_eq_true_iff' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.outerCheck_eq_true_iff

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_eq_some_iff_physicalChecks' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_eq_some_iff_physicalChecks

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_sound_or_outputUnbound_or_piCcsBadEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_sound_or_outputUnbound_or_piCcsBadEvent

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_sound_or_yRingUnbound_or_piCcsBadEvent_of_packedYZcolBound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_sound_or_yRingUnbound_or_piCcsBadEvent_of_packedYZcolBound

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs.Premises.exists_resultTransition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs.Premises.exists_resultTransition

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs.SemanticPremises.exists_resultTransition_or_samplerShortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.HonestNifs.SemanticPremises.exists_resultTransition_or_samplerShortfall

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_sound_of_closure' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_sound_of_closure

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.exists_run_and_holds_or_samplerShortfall' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.exists_run_and_holds_or_samplerShortfall

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_complete_of_outer_and_honestNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.run_complete_of_outer_and_honestNifs

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.exists_run_and_holds_of_outer_and_honestNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator.exists_run_and_holds_of_outer_and_honestNifs

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.exact' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.exact

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.activeIteration_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.activeIteration_necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.priorPublicLink_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.priorPublicLink_necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.dispatch_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.OuterPlan.dispatch_necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.obligations_iff_side_and_outer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.obligations_iff_side_and_outer

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.holds_iff_exists_side_and_outer' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.holds_iff_exists_side_and_outer

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.SideAnchor.liftCountermodel' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.SideAnchor.liftCountermodel

/-! Exact six-family active-obligation plan. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.accepts_iff_obligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.accepts_iff_obligations

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.exact

/-! Global actual-input obligation plan and minimality closure. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.accepts_iff_obligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.accepts_iff_obligations

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.exact' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.exact

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.lift_local_necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.lift_local_necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.inclusionMinimalSound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.Global.inclusionMinimalSound

/-! Actual outer-family lifts into the six-family plan. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.activeIteration_necessary_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.activeIteration_necessary_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.priorPublicLink_necessary_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.priorPublicLink_necessary_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.dispatch_necessary_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.OuterFamilies.dispatch_necessary_or_samplerShortfall_of_semanticPremises

/-! Actual-type selected-NIFS removal witness. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Realization.necessary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Realization.necessary

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.exists_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.exists_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.necessary_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.necessary_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.necessary_of_honestNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.necessary_of_honestNifs

/-! Global-language lift of the selected-NIFS removal witness. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global.necessary_of_realization' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global.necessary_of_realization

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global.necessary_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global.necessary_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global.necessary_of_honestNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SelectedNifs.Global.necessary_of_honestNifs

/-! Actual-type conditional side-family removal witnesses. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.priorSlot_necessary_of_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.priorSlot_necessary_of_transition

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.priorSlot_necessary_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.priorSlot_necessary_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.priorSlot_necessary_of_honestNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.priorSlot_necessary_of_honestNifs

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.expectedStructure_necessary_of_transition' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.expectedStructure_necessary_of_transition

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.expectedStructure_necessary_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.expectedStructure_necessary_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.expectedStructure_necessary_of_honestNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ObligationPlan.SideFamilies.expectedStructure_necessary_of_honestNifs

/-! Honest construction and actual-type realization of necessity side anchors. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.exists_sideAnchor_of_honestNifs' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.exists_sideAnchor_of_honestNifs

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.exists_sideAnchor_or_samplerShortfall_of_semanticPremises' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.exists_sideAnchor_or_samplerShortfall_of_semanticPremises

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.StableSideMutation.transport' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.StableSideMutation.transport

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ConcreteRealization.lift' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics.Necessity.ConcreteRealization.lift

/-- info: 'Nightstream.Protocol.FPrime.Paper.derivedOutput_base_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.derivedOutput_base_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.base_exists_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.base_exists_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.base_paperFPrimeStep' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.base_paperFPrimeStep

/-- info: 'Nightstream.Protocol.FPrime.Paper.derivedOutput_recursive_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.derivedOutput_recursive_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.recursive_exists_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.recursive_exists_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.recursive_paperFPrimeStep' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.recursive_paperFPrimeStep

/-- info: 'Nightstream.Protocol.FPrime.Paper.defaultValid_transfers_to_base' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.defaultValid_transfers_to_base

/-- info: 'Nightstream.Protocol.FPrime.Paper.setupValid_transfers_to_base' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.setupValid_transfers_to_base

/-! Certificate-oriented paper verifier equivalence. -/

/-- info: 'Nightstream.Protocol.FPrime.Paper.certificateRecursiveVerifier_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.certificateRecursiveVerifier_sound

/-- info: 'Nightstream.Protocol.FPrime.Paper.certificateRecursiveVerifier_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.certificateRecursiveVerifier_complete

/-- info: 'Nightstream.Protocol.FPrime.Paper.certificateRecursiveVerifier_iff_recursiveHolds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.certificateRecursiveVerifier_iff_recursiveHolds

/-- info: 'Nightstream.Protocol.FPrime.Paper.certificateFPrimeVerifier_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.certificateFPrimeVerifier_sound

/-- info: 'Nightstream.Protocol.FPrime.Paper.certificateFPrimeVerifier_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.certificateFPrimeVerifier_complete

/-- info: 'Nightstream.Protocol.FPrime.Paper.certificateFPrimeVerifier_iff_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.certificateFPrimeVerifier_iff_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.certificatePaperFPrimeStep_iff_paperFPrimeStep' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.certificatePaperFPrimeStep_iff_paperFPrimeStep

/-- info: 'Nightstream.Protocol.FPrime.Paper.RecursiveCertificate.inputsValid_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.RecursiveCertificate.inputsValid_or_badEvent

/-- info: 'Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash.outputHash_is_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash.outputHash_is_necessary

/-! Concrete fixed-active outer-to-NIFS context construction. -/

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct_fresh' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct_fresh

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct_running' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Context.Invocation.sourceProduct_running

/-- info: 'Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template.build_runningParent' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.ConcretePhi81.Context.Template.build_runningParent

/-- info: 'Nightstream.Protocol.FPrime.Step.check_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.check_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.check_eq_true_iff_holds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.check_eq_true_iff_holds

/-- info: 'Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing

/-- info: 'Nightstream.Protocol.FPrime.Step.checkLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.checkLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.closeLocal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.closeLocal

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeBase_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeBase_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeRecursive_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.fPrimeRecursive_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.next_state_pinned' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.next_state_pinned

/-- info: 'Nightstream.Protocol.FPrime.Step.holds_advance_facts' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Step.holds_advance_facts

/-- info: 'Nightstream.Protocol.TerminalCE.terminalCE_sound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.TerminalCE.terminalCE_sound

/-- info: 'Nightstream.Protocol.TerminalCE.terminalCE_complete' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.TerminalCE.terminalCE_complete
