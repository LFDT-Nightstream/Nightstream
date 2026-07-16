import Nightstream.HyperNova
import Nightstream.Protocol
import tests.Axioms.Support

/-!
Fail-closed protocol axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-- info: 'Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision

/-- info: 'Nightstream.HyperNova.Construction2.Default.emptyRunning_realizes_default' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.HyperNova.Construction2.Default.emptyRunning_realizes_default

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

/-- info: 'Nightstream.Protocol.FPrime.Paper.NifsVerifier.EdgeWitness.outputStructure' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.NifsVerifier.EdgeWitness.outputStructure

/-- info: 'Nightstream.Protocol.FPrime.Paper.selected_nifs_edge' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.selected_nifs_edge

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

/-- info: 'Nightstream.Protocol.FPrime.Paper.minimalRecursiveVerifier_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.minimalRecursiveVerifier_sound

/-- info: 'Nightstream.Protocol.FPrime.Paper.minimalRecursiveVerifier_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.minimalRecursiveVerifier_complete

/-- info: 'Nightstream.Protocol.FPrime.Paper.minimalRecursiveVerifier_iff_recursiveHolds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.minimalRecursiveVerifier_iff_recursiveHolds

/-- info: 'Nightstream.Protocol.FPrime.Paper.minimalFPrimeVerifier_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.minimalFPrimeVerifier_sound

/-- info: 'Nightstream.Protocol.FPrime.Paper.minimalFPrimeVerifier_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.minimalFPrimeVerifier_complete

/-- info: 'Nightstream.Protocol.FPrime.Paper.minimalFPrimeVerifier_iff_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.minimalFPrimeVerifier_iff_holds

/-- info: 'Nightstream.Protocol.FPrime.Paper.minimalPaperFPrimeStep_iff_paperFPrimeStep' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.minimalPaperFPrimeStep_iff_paperFPrimeStep

/-- info: 'Nightstream.Protocol.FPrime.Paper.RecursiveCertificate.inputsValid_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.RecursiveCertificate.inputsValid_or_badEvent

/-- info: 'Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash.outputHash_is_necessary' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Protocol.FPrime.Paper.Necessity.OutputHash.outputHash_is_necessary

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
