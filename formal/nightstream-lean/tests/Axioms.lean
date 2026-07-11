import Nightstream.Implementation.FPrime.Envelope
import Nightstream.Implementation.FPrime.CounterRefinement
import Nightstream.Implementation.R1CS.CanonicalU64Sound
import Nightstream.Implementation.R1CS.CanonicalU64Complete
import Nightstream.Implementation.R1CS.Program
import Nightstream.Implementation.R1CS.CheckedProgram
import Nightstream.Implementation.R1CS.SeededPhi81
import Nightstream.Implementation.R1CS.ShiftedTernarySound
import Nightstream.Implementation.R1CS.ShiftedTernaryComplete
import Nightstream.Implementation.R1CS.Poseidon2PermutationSound
import Nightstream.Implementation.R1CS.U64IncrementSound
import Nightstream.Implementation.R1CS.U64AddSound
import Nightstream.Implementation.R1CS.FPrimeCounterSound
import Nightstream.Implementation.Encoding.FPrime
import Nightstream.Implementation.R1CS.FPrimeEncodingSound
import Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound
import Nightstream.Implementation.R1CS.FPrimeStateLinkSound
import Nightstream.Implementation.R1CS.FPrimeBaseStateSound
import Nightstream.Implementation.R1CS.FPrimeBaseProgramSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound
import Nightstream.Implementation.R1CS.PiDecStrictSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuityArtifact
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound
import Nightstream.Implementation.R1CS.FPrimeChunkDigestSound
import Nightstream.Implementation.R1CS.FPrimeCeContinuitySound
import Nightstream.Implementation.R1CS.ProjectionBatchSound
import Nightstream.Implementation.R1CS.PiRLCProjectionSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles
import Nightstream.Assurance.FPrimeConcreteNifs
import Nightstream.Assurance.FPrimeFullHistorySemantics
import Nightstream.Assurance.FPrimeFullHistoryCircuitComplete
import Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound
import Nightstream.SuperNeo.Concrete.Relation
import Nightstream.SuperNeo.Concrete.Parameters
import Nightstream.SuperNeo.SumCheck
import Nightstream.SuperNeo.Folding.PiCCS
import Nightstream.SuperNeo.Folding.PiRLC
import Nightstream.SuperNeo.Folding.PiDEC
import Nightstream.SuperNeo.Folding.Composition
import Nightstream.Protocol.FPrime.XOut
import Nightstream.Protocol.FPrime.Step
import Nightstream.Protocol.Terminal.CE
import Nightstream.Implementation.Rust.FPrime
import Nightstream.Implementation.Rust.Terminal
import Nightstream.HyperNova.Construction2.Default
import Nightstream.Assurance.FPrimeTrace
import Nightstream.Assurance.FPrimeCircuit
import Nightstream.Assurance.FPrimeRecursiveCircuit

/-!
Fail-closed axioms gate: `#guard_msgs` fails the build if the axioms report of
a completed theorem ever differs from the recorded expectation — a theorem
that silently picks up `sorryAx`, `Classical.choice`, or other new axioms
breaks this file instead of printing an ignored info line.
-/

/-- info: 'Nightstream.Implementation.FPrime.Envelope.check_sound' does not depend on any axioms -/
#guard_msgs in
#print axioms Nightstream.Implementation.FPrime.Envelope.check_sound

/-- info: 'Nightstream.Implementation.FPrime.CounterRefinement.counter_refinement' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Implementation.FPrime.CounterRefinement.counter_refinement

/-- info: 'Nightstream.Implementation.R1CS.bitRow_le_one' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.bitRow_le_one

/-- info: 'Nightstream.Implementation.R1CS.Program.run_agrees_of_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.Program.run_agrees_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Program.run_agrees_of_builder_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.Program.run_agrees_of_builder_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.CheckedProgram.sound

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.CheckedProgram.complete

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.SeededPhi81.sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.SeededPhi81.complete

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_satisfies' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.commitmentHolds_of_satisfies' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.commitmentHolds_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.oneField_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.oneField_sound

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_canonicalRows' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryComplete.canonicalRows_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ShiftedTernaryComplete.canonicalRows_complete

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete

/-- info: 'Nightstream.Implementation.R1CS.canonicalU64_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.canonicalU64_sound

/-- info: 'Nightstream.Implementation.R1CS.u64Increment_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.u64Increment_sound

/-- info: 'Nightstream.Implementation.R1CS.u64Add_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.u64Add_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeCounterSound.fPrimeCounter_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeCounterSound.fPrimeCounter_sound

/-- info: 'Nightstream.Implementation.Encoding.FPrime.encInst_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.Encoding.FPrime.encInst_injective

/-- info: 'Nightstream.Implementation.Encoding.FPrime.encInst_bits_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.Encoding.FPrime.encInst_bits_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeEncodingSound.fPrimeEncoding_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeEncodingSound.fPrimeEncoding_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeEncodingSound.accepted_public_bits_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeEncodingSound.accepted_public_bits_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeStateLinkSound.fPrimeStateLink_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeStateLinkSound.fPrimeStateLink_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseStateSound.fPrimeBaseState_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeBaseStateSound.fPrimeBaseState_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound.fPrimeFullHistoryBase_step_local_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound.fPrimeFullHistoryBase_step_local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.outgoing_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.outgoing_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.base_step_holds' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.base_step_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.local_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.outgoing_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.outgoing_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.step_holds' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.step_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.output_binding' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.output_binding

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.recursive_exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.recursive_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.local_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.local_sound_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.step_holds_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound.step_holds_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_binding_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_binding_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.evaluation_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.evaluation_sound

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.census_batchAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.census_batchAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRLCProjection.exactRows_imply_batchAccepted' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.PiRLCProjection.exactRows_imply_batchAccepted

/-- info: 'Nightstream.SuperNeo.Concrete.ccsMembership_iff' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Concrete.ccsMembership_iff

/-- info: 'Nightstream.SuperNeo.Concrete.ceMembership_iff' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Concrete.ceMembership_iff

/-- info: 'Nightstream.SuperNeo.Concrete.canonicalCCS_holds' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Concrete.canonicalCCS_holds

/-- info: 'Nightstream.SuperNeo.Concrete.canonicalCE_holds' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Concrete.canonicalCE_holds

/-- info: 'Nightstream.SuperNeo.GlobalParams.rlc_bound_for' does not depend on any axioms -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.GlobalParams.rlc_bound_for

/-- info: 'Nightstream.SuperNeo.SumCheck.false_acceptance_implies_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.SumCheck.false_acceptance_implies_bad_challenge

/-- info: 'Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.SumCheck.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.strong_extract_or_bad_challenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiCCS.strong_extract_or_bad_challenge

/-- info: 'Nightstream.SuperNeo.Folding.BatchArity.total_le' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.BatchArity.total_le

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.product_complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiCCS.product_complete

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiCCS.complete

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.combinedOutput_holds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiRLC.combinedOutput_holds

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiRLC.complete

/-- info: 'Nightstream.SuperNeo.Folding.PiRLC.same_phi_extractions_unique_or_collision' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiRLC.same_phi_extractions_unique_or_collision

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.reduce_knowledge' depends on axioms: [Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiDEC.reduce_knowledge

/-- info: 'Nightstream.SuperNeo.Folding.PiDEC.complete' does not depend on any axioms -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.PiDEC.complete

/-- info: 'Nightstream.SuperNeo.Folding.Composition.fold_knowledge_or_bad_event' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.Folding.Composition.fold_knowledge_or_bad_event

/-- info: 'Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.XOut.xOut_binding_or_collision

/-- info: 'Nightstream.HyperNova.Construction2.Default.emptyRunning_realizes_default' does not depend on any axioms -/
#guard_msgs in
#print axioms Nightstream.HyperNova.Construction2.Default.emptyRunning_realizes_default

/-- info: 'Nightstream.Protocol.FPrime.Step.check_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.check_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.check_eq_true_iff_holds' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.check_eq_true_iff_holds

/-- info: 'Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.holds_iff_local_and_outgoing

/-- info: 'Nightstream.Protocol.FPrime.Step.checkLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.checkLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.fPrimeBaseLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.fPrimeRecursiveLocal_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.closeLocal' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.closeLocal

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeBase_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.fPrimeBase_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.fPrimeRecursive_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.fPrimeRecursive_sound

/-- info: 'Nightstream.Protocol.FPrime.Step.next_state_pinned' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.next_state_pinned

/-- info: 'Nightstream.Protocol.FPrime.Step.holds_advance_facts' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.FPrime.Step.holds_advance_facts

/-- info: 'Nightstream.Assurance.FPrimeTrace.accepted_trace_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeTrace.accepted_trace_sound

/-- info: 'Nightstream.Assurance.FPrimeTrace.accepted_trace_valid_execution' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeTrace.accepted_trace_valid_execution

/-- info: 'Nightstream.Assurance.FPrimeCircuit.split_check_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeCircuit.split_check_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program' does not depend on any axioms -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block' does not depend on any axioms -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block

/-- info: 'Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.SuperNeo.ProjectionCheck.batchAccepted_implies_exact_or_badRoot

/-- info: 'Nightstream.Protocol.TerminalCE.terminalCE_sound' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.TerminalCE.terminalCE_sound

/-- info: 'Nightstream.Protocol.TerminalCE.terminalCE_complete' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Protocol.TerminalCE.terminalCE_complete

/-- info: 'Nightstream.Implementation.Rust.FPrime.verify_eq_ok_iff_checkLocal' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.Rust.FPrime.verify_eq_ok_iff_checkLocal

/-- info: 'Nightstream.Implementation.Rust.FPrime.success_with_outgoing_refines_step' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.Rust.FPrime.success_with_outgoing_refines_step

/-- info: 'Nightstream.Implementation.Rust.FPrime.invalid_has_named_rejection' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.Rust.FPrime.invalid_has_named_rejection

/-- info: 'Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE

/-- info: 'Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.satisfies_iff_holds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.satisfies_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.satisfies_iff_assignmentHolds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.CheckedProgram.satisfies_iff_assignmentHolds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.satisfies_iff_holds' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.satisfies_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest_eq_outputDigest' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest_eq_outputDigest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.xOutValues_sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.xOutValues_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.local_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.Compiler.complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.Compiler.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.Compiler.complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.Compiler.complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.recursive_sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.recursive_sound

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.terminal_exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.terminal_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_roles_native_order' depends on axioms: [Lean.ofReduceBool,
 Lean.trustCompiler] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_roles_native_order

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_roles_native_order' depends on axioms: [Lean.ofReduceBool,
 Lean.trustCompiler] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_roles_native_order

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_glue_sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_glue_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_glue_sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_glue_sound

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_artifact_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_artifact_sound_or_badRoot

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminal_artifact_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.terminal_artifact_sound_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictCompiler.check_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.PiDecStrictCompiler.check_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursiveCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.recursiveCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminalCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.terminalCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursiveNativeCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.recursiveNativeCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminalNativeCheck_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.terminalNativeCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics_nifsVerify' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.stepSemantics_nifsVerify

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_verify_sound_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_verify_sound_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.native_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.native_complete

/-- info: 'Nightstream.Implementation.R1CS.SumcheckRoundSound.native_complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.SumcheckRoundSound.native_complete

/-- info: 'Nightstream.Implementation.R1CS.SumcheckChainSound.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.SumcheckChainSound.complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.native_complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.native_complete

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_complete

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.terminal_rows_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.terminal_rows_complete

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursiveContextCheck_eq_true_iff' depends on axioms: [propext] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.recursiveContextCheck_eq_true_iff

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentShapeAgrees' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentShapeAgrees

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentShapeAgrees' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentShapeAgrees

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentSerializes' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.RecursiveSemanticAccepted.parentSerializes

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentSerializes' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.TerminalSemanticAccepted.parentSerializes

/-- info: 'Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws_of_start' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws_of_start

/-- info: 'Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeFullHistorySemantics.recursiveCoreLaws

/-- info: 'Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_nifsVerify_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeConcreteNifs.recursive_rows_nifsVerify_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_sound

/-- info: 'Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.sound' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_complete

/-- info: 'Nightstream.Implementation.R1CS.CanonicalU64Complete.complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.CanonicalU64Complete.complete

/-- info: 'Nightstream.Implementation.R1CS.CanonicalU64Complete.mapped_complete' depends on axioms: [propext,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Implementation.R1CS.CanonicalU64Complete.mapped_complete

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_sound_or_bad

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_complete

/-- info: 'Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_execution_sound_or_bad' depends on axioms: [propext,
 Classical.choice,
 Lean.ofReduceBool,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#print axioms Nightstream.Assurance.FPrimeFullHistoryCircuit.fPrimeCircuit_execution_sound_or_bad
