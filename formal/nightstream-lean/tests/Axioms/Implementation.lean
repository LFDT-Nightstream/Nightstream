import Nightstream.Implementation
import tests.Axioms.Support

/-!
Fail-closed implementation correspondence axioms gate. Every expectation is checked when this
module is built; the aggregate entrypoint imports all ownership groups.
-/

/-- info: 'Nightstream.Implementation.FPrime.Envelope.check_sound' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.FPrime.Envelope.check_sound

/-- info: 'Nightstream.Implementation.FPrime.CounterRefinement.counter_refinement' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.FPrime.CounterRefinement.counter_refinement

/-- info: 'Nightstream.Implementation.R1CS.bitRow_le_one' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.bitRow_le_one

/-- info: 'Nightstream.Implementation.R1CS.Program.run_agrees_of_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Program.run_agrees_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Program.run_agrees_of_builder_satisfies' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Program.run_agrees_of_builder_satisfies

/-- info: 'Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Program.run_satisfies_builder_rows

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CheckedProgram.sound

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CheckedProgram.complete

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81.sound

/-- info: 'Nightstream.Implementation.R1CS.SeededPhi81.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SeededPhi81.complete

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_satisfies' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.commitmentHolds_of_satisfies' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.commitmentHolds_of_satisfies

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.oneField_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.oneField_sound

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_canonicalRows' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernarySound.canonicalOpening_of_canonicalRows

/-- info: 'Nightstream.Implementation.R1CS.ShiftedTernaryComplete.canonicalRows_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ShiftedTernaryComplete.canonicalRows_complete

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_sound

/-- info: 'Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_complete

/-- info: 'Nightstream.Implementation.R1CS.canonicalU64_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.canonicalU64_sound

/-- info: 'Nightstream.Implementation.R1CS.u64Increment_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.u64Increment_sound

/-- info: 'Nightstream.Implementation.R1CS.u64Add_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.u64Add_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeCounterSound.fPrimeCounter_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeCounterSound.fPrimeCounter_sound

/-- info: 'Nightstream.Implementation.Encoding.FPrime.encInst_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Encoding.FPrime.encInst_injective

/-- info: 'Nightstream.Implementation.Encoding.FPrime.encInst_bits_injective' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Encoding.FPrime.encInst_bits_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeEncodingSound.fPrimeEncoding_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeEncodingSound.fPrimeEncoding_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeEncodingSound.accepted_public_bits_injective' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeEncodingSound.accepted_public_bits_injective

/-- info: 'Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeTerminalLinkSound.fPrimeTerminalLink_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeStateLinkSound.fPrimeStateLink_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeStateLinkSound.fPrimeStateLink_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseStateSound.fPrimeBaseState_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeBaseStateSound.fPrimeBaseState_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeBaseProgramSound.fPrimeBaseProgram_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound.fPrimeFullHistoryBase_step_local_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseStepSound.fPrimeFullHistoryBase_step_local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.outgoing_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.outgoing_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.base_step_holds' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseOutgoingSound.base_step_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.local_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.outgoing_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.outgoing_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.step_holds' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound.step_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.recursive_exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.recursive_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_binding_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_binding_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeChunkDigestSound.fPrimeChunkDigest_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeCeContinuitySound.fPrimeCeContinuity_sound

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.evaluation_sound' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.evaluation_sound

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.census_batchAccepted' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.census_batchAccepted

/-- info: 'Nightstream.Implementation.R1CS.PiRLCProjection.exactRows_imply_batchAccepted' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiRLCProjection.exactRows_imply_batchAccepted

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.topLevel_covers_program

/-- info: 'Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeRecursiveManifest.nifs_covers_block

/-- info: 'Nightstream.Implementation.Rust.FPrime.verify_eq_ok_iff_checkLocal' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.FPrime.verify_eq_ok_iff_checkLocal

/-- info: 'Nightstream.Implementation.Rust.FPrime.success_with_outgoing_refines_step' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.FPrime.success_with_outgoing_refines_step

/-- info: 'Nightstream.Implementation.Rust.FPrime.invalid_has_named_rejection' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.FPrime.invalid_has_named_rejection

/-- info: 'Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.Terminal.success_refines_terminalCE

/-- info: 'Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.Rust.Terminal.invalid_has_named_rejection

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity.sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuity.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.satisfies_iff_holds' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalContinuitySound.satisfies_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.CheckedProgram.satisfies_iff_assignmentHolds' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CheckedProgram.satisfies_iff_assignmentHolds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.satisfies_iff_holds' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalLinkSound.satisfies_iff_holds

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest_eq_outputDigest' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryOutputEncodingSound.terminalFreshDigest_eq_outputDigest

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.xOutValues_sound' depends on axioms: [propext,
 Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.xOutValues_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.terminalFreshDigest_eq_xOut

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveOutputSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.local_sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.local_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.Compiler.complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterLocalSound.Compiler.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.Compiler.complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryCounterSound.Compiler.complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.recursive_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.recursive_sound

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPointBindingSound.recursive_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.terminal_exact_or_badRoot' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjection.terminal_exact_or_badRoot

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.canonical_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_sound' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Recursive.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryAffineSound.Terminal.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_roles_native_order' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_roles_native_order

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_roles_native_order' depends on axioms: [Lean.trustCompiler] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_roles_native_order

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_glue_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.recursive_glue_sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_glue_sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryProjectionRoles.terminal_glue_sound

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictCompiler.check_eq_true_iff' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictCompiler.check_eq_true_iff

/-- info: 'Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.native_complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.ProjectionProgram.ProjectionTrace.native_complete

/-- info: 'Nightstream.Implementation.R1CS.SumcheckRoundSound.native_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SumcheckRoundSound.native_complete

/-- info: 'Nightstream.Implementation.R1CS.SumcheckChainSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.SumcheckChainSound.complete

/-- info: 'Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.native_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.PiDecStrictSound.Exact.native_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryPublicPinsSound.Artifact.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound.complete' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryStateLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalParentLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_sound' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_sound

/-- info: 'Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.TranscriptCertificate.ordered_complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.sound' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.complete' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalRunningLinkSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.sound' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.sound

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalTranscriptSound.complete

/-- info: 'Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_complete' depends on axioms: [propext,
 Classical.choice,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.FPrimeFullHistoryTerminalCeSound.all_claims_complete

/-- info: 'Nightstream.Implementation.R1CS.CanonicalU64Complete.complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CanonicalU64Complete.complete

/-- info: 'Nightstream.Implementation.R1CS.CanonicalU64Complete.mapped_complete' depends on axioms: [propext,
 Lean.trustCompiler,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Nightstream.Implementation.R1CS.CanonicalU64Complete.mapped_complete
