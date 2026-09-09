import NightstreamFPrime
import tests.AxiomAudit

/-! PiCCS compiler declaration inventory, checked after a source build.
Each leaf lists its predicate, circuit, two directions, parent connection,
and physical costs. The final group checks composition, ownership, and export.
Independent formula review must check the statements and their premises.
This driver does not establish conformance or the full Stage 1 step theorem.
-/

-- Statement input and transcript binding.
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementBinding.spec_implies_keyStatement
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementBinding.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.StatementAbsorption.spec_implies_keyInitialState
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.StatementAbsorption.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.ChallengeDerivation.spec_implies_keyExecution_challenges
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.ChallengeDerivation.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.RoundTranscript.spec_implies_keyExecution_rounds
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.RoundTranscript.physicalRowCount_eq

-- Initial claim, generic round composition, and the separate terminal families.
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.InitialClaim.spec_implies_keyInitial
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.InitialClaim.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.InitialClaim.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.SumcheckChain.spec_implies_keyChain
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.SumcheckChain.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalKTerminal.spec_implies_keyPadAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalKTerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalKTerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.EvalATerminal.spec_implies_keyMatrixAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalATerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.EvalATerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.CcsTerminal.spec_implies_keyCcsAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.CcsTerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.NormTerminal.spec_implies_keyNormAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.NormTerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.NormTerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.FinalIdentity.spec_implies_keyTerminal
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.FinalIdentity.physicalRowCount_eq

-- All outgoing claims and the transcript state.
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.OutputBinding.spec_implies_keyOutgoingState
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Leaves.OutputBinding.physicalRowCount_eq

-- Complete phase composition and physical ownership.
#audit_axioms NightstreamFPrime.Spec.Folding.PiCCS.accepted_iff_coverage
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.Assumptions
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.PhaseHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.spec_implies_phaseHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_1.Formal.circuit
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.physical_implies_holdsFlat
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.physical_implies_phaseHolds
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.physical_complete
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.logicalConstraints_eq_ordered
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.logicalPrivateDeltas_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.physicalRowDeltas_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.physicalColumnDeltas_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.cumulativeFootprints_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.jointDomain_le_twoPow28
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.noBoundaryColumns
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.noBoundaryRows
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.ownedRows_length
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_1.Ownership.columnOwner_unique
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit.rowSpans_cover_layout
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit.rowSpans_pointwise_ownerAgreement
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit.columnSpans_cover_layout
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit.columnSpans_pointwise_ownerAgreement

-- Emitted rows, constructive completeness, and arbitrary assignment decoding.
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSCompleteness.lowerEmittedConstraints_rows
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSCompleteness.complete_arithmeticRows
#audit_axioms NightstreamFPrime.Export.Stage1.PackageCompleteness.complete_piCcsRows
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase.rowsZero_implies_specHolds
#audit_axioms NightstreamFPrime.Export.Stage1.PiCCSDecodedPhase.selectedRowsZero_implies_phaseHolds
#audit_axioms NightstreamFPrime.Export.Stage1.ActualPiCCSInputs.selectedRowsAndPublic_imply_phaseAndHashes
