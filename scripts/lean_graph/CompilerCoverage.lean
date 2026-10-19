import NightstreamFPrime
import tests.AxiomAudit

/-! PiCCS compiler declaration inventory, checked after a source build.
Each leaf lists its predicate, circuit, two directions, parent connection,
and physical costs. The final group checks composition, ownership, and export.
Independent formula review must check the statements and their premises.
This driver does not establish conformance or the full Stage 1 step theorem.
-/

-- Statement input and transcript binding.
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementBinding.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementBinding.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementBinding.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementBinding.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementBinding.spec_implies_keyStatement
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.StatementBinding.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.StatementBinding.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.StatementAbsorption.spec_implies_keyInitialState
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.StatementAbsorption.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.StatementAbsorption.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.ChallengeDerivation.spec_implies_keyExecution_challenges
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.ChallengeDerivation.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.ChallengeDerivation.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.RoundTranscript.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.RoundTranscript.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.RoundTranscript.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.RoundTranscript.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.RoundTranscript.spec_implies_keyExecution_rounds
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.RoundTranscript.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.RoundTranscript.physicalRowCount_eq

-- Initial claim, generic round composition, and the separate terminal families.
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.InitialClaim.spec_implies_keyInitial
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.InitialClaim.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.InitialClaim.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.SumcheckChain.spec_implies_keyChain
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.SumcheckChain.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.SumcheckChain.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalKTerminal.spec_implies_keyPadAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.EvalKTerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.EvalKTerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.EvalATerminal.spec_implies_keyMatrixAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.EvalATerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.EvalATerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.CcsTerminal.spec_implies_keyCcsAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.CcsTerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.CcsTerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.NormTerminal.spec_implies_keyNormAtMessage
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.NormTerminal.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.NormTerminal.physicalRowCount_eq

#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.FinalIdentity.spec_implies_keyTerminal
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.FinalIdentity.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.FinalIdentity.physicalRowCount_eq

-- All outgoing claims and the transcript state.
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding.circuit
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.OutputBinding.spec_implies_keyOutgoingState
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.OutputBinding.freshColumnCount_eq
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Leaves.OutputBinding.physicalRowCount_eq

-- Complete phase composition and physical ownership.
#audit_axioms NightstreamFPrime.Spec.Folding.PiCCS.accepted_iff_coverage
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.Assumptions
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.SpecHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.PhaseHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.soundness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.completeness
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.spec_implies_phaseHolds
#audit_axioms NightstreamFPrime.Lifecycle.PiCCS.v1_2.Formal.circuit
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.physical_implies_holdsFlat
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.physical_implies_phaseHolds
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.physical_complete
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.logicalConstraints_eq_ordered
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.logicalPrivateDeltas_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.physicalRowDeltas_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.physicalColumnDeltas_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.cumulativeFootprints_eq_production
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.jointDomain_le_twoPow28
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Ownership.noBoundaryColumns
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Ownership.noBoundaryRows
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Ownership.ownedRows_length
#audit_axioms NightstreamFPrime.Layout.PiCCS.v1_2.Ownership.columnOwner_unique
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
