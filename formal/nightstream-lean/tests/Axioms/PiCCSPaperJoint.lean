import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
import tests.Axioms.Support

/-!
Fail-closed trusted-dependency gate for the paper-anchored joint `Pi_CCS`
coefficient model.

Owns: dependency expectations for coefficient layout, canonical Boolean-table
zero residualization, shared Boolean-domain ordering, explicit CCS residual
construction, strict-norm residual semantics, carried-coordinate ordering,
the exponent-mismatch audit, finite target-shift identity, deterministic
mixing-root reduction, and output point binding.

Does not own: the explicitly open concrete arithmetization, SumCheck,
transcript, SplitNc, implementation, or security-reduction boundaries.

| Audited theorem | Model-level guarantee |
|---|---|
| `jointCoefficients_length` | the three gamma blocks exactly cover the finite coefficient list |
| `canonicalExponentVectors_nodup` | the verifier-owned squarefree basis has no duplicate monomial |
| `toAlphaPolynomial_coefficientZero_iff_allEntriesZero` | derived coefficients are zero iff all explicit leaves are zero |
| `toAlphaPolynomial_evaluate_eq_evaluate` | canonical polynomial evaluation equals the independent recursive table MLE |
| `BooleanVertex.all_nodup` | the shared semantic cube enumeration has no repeated point |
| `canonicalFinIndices_nodup` | the canonical finite-column enumeration has no repeated coordinate |
| `evaluate_eq_equalityWeightedSum` | the recursive MLE equals the explicit `sum_x eq(x,r) * table[x]` |
| `toAlphaPolynomial_evaluate_eq_equalityWeightedSum` | the canonical alpha polynomial equals that same explicit hypercube sum |
| `residualTable_allEntriesZero_iff_constraintSatisfied` | explicit matrix/polynomial CCS leaves are zero iff the independent CCS obligation holds |
| `residualPolynomial_coefficientZero_iff_constraintSatisfied` | canonical CCS alpha coefficients are zero iff the independent CCS obligation holds |
| `cubicResidual_eq_zero_iff_strictNormTwo` | the base cubic recognizes exactly the strict centered `b = 2` window under no zero divisors |
| `allCubicResidualsZero_iff_normBoundedTwo` | pointwise base cubics characterize the semantic assignment norm |
| `representedRoots_nodup` | the three canonical residue representatives are distinct |
| `residualTable_allEntriesZero_iff_normBounded` | each typed norm table is zero iff its canonical semantic assignment satisfies `normBounded 2` |
| `allResidualTablesZero_iff_allStrictNormBounded` | the complete `K+k` typed norm-table family is exact |
| `imageTable_evaluate_eq_computedCoefficient` | a carried matrix-image table MLE equals its explicit equality-weighted sum |
| `allResidualsZero_iff_allClaimsHold` | all claimed-minus-derived carried residuals vanish iff every evaluation equation holds |
| `canonicalCarriedCoordinates_localGammaExponents` | typed carried traversal is exactly the consecutive local gamma support `0..ktd-1` |
| `paperDifference_eq_signedResidualBlocks` | the corrected target minus explicit hypercube sum of pointwise `Q` equals the exact signed CCS/norm/carried residual blocks |
| `SignedCoefficientPolynomial.paperDifference_eq_evaluate` | that signed identity is exactly executable Horner evaluation of the constant-first three-block coefficient list |
| `SignedCoefficientObject.specializedCoefficients_eq` | independent finite alpha-polynomial/scalar coefficients specialize to the exact signed Horner list |
| `SignedCoefficientObject.coefficientTruth_iff_tableObligations` | signed coefficient truth is exactly the independent explicit table obligations |
| `SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot` | sampled signed equality is exactly coefficient truth or the named mixing root |
| `SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero` | verifier `T_abs` equals semantic `sum_x Q` iff the signed polynomial vanishes |
| `SumCheckInitial.checked_implies_tableObligations_or_mixingRoot_or_badChallenge` | executable finite acceptance plus an honest truth path yields table truth, a mixing root, or a named round collision |
| `SumCheckTruthPath.qAtPoint_toCubePoint_eq_qAt` | the explicit arbitrary-point joint polynomial restricts to the independent Boolean `Q` |
| `SumCheckTruthPath.sumCompletions_jointPolynomial_eq_summedQ` | the canonical Boolean completion sum is exactly the paper `sum_x Q` |
| `SumCheckTruthPath.canonicalGhosts_honest` | initial, expected rounds, and terminal come from the same explicit joint polynomial |
| `SumCheckInitial.checkedCanonical_implies_tableObligations_or_mixingRoot_or_badChallenge` | executable finite acceptance needs no caller-supplied expected callback or honesty proof |
| `SumCheckInitial.checkJoint_implies_tableObligations_or_mixingRoot_or_badChallenge` | typed challenge arity discharges exact one-round-per-variable shape |
| `ConcreteJointData.liftTable_allEntriesZero_iff` | zero-reflecting placement transports exact table truth across carriers |
| `ConcreteJointData.jointTableTruth_iff_semanticTruth` | the sole constructed joint tables are true iff independent CCS, norm, and carried semantics hold |
| `ConcreteJointData.coefficientTruth_iff_semanticTruth` | unsampled coefficient truth is exactly the independent semantic conjunction |
| `ConcreteJointData.checkJoint_implies_semanticTruth_or_badEvent` | executable one-joint acceptance reaches independent semantics or a named bad event |
| `MatrixSource.coefficientMatrix_constant_eq` | every constant carried coefficient matrix is the sole CCS matrix under the explicit kernel law |
| `ConnectedInputs.carriedImageConstantAt_eq_ccsImageAt` | the constant carried image equals the CCS image for the same authoritative assignment |
| `Phi81CoefficientKernel.basisConstantTerm` | actual Phi81 multiplication and the closed-form bar basis satisfy the 54-by-54 Kronecker law |
| `Phi81CarrierLayout.extendAssignment_tail_zero` | fresh assignment completion owns a canonical-zero suffix |
| `Phi81CarrierLayout.extendMatrix_tail_zero` | the sole CCS matrix is zero-extended to a complete 54-lane carrier |
| `Phi81CarrierLayout.layout_encode?_isSome` | every completed block/lane pair is a real carried CE coordinate |
| `Phi81MatrixSource.source_matrix_tail_zero` | the specialized sole matrix has no caller-controlled completed suffix |
| `Phi81MatrixSource.coefficientMatrix_constant_apply` | the derived constant coefficient equals that completed sole matrix everywhere |
| `PaperLinearAlgebra.matrixVectorAt_oneHot` | one selected carrier coordinate reduces to its exact matrix contribution |
| `omitting_completed_carrier_changes_coefficient_image` | original-width projection cannot determine a folded CE coefficient image |
| `ColumnLayout.columns_eq_twoPow` | the paper's square row/column bijection forces a power-of-two assignment width |
| `no_columnLayout_for_completeCarrier` | no complete 54-lane Phi81 carrier can use that square row/column bijection |
| `omitting_coefficient_connectivity_changes_semantic_truth` | identical non-coefficient sources have opposite semantic truth when coefficient matrices are disconnected |
| `residualizationBoundary` | the table constructor closes the arbitrary per-leaf iff at table level |
| `evaluateShifted_eq_shift_mul_evaluateLocal` | the shifted target is exactly `gamma^(2K+k)` times the literal target |
| `literalLocal_shifted_support_mismatch_witness` | exponent zero separates the two layouts under positive paper dimensions |
| `specializedGammaPolynomial_degreeUpperBound` | gamma degree metadata is derived from finite block length |
| `coefficientTruth_iff_allObligations` | conditional composition through `ResidualizationBoundary` |
| `sampledZero_iff_allObligations_or_mixingRoot` | deterministic sampled dichotomy, with no probability claim |
| `outputPoint_eq_roundChallenges` | every accepted output point is the SumCheck challenge vector |
-/

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices_nodup' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms canonicalFinIndices_nodup

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources.ColumnLayout.columns_eq_twoPow' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms UnifiedSources.ColumnLayout.columns_eq_twoPow

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.DomainSeparation.no_columnLayout_for_completeCarrier' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.DomainSeparation.no_columnLayout_for_completeCarrier

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.logicalWidth_le_carrierWidth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.logicalWidth_le_carrierWidth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.blockCount_carrierWidth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.blockCount_carrierWidth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.extendAssignment_embedLogical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.extendAssignment_embedLogical

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.extendAssignment_tail_zero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.extendAssignment_tail_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.extendMatrix_embedLogical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.extendMatrix_embedLogical

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.extendMatrix_tail_zero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.extendMatrix_tail_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.flatIndex_lt_carrierWidth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.flatIndex_lt_carrierWidth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CarrierLayout.layout_encode?_isSome' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81CarrierLayout.layout_encode?_isSome

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource.source_matrix_embedLogical' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81MatrixSource.source_matrix_embedLogical

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource.source_matrix_tail_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81MatrixSource.source_matrix_tail_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource.coefficientMatrix_constant_apply' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81MatrixSource.coefficientMatrix_constant_apply

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource.coefficientMatrix_constant_embedLogical' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81MatrixSource.coefficientMatrix_constant_embedLogical

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81MatrixSource.coefficientMatrix_constant_tail_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81MatrixSource.coefficientMatrix_constant_tail_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.matrixVectorAt_oneHot' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms PaperLinearAlgebra.matrixVectorAt_oneHot

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier.completed_matrix_tail_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.PaddedCarrier.completed_matrix_tail_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier.tail_coefficient_entry_eq_one' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.PaddedCarrier.tail_coefficient_entry_eq_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier.logicalProjections_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.PaddedCarrier.logicalProjections_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier.zeroAssignment_coefficientImage_eq_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.PaddedCarrier.zeroAssignment_coefficientImage_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier.tailAssignment_coefficientImage_eq_one' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.PaddedCarrier.tailAssignment_coefficientImage_eq_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.PaddedCarrier.omitting_completed_carrier_changes_coefficient_image' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.PaddedCarrier.omitting_completed_carrier_changes_coefficient_image

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Residuals.jointCoefficients_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Residuals.jointCoefficients_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Residuals.coefficientTruth_iff_allObligations' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Residuals.coefficientTruth_iff_allObligations

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.AlphaPolynomial.evaluate_eq_zero_of_coefficientZero' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms AlphaPolynomial.evaluate_eq_zero_of_coefficientZero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Residuals.specializedGammaPolynomial_degreeUpperBound' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms Residuals.specializedGammaPolynomial_degreeUpperBound

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Residuals.sampledZero_iff_allObligations_or_mixingRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Residuals.sampledZero_iff_allObligations_or_mixingRoot

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BoundOutputs.outputPoint_eq_roundChallenges' does not depend on any axioms -/
#guard_msgs in
#audit_axioms BoundOutputs.outputPoint_eq_roundChallenges

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BoundOutputs.outputs_share_point' does not depend on any axioms -/
#guard_msgs in
#audit_axioms BoundOutputs.outputs_share_point

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalExponentVectors_nodup' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms canonicalExponentVectors_nodup

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BooleanTable.toAlphaPolynomial_coefficientZero_iff_allEntriesZero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.toAlphaPolynomial_evaluate_eq_evaluate' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BooleanTable.toAlphaPolynomial_evaluate_eq_evaluate

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanVertex.all_nodup' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms BooleanVertex.all_nodup

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.evaluate_eq_equalityWeightedSum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BooleanTable.evaluate_eq_equalityWeightedSum

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.toAlphaPolynomial_evaluate_eq_equalityWeightedSum' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms BooleanTable.toAlphaPolynomial_evaluate_eq_equalityWeightedSum

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.residualTable_allEntriesZero_iff_constraintSatisfied' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CCSResidualTable.residualTable_allEntriesZero_iff_constraintSatisfied

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.residualPolynomial_coefficientZero_iff_constraintSatisfied' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CCSResidualTable.residualPolynomial_coefficientZero_iff_constraintSatisfied

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable.FreshBatch.allResidualTablesZero_iff_allConstraintsSatisfied' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CCSResidualTable.FreshBatch.allResidualTablesZero_iff_allConstraintsSatisfied

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.representedRoots_nodup' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NormRange.representedRoots_nodup

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.baseFieldNoZeroDivisors_of_modulusEuclid' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NormRange.baseFieldNoZeroDivisors_of_modulusEuclid

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.cubicResidual_eq_zero_iff_strictNormTwo' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms NormRange.cubicResidual_eq_zero_iff_strictNormTwo

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.allCubicResidualsZero_iff_normBoundedTwo' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms NormRange.allCubicResidualsZero_iff_normBoundedTwo

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable.strictNormBounded_iff_orderedValues_normBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms NormResidualTable.strictNormBounded_iff_orderedValues_normBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable.residualTable_allEntriesZero_iff_normBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms NormResidualTable.residualTable_allEntriesZero_iff_normBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable.residualPolynomial_coefficientZero_iff_strictNormBounded' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms NormResidualTable.residualPolynomial_coefficientZero_iff_strictNormBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormResidualTable.SourceBatch.allResidualTablesZero_iff_allStrictNormBounded' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms NormResidualTable.SourceBatch.allResidualTablesZero_iff_allStrictNormBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual.imageTable_evaluate_eq_computedCoefficient' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CarriedEvaluationResidual.imageTable_evaluate_eq_computedCoefficient

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CarriedEvaluationResidual.residual_eq_zero_iff_evaluationClaimHolds

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual.allResidualsZero_iff_allClaimsHold' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms CarriedEvaluationResidual.allResidualsZero_iff_allClaimsHold

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CarriedEvaluationResidual.orderedResiduals_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms CarriedEvaluationResidual.orderedResiduals_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalFinIndices_values' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms canonicalFinIndices_values

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalCarriedCoordinates_localGammaExponents' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms canonicalCarriedCoordinates_localGammaExponents

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity.paperDifference_eq_signedResidualBlocks' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SignedJointIdentity.paperDifference_eq_signedResidualBlocks

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial.coefficients_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SignedCoefficientPolynomial.coefficients_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientPolynomial.paperDifference_eq_evaluate' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SignedCoefficientPolynomial.paperDifference_eq_evaluate

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientObject.specializedCoefficients_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SignedCoefficientObject.specializedCoefficients_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientObject.coefficientTruth_iff_tableObligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SignedCoefficientObject.coefficientTruth_iff_tableObligations

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SignedCoefficientObject.evaluate_eq_zero_iff_coefficientTruth_or_mixingRoot

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckInitial.claimTrue_iff_polynomial_evaluate_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial.accepted_implies_polynomial_zero_or_badChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckInitial.accepted_implies_polynomial_zero_or_badChallenge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial.checked_implies_polynomial_zero_or_badChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckInitial.checked_implies_polynomial_zero_or_badChallenge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial.checked_implies_tableObligations_or_mixingRoot_or_badChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckInitial.checked_implies_tableObligations_or_mixingRoot_or_badChallenge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath.qAtPoint_toCubePoint_eq_qAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckTruthPath.qAtPoint_toCubePoint_eq_qAt

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath.sumCompletions_jointPolynomial_eq_summedQ' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckTruthPath.sumCompletions_jointPolynomial_eq_summedQ

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckTruthPath.canonicalGhosts_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckTruthPath.canonicalGhosts_honest

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial.checkedCanonical_implies_tableObligations_or_mixingRoot_or_badChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckInitial.checkedCanonical_implies_tableObligations_or_mixingRoot_or_badChallenge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SumCheckInitial.checkJoint_implies_tableObligations_or_mixingRoot_or_badChallenge' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms SumCheckInitial.checkJoint_implies_tableObligations_or_mixingRoot_or_badChallenge

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData.liftTable_allEntriesZero_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteJointData.liftTable_allEntriesZero_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData.jointTableTruth_iff_semanticTruth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteJointData.jointTableTruth_iff_semanticTruth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData.coefficientTruth_iff_semanticTruth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteJointData.coefficientTruth_iff_semanticTruth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteJointData.checkJoint_implies_semanticTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteJointData.checkJoint_implies_semanticTruth_or_badEvent

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial.evaluateLocal_eq_foldr' does not depend on any axioms -/
#guard_msgs in
#audit_axioms TargetPolynomial.evaluateLocal_eq_foldr

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.literalTargetExponent_ne_declaredCarriedExponent' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms literalTargetExponent_ne_declaredCarriedExponent

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial.evaluateShifted_eq_shift_mul_evaluateLocal' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TargetPolynomial.evaluateShifted_eq_shift_mul_evaluateLocal

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TargetPolynomial.literalLocal_shifted_support_mismatch_witness' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TargetPolynomial.literalLocal_shifted_support_mismatch_witness

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.canonicalCarriedCoordinates_length' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms canonicalCarriedCoordinates_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TableResidualData.orderedCarriedEvaluation_eq_formulaOrder' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TableResidualData.orderedCarriedEvaluation_eq_formulaOrder

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TableResidualData.residualizationBoundary' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TableResidualData.residualizationBoundary

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.TableResidualData.coefficientTruth_iff_tableObligations' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms TableResidualData.coefficientTruth_iff_tableObligations

/-! Actual off-cube paper polynomial and output-message terminal. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable.valueAt_tabulate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms BooleanTable.valueAt_tabulate

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.OutputMessage.ext' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolPolynomial.OutputMessage.ext

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.qAtPoint_toCubePoint_eq_tableQ' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolPolynomial.qAtPoint_toCubePoint_eq_tableQ

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolPolynomial.sumCompletions_polynomial_eq_summedQ

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.canonicalGhosts_honest' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolPolynomial.canonicalGhosts_honest

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.check_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolPolynomial.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.check_implies_tableTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolPolynomial.check_implies_tableTruth_or_badEvent

/-! Necessity of the nonlinear off-cube construction order. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.Necessity.NonlinearTerminal.residualTableTerminal_ne_protocolTerminal' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolPolynomial.Necessity.NonlinearTerminal.residualTableTerminal_ne_protocolTerminal

/-! Verifier-owned paper joint-`Pi_CCS` challenge schedule. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir.Certificate.toFinite_rounds_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms FiatShamir.Certificate.toFinite_rounds_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir.checkResidualTableAudit_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms FiatShamir.checkResidualTableAudit_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir.checkResidualTableAudit_complete_of_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms FiatShamir.checkResidualTableAudit_complete_of_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiatShamir.checkResidualTableAudit_implies_semanticTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms FiatShamir.checkResidualTableAudit_implies_semanticTruth_or_badEvent

/-! Transcript-bound actual paper-polynomial verifier. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.Certificate.toFinite_rounds_length' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolVerifier.Certificate.toFinite_rounds_length

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.derive_coins_eq_transcript' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolVerifier.derive_coins_eq_transcript

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.derive_outgoingState_eq_absorbOutput' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolVerifier.derive_outgoingState_eq_absorbOutput

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.check_eq_true_iff_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolVerifier.check_eq_true_iff_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.check_complete_of_accepted' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolVerifier.check_complete_of_accepted

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolVerifier.check_implies_tableTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolVerifier.check_implies_tableTruth_or_badEvent

/-! One authoritative source family across CCS, norm, and carried checks. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources.freshSourceIndex_ne_runningSourceIndex' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms UnifiedSources.freshSourceIndex_ne_runningSourceIndex

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources.source_eq_fresh_or_running' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms UnifiedSources.source_eq_fresh_or_running

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources.UnifiedInputs.normBatch_at_toVertex_eq_assignment' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms UnifiedSources.UnifiedInputs.normBatch_at_toVertex_eq_assignment

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources.UnifiedInputs.normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded' does not depend on any axioms -/
#guard_msgs in
#audit_axioms UnifiedSources.UnifiedInputs.normBatch_allStrictNormBounded_iff_allAssignmentsStrictNormBounded

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedSources.UnifiedInputs.toIndependentInputs_semanticTruth_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms UnifiedSources.UnifiedInputs.toIndependentInputs_semanticTruth_iff

/-! Actual-protocol data refinement and unified verifier. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement.ProtocolLift.map_zero' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ProtocolDataRefinement.ProtocolLift.map_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement.liftMonomial_totalDegree' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ProtocolDataRefinement.liftMonomial_totalDegree

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement.evaluatePolynomial_lift' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolDataRefinement.evaluatePolynomial_lift

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement.liftTable_tabulate' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ProtocolDataRefinement.liftTable_tabulate

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement.ccsTable_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolDataRefinement.ccsTable_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement.normTable_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolDataRefinement.normTable_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolDataRefinement.toProtocolData_toJointData_eq' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ProtocolDataRefinement.toProtocolData_toJointData_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.UnifiedProtocolVerifier.check_implies_semanticTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms UnifiedProtocolVerifier.check_implies_semanticTruth_or_badEvent

/-! Concrete carrier and its semantic leaf refinements. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.SignedJointIdentity.JointData.ext' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms SignedJointIdentity.JointData.ext

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.NormRange.embed_cubicResidual' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms NormRange.embed_cubicResidual

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.derived_sub_eq_concrete_sub' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.derived_sub_eq_concrete_sub

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.baseLaws' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.baseLaws

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionLaws' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.extensionLaws

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.extensionZeroLaws' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.extensionZeroLaws

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.zeroReflectingLift' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ConcreteCarrier.zeroReflectingLift

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.embed_one' does not depend on any axioms -/
#guard_msgs in
#audit_axioms ConcreteCarrier.embed_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.embed_add' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.embed_add

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.embed_mul' depends on axioms: [propext] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.embed_mul

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.embed_strictNorm' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.embed_strictNorm

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.protocolLift' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.protocolLift

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.check_implies_semanticTruth_or_badEvent' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms ConcreteCarrier.check_implies_semanticTruth_or_badEvent

/-! Single-source field-matrix to carried-coefficient connection. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource.sumRange_select' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MatrixCoefficientSource.sumRange_select

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource.MatrixSource.coefficientMatrix_constant_apply' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MatrixCoefficientSource.MatrixSource.coefficientMatrix_constant_apply

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource.MatrixSource.coefficientMatrix_constant_eq' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MatrixCoefficientSource.MatrixSource.coefficientMatrix_constant_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource.ConnectedInputs.toUnifiedInputs_system_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms MatrixCoefficientSource.ConnectedInputs.toUnifiedInputs_system_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource.ConnectedInputs.toUnifiedInputs_coefficientMatrices_eq' does not depend on any axioms -/
#guard_msgs in
#audit_axioms MatrixCoefficientSource.ConnectedInputs.toUnifiedInputs_coefficientMatrices_eq

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.MatrixCoefficientSource.ConnectedInputs.carriedImageConstantAt_eq_ccsImageAt' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms MatrixCoefficientSource.ConnectedInputs.carriedImageConstantAt_eq_ccsImageAt

/-! Concrete Phi81 logical-column layout. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.flatIndex_decode' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81ColumnLayout.flatIndex_decode

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.decode_encode' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81ColumnLayout.decode_encode

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.encode_decode' depends on axioms: [propext, Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81ColumnLayout.encode_decode

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.encode_eq_none_iff' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81ColumnLayout.encode_eq_none_iff

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81ColumnLayout.flatIndex_lt_paddedWidth' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Phi81ColumnLayout.flatIndex_lt_paddedWidth

/-! Concrete Phi81 coefficient kernel. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel.basisConstantTerm' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Phi81CoefficientKernel.basisConstantTerm

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel.phi81ConstantTermLaw' does not depend on any axioms -/
#guard_msgs in
#audit_axioms Phi81CoefficientKernel.phi81ConstantTermLaw

/-! Carried coefficient-matrix connectivity necessity. -/

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity.sameNonCoefficientInputs' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.CoefficientConnectivity.sameNonCoefficientInputs

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity.coefficientMatrices_ne' depends on axioms: [propext,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.CoefficientConnectivity.coefficientMatrices_ne

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity.validInputs_computedCoefficient_eq_zero' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.CoefficientConnectivity.validInputs_computedCoefficient_eq_zero

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity.disconnectedInputs_computedCoefficient_eq_one' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.CoefficientConnectivity.disconnectedInputs_computedCoefficient_eq_one

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity.validInputs_semanticTruth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.CoefficientConnectivity.validInputs_semanticTruth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity.disconnectedInputs_not_semanticTruth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.CoefficientConnectivity.disconnectedInputs_not_semanticTruth

/-- info: 'Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.Necessity.CoefficientConnectivity.omitting_coefficient_connectivity_changes_semantic_truth' depends on axioms: [propext,
 Classical.choice,
 Quot.sound] -/
#guard_msgs in
#audit_axioms Necessity.CoefficientConnectivity.omitting_coefficient_connectivity_changes_semantic_truth
