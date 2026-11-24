import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingJoin
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatch
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchFold
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchOffload
import NightstreamFPrime.Spec.Compression.Akita.Key30
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules
import NightstreamFPrime.Spec.Compression.Akita.Replay.FusedJoin

/-! Axiom audits for the weight at `r_2` of the packing levels of the replay of Akita openings
(`Replay/WeightPacking.lean`, `Replay/WeightPackingSpec.lean`, `Replay/WeightPackingRows.lean`,
`Replay/WeightPackingGroup.lean`, `Replay/WeightPackingQuotient.lean`,
`Replay/WeightPackingLevel.lean`, `Replay/WeightPackingJoin.lean`, `Replay/WeightPackingBatch.lean`, `Replay/WeightPackingBatchReplay.lean`,
`Replay/WeightPackingBatchLevel.lean`, `Replay/WeightPackingBatchFold.lean`,
`Replay/WeightPackingBatchOffload.lean`), its end check in the replay
(`Replay/Transcript.lean`, `Replay/FusedJoin.lean`) and the setup prefix in tasks
(`Replay/TerminalSetup.lean`). The examples at the end show that the hypotheses of
`foldWeight_one`, `packWeight_one` and `packingFused_holds_iff` hold at the packing levels with one
group (level 1 of `layout20`, the levels 2 and 3 of `layout30` for each row): `PackGroup.Fits`,
the base `2^logBasis` of `packInputsIn`, the claim coefficient 1 of `packGroupsIn`, the basis
`(1, u)` of `Quadratic` and the zero fold challenges. The examples after them show the hypotheses
of the theorems of the folds with opening groups (`batchWeight_groups`, `batchOffload_groups`,
`batchAmbiented`) at the root of `layout20` and at the root and level 1 of `layout30`. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dotFrom_toE
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dotFq_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.powersOf_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lowLevel_size
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqAt_take_succ
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lowLevel_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lowSums_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.of_high
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.run_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.read_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.liftRead_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_toRq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Lift.value_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.RingRow.mle_liftWeight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.RingRow.scan_relocate
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.LiftRelation.mle_lift
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Runs.moved_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Gaps.place_shift
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.run_of_start
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.run_of_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.add_lt_of_dvd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Shape.responseCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.OpeningShape.partialCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.OpeningShape.outerCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.OpeningShape.innerCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.OpeningShape.partialPosition_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.OpeningShape.partialPosition_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.OpeningShape.partialPosition_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_smul
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_mul_natCast
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_natCast
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_algebraMap
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.aeval_rep_neg'
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.partialForm_scan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.contractForm_scan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.imageForm_scan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recompose_scan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.partialLifts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.responseLifts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.imageLifts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outerLifts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.chainLifts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.contractionValue
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openLifts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openParts_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupParts_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Placement.move_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.reads_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.reads_single
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.pow_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Lift.ofTerm_value_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Runs.main_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.partialLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.basis_collapse
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofFn_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.responseCell_lt_chainStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.contractionValue_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outerCell_lt_chainStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.innerCell_lt_chainStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sliceCell_lt_chainStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.partialCell_lt_chainStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_comm_in2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_comm_in3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.responseLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.imageLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_comm_in4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.outerLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.segCell
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.segCell_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Runs.seg
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.recomposeLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.columnLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.imageRow_scan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.chainLifts_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.scalarPart_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatSlot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatSlot_left
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatSlot_right
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.natAdd_ne_castAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_range_ite_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatSlot_width_left
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatSlot_width_right
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatSlot_entry_plane
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatSlot_entry_tail
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatJoin_entry_plane
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatJoin_entry_tail
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Gaps.slot_main
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Gaps.slot_middle
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Gaps.slot_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.entry_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_mainRow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_middle
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_mainRow₀
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_plane
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_row_main
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_row_main₀
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_row_middle
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_row_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_row_plane
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.moved_segCell
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Runs.of_start
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.unitLift_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tables.of_alpha
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientLift_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.carrierQuotient_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientRows_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zipIdx_range_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowStarts_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowStarts_nil
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowStarts_sum_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lv_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowQuotients_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowQuotients_two
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatTail
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatRows_castAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.evaluationRow_degree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relocate_degree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatRows_natAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.liftedReduction_weight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.innerRow_degree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_seven
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_castAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_natAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_const
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatTail_offset
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_append_castAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_append_natAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flatTail_weight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.filterMap_some_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zipIdx_map_fst
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lv_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zipIdx_map_lift
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.segCell_eta
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.firstCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.secondCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openFirstCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openSecondCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Placement.of_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Placement.of_one_spans
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldSupport_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.PackGroup.Fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pow_log_top
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.log_le_top
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.log_le_cube
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotient_insides
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldWeight_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packingFused_holds_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packWeight_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packSupport_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.PackShape.setupCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FusedEnd.passing
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fold.setupCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.claimPointIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packGroupsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packInputsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packingValuesIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.setupClaimIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.packingIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.fusedEndIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_packing_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Setup.expandTasks
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Setup.expandTasks_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.place_symm_code
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.codeOffset_le_of_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_before_place
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.codeOffset_eq_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.castAdd_ne_natAdd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.appendSlot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.appendSlot_left
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.appendSlot_right
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotWidth_append_left
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotWidth_append_right
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joinEntry_append_left
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joinEntry_append_right
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joinEntry_plain
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.gatherSlot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.gatherSlot_place
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.join_gather_basis
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slotWidth_gather
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joinEntry_gather
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_groupPlace
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.gather_place
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchJoinSlot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchJoinBasis
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchJoinSlot_group
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchJoinSlot_rest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchJoinBasis_group
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchJoinBasis_rest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batch_flatRows_group
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupRows_degree_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchLifted_rows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ringTerm
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.entry_below
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.coefficient_below
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_slot_main
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_slot_middle
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotientScan_slot_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupCells
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ringTerm_relocate
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchEntry_group
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupCells_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupDegrees_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_plane
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_inner
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_image
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_gather
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupDegrees_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchEntry_rest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.degree_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.restDegrees
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_groupPlace_const
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_openImage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_openFirst
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_second
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchTerm_openSecond
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroup_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchRest_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchReduction_weight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.take_finRange_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.before_finRange
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zipIdx_finRange
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.getElem?_finRange_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_finRange_order
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_order
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroup
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroups_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroups_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroups_before
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Placement.move_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openLifts_columns
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.scalarPart_moved
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mle_batchScalar
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offset_succ_val
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowStarts_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relocated_shift
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relocated_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relocated_second
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relocated_bounded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupWeight_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openSetupCode_code
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openRows_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.scalar_place
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Placement.of_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Placement.of_batch_spans
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldSupport_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchSupport_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flat_parts_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.getD_range_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.le_foldl_max
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldl_max_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.quotients_split
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.cells_le_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchScalar_weight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchFirstCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchSecondCell_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchOpen_weight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroup_insides
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupFamilies_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupsWeight_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldWeight_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawnCarrier
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawnAmbient
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchGroup_fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packGroupsIn_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchWeight_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchSupport_in
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchOpenLifts_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupParts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.filterMap_parts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skip_rest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skip_parts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.list_sum_map_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.flat_filter_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldParts_offload
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.setupParts_batch
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchSetup_weight
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.common_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldWeight_offload
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchOffload_groups
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.codeScan_cast
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.embed_stride
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawn_ambiented
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawn_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchAmbiented
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packAmbiented
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchFused_holds_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offloadFused_holds_iff

section Vacuity

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Replay
open NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer NightstreamFPrime.Spec.Compression.Lib

example : PackGroup.Fits ⟨Key20.level₁.shape, Key20.level₁.opening, Key20.level₁.fold, [], [], [],
    Ext.one⟩ Key20.level₁.vars :=
  ⟨⟨6, rfl⟩, ⟨10, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide⟩

example : PackGroup.Fits ⟨(Key30.level₂ Key30.w27).shape, (Key30.level₂ Key30.w27).opening,
    (Key30.level₂ Key30.w27).fold, [], [], [], Ext.one⟩ (Key30.level₂ Key30.w27).vars :=
  ⟨⟨6, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide⟩

example : PackGroup.Fits ⟨(Key30.level₂ Key30.w26).shape, (Key30.level₂ Key30.w26).opening,
    (Key30.level₂ Key30.w26).fold, [], [], [], Ext.one⟩ (Key30.level₂ Key30.w26).vars :=
  ⟨⟨6, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide⟩

example : PackGroup.Fits ⟨Key30.level₃.shape, Key30.level₃.opening, Key30.level₃.fold, [], [], [],
    Ext.one⟩ Key30.level₃.vars :=
  ⟨⟨6, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
    by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    by decide, by decide⟩

example : 2 ^ Nat.log 2 Key20.level₁.shape.openingBase = Key20.level₁.shape.certifiedBase ∧
    2 ^ Nat.log 2 Key30.level₃.shape.openingBase = Key30.level₃.shape.certifiedBase :=
  ⟨pow_log_top ⟨3, rfl⟩, pow_log_top ⟨4, rfl⟩⟩

example (r : Key30.Row) :
    2 ^ Nat.log 2 (Key30.level₂ r).shape.openingBase = (Key30.level₂ r).shape.certifiedBase :=
  pow_log_top ⟨4, rfl⟩

example : (#[Ext.one] : Array Ext).getD 0 Ext.zero = Ext.one := rfl

example : ∀ t : Fin 2,
    QuadraticAlgebra.basis (2 : Zq) 0 t = ((extBasis.getD t Ext.zero).into : Quadratic) := by
  intro t
  fin_cases t
  · exact (QuadraticAlgebra.basis_apply_zero _ _).trans Ext.into_one.symm
  · refine QuadraticAlgebra.basis_apply_one.trans ?_
    ext <;> simp [extBasis, Ext.into, Degree2.coords, Ext.toE, Fq.toZq_zero, Fq.toZq_one]

example (S : Shape) (O : OpeningShape S) : ∀ i : Fin S.blocks,
    (0 : Fin S.blocks → Rq O.carrierDegree) i =
      Poly.toRq _ (([] : List (Array Fq)).getD i (Poly.zero O.carrierDegree)) :=
  fun _ => Poly.toRq_zero.symm

example (S : Shape) (O : OpeningShape S) : ∀ i : Fin S.blocks,
    Poly.toRq S.innerDegree (([] : List (Array Fq)).getD i.val (Poly.zero S.innerDegree)) =
      embed O.stride O.stride_tower ((0 : Fin S.blocks → Rq O.carrierDegree) i) :=
  fun _ => Poly.toRq_zero.trans (map_zero _).symm

/-- The hypotheses of `batchWeight_groups`, `batchOffload_groups`, `batchAmbiented` and of the
theorems that they apply hold at the root of `layout20` and at the root and level 1 of `layout30`
(both rows): the slots and the path of the packing shape of the layout, `numbers` for each group of
`packGroupsIn`, `lowFits` and `lowShort` at the root of `key30.aks`, a challenge with
`challenged`, the drawn list after a draw (`drawnAt`, `drawn_sample`), the ring and the degree of
the fold groups of `ofBatch`, and `parted` and `scaled` of `Batch.codeScan`. -/
example (zMax rice : ℕ) : (layout20 zMax rice).folds.head?.map (·.fused) =
    some (.packing ⟨(List.finRange 3).map fun k => ⟨(groupOrder 3 k).val,
      Key20.root.shape (groupOrder 3 k), Key20.root.opening (groupOrder 3 k),
      Key20.root.fold (groupOrder 3 k)⟩, ⟨Key20.root.openDegree, Key20.root.openRank, Key20.root.chain⟩,
      Key20.quotientDigits⟩) := rfl

example (r : Key30.Row) (zMax rice : ℕ) : (layout30 r zMax rice).folds.head?.map (·.fused) =
    some (.packing ⟨(List.finRange 3).map fun k => ⟨(groupOrder 3 k).val,
      (Key30.root r).shape (groupOrder 3 k), (Key30.root r).opening (groupOrder 3 k),
      (Key30.root r).fold (groupOrder 3 k)⟩,
      ⟨(Key30.root r).openDegree, (Key30.root r).openRank, (Key30.root r).chain⟩,
      Key30.quotientDigits₀⟩) := rfl

example (r : Key30.Row) (zMax rice : ℕ) : (layout30 r zMax rice).folds[1]?.map (·.fused) =
    some (.packing ⟨(List.finRange 2).map fun k => ⟨(groupOrder 2 k).val,
      (Key30.level₁ r).shape (groupOrder 2 k), (Key30.level₁ r).opening (groupOrder 2 k),
      (Key30.level₁ r).fold (groupOrder 2 k)⟩,
      ⟨(Key30.level₁ r).openDegree, (Key30.level₁ r).openRank, (Key30.level₁ r).chain⟩,
      Key30.quotientDigits⟩) := rfl

example (s : Replayer) (level : ℕ) (F : Fold) : ∀ g ∈ packGroupsIn ⟨(List.finRange 3).map fun k =>
    ⟨(groupOrder 3 k).val, Key20.root.shape (groupOrder 3 k), Key20.root.opening (groupOrder 3 k),
      Key20.root.fold (groupOrder 3 k)⟩, ⟨Key20.root.openDegree, Key20.root.openRank, Key20.root.chain⟩, Key20.quotientDigits⟩ s level F,
      g.Fits Key20.rootVars := by
  rw [packGroupsIn_batch Key20.root s level _ F rfl]
  intro g member
  obtain ⟨k, -, rfl⟩ := List.mem_map.mp member
  fin_cases k
  · exact batchGroup_fits Key20.root _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨6, rfl⟩, ⟨10, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits Key20.root _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨10, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits Key20.root _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨6, rfl⟩, ⟨9, rfl⟩, ⟨7, rfl⟩, ⟨7, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩

example (s : Replayer) (level : ℕ) (F : Fold) : ∀ g ∈ packGroupsIn ⟨(List.finRange 3).map fun k =>
    ⟨(groupOrder 3 k).val, (Key30.root Key30.w27).shape (groupOrder 3 k), (Key30.root Key30.w27).opening (groupOrder 3 k),
      (Key30.root Key30.w27).fold (groupOrder 3 k)⟩, ⟨(Key30.root Key30.w27).openDegree, (Key30.root Key30.w27).openRank, (Key30.root Key30.w27).chain⟩, Key30.quotientDigits₀⟩ s level F,
      g.Fits Key30.rootVars := by
  rw [packGroupsIn_batch (Key30.root Key30.w27) s level _ F rfl]
  intro g member
  obtain ⟨k, -, rfl⟩ := List.mem_map.mp member
  fin_cases k
  · exact batchGroup_fits (Key30.root Key30.w27) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits (Key30.root Key30.w27) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits (Key30.root Key30.w27) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨7, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩

example (s : Replayer) (level : ℕ) (F : Fold) : ∀ g ∈ packGroupsIn ⟨(List.finRange 3).map fun k =>
    ⟨(groupOrder 3 k).val, (Key30.root Key30.w26).shape (groupOrder 3 k), (Key30.root Key30.w26).opening (groupOrder 3 k),
      (Key30.root Key30.w26).fold (groupOrder 3 k)⟩, ⟨(Key30.root Key30.w26).openDegree, (Key30.root Key30.w26).openRank, (Key30.root Key30.w26).chain⟩, Key30.quotientDigits₀⟩ s level F,
      g.Fits Key30.rootVars := by
  rw [packGroupsIn_batch (Key30.root Key30.w26) s level _ F rfl]
  intro g member
  obtain ⟨k, -, rfl⟩ := List.mem_map.mp member
  fin_cases k
  · exact batchGroup_fits (Key30.root Key30.w26) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits (Key30.root Key30.w26) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits (Key30.root Key30.w26) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨7, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩

example (s : Replayer) (level : ℕ) (F : Fold) : ∀ g ∈ packGroupsIn ⟨(List.finRange 2).map fun k =>
    ⟨(groupOrder 2 k).val, (Key30.level₁ Key30.w27).shape (groupOrder 2 k), (Key30.level₁ Key30.w27).opening (groupOrder 2 k),
      (Key30.level₁ Key30.w27).fold (groupOrder 2 k)⟩, ⟨(Key30.level₁ Key30.w27).openDegree, (Key30.level₁ Key30.w27).openRank, (Key30.level₁ Key30.w27).chain⟩, Key30.quotientDigits⟩ s level F,
      g.Fits Key30.vars₁ := by
  rw [packGroupsIn_batch (Key30.level₁ Key30.w27) s level _ F rfl]
  intro g member
  obtain ⟨k, -, rfl⟩ := List.mem_map.mp member
  fin_cases k
  · exact batchGroup_fits (Key30.level₁ Key30.w27) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits (Key30.level₁ Key30.w27) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩

example (s : Replayer) (level : ℕ) (F : Fold) : ∀ g ∈ packGroupsIn ⟨(List.finRange 2).map fun k =>
    ⟨(groupOrder 2 k).val, (Key30.level₁ Key30.w26).shape (groupOrder 2 k), (Key30.level₁ Key30.w26).opening (groupOrder 2 k),
      (Key30.level₁ Key30.w26).fold (groupOrder 2 k)⟩, ⟨(Key30.level₁ Key30.w26).openDegree, (Key30.level₁ Key30.w26).openRank, (Key30.level₁ Key30.w26).chain⟩, Key30.quotientDigits⟩ s level F,
      g.Fits Key30.vars₁ := by
  rw [packGroupsIn_batch (Key30.level₁ Key30.w26) s level _ F rfl]
  intro g member
  obtain ⟨k, -, rfl⟩ := List.mem_map.mp member
  fin_cases k
  · exact batchGroup_fits (Key30.level₁ Key30.w26) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩
  · exact batchGroup_fits (Key30.level₁ Key30.w26) _ _ _ _ (cs := fun _ => []) (amb := fun _ => [])
      (pts := fun _ => []) (w := fun _ => Ext.one)
      ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, by decide,
      by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
      by decide, by decide⟩

example : ∀ r ∈ Key30.rows, 2 ^ (Key30.root r).low ≤ (Key30.root r).openDegree ∧
    (Key30.root r).low ≤ Key30.rootVars := by decide

example {n : ℕ} (B : Batch (Fin n)) (s : Replayer) (level : ℕ) :
    ∀ k (i : Fin (B.shape (groupOrder n k)).blocks),
      (fun g (i : Fin (B.shape g).blocks) => Poly.toRq (B.carrierDegree g)
        ((drawnCarrier B s level ((groupOrder n).symm g)).getD i (Poly.zero (B.carrierDegree g))) :
          B.Challenge) (groupOrder n k) i = Poly.toRq _
        ((drawnCarrier B s level k).getD i (Poly.zero (B.carrierDegree (groupOrder n k)))) := by
  intro k i
  simp only [Equiv.symm_apply_apply]

example (s₀ : Replayer) (empty : s₀.derived.drawn = #[]) (level g : ℕ) (G : FoldGroup)
    (cs : List Sparse) : drawnIn (s₀.withDrawn level g (cs.map (ambient G))) level g =
      cs.map (ambient G) := by
  simp [drawnIn, Replayer.withDrawn, empty]

example {n : ℕ} (order : Equiv.Perm (Fin n)) (B : Batch (Fin n)) (R : Level) (pairs : ℕ → ℕ × ℕ)
    (vars batch base : ℕ) (root : Bool) (digits : ℕ) :
    (ofBatch order B R pairs vars batch base root digits).groups = (List.finRange n).map fun g =>
      ⟨(B.shape g).blocks, (B.shape g).innerDegree, B.carrierDegree g, (pairs g).1, (pairs g).2,
        false⟩ := rfl

example {E : Type} [Field E] [Fintype E] [Algebra Zq E] {n : ℕ} (B : Batch (Fin n))
    (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E) (challenge : B.Challenge)
    {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ) (room : B.quotientRoom digits ≤ 2 ^ μ)
    (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) (ν : ℕ) (flat : Table) (κ : ℕ)
    (blocked : B.Blocked κ) (low : κ ≤ μ) :
    (∀ α τ r2, (Batch.codeScan β st challenge μ fits base digits room batch rowsFit ν flat κ blocked
      low).part α τ r2 = mle r2 (((Batch.liftedReduction β (st.over (0 : Fin (2 ^ ν) → Zq)) challenge
        μ fits base digits room batch rowsFit).rows α).batchWeight τ)) ∧
    ∀ α τ r2, (Batch.codeScan β st challenge μ fits base digits room batch rowsFit ν flat κ blocked
      low).scale α τ r2 = powerScale α (r2.take κ) :=
  ⟨fun _ _ _ => rfl, fun _ _ _ => rfl⟩

end Vacuity
