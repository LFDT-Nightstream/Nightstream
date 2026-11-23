import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.Plan
import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation
import NightstreamFPrime.Spec.Compression.Akita.Replay.Rows
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules
import NightstreamFPrime.Spec.Compression.Akita.Replay.SumCheck
import NightstreamFPrime.Spec.Compression.Akita.Replay.Transcript
import NightstreamFPrime.Spec.Compression.Akita.Replay.Grows
import NightstreamFPrime.Spec.Compression.Akita.Replay.Readback

/-! Axiom audits for the replay of Akita openings (`Spec/Compression/Akita/Replay/`): the
reduction of the words of `Z_q` and `E` to `Zq` and `Quadratic` (`Field`), the binomial of the
fold messages (`Layout`), the outcomes of the byte layer (`Wire`), the claims of Stage 2 against
`reducedEval` and `FieldRows.batchTarget` (`Relation`), the targets of the field rows of the
reductions of the claims at the public rows of the code (`Rows`), the join of the replay with the
stage functions and the games of the specification (`Transcript`), the work outcome of the replay
(`Transcript`), the absorbed values read back (`Readback`), and the definitions of the layouts, the script, the decoder, the sponge and the
replay. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toNat_ofNat
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toZq_ofNat
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.ofNat_canonical
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toZq_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toZq_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toZq_add
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toZq_neg
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toZq_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.toZq_mul
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fq.ofWord?_canonical
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_ofBase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_add
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_neg
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_mul
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.toE_injective
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FoldGroup.binomial_eq_choose
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyBytes_malformed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyBytes_trailing
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyBytes_verdict
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyBytes_rejects
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fields_canonical
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FoldGroup.messageBytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.script
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decode
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyBytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Duplex.absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Duplex.squeeze
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Keccak.shake128
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Keccak.sha3_256
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.poolBytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decodePool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.replay
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.planDifferences
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.into_add
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.into_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.into_mul
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.into_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.into_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.into_ofBase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqIndex_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ringEval_into_sum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ringEval_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mle_delta_at
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchEntries_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchEntries_batchTarget
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchEntries_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage2_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyBytes_noWork
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqIndex
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ringEval
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchEntries
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rhsEntries
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.dot
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relation
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.commitmentRows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.works
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verdict
#audit_axioms NightstreamFPrime.Spec.Compression.Game.runs_unique
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqRun_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_none
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_equality_eqRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainRun_point
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqRun_point
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_point
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.orShift
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldWord
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.word_of_data
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decodeExts_extBytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.formIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.endRoundsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.mapM_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_ended
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.toE_extHom
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.degree2_extHom
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.listPoly_eval_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.listPoly_eval_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.listPoly_coeff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.listPoly_natDegree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.coeffPoly_coefficients
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainPoly_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainPoly_natDegree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_total
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_horner
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_headD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_plainStep
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_eqStep
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_eqVal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.listFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.coeffPoly_listFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_listFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqRun_factor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_equality_factor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.equality_end
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs_recv
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainRun_runs_with
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqRun_runs_with
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqRun_eqSumcheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_eqSumcheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqCodePath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_plain
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_plain_with
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_equality
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun_equality_with
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.coefficients_plainPoly_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainCode_coefficients
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.coefficients_plainCode
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.same
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Ext.same_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.natExt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.powExt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumTo
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.prodTo
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.nodeEval
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageEnd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batched
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.leafEnd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.treeRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.leafRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangeRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.entry_listFn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_natExt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_powExt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_sumTo
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_prodTo
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_getD
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_nodeEval
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_stageEnd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_batched
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_leafEnd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.treePath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.leafPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.leafRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.treeRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangeRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqList
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.normRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plainRun_point_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.normPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.map_eqList
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.normRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.energyFits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.energyFits_iff
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.same_degree2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.nat_degree2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorInput
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.into_pair
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.re_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.im_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.into_tensorInput
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorRun_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.pointBefore
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.weightsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.holds
#audit_axioms NightstreamFPrime.Spec.Compression.Game.runs_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Game.runs_bind_end
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowsBefore
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pathRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupRows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mainRows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.secondStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowTotal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openRun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relation
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowsBefore_add_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.relation_below
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tableOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_eq_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tableOf_eq_spread
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_fin
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tableOf_append_left
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tableOf_append_right
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tableOf_lastCases_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tableOf_lastCases_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_lastCases
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.place_symm_val
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_gather
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.list_sum_range
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joinTargets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.plain_targets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.all_targets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.append_targets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.gather_targets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.targetsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.targetsOf_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.gather_rows_targets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.firstRow_target
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ringOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.payloadOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runValues
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.chunk_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.into_ringOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.path_spread
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_targets_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runsAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.entries_at
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fused_of_targets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupRun_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.openRun_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowTotal_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runsAt_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.carried_spread
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.freeRows_targetAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceRows_targetAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packingRows_targetAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.trace_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packing_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.path_spread_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.spread_gather_rows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runsAt_relation
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sum_regroup
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mainRows_order
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.secondStart_order
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.secondSum_order
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rowTotal_order
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchFirst
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchSecond
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.groupTargets
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.group_spread
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchRows_targetAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batch_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batch_flat_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.chunk
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.batchSource
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.levelRelation
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout20_relations
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout30_relations
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.factorStep
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorFactor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.factorSum
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.factorSum_linear
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.factorSum_cons
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.factorSum_nil
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.re_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.im_sub
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.factor_fold
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorFactor_into
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.recordedIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.firstIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.roundsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.pointIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.productsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stageValue
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.StageOut.accepts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.checked
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.checkWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.checkStep
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.bind_get_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.checkStep_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.checkWith_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_stage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Terminal.zCoords
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Terminal.setupCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Terminal.params
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.bytesIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.drawnIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.terminalMessagesIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.terminalChallengesIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.terminalValue
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.terminalSites
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.terminalWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.step_range_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.step_norm_runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.step_tensor_runs

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The hypotheses `reads`, `openReads` and `value` of `trace_fused` hold for the statement with
the payloads of the runs and the value of the claims. -/
example {E : Type} [Field E] [Algebra Zq E] [Degree2 E] {S : Shape} {O : OpeningShape S}
    (st : TraceStatement O E) (data : RhsRun → Array Fq) (source : RhsSource)
    (coefficients claims : Array Ext) :
    let st' : TraceStatement O E := { st with
      payload := payloadOf S.chain _ _
        (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0))
      openPayload := payloadOf O.chain _ _
        (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain))
      value := (dot coefficients claims).into }
    st'.payload = payloadOf S.chain _ _
        (data (groupRun (fun _ : Fin 1 => S) O.openRank O.chain (fun _ => source) 0)) ∧
      st'.openPayload = payloadOf O.chain _ _
        (data (openRun (fun _ : Fin 1 => S) O.openRank O.openDegree O.chain)) ∧
      st'.value = (dot coefficients claims).into :=
  ⟨rfl, rfl, rfl⟩

/-- The hypothesis `reads` of `batch_fused` holds for the statement whose group `g` has the
payload of the run at its place `(groupOrder n).symm g`. -/
example {E : Type} [Field E] [Algebra Zq E] {n : ℕ} {B : Batch (Fin n)} (st : B.Statement E)
    (data : RhsRun → Array Fq) (source : Fin n → RhsSource) :
    let st' : B.Statement E := { st with
      payload := fun g => payloadOf (B.shape g).chain _ _ (data
        (groupRun (fun k => B.shape (groupOrder n k)) B.openRank B.chain source
          ((groupOrder n).symm g))) }
    ∀ k, st'.payload (groupOrder n k) = payloadOf (B.shape (groupOrder n k)).chain _ _
      (data (groupRun (fun k => B.shape (groupOrder n k)) B.openRank B.chain source k)) :=
  fun k => by simp only [Equiv.symm_apply_apply]

end NightstreamFPrime.Spec.Compression.Akita.Replay
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.writes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.run_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Grows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Grows.of_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Grows.refl
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Grows.trans
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Grows.mono
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Keeps
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_pure
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_throw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_get
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_mono
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_get_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_modify
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_mapM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_foldlM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_fieldsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_pointBefore
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_squeeze
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_message
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_record
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_stage2Of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.grows_fields
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.grows_integers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.grows_bytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.grows_rounds
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.grows_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_weightsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_holds
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_checkWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_terminalWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_decodeExts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_keeps
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.forM_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.keeps_forM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.runOps
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.runOps_grows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.lookup_later
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.forM_array
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.replay_runOps
