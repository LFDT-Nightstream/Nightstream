import tests.AxiomAudit
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeKey20
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeKey30
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeStages
import NightstreamFPrime.Spec.Compression.Akita.Replay.ShapeAccept

/-! Axiom audits for the schedule facts of the scripts as theorems: the skeleton of a layout
(`Replay/ShapeScript.lean`), the shapes on the erased operations (`Replay/ShapeErase.lean`), the
kernel computations and the facts of each script of `layout20` (`Replay/ShapeLayout20.lean`), and
the games of the levels of `key20.aks` with the layout as the only schedule hypothesis
(`Replay/ShapeKey20.lean`), the same for the two rows of `key30.aks` (`Replay/ShapeLayout30.lean`,
`Replay/ShapeKey30.lean`), the shape of `range.k.end` (`Replay/ShapeRange.lean`,
`Replay/ShapeStages.lean`), the converse of `verifyProof_accept` from the level functions
(`Replay/ShapeAccept.lean`), with the guard that the bundles `FusedAt`, `RangeAt`, `TensorAt` and
`OffloadAt` hold on the skeletons of `layout20` and `layout30` as operations. -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.challenges
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.round
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.sumcheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.foldChallenges
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.tensor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.range
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.close
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.fold
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.prefix_
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.level
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.Emits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.Steps
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.Emits.of_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.Emits.bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.Steps.of_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.Steps.bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_pure
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.steps_yield
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_throw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_throw_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.steps_throw_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_ite
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.steps_ite
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_emit
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_emit_work
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_emit_nonce
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_emit_bounded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_get
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_take
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_forIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_forIn_range
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_work
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_challenges
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_sumcheck
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_foldChallenges
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_tensor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_range
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_fold
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_prefix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Build.emits_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.except_bind_ok
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.script_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.writes_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.ends_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.drawsInto_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.extsInto_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.drawsGroup_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isRangeCheck_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isFusedClaim_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isPlainEnd_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isStage3End_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isSetupClaim_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isEndOf_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eraseAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.frees_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawnCount_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.extsCount_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeezeOf_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.roundsShape_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.roundsAt_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldShape_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offloadShape_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.gameShape_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangeShape_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawnShape_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.endFits_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.endsFit_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zipIdx_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.size_skeleton
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.erase_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.erase_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.take_erase
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.blocks
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedIndex
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Skeleton.level_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.blocks_getElem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton_fused
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldShapeAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldShapeAt_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offloadShapeAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.offloadShapeAt_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangeShapeAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isTensorEnd
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isTensorEnd_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorShapeAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.levelShaped
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.FusedAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.OffloadAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.RangeAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.TensorAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_tensor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_range
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_level
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_ends
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_last
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Public.ops
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_scripted
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_level0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_level1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_level2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_level3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_level4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_level5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_ends
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_tensor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.folds20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedIndex20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.length20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fold20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout20_fold
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_shaped
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.script20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_script20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.replay_matches20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_accept_iff20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_root_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_level1_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.shape20_cast_ne_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_trace_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key20_traceSq_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_level0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_level1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_level2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_level3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_level4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_level5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_level6
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_ends
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_tensor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_level0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_level1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_level2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_level3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_level4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_level5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_level6
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_ends
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_tensor
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.folds30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedIndex30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.length30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fold30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.layout30_fold
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_shaped
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fold30_root
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.script30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_script30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.replay_matches30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_accept_iff30
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_key30_root_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageHolds
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.treeInputs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.treeRun_stage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.treeRun_none_of_stage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_absorb_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_absorb_batched
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stateBefore_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.recorded_between
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.rounds_between
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.first_between
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stateBefore_succ
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.recorded_fresh
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.first_of_writer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.rounds_of_end
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.weightsIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.run_holds_stage
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.output_of_end
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage0_ident
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.output_key
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage0_forward
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage0_backward
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.mem_drop_of_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage1_ident
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.treeInputs_one
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage1_forward
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stage1_backward
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.foldRunIn_none_of_tree
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isClaim0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isClaim1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isEnd0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isEnd1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isCheck0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isCheck1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isClaim0_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isClaim1_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isEnd0_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isEnd1_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isCheck0_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isCheck1_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage0At
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage1At
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.indexOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stagesShape
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.mem_drop_take_of_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.erase_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.erase_ended
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_at
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_frees_take
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_frees_drop
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_drawnCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_extsCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tree_of_accepts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage0_of_shape
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stage1_of_shape
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_stages
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_stages0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_stages1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_stages2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_stages3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_stages4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_stages5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_stages0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_stages1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_stages2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_stages3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_stages4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_stages5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w27_stages6
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_stages0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_stages1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_stages2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_stages3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_stages4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_stages5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_w26_stages6
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_stages
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton30_stages
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.script20_stages
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.script30_stages
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_reject_stage20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.tensorRunIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.holds_run
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.check_keeps
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.stateBefore_stopped
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.holds_of_stopped
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.holds_of_later
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.levelStart
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.inLevel
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageIndices
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isTensorOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isTensorOf_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.coveredAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_stage_reject
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_checks
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runProof_accept_of_levels
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_covered
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.levelStart20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fold20_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_accept_of_levels
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_levels_of_accept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_accept_iff_levels

namespace AkitaShapeGuards

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The operation has no bits of the plan and no most bytes of `z`. -/
def erased : Op → Bool
  | .work _ _ bits => bits == 0
  | .foldNonce _ width => width == 0
  | .bounded _ max => max == 0
  | _ => true

theorem erase_of_erased {op : Op} (h : erased op = true) : op.erase = op := by
  cases op <;> simp_all [erased, Op.erase]

theorem skeleton20_erased : (skeleton (layout20 0 0)).all erased = true := by decide +kernel

theorem skeleton20_fixed :
    (skeleton (layout20 0 0)).toArray.toList.map Op.erase = skeleton (layout20 0 0) := by
  rw [List.toList_toArray]
  conv_rhs => rw [← List.map_id (skeleton (layout20 0 0))]
  exact List.map_congr_left fun op member =>
    erase_of_erased (List.all_eq_true.mp skeleton20_erased op member)

/-- The skeleton of `layout20` as operations has the bundles of the shapes at the root and the
tensor reduction of the terminal. -/
example : FusedAt (skeleton (layout20 0 0)).toArray 0 (fold20 0) (fusedIndex (layout20 0 0) 0) ∧
    (∃ c, RangeAt (skeleton (layout20 0 0)).toArray 0 (fold20 0) c) ∧
    ∃ c, TensorAt (skeleton (layout20 0 0)).toArray 6 (layout20 0 0).terminal.rounds c :=
  ⟨(erased_level skeleton20_fixed (layout20_fold 0 0 0) (skeleton20_shaped 0)).1,
    (erased_level skeleton20_fixed (layout20_fold 0 0 0) (skeleton20_shaped 0)).2.1,
    erased_tensor skeleton20_fixed skeleton20_tensor⟩

theorem skeleton30_erased {r : Key30.Row} (member : r ∈ Key30.rows) :
    (skeleton (layout30 r 0 0)).all erased = true := by
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;> decide +kernel

theorem skeleton30_fixed {r : Key30.Row} (member : r ∈ Key30.rows) :
    (skeleton (layout30 r 0 0)).toArray.toList.map Op.erase = skeleton (layout30 r 0 0) := by
  rw [List.toList_toArray]
  conv_rhs => rw [← List.map_id (skeleton (layout30 r 0 0))]
  exact List.map_congr_left fun op h =>
    erase_of_erased (List.all_eq_true.mp (skeleton30_erased member) op h)

/-- The skeleton of each row of `layout30` has the bundle of the offloaded root. -/
example {r : Key30.Row} (member : r ∈ Key30.rows) :
    OffloadAt (skeleton (layout30 r 0 0)).toArray 0 (fold30 r 0)
      (SetupPlan.ofBatch (Key30.root r) (groupOrder 3) (fold30 r 0).low 3)
      Key30.setupVars (fusedIndex (layout30 r 0 0) 0) :=
  (erased_level (skeleton30_fixed member) (layout30_fold r 0 0 0)
    (skeleton30_shaped member 0).1).2.2.2 _ _ (fold30_root member).1 (fold30_root member).2

example : Build.Emits (pure () : Build Unit) [] := Build.emits_pure ()

example : Build.Steps (pure (.yield ()) : Build (ForInStep Unit)) [] := Build.steps_yield

end AkitaShapeGuards
