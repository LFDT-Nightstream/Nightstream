import tests.AxiomAudit
import tests.AxiomsAkitaShape
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinTerminal
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinConverse
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinCount
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinSlots
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinExists
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinStage
import NightstreamFPrime.Spec.Compression.Akita.Replay.Causal
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBound

/-! Axiom audits for the join of the replay of `key20.aks` to the tower data of the model
(`Replay/JoinNote.lean`) the work bits of the model (`Replay/JoinBits.lean`) and the frames of the links of `layout20`
(`Replay/JoinShape.lean`), the transcripts that present the replay (`Replay/JoinPresent.lean`) and
theorem B "→" (`Replay/JoinAccept.lean`, `Replay/JoinLevels.lean`,
`Replay/JoinDrawn.lean`, `Replay/JoinTerminal.lean`), with the guard that the skeleton of `layout20` as operations has the
frames, the notes of the stages (`Replay/JoinStage.lean`) and the state of the replay as a function
of the events of the sponge (`Replay/Causal.lean`). -/

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rootClaim20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.claim20₁
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceClaim20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.terminalStatement20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.response20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.note20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rootClaimIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.claimIn₁
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceClaimIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.terminalStatementIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.responseIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.noteOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.basis20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.basis20_apply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tower20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tower20_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squaredBits20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bits20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stages20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tower_bits20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isAbsorbPoint
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isAbsorbValues
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isAbsorbRecorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isAbsorbPoint_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isAbsorbValues_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isAbsorbRecorded_sound
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.erase_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.erased_matched
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joined_frees
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joined_frees_take
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joined_drawnCount
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.joined_roundsAt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.RootFrame
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.PackFrame
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.TensorFrame
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sk20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedIndex20_values
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rootShape20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packShape20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorShape20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_root
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_pack
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_tensor1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_tensor2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_tensor3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_tensor4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_tensor5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorFrame_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rootFrame_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packFrame_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.script20_frames
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_frames20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isWork
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isFoldNonce
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.workIndices20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.nonceIndices20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.starts20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.workIndices20_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.nonceIndices20_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.StepOk
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fused20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.receiver20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tail20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.path20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20₀
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20₁
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challengeTrace20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20L
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.weights20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.Shows
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.PresentsAt20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Presents20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stepOk_of_accept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofFn_getD_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.path20_zero
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zeroth_numbers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_root
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_level0
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.path20_trace
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.receiver20_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.first20_numbers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_level1
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.second20_numbers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.third20_numbers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fourth20_numbers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_level2
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_level3
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_level4
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Last20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fifth20_numbers
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_level5
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_passed
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_accepts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.bind_draw_sampled
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.drawnIn_sampled
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.squeeze20_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_drawn20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.payload20_of_accept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.last20_of_accept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_presents20

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.DrawsLe
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.DrawsLe.mono
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.DrawsLe.bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.DrawsLe.ite
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.DrawsLe.runs
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheck_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sumcheckWith_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.eqSumcheck_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangeTree_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.leafStage_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rangeCheck_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.normLeafStage_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fusedCheck_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawBin_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldCheck_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldCheckSq_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.tensorCheck_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.tensored_draws
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.narrow
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.packs_mono
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.nextIndex_mono
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.nextIndex_append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fits.mono
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_nil
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_recv
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Fits.append
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_ite
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_binPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_plainPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_eqCodePath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_treePath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_leafPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_normPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_fusedPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_tensorPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_tensoredPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_foldPath
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_foldPathSq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.productsIn_arity
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_foldPathIn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_open_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_kept64_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_kept128_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.carrier_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.trace_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.terminal_drawn
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20₀_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20₁_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challengeTrace20_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.challenge20L_eq
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.drawn20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slots20₀
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slots20₁
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slots20₂
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slots20₃
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slots20₄
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slots20₅
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.slots20L

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.energy20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.narrow20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.leaves20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fits_path20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.exists_block20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.tensored_presentable
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.checks20_zeroth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.checks20_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.checks20_second
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.checks20_third
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.checks20_fourth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.checks20_fifth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs20_zeroth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs20_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs20_second
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs20_third
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs20_fourth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.runs20_fifth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.block20_zeroth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.block20_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.block20_second
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.block20_third
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.block20_fourth
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.block20_fifth

#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix.wire_all
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.lastWire_all
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ofBlocks_at
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Chained
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Chained.getElem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.bits20_le
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_presents20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_exists_accepts20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Reads20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Reads20.of_all
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceKeys
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.terminalKeys
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.works20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_claim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_commit₁
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_claim₁
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_commit₂
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_claim₂
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_commit₃
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_claim₃
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_commit₄
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_claim₄
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_commit₅
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_claim₅
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_state
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_statement
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held20_partials
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isDraw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.between20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ordered20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.skeleton20_ordered
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.held_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.rootClaimIn_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.claimIn₁_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.claimIn₁_payload
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceClaimIn_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.traceClaimIn_payload
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stateIn_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.partialsIn_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.terminalStatementIn_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageIndex20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stageIndex20_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.noteAt20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.noteAt20_reads
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix.mapWire
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix.mapWire_length
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix.mapWire_reply
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Prefix.mapWire_wire
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.forall₂_zipIdx_map
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.SameRead
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pathOf_congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.sameRead_page
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.Shows.mapWire
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.zeroth_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.first_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.second_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.third_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fourth_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fifth_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.renote
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Reads20.congr
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.presents20_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.exists_presents20_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_exists_accepts20_staged
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_accepts20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_bound20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Core
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.core
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Alike
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Alike.refl
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Alike.symm
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Alike.trans
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.get?_insert_fun
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.recordedIn_insert
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Sim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_pure
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_throw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_modify
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_mapM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_foldlM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_fieldsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.pointIn_alike
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_pointBefore
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_squeeze
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_record
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_decodeExts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_stage2Of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.sim_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_sim
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Path
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Moves
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_bind
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_pure
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_throw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_get
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_modify
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_mapM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_foldlM
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_recorded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_first
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_fieldsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_pointBefore
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_record
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_message
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.Suffix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.suffix_path
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.eq_path
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_absorb
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_squeeze
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_stage2Of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_value
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_weightsOf
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_holds
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_checkWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_draw
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_terminalWith
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.moves_decodeExts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_suffix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_fixed_check
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_fixed_terminal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_core
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_inert
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.runFrom_suffix
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.stores
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.input
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.tail
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.store
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.store_quiet
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.tail_events
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Faithful
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Faithful.refl
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.core_fields
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.core_env
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.core_integer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.core_bytes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_stores
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_foldNonce
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_work
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.step_read
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.read_alike
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.run_alike
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.state_causal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Replayer.core_faithful
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.noteOf_core
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.noteAt20_faithful
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.ReadsFrom
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decode_step_from
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decode_from
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Op.Parses
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.faithful_of_reads
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldr_congr_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.foldlM_congr_mem
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.get!_extract
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.word_extract
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.number_extract
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fields_extract
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.fields_fits
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.parses_fields
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.parses_exts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.parses_integer
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.parses_digest
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.number_lt
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.le64_inj
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.parses_bounded
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.parses_of_stores
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Faithful.take
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.decode_faithful
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.core_causal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.noteAt20_causal
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.works_of_decodes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Tier.tried_of_decodes
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.isPool
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.Pools20
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.pools20_of_accept
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.index20_of
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.works20_of_stepOk
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.stepOk20_of_accepts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.works20_of_accepts
#audit_axioms NightstreamFPrime.Spec.Compression.Akita.Replay.verifyProof_accept_of_presents20

namespace AkitaJoinGuards

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Akita.Replay
open NightstreamFPrime.Spec.Compression (Game)

/-- The skeleton of `layout20` as operations has the frames of the links. -/
example : RootFrame (skeleton (layout20 0 0)).toArray 165 ∧
    PackFrame (skeleton (layout20 0 0)).toArray 165 316 ∧
    TensorFrame (skeleton (layout20 0 0)).toArray 5 Key20.level₅.tail 1456 1513 1521 :=
  ⟨rootFrame_of AkitaShapeGuards.skeleton20_fixed, packFrame_of AkitaShapeGuards.skeleton20_fixed,
    tensorFrame_of AkitaShapeGuards.skeleton20_fixed skeleton20_tensor5⟩

/-- `DrawsLe` and `Fits` have instances: the sum-check of one round draws at most once, and the
path of one plain round from the message 0 fits the cells of `Key20.Wire`. -/
example : DrawsLe (Game.sumcheck 1 (fun _ => 0) 1 [] (0 : ZMod 2)) 1 :=
  sumcheck_draws 1 _ 1 [] 0

example : Fits (ZMod 2) 8 3029804899 4 (plainPath (fun _ : Ext => (0 : ZMod 2)) 1 Ext.zero
    [([Ext.zero], Ext.zero)]) 0 0 :=
  fits_plainPath (by decide) _ 1 (by decide) _ _ (by decide)

/-- `Chained` has an instance: one block of one entry from the start 0. -/
example : Chained (fun i (x : ℕ) => x = i) 0 [[0]] :=
  ⟨fun i inside => by simp at inside; subst inside; rfl, trivial⟩

/-- `Reads20` has an instance: the notes that are `note20` at each wire message. -/
example {E : Type} [Field E] [Algebra Zq E] [Degree2 E] (pub : Public) (proof : ByteArray)
    (messages : Array Message) (tLast : ℕ → Rq Key20.terminal.innerDegree)
    (eLast : Rq Key20.terminal.innerDegree) :
    Reads20 (E := E) pub proof messages tLast eLast fun _ => note20 pub proof messages tLast eLast :=
  Reads20.of_all pub proof messages tLast eLast fun _ => rfl

/-- `SameRead` has an instance: a wire message and itself. -/
example {E : Type} [Field E] [Algebra Zq E] (w : Key20.Wire E) :
    SameRead Extract.Page.read w w := sameRead_page rfl

/-- `Faithful` has an instance: the messages of one proof with themselves. `Path` has the
instances `Suffix` and equality, and `Sim` the instance of a computation with no effect. -/
example (proof : ByteArray) (messages : Array Message) (l : List (Op × ℕ)) :
    Faithful proof proof messages messages l 0 := Faithful.refl proof messages l 0

/-- `Op.Parses` has an instance: a digest. `ReadsFrom` has an instance: no operation. -/
example (key : Key) : (Op.digest key).Parses := parses_digest key

example (proof : ByteArray) (messages : Array Message) : ReadsFrom proof messages [] 0 := trivial

example : Replayer.Path Replayer.Suffix ∧ Replayer.Path (fun a b : Duplex => a = b) :=
  ⟨Replayer.suffix_path, Replayer.eq_path⟩

example : Replayer.Sim (pure () : ReplayM Unit) (pure ()) := Replayer.sim_pure ()

/-- `skeleton20_ordered` is not empty: four operations are between the works of the stages 1 and
2 (three squeezes of fold challenges, then the message `next_witness`), and the last one reads a
message. -/
example : (between20 2).length = 4 ∧ ((between20 2).map fun x => x.1.reads) =
    [false, false, false, true] := by decide +kernel

end AkitaJoinGuards
