(* Public refinement: one program ROM binding and a changing source position. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Range MemoryTrace TraceRefinement Wasm2 StraightLineSpec
  CallerSpec CallerRefinement ProgramSpec ProgramEncoding TraceSegmentation
  MemoryCatalog OpcodeCatalog ProgramValidation.
Import ListNotations.
Open Scope Z_scope.

Lemma code_at_advance : forall f offset executed remaining,
  codeAt f offset (executed ++ remaining) ->
  codeAt f (offset + length executed) remaining.
Proof.
  intros f offset executed remaining [Hbound Hcode]. split.
  - pose proof (f_equal (@length instr) Hcode) as Hlength.
    rewrite length_skipn, length_app in Hlength. lia.
  - replace (offset + length executed)%nat with (length executed + offset)%nat by lia.
    rewrite <- skipn_skipn, Hcode, skipn_app, skipn_all, Nat.sub_diag. reflexivity.
Qed.

Lemma frame_matches_preserved : forall program f before after,
  sameCaller before after -> frameMatches program f before -> frameMatches program f after.
Proof.
  intros program f before after [Hs [Ha Hl]] [Hfunctions [Haddresses Hlocals]].
  unfold frameMatches. rewrite Ha, Hl. split; [ | auto].
  destruct before, after. cbn [fun_store fun_funcinst] in *. congruence.
Qed.

(* Generated validation and opcode support are separate parts of SourceState.
   programLoaded fixes the ROM once; SourceState contains no ROM lookups. *)
Lemma segmented_run_refinement : forall program position executed remaining segments z initial final boundary c0,
  fieldPrime -> programLoaded program initial ->
  validCircuitRun initial (callerRows segments) final ->
  Forall2 segmentMatches executed segments -> callerStartsAt boundary segments ->
  SourceState program position z initial boundary (executed ++ remaining) c0 ->
  exists z' c1, Steps c0 c1 /\ programLoaded program final /\
    SourceState program (advanceSource position (length executed)) z' final
      (callerFinalBoundary boundary segments) remaining c1.
Proof.
  intros program [index offset] executed remaining segments z initial final boundary c0
    Hprime Hloaded Hrun Hmatches Hstart Hsource.
  unfold SourceState in Hsource. cbn [sourceFunction sourceOffset] in Hsource.
  destruct (functionAt program index) as [f | ] eqn:Hf; [ | contradiction].
  destruct Hsource as [Hframe [Hcode [Hvalidation [Hrep Hpc]]]].
  pose proof (validated_code_facts _ _ _ _ _ Hframe
    (represented_state_wf _ _ _ _ _ Hrep) Hvalidation) as Htyped.
  assert (Hcaller : CallerState (programCallees program (guestModule f)) z initial boundary
      (guestEntry f + Z.of_nat offset) (executed ++ remaining) c0).
  { constructor; try assumption. eapply program_loaded_caller; eassumption. }
  destruct (caller_run_refinement executed remaining segments (programCallees program (guestModule f)) z
    initial final boundary (guestEntry f + Z.of_nat offset) c0 Hprime Hrun Hmatches Hstart Hcaller)
    as [z' [c1 [Hsteps [[_ Htyped' Hrep' Hpc'] Hidentity]]]].
  exists z', c1. split; [exact Hsteps | ]. split.
  - eapply program_loaded_preserved; [eapply run_preserves_roms; exact Hrun | exact Hloaded].
  - unfold SourceState. cbn [advanceSource sourceFunction sourceOffset]. rewrite Hf.
    split; [eapply frame_matches_preserved; eassumption | ].
    split; [apply code_at_advance; exact Hcode | ].
    split.
    + eapply validated_code_remaining; [exact Hidentity | eapply frame_matches_preserved; eassumption | exact Hvalidation | exact Htyped'].
    + split; [exact Hrep' | ]. rewrite Nat2Z.inj_add. lia.
Qed.

(* Every row is consumed. The executed prefix and its grouping are consequences
   of validity and the ROM binding, rather than inputs supplied by the caller. *)
Theorem run_refinement : forall program position code rows z initial final boundary depth c0,
  fieldPrime -> programLoaded program initial -> validCircuitRun initial rows final ->
  callerFragment boundary depth rows ->
  SourceState program position z initial boundary code c0 ->
  exists executed remaining z' c1,
    code = executed ++ remaining /\ Steps c0 c1 /\ programLoaded program final /\
    SourceState program (advanceSource position (length executed)) z' final
      (rowsFinalBoundary boundary rows) remaining c1.
Proof.
  intros program [index offset] code rows z initial final boundary depth c0
    Hprime Hloaded Hrun Hfragment Hsource.
  pose proof Hsource as Hparts. unfold SourceState in Hparts.
  cbn [sourceFunction sourceOffset] in Hparts.
  destruct (functionAt program index) as [f | ] eqn:Hf; [ | contradiction].
  destruct Hparts as [Hframe [Hcode [Hvalidation [Hrep Hpc]]]].
  pose proof (validated_code_facts _ _ _ _ _ Hframe
    (represented_state_wf _ _ _ _ _ Hrep) Hvalidation) as Htyped.
  assert (Hcaller : CallerState (programCallees program (guestModule f)) z initial boundary
      (guestEntry f + Z.of_nat offset) code c0).
  { constructor; try assumption. eapply program_loaded_caller; eassumption. }
  assert (Hterminal :
    initial programOpcodeId [guestEntry f + Z.of_nat offset + Z.of_nat (length code)] = Some CODE_END /\
    initial pcEdgeKindId [guestEntry f + Z.of_nat offset + Z.of_nat (length code)] = Some 1).
  { destruct Hcode as [Hbound Hsuffix].
    pose proof (f_equal (@length instr) Hsuffix) as Hlength. rewrite length_skipn in Hlength.
    replace (guestEntry f + Z.of_nat offset + Z.of_nat (length code)) with
      (guestEntry f + Z.of_nat (length (guestBody f))) by lia.
    exact (encoded_function_end _ _ _ _ Hloaded Hf). }
  destruct (valid_caller_run_has_segments code rows (programCallees program (guestModule f)) z
    initial final boundary (guestEntry f + Z.of_nat offset) depth c0
    Hprime Hrun Hfragment Hcaller (proj1 Hterminal) (proj2 Hterminal))
    as [executed [remaining [segments [Hsplit [Hrows [Hmatches Hstart]]]]]].
  rewrite Hsplit in Hsource. rewrite Hrows in Hrun.
  destruct (segmented_run_refinement program (sourcePosition index offset) executed remaining segments z
    initial final boundary c0 Hprime Hloaded Hrun Hmatches Hstart Hsource)
    as [z' [c1 [Hsteps [Hloaded' Hsource']]]].
  exists executed, remaining, z', c1. split; [exact Hsplit | ].
  split; [exact Hsteps | ]. split; [exact Hloaded' | ].
  rewrite Hrows, caller_rows_final_boundary. exact Hsource'.
Qed.
