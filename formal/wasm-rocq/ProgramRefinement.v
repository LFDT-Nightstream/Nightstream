(* Public refinement: one program ROM binding and a changing source position. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Range MemoryTrace TraceRefinement Wasm2 StraightLineSpec
  CallerSpec CallerRefinement ProgramSpec ProgramEncoding.
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

(* The supported fragment is restricted by callerTyped and segmentMatches.
   programLoaded fixes the ROM once; SourceState contains no ROM lookups. *)
Theorem run_refinement : forall program position executed remaining segments z initial final boundary c0,
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
  destruct Hsource as [Hframe [Hcode [Htyped [Hrep Hpc]]]].
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
    split; [exact Htyped' | ]. split; [exact Hrep' | ]. rewrite Nat2Z.inj_add. lia.
Qed.
