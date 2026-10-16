(* Bind validation's CALL types to the same store entries used by ROM encoding. *)
From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 StraightLineSpec SpecTecCalls CallSpec CallerSpec CallerRefinement
  FunctionValidation CallerValidation ProgramSpec ProgramEncoding SourceInstructions SourceValidation.
Import ListNotations.

Lemma state_function_store : forall z, store_FUNCS (fun_store z) = fun_funcinst z.
Proof. intros []; reflexivity. Qed.

Lemma program_callee_types : forall program s module C,
  store_FUNCS s = program -> Moduleinst_ok s module C ->
  calleeTypes C (programCallees program module).
Proof.
  intros program s module C Hstore Hmodule x f Hf.
  unfold programCallees in Hf.
  destruct (nth_error (FUNCS module) (N.to_nat x)) as [address | ] eqn:Ha; [ | discriminate].
  destruct (function_at_sound _ _ _ Hf) as [Hfi _].
  pose proof (module_function_type _ _ _ _ _ _ Hmodule Ha ltac:(rewrite Hstore; exact Hfi)) as Htype.
  exact Htype.
Qed.

Lemma program_callee_validates : forall program s module x f,
  Forall (fun fi => Funcinst_ok s fi (funcinst_TYPE fi)) program ->
  programCallees program module x = Some f ->
  Funcinst_ok s (i32Function (guestModule f) (guestTypeIndex f)
    (guestParams f) (guestResults f) (guestLocals f) (guestBody f))
    (i32StackType (guestParams f) (guestResults f)).
Proof.
  intros program s module x f Hvalid Hf. unfold programCallees in Hf.
  destruct (nth_error (FUNCS module) (N.to_nat x)) as [address | ]; [ | discriminate].
  destruct (function_at_sound _ _ _ Hf) as [Hfi _].
  rewrite Forall_forall in Hvalid. apply Hvalid. eapply nth_error_In; exact Hfi.
Qed.

Lemma represented_state_wf : forall z memory boundary code c,
  represents z memory boundary code c -> wf_state z.
Proof.
  intros z memory boundary code c [values [_ [-> [Hwf _]]]].
  inversion Hwf; assumption.
Qed.

Lemma validated_code_facts : forall program f z height code,
  frameMatches program f z -> wf_state z -> validatedCode program f z height code ->
  sourceFacts (programCallees program (guestModule f)) z height code.
Proof.
  intros program f z height code [Hstore [_ Hlocals]] Hz [Hfunctions [Hsupport [C [Hmodule Hcode]]]].
  assert (Htypes : calleeTypes (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
    (guestParams f) (guestResults f) (guestLocals f) (guestBody f))) (programCallees program (guestModule f))).
  { apply (program_callee_types program (fun_store z) (guestModule f) C ltac:(rewrite state_function_store; exact Hstore) Hmodule). }
  pose proof (source_validation_facts _ _ _ Hcode (programCallees program (guestModule f)) z) as H.
  cbn zeta in H. rewrite !repeat_length in H.
  specialize (H ltac:(rewrite function_context_locals by (eapply module_context_locals; eassumption);
      now rewrite Hlocals) Htypes).
  assert (Hcalls : forall x g, programCallees program (guestModule f) (x :> N) = Some g ->
    wf_uN 32%N x -> supportedStraightLineProgram (guestBody g) ->
    (guestParams g < 256)%nat -> (guestResults g < 256)%nat ->
    typedCall z x (guestModule g) (guestTypeIndex g) (guestParams g)
      (guestResults g) (guestLocals g) (guestBody g)).
  { intros x g Hg Hx Hs Hp Hr.
    pose proof (program_callee_validates _ _ _ _ _ Hfunctions Hg) as Hinst.
    split; [exact Hz | ]. split; [exact Hx | ]. split.
    - inversion Hinst; assumption.
    - split; [exact Hs | ]. split; [eapply funcinst_body_validates; eassumption | auto]. }
  destruct (H Hcalls Hsupport) as [Hfacts _].
  specialize (Hfacts 0%nat [] I). now rewrite app_nil_r in Hfacts.
Qed.

Lemma validated_code_remaining : forall program f z z' before after executed remaining,
  sameCaller z z' -> frameMatches program f z' ->
  validatedCode program f z before (executed ++ remaining) ->
  sourceHeights (programCallees program (guestModule f)) executed before = Some after ->
  sourceFacts (programCallees program (guestModule f)) z' after remaining ->
  validatedCode program f z' after remaining.
Proof.
  intros program f z z' before after executed remaining [Hstore _] [Hfunctions [_ Hlocals]]
    [Hvalid [Hsupport [C [Hmodule Hcode]]]] Hexecuted Hfacts.
  unfold validatedCode. rewrite Hstore.
  split; [exact Hvalid | ]. split; [apply Forall_app in Hsupport; exact (proj2 Hsupport) | ].
  assert (Htypes : calleeTypes (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
    (guestParams f) (guestResults f) (guestLocals f) (guestBody f))) (programCallees program (guestModule f))).
  { apply (program_callee_types program (fun_store z) (guestModule f) C); [ | exact Hmodule].
    rewrite <- state_function_store, Hstore in Hfunctions. exact Hfunctions. }
  destruct (source_facts_validate _ _ _ _ _ ltac:(rewrite function_context_locals by
      (eapply module_context_locals; eassumption); now rewrite Hlocals)
    ltac:(apply function_context_wf; eapply module_context_wf; eassumption) Htypes Hfacts) as [result Hresult].
  assert (Hcontext : context_LOCALS (functionContext C (i32Function (guestModule f) (guestTypeIndex f)
      (guestParams f) (guestResults f) (guestLocals f) (guestBody f))) = repeat valtype_I32 (length (stateLocals z'))).
  { rewrite function_context_locals by (eapply module_context_locals; eassumption). now rewrite Hlocals. }
  (* Both typings determine the remaining heights, so the result type is preserved. *)
  pose proof (source_validation_height _ _ _ Hcode _ z' Hcontext Htypes Hsupport 0%nat) as Hwhole.
  pose proof (source_validation_height _ _ _ Hresult _ z' Hcontext Htypes
    ltac:(apply Forall_app in Hsupport; exact (proj2 Hsupport)) 0%nat) as Hrest.
  rewrite !repeat_length in Hwhole, Hrest. cbn [Nat.add] in Hwhole, Hrest.
  rewrite source_heights_app, Hexecuted, Hrest in Hwhole. injection Hwhole as ->.
  exists C; auto.
Qed.
