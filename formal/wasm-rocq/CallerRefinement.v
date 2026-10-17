(* Facts shared by frames across rows: unchanged ROMs, the same caller frame shape,
   and SpecTec steps under a code suffix. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement
  Wasm2 WasmIntegers StraightLineSpec StackSimulation LocalSimulation SpecTecCalls
  CallSpec CallBody CallScope CallerSpec SourceInstructions StraightLineInstructions.
Import ListNotations.
Open Scope Z_scope.

Definition sameCaller (before after : state) : Prop :=
  fun_store after = fun_store before /\ fun_moduleinst after = fun_moduleinst before /\
  length (stateLocals after) = length (stateLocals before).

Definition sameRoms (before after : MemoryState) : Prop :=
  forall memory, readOnlyMemory memory memories = true ->
    forall address, after memory address = before memory address.

Lemma run_preserves_roms : forall initial rows final,
  validCircuitRun initial rows final -> sameRoms initial final.
Proof.
  intros initial rows final [_ [_ Hreplay]] memory Hreadonly address.
  eapply replay_preserves; [exact Hreplay | apply read_only_trace; exact Hreadonly].
Qed.

Ltac preserve_roms H :=
  repeat match goal with
  | |- context [ ?after ?memory ] =>
      progress setoid_rewrite (H memory ltac:(vm_compute; reflexivity))
  end.

Lemma caller_typed_preserved : forall program callees z z' height,
  length (stateLocals z') = length (stateLocals z) -> wf_state z' ->
  callerFacts callees z height program -> callerFacts callees z' height program.
Proof.
  induction program as [ | instruction rest IH]; intros callees z z' height Hcount Hwf Htyped; [exact I | ].
  assert (Hcontinue : forall next,
    (match next with Some h => callerFacts callees z h rest | None => False end) ->
    (match next with Some h => callerFacts callees z' h rest | None => False end)).
  { intros [h | ] H; [eapply IH; eassumption | contradiction]. }
  pose proof (Hcontinue (instructionStackHeight instruction height)) as Hnext.
  destruct instruction; cbn [callerFacts] in *;
    try (destruct Htyped as [Hs [Hi [Ht Hr]]];
      split; [exact Hs | ]; split; [exact Hi | ]; split; [now rewrite Hcount | exact (Hnext Hr)]).
  match goal with |- context [callees ?x] => destruct (callees x) as [f | ] end; [ | contradiction].
  destruct Htyped as [[_ Hcall] [Hheight Hrest]].
  split; [split; assumption | ]. split; [exact Hheight | eapply IH; eassumption].
Qed.

Lemma caller_program_wf : forall program callees z height,
  callerFacts callees z height program -> Forall wf_instr program.
Proof.
  induction program as [ | instruction rest IH]; intros callees z height Htyped; [constructor | ].
  assert (Hcontinue : forall next,
    (match next with Some h => callerFacts callees z h rest | None => False end) ->
    Forall wf_instr rest).
  { intros [h | ] H; [eapply IH; exact H | contradiction]. }
  pose proof (Hcontinue (instructionStackHeight instruction height)) as Hnext.
  destruct instruction; cbn [callerFacts] in Htyped;
    try (destruct Htyped as [_ [Hi [_ Hr]]]; constructor; [exact Hi | exact (Hnext Hr)]).
  match type of Htyped with context [callees ?x] => destruct (callees x) as [f | ] end; [ | contradiction].
  destruct Htyped as [[_ [Hindex _]] [_ Hrest]].
  constructor; [constructor; exact Hindex | eapply IH; exact Hrest].
Qed.

Lemma stack_suffix_steps : forall z z' values values' code rest,
  Steps (stackConfig z values code) (stackConfig z' values' []) ->
  Forall wf_instr rest ->
  Steps (stackConfig z values (code ++ rest)) (stackConfig z' values' rest).
Proof.
  intros z z' values values' code rest Hsteps Hrest.
  pose proof (steps_with_instruction_suffix rest _ _ Hrest Hsteps) as Hlift.
  unfold instructionSuffix, stackConfig in Hlift |- *.
  rewrite map_app. cbn [map] in Hlift. rewrite app_nil_r in Hlift.
  rewrite <- app_assoc in Hlift. exact Hlift.
Qed.
