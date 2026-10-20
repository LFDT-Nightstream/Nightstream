(* Facts shared by frames across rows: unchanged ROMs, the same caller frame shape,
   and SpecTec steps under a code suffix. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range MemoryTrace MemoryCatalog TraceRefinement
  Wasm2 WasmIntegers StraightLineSpec StackSimulation LocalSimulation SpecTecCalls
  CallSpec CallBody CallScope CallerSpec StraightLineInstructions.
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
