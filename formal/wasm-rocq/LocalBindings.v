(* Local gates, frame-base preservation, and the stack-memory writes of a row.
   Locals live in their frame's stack cells, so a row's local writes are stack writes. *)
From Stdlib Require Import List ZArith Lia Bool.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts MemoryTrace MemoryCatalog
  CallInitialization.
Import ListNotations.
Open Scope Z_scope.

Definition straightLineSelector (chosen : Column) : Prop :=
  In chosen [COL_SEL_I32_ADD; COL_SEL_I32_SUB; COL_SEL_LOCAL_GET; COL_SEL_LOCAL_SET].

Lemma straight_line_selector_in : forall chosen,
  straightLineSelector chosen -> In chosen selectorColumns.
Proof. intros chosen H; destruct H as [<- | [<- | [<- | [<- | []]]]]; vm_compute; tauto. Qed.

Lemma full_straight_line_local_gates : forall (w : Witness) chosen,
  Range.fieldPrime -> FullRelation.satFull w ->
  straightLineSelector chosen -> w chosen = 1 ->
  w COL_LOCALS_FBP_AFTER = w COL_LOCALS_FBP_BEFORE /\
  w COL_LOCAL_WRITE_ENABLED = (if Nat.eqb chosen COL_SEL_LOCAL_SET then 1 else 0) /\
  w COL_PROGRAM_LOCAL_INDEX_ACTIVE =
    (if Nat.eqb chosen COL_SEL_LOCAL_GET || Nat.eqb chosen COL_SEL_LOCAL_SET then 1 else 0).
Proof.
  intros w chosen Hprime Hfull Hchoice Hchosen.
  pose proof (straight_line_selector_in chosen Hchoice) as Hin.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  destruct Hchoice as [<- | [<- | [<- | [<- | []]]]];
    cbn [COL_SEL_I32_ADD COL_SEL_I32_SUB COL_SEL_LOCAL_GET COL_SEL_LOCAL_SET Nat.eqb orb];
    match goal with Hchosen : value w ?chosen = 1 |- _ =>
      destruct (full_body_frame_sp_drop_idle w chosen Hprime Hfull Hin Hchosen
        ltac:(vm_compute; intuition discriminate)) as [Hpop Hpush _ Hturn _];
      split;
      [ selected_row chosen localsBasePreservedRow Hrow; zero_host_switch Hrow; rewrite Hone, Hpush, Hpop, Hturn in Hrow;
        pose proof (value_bound w COL_LOCALS_FBP_BEFORE);
        pose proof (value_bound w COL_LOCALS_FBP_AFTER); field_linear Hrow
      | split;
        [ selected_row chosen localWriteGateIndex Hrow; rewrite Hone in Hrow;
          pose proof (value_bound w COL_LOCAL_WRITE_ENABLED); field_linear Hrow
        | selected_row chosen localIndexGateRow Hrow; rewrite Hone in Hrow;
          pose proof (value_bound w COL_PROGRAM_LOCAL_INDEX_ACTIVE); field_linear Hrow ] ]
    end.
Qed.

(* Locals share the stack memory; their address columns follow the frame base. *)
Lemma full_local_addresses : forall (w : Witness),
  Range.fieldPrime -> FullRelation.satFull w ->
  w COL_LOCAL_ADDR_LO = 2 * (w COL_LOCALS_FBP_BEFORE + w COL_LOCAL_INDEX) /\
  w COL_LOCAL_ADDR_HI = w COL_LOCAL_ADDR_LO + 1.
Proof.
  intros w Hprime Hfull.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_range COL_LOCAL_ADDR_LO Hlo. circuit_range COL_LOCAL_ADDR_HI Hhi.
  circuit_range COL_LOCALS_FBP_BEFORE Hfbp. circuit_range COL_LOCAL_INDEX Hindex.
  circuit_row ([(COL_LOCAL_ADDR_LO, 1); (COL_LOCALS_FBP_BEFORE, -2); (COL_LOCAL_INDEX, -2)], [(COL_ONE, 1)],
    ([] : SparseRow)) Hrow.
  circuit_row ([(COL_LOCAL_ADDR_HI, 1); (COL_LOCAL_ADDR_LO, -1); (COL_ONE, -1)], [(COL_ONE, 1)],
    ([] : SparseRow)) Hnext.
  expand_row Hrow. expand_row Hnext. rewrite Hone in Hrow, Hnext.
  split; [field_linear Hrow | field_linear Hnext].
Qed.

(* Gather rows alone write entry inputs into locals. *)
Lemma full_no_gather_local_ports : forall (w : Witness),
  FullRelation.satFull w -> w COL_GATHER_ACTIVE = 0 ->
  w COL_GATHER_LOCAL_WRITE = 0 /\ w COL_GATHER_LOCAL_WRITE_LO = 0.
Proof.
  intros w Hfull Hgather.
  assert (Hone : w COL_ONE = 1) by exact (proj1 Hfull).
  circuit_row ([(COL_GATHER_LOCAL_WRITE, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) Hall.
  circuit_row ([(COL_GATHER_LOCAL_WRITE_LO, 1)], [(COL_ONE, 1); (COL_GATHER_ACTIVE, -1)], ([] : SparseRow)) Hlo.
  expand_row Hall. expand_row Hlo. rewrite Hone, Hgather in Hall, Hlo.
  pose proof (value_bound w COL_GATHER_LOCAL_WRITE). pose proof (value_bound w COL_GATHER_LOCAL_WRITE_LO).
  split; [field_linear Hall | field_linear Hlo].
Qed.

Definition isStackWrite (event : Event) : bool :=
  match event with
  | WriteEvent memory _ _ _ => Nat.eqb memory MemoryCatalog.stackId
  | ReadEvent _ _ _ => false
  end.

Definition writeIf (gate : Z) (event : Event) : list Event := if Z.eqb gate 1 then [event] else [].

(* The stack-memory writes of a row, in port order: the operand write, then the
   local's cell from local.set/tee or an entry input. *)
Definition stackPortWrites (w : Witness) : list Event :=
  writeIf (w COL_STACK_WRITE0_ACTIVE)
    (WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_LO] (w COL_STACK_WRITE0_VALUE_LO) None) ++
  writeIf (w COL_STACK_WRITE0_HI_ACTIVE)
    (WriteEvent MemoryCatalog.stackId [w COL_STACK_WRITE0_ADDR_HI] (w COL_STACK_WRITE0_VALUE_HI) None) ++
  writeIf (w COL_LOCAL_WRITE_ENABLED)
    (WriteEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_LO] (w COL_LOCAL_VALUE) None) ++
  writeIf (w COL_GATHER_LOCAL_WRITE_LO)
    (WriteEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_LO] (w COL_LOCAL_VALUE) None) ++
  writeIf (w COL_LOCAL_WRITE_ENABLED)
    (WriteEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_HI] (w COL_LOCAL_VALUE_HI) None) ++
  writeIf (w COL_GATHER_LOCAL_WRITE)
    (WriteEvent MemoryCatalog.stackId [w COL_LOCAL_ADDR_HI] (w COL_LOCAL_VALUE_HI) None).

Lemma filter_read_only_memory : forall w catalog,
  readOnlyMemory MemoryCatalog.stackId catalog = true ->
  filter isStackWrite (eventsForRow catalog w) = [].
Proof.
  intros w catalog. induction catalog as [ | specification rest IH]; [reflexivity | ].
  intros Hread. cbn [readOnlyMemory forallb] in Hread. apply andb_prop in Hread as [Hhead Hrest].
  unfold eventsForRow in *. cbn [flat_map]. rewrite filter_app, (IH Hrest), app_nil_r.
  destruct (Nat.eqb (memoryId specification) MemoryCatalog.stackId) eqn:Hsame.
  - induction (ports specification) as [ | port ports IHports]; [reflexivity | ].
    cbn [forallb] in Hhead. apply andb_prop in Hhead as [Hport Hports].
    cbn [flat_map]. rewrite filter_app, (IHports Hports).
    unfold portEvent. destruct (active w (activation port)); [ | reflexivity].
    destruct (portKind port); [reflexivity | discriminate].
  - induction (ports specification) as [ | port ports IHports]; [reflexivity | ].
    cbn [flat_map]. rewrite filter_app, IHports.
    unfold portEvent. destruct (active w (activation port)); [ | reflexivity].
    destruct (portKind port); cbn [filter isStackWrite]; [reflexivity | rewrite Hsame; reflexivity].
Qed.

Lemma stack_writes_in_row : forall (w : Witness),
  filter isStackWrite (eventsForRow MemoryCatalog.memories w) = stackPortWrites w.
Proof.
  intros w.
  change (filter isStackWrite (flat_map (fun p => match portEvent MemoryCatalog.stackId w p with
      Some event => [event] | None => [] end) (ports (hd (mkMemory 0%nat Ram []) MemoryCatalog.memories)) ++
    eventsForRow (tl MemoryCatalog.memories) w) = stackPortWrites w).
  rewrite filter_app, filter_read_only_memory by (vm_compute; reflexivity). rewrite app_nil_r.
  cbn [hd MemoryCatalog.memories ports]. unfold portEvent, active, stackPortWrites, writeIf.
  cbn [activation addressColumns valueColumn portKind map option_map flat_map].
  normalize_columns.
  repeat match goal with |- context [Z.eqb (value w ?c) 1] =>
    let b := fresh "b" in destruct (Z.eqb (value w c) 1) eqn:b end;
  reflexivity.
Qed.

Lemma no_stack_writes_preserve : forall initial events final,
  Replay initial events final -> filter isStackWrite events = [] ->
  forall address, final MemoryCatalog.stackId address = initial MemoryCatalog.stackId address.
Proof.
  intros initial events final Hreplay. induction Hreplay as [ | before middle after event events Hstep _ IH];
    intros Hfilter address; [reflexivity | ].
  cbn [filter] in Hfilter. destruct (isStackWrite event) eqn:Hwrite; [discriminate | ].
  rewrite (IH Hfilter address). eapply event_step_preserves; [exact Hstep | ].
  destruct event as [memory addr value | memory addr value prior]; cbn; auto.
  unfold isStackWrite in Hwrite. apply Nat.eqb_neq in Hwrite. auto.
Qed.

Lemma events_for_row_app : forall first second w,
  eventsForRow (first ++ second) w =
    eventsForRow first w ++ eventsForRow second w.
Proof. intros. unfold eventsForRow. apply flat_map_app. Qed.

Lemma events_for_row_cons : forall specification rest w,
  eventsForRow (specification :: rest) w =
    flat_map (fun p => match portEvent (memoryId specification) w p with
      | Some event => [event] | None => [] end) (ports specification) ++
    eventsForRow rest w.
Proof. reflexivity. Qed.

(* A row with every stack write gate off leaves the stack memory unchanged. *)
Lemma stack_writes_inactive : forall (w : Witness) address,
  w COL_STACK_WRITE0_ACTIVE = 0 -> w COL_STACK_WRITE0_HI_ACTIVE = 0 -> w COL_LOCAL_WRITE_ENABLED = 0 ->
  w COL_GATHER_LOCAL_WRITE_LO = 0 -> w COL_GATHER_LOCAL_WRITE = 0 ->
  Forall (NoWriteTo MemoryCatalog.stackId address) (eventsForRow MemoryCatalog.memories w).
Proof.
  intros w address Hlo Hhi Hlocal Hgatherlo Hgather.
  assert (Hwrites : filter isStackWrite (eventsForRow MemoryCatalog.memories w) = []).
  { rewrite stack_writes_in_row. unfold stackPortWrites, writeIf.
    rewrite Hlo, Hhi, Hlocal, Hgatherlo, Hgather. reflexivity. }
  apply Forall_forall. intros event Hin.
  destruct event as [memory addr value | memory addr value prior]; [exact I | ].
  cbn. left. intro Hsame. subst memory.
  assert (Hfiltered : In (WriteEvent MemoryCatalog.stackId addr value prior)
    (filter isStackWrite (eventsForRow MemoryCatalog.memories w))).
  { apply filter_In. split; [exact Hin | reflexivity]. }
  rewrite Hwrites in Hfiltered. contradiction.
Qed.
