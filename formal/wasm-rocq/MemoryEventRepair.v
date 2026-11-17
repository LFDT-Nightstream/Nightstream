From Stdlib Require Import List ZArith Bool Lia.
Require Import R1CS MemoryTrace.
Import ListNotations.
Open Scope Z_scope.

Definition portColumns (port : PortSpec) : list Column :=
 addressColumns port ++ [valueColumn port] ++
 (match portKind port with Write (Some c) => [c] | _ => [] end) ++
 (match activation port with Always => [] | When c | Unless c => [c] end).

Lemma port_event_agrees : forall memory port (w w' : Witness),
 (forall c, In c (portColumns port) -> w' c = w c) ->
 portEvent memory w' port = portEvent memory w port.
Proof.
 intros memory [address value kind gate] w w' Hagree.
 assert (Haddress : map w' address = map w address).
 { apply map_ext_in. intros c Hin. apply Hagree.
   unfold portColumns; cbn. apply in_or_app; auto. }
 destruct kind as [|[prior|]]; destruct gate as [|gate|gate];
 unfold portEvent; cbn [activation active addressColumns valueColumn portKind map option_map];
 rewrite Haddress;
 repeat rewrite Hagree by (unfold portColumns; cbn; rewrite ?in_app_iff; simpl; tauto);
 reflexivity.
Qed.

Definition portRepairSafe (column : Column) (off : list Column) (port : PortSpec) : bool :=
 forallb (fun c => negb (Nat.eqb c column)) (portColumns port) ||
 match activation port with
 | When gate => negb (Nat.eqb gate column) && existsb (Nat.eqb gate) off
 | _ => false
 end.

Lemma port_repair_safe : forall column off memory port (w w' : Witness),
 (forall c, c <> column -> w' c = w c) ->
 Forall (fun gate => w gate = 0) off ->
 portRepairSafe column off port = true ->
 portEvent memory w' port = portEvent memory w port.
Proof.
 intros column off memory port w w' Hagree Hoff Hsafe.
 unfold portRepairSafe in Hsafe. apply orb_true_iff in Hsafe as [Hsafe|Hsafe].
 - apply port_event_agrees. intros c Hin. apply Hagree.
   pose proof (proj1 (forallb_forall _ _) Hsafe c Hin) as Hneq.
   apply negb_true_iff in Hneq. apply Nat.eqb_neq in Hneq. exact Hneq.
 - destruct (activation port) as [|gate|gate] eqn:Hgate; try discriminate.
   apply andb_true_iff in Hsafe as [Hneq Hin].
   apply negb_true_iff in Hneq. apply Nat.eqb_neq in Hneq.
   apply existsb_exists in Hin as [g [Hin Heq]]. apply Nat.eqb_eq in Heq. subst g.
   pose proof (proj1 (Forall_forall _ _) Hoff gate Hin) as Hz.
   unfold portEvent. rewrite Hgate. cbn [active]. rewrite Hagree by exact Hneq.
   rewrite Hz. reflexivity.
Qed.

Lemma events_repair_safe : forall catalog column off (w w' : Witness),
 (forall c, c <> column -> w' c = w c) ->
 Forall (fun gate => w gate = 0) off ->
 forallb (fun memory => forallb (portRepairSafe column off) (ports memory)) catalog = true ->
 eventsForRow catalog w' = eventsForRow catalog w.
Proof.
 intros catalog column off w w' Hagree Hoff Hsafe. unfold eventsForRow.
 induction catalog as [|memory rest IH]; [reflexivity|].
 cbn [forallb] in Hsafe. apply andb_true_iff in Hsafe as [Hports Hrest].
 cbn [flat_map]. rewrite (IH Hrest). f_equal.
 induction (ports memory) as [|port restPorts IHports]; [reflexivity|].
 cbn [forallb] in Hports. apply andb_true_iff in Hports as [Hport Htail].
 cbn [flat_map]. rewrite (port_repair_safe column off (memoryId memory) port w w' Hagree Hoff Hport).
 rewrite (IHports Htail). reflexivity.
Qed.
