(* Trap accounting for the supported fragment: UNREACHABLE and an indirect call's
   resolve step trap; the other supported instructions cannot. Coverage is checked against the ROM encoder,
   and each trap cause is constrained by its own selectors. *)
From Stdlib Require Import List ZArith Lia.
Require Import R1CS Columns FullRelation Range Selectors CircuitFacts Wasm2 OpcodeCatalog
  StraightLineSpec CodeLayout Fragment ProgramSpec CallInitialization.
Import ListNotations.
Open Scope Z_scope.

(* Circuit selectors proved nontrapping below. Coverage of the source fragment
   is checked against the actual opcode ROM encoder by supported_opcode_trap_classification. *)
Definition nonTrappingSelectors : list Column :=
  [COL_SEL_I32_ADD; COL_SEL_I32_SUB; COL_SEL_LOCAL_GET; COL_SEL_LOCAL_SET; COL_SEL_CALL; COL_SEL_BLOCK;
   COL_SEL_LOOP; COL_SEL_IF; COL_SEL_ELSE; COL_SEL_END; COL_SEL_BR; COL_SEL_BR_IF_EQZ; COL_SEL_BR_TABLE;
   COL_SEL_RETURN; COL_SEL_NOP].

Definition factorsAre (a b c : option SparseRow) (row : Row) : bool :=
  let '(a', b', c') := row in
  let matches (expected : option SparseRow) actual :=
    match expected with Some terms => termsEqual actual terms | None => true end in
  matches a a' && matches b b' && matches c c'.

(* The row whose factors are given, with the selector `chosen` set. *)
Ltac selected_row_where chosen a b c name :=
  let found := eval vm_compute in (findRow (factorsAre a b c) FullRelation.rows) in
  lazymatch found with
  | Some ?index => selected_row chosen index name
  | None => fail "Missing circuit row" a b c
  end.

Lemma full_nontrapping_untrapped : forall (w : Witness) chosen,
  fieldPrime -> FullRelation.satFull w -> In chosen nonTrappingSelectors -> w chosen = 1 ->
  w COL_TRAPPED_AFTER = 0.
Proof.
  intros w chosen Hprime Hfull Hsupported Hchosen.
  assert (Hin : In chosen selectorColumns) by (vm_compute in Hsupported; vm_compute; intuition).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_selected_is_program_row w chosen Hprime Hfull Hin Hchosen) as Hprogram.
  destruct (full_program_row_quiet w chosen Hprime Hfull Hin Hchosen) as [Hkinds _ _ _].
  (* A program row starts untrapped. *)
  assert (Hbefore : w COL_TRAPPED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_TRAPPED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_TRAPPED_BEFORE). field_linear Hrow. }
  repeat (destruct Hsupported as [<- | Hsupported]; [ | ]); [ .. | contradiction];
    match goal with Hchosen : value w ?chosen = 1 |- _ =>
      selected_row_where chosen (@None SparseRow) (@None SparseRow)
        (Some [(COL_DIV_TRAP, 1); (COL_DIV_OVERFLOW, -1)]) Hdiv;
      selected_row_where chosen (@None SparseRow) (Some [(COL_DIV_OVERFLOW_COND, 1)])
        (Some [(COL_DIV_OVERFLOW, 1)]) Hoverflow;
      selected_row_where chosen (@None SparseRow) (Some [(COL_CMP_GE, 1)]) (Some [(COL_MEM_OOB, 1)]) Hmem;
      selected_row_where chosen (@None SparseRow) (Some [(COL_CMP_GE, 1)]) (Some [(COL_TABLE_OOB, 1)]) Htable;
      selected_row_where chosen (@None SparseRow) (Some [(COL_CMP_GE, 1)]) (Some [(COL_CI_OOB, 1)]) Hoob;
      selected_row_where chosen (@None SparseRow) (Some [(COL_ONE, 1); (COL_CI_ENTRY_IS_NULL, -1)])
        (Some [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, 1)]) Hgate;
      selected_row_where chosen (Some [(COL_FUNCTION_CALL_TYPE_LOOKUP_GATE, 1)]) (Some [(COL_CI_TYPE_EQ, 1)])
        (@None SparseRow) Htype;
      selected_circuit_row chosen
        ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1); (COL_DIV_TRAP, -1);
          (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Htrap
    end.
  all: zero_gather_kinds Hkinds Hmem; rewrite ?Hone in *.
  all: pose proof (value_bound w COL_DIV_OVERFLOW); pose proof (value_bound w COL_DIV_TRAP);
    pose proof (value_bound w COL_MEM_OOB); pose proof (value_bound w COL_TABLE_OOB); pose proof (value_bound w COL_CI_OOB);
    pose proof (value_bound w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE);
    pose proof (value_bound w COL_CALL_INDIRECT_IS_TRAP); pose proof (value_bound w COL_TRAPPED_AFTER).
  all: assert (Hov : w COL_DIV_OVERFLOW = 0) by field_linear Hoverflow.
  all: assert (Hdt : w COL_DIV_TRAP = 0) by field_linear Hdiv.
  all: assert (Hmo : w COL_MEM_OOB = 0) by field_linear Hmem.
  all: assert (Hto : w COL_TABLE_OOB = 0) by field_linear Htable.
  all: assert (Hco : w COL_CI_OOB = 0) by field_linear Hoob.
  all: normalize_columns; rewrite Hco in Hgate.
  all: assert (Hg : w COL_FUNCTION_CALL_TYPE_LOOKUP_GATE = 0) by field_linear Hgate.
  all: normalize_columns; rewrite Hg in Htype.
  all: assert (Hct : w COL_CALL_INDIRECT_IS_TRAP = 0) by field_linear Htype.
  all: field_linear Htrap.
Qed.

Ltac nontrapping_selector :=
  match goal with
  | |- exists chosen, In chosen nonTrappingSelectors /\ In (chosen, ?opcode) selectorCodes =>
    let found := eval vm_compute in (find (fun entry => Z.eqb (snd entry) opcode) selectorCodes) in
    lazymatch found with
    | Some (?chosen, _) => exists chosen; split; vm_compute; tauto
    end
  end.

(* Classify every supported source instruction's encoded opcode. UNREACHABLE and
   an indirect call's resolve step are the trapping cases; a function-label BR may
   encode RETURN. Adding a constructor
   to Supported must extend this proof; it cannot silently escape trap coverage. *)
Lemma supported_opcode_trap_classification : forall callees i program f pc targets opcode,
  Supported callees i ->
  instructionAtPC program pc = Some (f, Some (flatInstruction (opInstr i) targets)) ->
  instructionRom program MemoryCatalog.programOpcodeId pc = Some opcode ->
  (i = UNREACHABLE /\ opcode = CODE_UNREACHABLE) \/
  (exists x y, i = CALL_INDIRECT x y /\ opcode = CODE_CALL_INDIRECT_RESOLVE) \/
  exists chosen, In chosen nonTrappingSelectors /\ In (chosen, opcode) selectorCodes.
Proof.
  intros callees i program f pc targets opcode Hsupported Hslot Hcode.
  unfold instructionRom in Hcode. rewrite Hslot in Hcode.
  inversion Hsupported; subst;
    destruct targets as [ | [target | ] [ | next rest]]; cbn in Hcode;
    injection Hcode as <-;
    first [left; split; reflexivity | right; left; do 2 eexists; split; reflexivity |
      right; right; nontrapping_selector].
Qed.

(* END and ELSE are synthetic markers, not source instructions. *)
Lemma marker_opcode_nontrapping : forall program f pc marker targets opcode,
  (marker = opEnd \/ marker = opElse) ->
  instructionAtPC program pc = Some (f, Some (flatInstruction marker targets)) ->
  instructionRom program MemoryCatalog.programOpcodeId pc = Some opcode ->
  exists chosen, In chosen nonTrappingSelectors /\ In (chosen, opcode) selectorCodes.
Proof.
  intros program f pc marker targets opcode Hmarker Hslot Hcode.
  unfold instructionRom in Hcode. rewrite Hslot in Hcode.
  destruct Hmarker as [-> | ->]; cbn in Hcode; injection Hcode as <-;
    nontrapping_selector.
Qed.

(* UNREACHABLE is the fragment's sole trapping instruction. Its terminal flags
   follow from the constraints, rather than being supplied by the decoder. *)
Lemma full_unreachable_traps : forall (w : Witness),
  fieldPrime -> FullRelation.satFull w -> w COL_SEL_UNREACHABLE = 1 ->
  w COL_TRAPPED_AFTER = 1 /\ w COL_ACTIVATION_FINISHED_AFTER = 1.
Proof.
  intros w Hprime Hfull Hchosen.
  assert (Hin : In COL_SEL_UNREACHABLE selectorColumns) by (vm_compute; tauto).
  pose proof (proj1 Hfull) as Hone.
  pose proof (full_selected_is_program_row w _ Hprime Hfull Hin Hchosen) as Hprogram.
  assert (Htb : w COL_TRAPPED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_TRAPPED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_TRAPPED_BEFORE). field_linear Hrow. }
  assert (Hfb : w COL_ACTIVATION_FINISHED_BEFORE = 0).
  { circuit_row ([(COL_IS_PROGRAM_ROW, 1)], [(COL_ACTIVATION_FINISHED_BEFORE, 1)], ([] : SparseRow)) Hrow.
    expand_row Hrow. rewrite Hprogram in Hrow.
    pose proof (value_bound w COL_ACTIVATION_FINISHED_BEFORE). field_linear Hrow. }
  selected_circuit_row COL_SEL_UNREACHABLE
    ([(COL_TRAPPED_AFTER, 1); (COL_TRAPPED_BEFORE, -1); (COL_SEL_UNREACHABLE, -1); (COL_DIV_TRAP, -1);
      (COL_CALL_INDIRECT_IS_TRAP, -1); (COL_MEM_OOB, -1); (COL_TABLE_OOB, -1)], [(COL_ONE, 1)], ([] : SparseRow)) Htrap.
  circuit_boolean COL_TRAPPED_AFTER Hta. circuit_boolean COL_DIV_TRAP Hdiv.
  circuit_boolean COL_CALL_INDIRECT_IS_TRAP Hci. circuit_boolean COL_MEM_OOB Hmem.
  circuit_boolean COL_TABLE_OOB Htable.
  normalize_columns. rewrite Hone, Htb in Htrap.
  assert (Htrapped : w COL_TRAPPED_AFTER = 1).
  { normalize_columns. apply Z.mod_divide in Htrap; [ | unfold modulus; lia]. destruct Htrap as [k Htrap]. unfold modulus in *; lia. }
  split; [exact Htrapped | ].
  assert (Hturn : w COL_TURN_BOUNDARY = 0).
  { exact (full_row_kind_exclusive w COL_IS_PROGRAM_ROW COL_TURN_BOUNDARY Hprime Hfull
      ltac:(cbn; tauto) ltac:(cbn; tauto) ltac:(discriminate) Hprogram). }
  destruct (full_host_switch_absent w COL_IS_PROGRAM_ROW Hprime Hfull ltac:(cbn; auto 20)
    ltac:(cbn; discriminate) ltac:(cbn; discriminate) ltac:(cbn; discriminate) Hprogram) as [_ [Hresume _]].
  pose proof (full_no_activation_clear w Hfull Hturn Hresume) as Hclear.
  circuit_row ([(COL_ACTIVATION_FINISHED_AFTER, 1); (COL_ACTIVATION_FINISHED_BEFORE, -1); (COL_TRAPPED_AFTER, -1);
    (COL_TRAPPED_BEFORE, 1); (COL_HOST_CLEAR_ACTIVATION_FINISHED, 1); (COL_HOST_EVENT_EXIT_LATCH, -1)],
    [(COL_ONE, 1)], ([] : SparseRow)) Hstate.
  expand_row Hstate. normalize_columns. rewrite Hone, Hfb, Htrapped, Htb, Hclear in Hstate.
  circuit_boolean COL_HOST_EVENT_EXIT_LATCH Hlatch. circuit_boolean COL_ACTIVATION_FINISHED_AFTER Hfa.
  normalize_columns. apply Z.mod_divide in Hstate; [ | unfold modulus; lia]. destruct Hstate as [k Hstate]. unfold modulus in *; lia.
Qed.
