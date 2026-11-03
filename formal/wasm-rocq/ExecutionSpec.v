(* A running activation has an outer SpecTec function frame. A terminal outcome
   is bound to the final row's trap flag or captured i32 result. Runs stop on their
   terminal row; host transitions and padding after termination are not covered. *)
From Stdlib Require Import List ZArith.
Require Import R1CS Columns Wasm2 MemoryTrace StraightLineSpec SpecTecCalls CallerSpec ProgramSpec ControlDecode.
Import ListNotations.
Open Scope Z_scope.

Fixpoint rootSite (sites : list CodePosition) (active : nat) : nat :=
  match sites with [] => active | top :: rest => rootSite rest (positionFunction top) end.

Definition shapeRoot (shape : ControlShape) : nat := rootSite (callSites shape) (siteFunction (activeSite shape)).

(* Host depth identifies this activation's base; guest rows preserve it. *)
Definition ActivationConfig (program : Program) (s : store) (caller : frame) (baseDepth : Z)
    (root : nat) (shape : ControlShape) (memory : MemoryState) (boundary : Boundary) (c : config) : Prop :=
  boundaryHostDepth boundary = baseDepth /\
  wf_frame caller /\ shapeRoot shape = root /\
  exists f inner, functionAt program root = Some f /\
    CircuitConfig program s baseDepth shape memory boundary inner /\
    c = callContext caller (N.of_nat (guestResults f)) inner.

Definition terminalRow (row : Witness) : Prop :=
  row COL_IS_PROGRAM_ROW = 1 /\ row COL_ACTIVATION_FINISHED_AFTER = 1.

Fixpoint executionRows (baseDepth : Z) (rows : list Witness) : Prop :=
  match rows with
  | [] => True
  | row :: rest =>
      match rest with [] => callerRow baseDepth row \/ terminalRow row
      | _ => callerRow baseDepth row /\ executionRows baseDepth rest end
  end.

Definition executionFragment (boundary : Boundary) (baseDepth : Z) (rows : list Witness) : Prop :=
  runStartsAt boundary rows /\ executionRows baseDepth rows.

Inductive RunOutcome :=
| running (shape : ControlShape)
| trapped
| returned (result : option Z).

Definition lastRow (rows : list Witness) : option Witness := hd_error (rev rows).

Definition TerminalConfig (s : store) (caller : frame) (row : Witness) (outcome : RunOutcome) (c : config) : Prop :=
  row COL_ACTIVATION_FINISHED_AFTER = 1 /\
  match outcome with
  | running _ => False
  | trapped => row COL_TRAPPED_AFTER = 1 /\ c = mk_config (mk_state s caller) [admininstr_TRAP]
  | returned result => row COL_TRAPPED_AFTER = 0 /\
      (match result with None => row COL_OUTPUT_CAPTURED = 0
      | Some value => row COL_OUTPUT_CAPTURED = 1 /\ row COL_OUTPUT_VALUE_LO_AFTER = value end) /\
      c = mk_config (mk_state s caller)
        (match result with None => [] | Some value => [i32Constant value] end)
  end.

Definition ExecutionConfig (program : Program) (s : store) (caller : frame) (baseDepth : Z) (root : nat)
    (memory : MemoryState) (boundary : Boundary) (rows : list Witness) (outcome : RunOutcome) (c : config) : Prop :=
  match outcome with
  | running shape =>
      (match lastRow rows with None => True | Some row => row COL_ACTIVATION_FINISHED_AFTER = 0 end) /\
      ActivationConfig program s caller baseDepth root shape memory boundary c
  | _ => exists row, lastRow rows = Some row /\ TerminalConfig s caller row outcome c
  end.
