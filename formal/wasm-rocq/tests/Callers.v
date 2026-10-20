From Stdlib Require Import List ZArith Lia.
Require Import Wasm2 StraightLineSpec CallerSpec Fragment tests.Calls.
Import ListNotations.
Open Scope Z_scope.

Definition readZeroCallee := mkGuestFunction 0%N callModule (mk_uN 0) 0 1 1 readZeroBody 10.
Definition callerCallees : CalleeEnvironment := fun index =>
  if N.eqb index 0 then Some readZeroCallee else None.
Definition getCallAdd : list instr :=
  [LOCAL_GET (mk_uN 0); CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)].

(* Operand heights through calls, from the callees' parameter and result counts. *)
Example get_call_add_height : sourceHeights callerCallees getCallAdd 0 = Some 1%nat.
Proof. reflexivity. Qed.

Example add_after_call_needs_two_values :
  sourceHeights callerCallees [CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)] 0 = None.
Proof. reflexivity. Qed.

Example missing_callee_rejected : sourceHeights callerCallees [CALL (mk_uN 1)] 0 = None.
Proof. reflexivity. Qed.

Example calls_share_one_environment_entry :
  sourceHeights callerCallees [CALL (mk_uN 0); CALL (mk_uN 0); BINOP I32 (mk_binop__0 Inn_I32 ADD)] 0 = Some 1%nat.
Proof. reflexivity. Qed.
