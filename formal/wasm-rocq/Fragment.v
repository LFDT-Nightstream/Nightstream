(* The supported Wasm fragment: one constructor per supported instruction, with the
   restriction on where it may occur. Supporting a new instruction, or relaxing a
   restriction, is a change to this list. Validation is separate (ProgramSpec). *)
From Stdlib Require Import List ZArith.
Require Import Wasm2 CallerSpec BlockSpec.
Import ListNotations.

(* Where an instruction sits: a function body, whose calls resolve through its
   module's callees, or inside a block. *)
Inductive Site :=
| functionSite (callees : CalleeEnvironment)
| blockSite.

Inductive Supported : Site -> instr -> Prop :=
| supported_i32_add : forall site, Supported site (BINOP I32 (mk_binop__0 Inn_I32 ADD))
| supported_i32_sub : forall site, Supported site (BINOP I32 (mk_binop__0 Inn_I32 SUB))
| supported_local_get : forall site x, Supported site (LOCAL_GET x)
| supported_local_set : forall site x, Supported site (LOCAL_SET x)
(* Direct calls from a function body, not from inside a block. The circuit's call
   metadata holds the parameter and result counts in one byte each. *)
| supported_call : forall callees x f, callees (x :> N) = Some f ->
    (guestParams f < 256)%nat -> (guestResults f < 256)%nat ->
    Supported (functionSite callees) (CALL x)
(* Zero-result blocks in a function body, not nested. At each branch site only the
   condition may be above the label's stack prefix: no unwinding. *)
| supported_block : forall callees body,
    Forall (Supported blockSite) body -> branchSitesBalanced body 0 ->
    Supported (functionSite callees) (BLOCK (_RESULT None) body)
| supported_br_if : Supported blockSite (BR_IF (mk_uN 0)).

Definition supportedCode (site : Site) (code : list instr) : Prop := Forall (Supported site) code.
