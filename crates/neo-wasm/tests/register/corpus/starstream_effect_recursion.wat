(module
  (type (;0;) (func (param i32)))
  (type (;1;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;2;) (func (param i64 i64 i64 i64)))
  (type (;3;) (func))
  (type (;4;) (func (result i32)))
  (type (;5;) (func (param i32) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;0;) (type 0) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;1;) (type 1) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;2;) (type 0) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;3;) (type 0) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;4;) (type 2) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;5;) (type 3)))
  (import "starstream:effects/effects" "hello" (func (;6;) (type 3)))
  (import "starstream:events/effects" "log" (func (;7;) (type 0) (param i32)))
  (export "test0" (func 10))
  (export "starstream:effect-handler/effects#hello" (func 11))
  (func (;8;) (type 4) (result i32)
    (i32.const 1)
    (call 7
      (i32.const 4))
  )
  (func (;9;) (type 4) (result i32)
    (i32.const 1)
    (call 7
      (i32.const 3))
    (call 6)
    (call 7
      (i32.const 5))
  )
  (func (;10;) (type 3)
    (call 7
      (i32.const 1))
    (call 3
      (i32.const 1))
    (call 4
      (i64.const -845517842285046603)
      (i64.const -917691796666542500)
      (i64.const 6971151319575858812)
      (i64.const 7575422253805601197))
    (call 3
      (i32.const 2))
    (call 4
      (i64.const -845517842285046603)
      (i64.const -917691796666542500)
      (i64.const 6971151319575858812)
      (i64.const 7575422253805601197))
    (call 7
      (i32.const 2))
    (call 6)
    (call 7
      (i32.const 6))
    (call 5)
    (call 5)
    (call 7
      (i32.const 7))
  )
  (func (;11;) (type 5) (param i32) (result i32)
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (local.get 0)
          (i32.const 1)))
      (return_call 8))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (local.get 0)
          (i32.const 2)))
      (return_call 9))
    (i32.const 0)
  )
)
