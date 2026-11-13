(module
  (type (;0;) (func (param i32)))
  (type (;1;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;2;) (func (param i64 i64 i64 i64)))
  (type (;3;) (func))
  (type (;4;) (func (param i64 i32) (result i32 i64)))
  (type (;5;) (func (param i64 i64) (result i64)))
  (type (;6;) (func (param i64 i64 i32) (result i64)))
  (type (;7;) (func (result i64)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;0;) (type 0) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;1;) (type 1) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;2;) (type 0) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;3;) (type 0) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;4;) (type 2) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;5;) (type 3)))
  (export "run-tuples" (func 10))
  (func (;6;) (type 4) (param i64 i32) (result i32 i64)
    (local i64 i32)
    (local.set 2
      (local.get 0)
      (local.set 3
        (local.get 1)))
    (local.get 3)
    (local.get 2)
  )
  (func (;7;) (type 5) (param i64 i64) (result i64)
    (local i64)
    (local.tee 2
      (i64.add
        (local.get 0)
        (local.get 1)))
    (if ;; label = @1
      (i32.ne
        (i32.and
          (i64.ge_s
            (i64.xor
              (local.get 0)
              (local.get 1))
            (i64.const 0))
          (i64.lt_s
            (i64.xor
              (local.get 2)
              (local.get 0))
            (i64.const 0)))
        (i32.const 0))
      (then
        (unreachable)))
  )
  (func (;8;) (type 6) (param i64 i64 i32) (result i64)
    (local i64 i64 i32)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (local.set 3
          (local.get 0)
          (local.set 4
            (local.get 1)
            (local.set 5
              (local.get 2))))
        (br_if 0 (;@2;)
          (local.get 5))
        (block ;; label = @3
          (br_if 0 (;@3;)
            (i32.eqz
              (local.get 5)))
          (unreachable))
        (br 1 (;@1;)
          (local.get 3)))
      (call 7
        (local.get 3)
        (local.get 4)))
  )
  (func (;9;) (type 7) (result i64)
    (local i32 i64)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (call 6
          (i64.const 1)
          (i32.const 1))
        (local.set 1)
        (local.tee 0)
        (br_if 0 (;@2;))
        (block ;; label = @3
          (br_if 0 (;@3;)
            (i32.eqz
              (local.get 0)))
          (unreachable))
        (br 1 (;@1;)
          (i64.const 0)))
      (local.get 1))
  )
  (func (;10;) (type 3)
    (drop
      (call 9))
    (drop
      (call 8
        (i64.const 1)
        (i64.const 2)
        (i32.const 1)))
  )
)
