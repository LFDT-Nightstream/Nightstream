(module
  (type (;0;) (func (param i32)))
  (type (;1;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;2;) (func (param i64 i64 i64 i64)))
  (type (;3;) (func))
  (type (;4;) (func (result i32 i64)))
  (type (;5;) (func (param i32 i64) (result i64)))
  (type (;6;) (func (param i32 i64 i32 i64)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;0;) (type 0) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;1;) (type 1) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;2;) (type 0) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;3;) (type 0) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;4;) (type 2) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;5;) (type 3)))
  (export "run" (func 12))
  (func (;6;) (type 4) (result i32 i64)
    (i32.const 1)
    (i64.const 42)
  )
  (func (;7;) (type 4) (result i32 i64)
    (i32.const 0)
    (i64.const 0)
  )
  (func (;8;) (type 5) (param i32 i64) (result i64)
    (local i32 i64)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (br_if 0 (;@2;)
          (i32.eq
            (local.tee 2
              (local.get 0)
              (local.set 3
                (local.get 1)))
            (i32.const 1)))
        (block ;; label = @3
          (br_if 0 (;@3;)
            (i32.eq
              (local.get 2)
              (i32.const 0)))
          (unreachable))
        (br 1 (;@1;)
          (i64.const 0)))
      (local.get 3))
  )
  (func (;9;) (type 4) (result i32 i64)
    (i32.const 0)
    (i64.const 42)
  )
  (func (;10;) (type 4) (result i32 i64)
    (i32.const 1)
    (i64.extend_i32_u
      (i32.const 0))
  )
  (func (;11;) (type 5) (param i32 i64) (result i64)
    (local i32 i64)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (br_if 0 (;@2;)
          (i32.eq
            (local.tee 2
              (local.get 0)
              (local.set 3
                (local.get 1)))
            (i32.const 0)))
        (block ;; label = @3
          (br_if 0 (;@3;)
            (i32.eq
              (local.get 2)
              (i32.const 1)))
          (unreachable))
        (drop
          (i32.wrap_i64
            (local.get 3)))
        (br 1 (;@1;)
          (i64.const 0)))
      (local.get 3))
  )
  (func (;12;) (type 6) (param i32 i64 i32 i64)
    (drop
      (call 8
        (call 6)))
    (drop
      (call 11
        (call 9)))
  )
)
