(module
  (type (;0;) (func (param i32)))
  (type (;1;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;2;) (func (param i64 i64 i64 i64)))
  (type (;3;) (func))
  (type (;4;) (func (param i32 i64 i64) (result i64)))
  (type (;5;) (func (param i64) (result i64)))
  (type (;6;) (func (param i64 i64) (result i64)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;0;) (type 0) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;1;) (type 1) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;2;) (type 0) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;3;) (type 0) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;4;) (type 2) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;5;) (type 3)))
  (export "get-bar" (func 11))
  (func (;6;) (type 4) (param i32 i64 i64) (result i64)
    (local i32 i64 i64 i32 i64)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (block ;; label = @3
          (br_if 0 (;@3;)
            (i32.eq
              (local.tee 3
                (local.get 0)
                (local.set 4
                  (local.get 1)
                  (local.set 5
                    (local.get 2))))
              (i32.const 2)))
          (block ;; label = @4
            (br_if 0 (;@4;)
              (i32.eq
                (local.get 3)
                (i32.const 0)))
            (block ;; label = @5
              (br_if 0 (;@5;)
                (i32.eq
                  (local.get 3)
                  (i32.const 1)))
              (unreachable))
            (br 3 (;@1;)
              (local.get 4)))
          (drop
            (i32.wrap_i64
              (local.get 4)))
          (br 2 (;@1;)
            (i64.const 0)))
        (local.set 6
          (i32.wrap_i64
            (local.get 4)))
        (local.set 7
          (local.get 5))
        (br_if 0 (;@2;)
          (i32.eq
            (local.get 6)
            (i32.const 0)))
        (br 1 (;@1;)
          (i64.const 2)))
      (local.get 7))
  )
  (func (;7;) (type 4) (param i32 i64 i64) (result i64)
    (local i64)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (br_if 0 (;@2;)
          (i32.eq
            (local.get 0)
            (local.set 3
              (local.get 1))
            (i32.const 0)))
        (br 1 (;@1;)
          (i64.const 0)))
      (drop
        (i32.wrap_i64
          (local.get 3)))
      (i64.const 1))
  )
  (func (;8;) (type 5) (param i64) (result i64)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (br_if 0 (;@2;)
          (i64.eq
            (local.get 0)
            (i64.const 42)))
        (br 1 (;@1;)
          (i64.const 0)))
      (i64.const 1))
  )
  (func (;9;) (type 6) (param i64 i64) (result i64)
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
  (func (;10;) (type 4) (param i32 i64 i64) (result i64)
    (local i64 i64 i64 i64)
    (block (result i64) ;; label = @1
      (block ;; label = @2
        (br_if 0 (;@2;)
          (i32.eq
            (local.get 0)
            (local.set 3
              (local.get 1)
              (local.set 4
                (local.get 2)))
            (i32.const 1)))
        (br 1 (;@1;)
          (i64.const 0)))
      (local.set 5
        (local.get 3))
      (local.set 6
        (local.get 4))
      (call 9
        (local.get 5)
        (local.get 6)))
  )
  (func (;11;) (type 3)
    (drop
      (call 6
        (i32.const 0)
        (i64.extend_i32_u
          (i32.const 1))
        (i64.const 0)))
  )
)
