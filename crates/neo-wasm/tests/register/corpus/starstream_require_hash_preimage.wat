(module
  (type (;0;) (func (param i64) (result i64)))
  (type (;1;) (func (param i32)))
  (type (;2;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;3;) (func (param i64 i64 i64 i64)))
  (type (;4;) (func))
  (type (;5;) (func (param i32 i64 i64 i64 i64)))
  (type (;6;) (func (param i32 i64)))
  (type (;7;) (func (param i32) (result i32)))
  (type (;8;) (func (param i64) (result i32)))
  (type (;9;) (func (param i32 i64) (result i32)))
  (type (;10;) (func (param i32) (result i32 i64 i32 i64 i32)))
  (type (;11;) (func (param i32 i32 i64 i32 i64 i32) (result i32)))
  (import "sha256lib" "sha256-u64" (func (;0;) (type 0) (param i64) (result i64)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;1;) (type 1) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;2;) (type 2) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;3;) (type 1) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;4;) (type 1) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;5;) (type 3) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;6;) (type 4)))
  (import "starstream:std/utxo-context" "[resource-drop]utxo-context" (func (;7;) (type 1) (param i32)))
  (import "starstream:std/utxo-context" "[method]utxo-context.resume" (func (;8;) (type 1) (param i32)))
  (import "starstream:std/utxo-context" "[method]utxo-context.implements-method" (func (;9;) (type 5) (param i32 i64 i64 i64 i64)))
  (import "starstream:contract/dynamic-utxo" "consume" (func (;10;) (type 6) (param i32 i64)))
  (import "starstream:self/require-hash-preimage" "[resource-drop]utxo" (func (;11;) (type 1) (param i32)))
  (import "[export]require-hash-preimage" "[resource-new]utxo" (func (;12;) (type 7) (param i32) (result i32)))
  (import "[export]require-hash-preimage" "[resource-drop]utxo" (func (;13;) (type 1) (param i32)))
  (import "starstream:self/require-hash-preimage" "[static]utxo.create" (func (;14;) (type 8) (param i64) (result i32)))
  (import "starstream:self/require-hash-preimage" "[method]utxo.consume" (func (;15;) (type 6) (param i32 i64)))
  (memory (;0;) 1)
  (global (;0;) (mut i32) (i32.const 0))
  (global (;1;) (mut i32) (i32.const 0))
  (global (;2;) (mut i64) (i64.const 0))
  (global (;3;) (mut i32) (i32.const 0))
  (global (;4;) (mut i64) (i64.const 0))
  (global (;5;) (mut i32) (i32.const 0))
  (export "require-hash-preimage#[static]utxo.create" (func 17))
  (export "require-hash-preimage#[method]utxo.consume" (func 19))
  (export "require-hash-preimage#get-storage" (func 21))
  (export "require-hash-preimage#set-storage" (func 22))
  (export "create-hash" (func 23))
  (export "consume-hash" (func 25))
  (export "test10" (func 26))
  (export "test11" (func 27))
  (export "test12" (func 28))
  (export "memory" (memory 0))
  (func (;16;) (type 4)
    (call 8
      (global.get 1))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (global.get 0)
          (i32.const 1)))
      (return_call 18))
    (unreachable)
  )
  (func (;17;) (type 9) (param i32 i64) (result i32)
    (local i32)
    (local.set 2
      (call 12
        (i32.const 0)))
    (global.set 1
      (local.get 0))
    (global.set 2
      (local.get 1))
    (global.set 0
      (i32.const 1))
    (global.set 3
      (local.get 0))
    (global.set 4
      (local.get 1))
    (global.set 5
      (local.get 2))
    (call 9
      (global.get 1)
      (i64.const -8176306661852515434)
      (i64.const -6925212332898645045)
      (i64.const -908562153280790436)
      (i64.const -2385339763266749307))
    (local.get 2)
  )
  (func (;18;) (type 4)
    (local i32)
    (local.set 0
      (global.get 3))
    (global.set 3
      (i32.const 0))
    (global.set 4
      (i64.const 0))
    (global.set 5
      (i32.const 0))
    (call 7
      (local.get 0))
  )
  (func (;19;) (type 6) (param i32 i64)
    (block ;; label = @1
      (block ;; label = @2
        (br_if 0 (;@2;)
          (i64.ne
            (call 0
              (local.get 1))
            (global.get 2)))
        (br 1 (;@1;)))
      (unreachable))
    (return_call 16)
  )
  (func (;20;) (type 10) (param i32) (result i32 i64 i32 i64 i32)
    (global.get 0)
    (global.get 2)
    (global.get 3)
    (global.get 4)
    (global.get 5)
  )
  (func (;21;) (type 7) (param i32) (result i32)
    (local i32 i32 i64 i32 i64 i32)
    (i32.const 8)
    (call 20
      (local.get 0))
    (local.set 6)
    (local.set 5)
    (local.set 4)
    (local.set 3)
    (local.set 2)
    (local.tee 1)
    (local.get 2)
    (i32.store)
    (i64.store offset=8
      (local.get 1)
      (local.get 3))
    (i32.store offset=16
      (local.get 1)
      (local.get 4))
    (i64.store offset=24
      (local.get 1)
      (local.get 5))
    (i32.store offset=32
      (local.get 1)
      (local.get 6))
    (i32.const 8)
  )
  (func (;22;) (type 11) (param i32 i32 i64 i32 i64 i32) (result i32)
    (global.set 1
      (local.get 0))
    (global.set 0
      (local.get 1))
    (global.set 2
      (local.get 2))
    (global.set 3
      (local.get 3))
    (global.set 4
      (local.get 4))
    (global.set 5
      (local.get 5))
    (return
      (call 12
        (i32.const 0)))
  )
  (func (;23;) (type 8) (param i64) (result i32)
    (call 14
      (call 0
        (local.get 0)))
  )
  (func (;24;) (type 6) (param i32 i64)
    (call 15
      (local.get 0)
      (local.get 1))
  )
  (func (;25;) (type 6) (param i32 i64)
    (call 24
      (local.get 0)
      (local.get 1))
    (call 11
      (local.get 0))
  )
  (func (;26;) (type 4)
    (call 24
      (call 23
        (i64.const 1))
      (i64.const 1))
  )
  (func (;27;) (type 4)
    (call 24
      (call 23
        (i64.const 1))
      (i64.const 2))
  )
  (func (;28;) (type 4)
    (drop
      (call 23
        (i64.const 1)))
  )
)
