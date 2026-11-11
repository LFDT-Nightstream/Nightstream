(module
  (type (;0;) (func (param i32)))
  (type (;1;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;2;) (func (param i64 i64 i64 i64)))
  (type (;3;) (func))
  (type (;4;) (func (param i64)))
  (type (;5;) (func (param i32) (result i32)))
  (type (;6;) (func (result i32)))
  (type (;7;) (func (param i32) (result i64)))
  (type (;8;) (func (param i32 i32)))
  (type (;9;) (func (param i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;0;) (type 0) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;1;) (type 1) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;2;) (type 0) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;3;) (type 0) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;4;) (type 2) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;5;) (type 3)))
  (import "starstream:events/my-token-events" "test" (func (;6;) (type 4) (param i64)))
  (import "starstream:self/my-token" "[resource-drop]token" (func (;7;) (type 0) (param i32)))
  (import "[export]my-token" "[resource-new]token" (func (;8;) (type 5) (param i32) (result i32)))
  (import "[export]my-token" "[resource-drop]token" (func (;9;) (type 0) (param i32)))
  (import "starstream:self/my-token" "[static]token.my-mint" (func (;10;) (type 6) (result i32)))
  (import "starstream:self/my-token" "[static]token.my-burn" (func (;11;) (type 7) (param i32) (result i64)))
  (global (;0;) (mut i64) (i64.const 0))
  (export "my-token#[static]token.my-mint" (func 12))
  (export "my-token#[static]token.my-burn" (func 13))
  (export "my-token#[method]token.attach" (func 14))
  (export "my-token#[method]token.detach" (func 15))
  (export "my-token#get-storage" (func 16))
  (export "my-token#set-storage" (func 17))
  (export "call-token-mint" (func 18))
  (func (;12;) (type 6) (result i32)
    (local i32)
    (local.set 0
      (call 8
        (i32.const 0)))
    (global.set 0
      (i64.const 1))
    (local.get 0)
  )
  (func (;13;) (type 7) (param i32) (result i64)
    (call 6
      (global.get 0))
    (global.get 0)
  )
  (func (;14;) (type 8) (param i32 i32))
  (func (;15;) (type 8) (param i32 i32))
  (func (;16;) (type 7) (param i32) (result i64)
    (global.get 0)
  )
  (func (;17;) (type 9) (param i64) (result i32)
    (global.set 0
      (local.get 0))
    (return
      (call 8
        (i32.const 0)))
  )
  (func (;18;) (type 3)
    (local i32 i64)
    (local.set 0
      (call 10))
    (call 6
      (i64.const 10))
    (local.set 1
      (call 11
        (local.get 0)))
    (call 6
      (i64.const 20))
    (call 6
      (local.get 1))
  )
)
