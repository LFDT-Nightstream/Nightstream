(module
  (type (;0;) (func (param i32)))
  (type (;1;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;2;) (func (param i64 i64 i64 i64)))
  (type (;3;) (func))
  (type (;4;) (func (param i32 i64 i64 i64 i64)))
  (type (;5;) (func (param i32 i64)))
  (type (;6;) (func (param i64)))
  (type (;7;) (func (param i32) (result i32)))
  (type (;8;) (func (result i32)))
  (type (;9;) (func (param i64 i64) (result i64)))
  (type (;10;) (func (param i32) (result i32 i64 i64 i32 i32)))
  (type (;11;) (func (param i32 i32 i64 i64 i32 i32) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;0;) (type 0) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;1;) (type 1) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;2;) (type 0) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;3;) (type 0) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;4;) (type 2) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;5;) (type 3)))
  (import "starstream:std/utxo-context" "[resource-drop]utxo-context" (func (;6;) (type 0) (param i32)))
  (import "starstream:std/utxo-context" "[method]utxo-context.resume" (func (;7;) (type 0) (param i32)))
  (import "starstream:std/utxo-context" "[method]utxo-context.implements-method" (func (;8;) (type 4) (param i32 i64 i64 i64 i64)))
  (import "starstream:contract/dynamic-utxo" "plus-chips" (func (;9;) (type 5) (param i32 i64)))
  (import "starstream:contract/dynamic-utxo" "plus-mult" (func (;10;) (type 5) (param i32 i64)))
  (import "starstream:contract/dynamic-utxo" "mult-mult" (func (;11;) (type 5) (param i32 i64)))
  (import "starstream:contract/dynamic-utxo" "finish" (func (;12;) (type 0) (param i32)))
  (import "starstream:events/score" "finish" (func (;13;) (type 6) (param i64)))
  (import "starstream:self/score-progress" "[resource-drop]utxo" (func (;14;) (type 0) (param i32)))
  (import "[export]score-progress" "[resource-new]utxo" (func (;15;) (type 7) (param i32) (result i32)))
  (import "[export]score-progress" "[resource-drop]utxo" (func (;16;) (type 0) (param i32)))
  (import "starstream:self/score-progress" "[static]utxo.new" (func (;17;) (type 8) (result i32)))
  (import "starstream:self/score-progress" "[method]utxo.plus-chips" (func (;18;) (type 5) (param i32 i64)))
  (import "starstream:self/score-progress" "[method]utxo.plus-mult" (func (;19;) (type 5) (param i32 i64)))
  (import "starstream:self/score-progress" "[method]utxo.mult-mult" (func (;20;) (type 5) (param i32 i64)))
  (import "starstream:self/score-progress" "[method]utxo.finish" (func (;21;) (type 0) (param i32)))
  (memory (;0;) 1)
  (global (;0;) (mut i32) (i32.const 0))
  (global (;1;) (mut i32) (i32.const 0))
  (global (;2;) (mut i64) (i64.const 0))
  (global (;3;) (mut i64) (i64.const 0))
  (global (;4;) (mut i32) (i32.const 0))
  (global (;5;) (mut i32) (i32.const 0))
  (export "score-progress#[static]utxo.new" (func 24))
  (export "score-progress#[method]utxo.plus-chips" (func 27))
  (export "score-progress#[method]utxo.plus-mult" (func 28))
  (export "score-progress#[method]utxo.mult-mult" (func 29))
  (export "score-progress#[method]utxo.finish" (func 30))
  (export "score-progress#get-storage" (func 32))
  (export "score-progress#set-storage" (func 33))
  (export "example" (func 34))
  (export "test13" (func 35))
  (export "memory" (memory 0))
  (func (;22;) (type 3)
    (call 7
      (global.get 1))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (global.get 0)
          (i32.const 1)))
      (return_call 25))
    (unreachable)
  )
  (func (;23;) (type 9) (param i64 i64) (result i64)
    (local i64)
    (local.tee 2
      (i64.mul
        (local.get 0)
        (local.get 1)))
    (if ;; label = @1
      (if (result i32) ;; label = @1
        (i64.eqz
          (local.get 1))
        (then
          (i32.const 0))
        (else
          (i64.ne
            (i64.div_u
              (local.get 2)
              (local.get 1))
            (local.get 0))))
      (then
        (unreachable)))
  )
  (func (;24;) (type 7) (param i32) (result i32)
    (local i32)
    (local.set 1
      (call 15
        (i32.const 0)))
    (global.set 1
      (local.get 0))
    (global.set 0
      (i32.const 1))
    (global.set 4
      (local.get 0))
    (global.set 5
      (local.get 1))
    (call 8
      (global.get 1)
      (i64.const -8462002850237496965)
      (i64.const 3777191895585416912)
      (i64.const -7032313579976096834)
      (i64.const -593716351821793027))
    (call 8
      (global.get 1)
      (i64.const -8415072477063627739)
      (i64.const 5859691412821126436)
      (i64.const -3810519776703246304)
      (i64.const -46505724939742969))
    (call 8
      (global.get 1)
      (i64.const -7959776885903164596)
      (i64.const -5373630742727988565)
      (i64.const -4027567226927148146)
      (i64.const -2641969232007030050))
    (call 8
      (global.get 1)
      (i64.const 5384737671178596162)
      (i64.const 7822374597054249491)
      (i64.const -6997249775370434463)
      (i64.const -2683880554626369430))
    (local.get 1)
  )
  (func (;25;) (type 3)
    (local i32)
    (local.set 0
      (global.get 4))
    (global.set 4
      (i32.const 0))
    (global.set 5
      (i32.const 0))
    (call 13
      (call 23
        (global.get 2)
        (global.get 3)))
    (call 6
      (local.get 0))
  )
  (func (;26;) (type 9) (param i64 i64) (result i64)
    (local i64)
    (local.tee 2
      (i64.add
        (local.get 0)
        (local.get 1)))
    (if ;; label = @1
      (i64.lt_u
        (local.get 2)
        (local.get 0))
      (then
        (unreachable)))
  )
  (func (;27;) (type 5) (param i32 i64)
    (global.set 2
      (call 26
        (global.get 2)
        (local.get 1)))
  )
  (func (;28;) (type 5) (param i32 i64)
    (global.set 3
      (call 26
        (global.get 3)
        (local.get 1)))
  )
  (func (;29;) (type 5) (param i32 i64)
    (global.set 3
      (i64.div_u
        (call 23
          (global.get 3)
          (local.get 1))
        (i64.const 100)))
  )
  (func (;30;) (type 0) (param i32)
    (return_call 22)
  )
  (func (;31;) (type 10) (param i32) (result i32 i64 i64 i32 i32)
    (global.get 0)
    (global.get 2)
    (global.get 3)
    (global.get 4)
    (global.get 5)
  )
  (func (;32;) (type 7) (param i32) (result i32)
    (local i32 i32 i64 i64 i32 i32)
    (i32.const 8)
    (call 31
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
    (i64.store offset=16
      (local.get 1)
      (local.get 4))
    (i32.store offset=24
      (local.get 1)
      (local.get 5))
    (i32.store offset=28
      (local.get 1)
      (local.get 6))
    (i32.const 8)
  )
  (func (;33;) (type 11) (param i32 i32 i64 i64 i32 i32) (result i32)
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
      (call 15
        (i32.const 0)))
  )
  (func (;34;) (type 3)
    (call 18
      (call 17)
      (i64.const 10))
  )
  (func (;35;) (type 3)
    (call 34)
  )
)
