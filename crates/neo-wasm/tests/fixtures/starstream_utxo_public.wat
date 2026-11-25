(module
  (type (;0;) (func (param i32)))
  (type (;1;) (func (param i32 i64 i64 i64 i64) (result i32)))
  (type (;2;) (func (param i64 i64 i64 i64)))
  (type (;3;) (func))
  (type (;4;) (func (param i32 i64 i64 i64 i64)))
  (type (;5;) (func (param i64)))
  (type (;6;) (func (param i32) (result i32)))
  (type (;7;) (func (result i32)))
  (type (;8;) (func (param i32 i64) (result i64)))
  (type (;9;) (func (param i64 i64) (result i64)))
  (type (;10;) (func (param i32) (result i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)))
  (type (;11;) (func (param i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]utxo" (func (;0;) (type 0) (param i32)))
  (import "starstream:std/builtin" "[method]utxo.has-method" (func (;1;) (type 1) (param i32 i64 i64 i64 i64) (result i32)))
  (import "starstream:std/builtin" "[resource-drop]token" (func (;2;) (type 0) (param i32)))
  (import "starstream:std/builtin" "push-effect-handler-frame" (func (;3;) (type 0) (param i32)))
  (import "starstream:std/builtin" "add-effect-handler" (func (;4;) (type 2) (param i64 i64 i64 i64)))
  (import "starstream:std/builtin" "pop-effect-handler-frame" (func (;5;) (type 3)))
  (import "starstream:std/utxo-context" "[resource-drop]utxo-context" (func (;6;) (type 0) (param i32)))
  (import "starstream:std/utxo-context" "[method]utxo-context.resume" (func (;7;) (type 0) (param i32)))
  (import "starstream:std/utxo-context" "[method]utxo-context.implements-method" (func (;8;) (type 4) (param i32 i64 i64 i64 i64)))
  (import "starstream:contract/dynamic-utxo" "extra" (func (;9;) (type 0) (param i32)))
  (import "starstream:events/observations" "value" (func (;10;) (type 5) (param i64)))
  (import "starstream:self/counter" "[resource-drop]utxo" (func (;11;) (type 0) (param i32)))
  (import "[export]counter" "[resource-new]utxo" (func (;12;) (type 6) (param i32) (result i32)))
  (import "[export]counter" "[resource-drop]utxo" (func (;13;) (type 0) (param i32)))
  (import "starstream:self/counter" "[method]utxo.advance" (func (;14;) (type 0) (param i32)))
  (import "starstream:self/counter" "[static]utxo.new" (func (;15;) (type 7) (result i32)))
  (import "starstream:self/counter" "[static]utxo.other" (func (;16;) (type 7) (result i32)))
  (import "starstream:self/counter" "[method]utxo.add" (func (;17;) (type 8) (param i32 i64) (result i64)))
  (memory (;0;) 1)
  (global (;0;) (mut i32) (i32.const 0))
  (global (;1;) (mut i32) (i32.const 0))
  (global (;2;) (mut i64) (i64.const 0))
  (global (;3;) (mut i32) (i32.const 0))
  (global (;4;) (mut i32) (i32.const 0))
  (global (;5;) (mut i32) (i32.const 0))
  (global (;6;) (mut i32) (i32.const 0))
  (global (;7;) (mut i32) (i32.const 0))
  (global (;8;) (mut i32) (i32.const 0))
  (global (;9;) (mut i32) (i32.const 0))
  (global (;10;) (mut i32) (i32.const 0))
  (global (;11;) (mut i32) (i32.const 0))
  (global (;12;) (mut i32) (i32.const 0))
  (export "counter#[method]utxo.advance" (func 19))
  (export "counter#[static]utxo.new" (func 21))
  (export "counter#[static]utxo.other" (func 25))
  (export "counter#[method]utxo.add" (func 28))
  (export "counter#[method]utxo.extra" (func 29))
  (export "counter#get-storage" (func 31))
  (export "counter#set-storage" (func 32))
  (export "test15" (func 33))
  (export "test16" (func 34))
  (export "test17" (func 35))
  (export "test18" (func 36))
  (export "test19" (func 37))
  (export "memory" (memory 0))
  (func (;18;) (type 3)
    (call 7
      (global.get 1))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (global.get 0)
          (i32.const 1)))
      (return_call 22))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (global.get 0)
          (i32.const 2)))
      (return_call 23))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (global.get 0)
          (i32.const 3)))
      (return_call 24))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (global.get 0)
          (i32.const 4)))
      (return_call 26))
    (block ;; label = @1
      (br_if 0 (;@1;)
        (i32.ne
          (global.get 0)
          (i32.const 5)))
      (return_call 27))
    (unreachable)
  )
  (func (;19;) (type 0) (param i32)
    (return_call 18)
  )
  (func (;20;) (type 9) (param i64 i64) (result i64)
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
  (func (;21;) (type 6) (param i32) (result i32)
    (local i32)
    (local.set 1
      (call 12
        (i32.const 0)))
    (global.set 1
      (local.get 0))
    (global.set 2
      (i64.const 10))
    (global.set 0
      (i32.const 1))
    (global.set 3
      (local.get 0))
    (global.set 4
      (local.get 1))
    (call 8
      (global.get 1)
      (i64.const -4992487955248509468)
      (i64.const -2486664750538924298)
      (i64.const 6169022919517591603)
      (i64.const -8827430911239586701))
    (call 8
      (global.get 1)
      (i64.const -209942881528799618)
      (i64.const 3273890352309065231)
      (i64.const 3729828498835407705)
      (i64.const 7442122955733531148))
    (local.get 1)
  )
  (func (;22;) (type 3)
    (local i32 i32)
    (local.set 0
      (global.get 3))
    (global.set 3
      (i32.const 0))
    (local.set 1
      (global.get 4))
    (global.set 4
      (i32.const 0))
    (global.set 2
      (call 20
        (global.get 2)
        (i64.const 100)))
    (global.set 0
      (i32.const 2))
    (global.set 5
      (local.get 0))
    (global.set 6
      (local.get 1))
    (call 8
      (global.get 1)
      (i64.const 7802059248105938632)
      (i64.const -7464406540535523192)
      (i64.const -7930903170017996614)
      (i64.const -7603060425757492419))
    (call 8
      (global.get 1)
      (i64.const -4992487955248509468)
      (i64.const -2486664750538924298)
      (i64.const 6169022919517591603)
      (i64.const -8827430911239586701))
    (call 8
      (global.get 1)
      (i64.const -209942881528799618)
      (i64.const 3273890352309065231)
      (i64.const 3729828498835407705)
      (i64.const 7442122955733531148))
  )
  (func (;23;) (type 3)
    (local i32 i32)
    (local.set 0
      (global.get 5))
    (global.set 5
      (i32.const 0))
    (local.set 1
      (global.get 6))
    (global.set 6
      (i32.const 0))
    (global.set 2
      (call 20
        (global.get 2)
        (i64.const 1000)))
    (global.set 0
      (i32.const 3))
    (global.set 7
      (local.get 0))
    (global.set 8
      (local.get 1))
    (call 8
      (global.get 1)
      (i64.const -4992487955248509468)
      (i64.const -2486664750538924298)
      (i64.const 6169022919517591603)
      (i64.const -8827430911239586701))
    (call 8
      (global.get 1)
      (i64.const -209942881528799618)
      (i64.const 3273890352309065231)
      (i64.const 3729828498835407705)
      (i64.const 7442122955733531148))
  )
  (func (;24;) (type 3)
    (local i32)
    (local.set 0
      (global.get 7))
    (global.set 7
      (i32.const 0))
    (global.set 8
      (i32.const 0))
    (call 6
      (local.get 0))
  )
  (func (;25;) (type 6) (param i32) (result i32)
    (local i32)
    (block ;; label = @1
      (local.set 1
        (call 12
          (i32.const 0)))
      (global.set 1
        (local.get 0))
      (global.set 2
        (i64.const 40))
      (br_if 0 (;@1;)
        (i32.const 1))
      (global.set 0
        (i32.const 5))
      (global.set 11
        (local.get 0))
      (global.set 12
        (local.get 1))
      (call 8
        (global.get 1)
        (i64.const 7802059248105938632)
        (i64.const -7464406540535523192)
        (i64.const -7930903170017996614)
        (i64.const -7603060425757492419))
      (call 8
        (global.get 1)
        (i64.const -4992487955248509468)
        (i64.const -2486664750538924298)
        (i64.const 6169022919517591603)
        (i64.const -8827430911239586701))
      (call 8
        (global.get 1)
        (i64.const -209942881528799618)
        (i64.const 3273890352309065231)
        (i64.const 3729828498835407705)
        (i64.const 7442122955733531148))
      (return
        (local.get 1)))
    (global.set 0
      (i32.const 4))
    (global.set 9
      (local.get 0))
    (global.set 10
      (local.get 1))
    (call 8
      (global.get 1)
      (i64.const -4992487955248509468)
      (i64.const -2486664750538924298)
      (i64.const 6169022919517591603)
      (i64.const -8827430911239586701))
    (call 8
      (global.get 1)
      (i64.const -209942881528799618)
      (i64.const 3273890352309065231)
      (i64.const 3729828498835407705)
      (i64.const 7442122955733531148))
    (local.get 1)
  )
  (func (;26;) (type 3)
    (local i32)
    (local.set 0
      (global.get 9))
    (global.set 9
      (i32.const 0))
    (global.set 10
      (i32.const 0))
    (call 6
      (local.get 0))
  )
  (func (;27;) (type 3)
    (local i32)
    (local.set 0
      (global.get 11))
    (global.set 11
      (i32.const 0))
    (global.set 12
      (i32.const 0))
    (call 6
      (local.get 0))
  )
  (func (;28;) (type 8) (param i32 i64) (result i64)
    (global.set 2
      (call 20
        (global.get 2)
        (local.get 1)))
    (global.get 2)
  )
  (func (;29;) (type 0) (param i32))
  (func (;30;) (type 10) (param i32) (result i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (global.get 0)
    (global.get 2)
    (global.get 3)
    (global.get 4)
    (global.get 5)
    (global.get 6)
    (global.get 7)
    (global.get 8)
    (global.get 9)
    (global.get 10)
    (global.get 11)
    (global.get 12)
  )
  (func (;31;) (type 6) (param i32) (result i32)
    (local i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32)
    (i32.const 8)
    (call 30
      (local.get 0))
    (local.set 13)
    (local.set 12)
    (local.set 11)
    (local.set 10)
    (local.set 9)
    (local.set 8)
    (local.set 7)
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
    (i32.store offset=20
      (local.get 1)
      (local.get 5))
    (i32.store offset=24
      (local.get 1)
      (local.get 6))
    (i32.store offset=28
      (local.get 1)
      (local.get 7))
    (i32.store offset=32
      (local.get 1)
      (local.get 8))
    (i32.store offset=36
      (local.get 1)
      (local.get 9))
    (i32.store offset=40
      (local.get 1)
      (local.get 10))
    (i32.store offset=44
      (local.get 1)
      (local.get 11))
    (i32.store offset=48
      (local.get 1)
      (local.get 12))
    (i32.store offset=52
      (local.get 1)
      (local.get 13))
    (i32.const 8)
  )
  (func (;32;) (type 11) (param i32 i32 i64 i32 i32 i32 i32 i32 i32 i32 i32 i32 i32) (result i32)
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
    (global.set 6
      (local.get 6))
    (global.set 7
      (local.get 7))
    (global.set 8
      (local.get 8))
    (global.set 9
      (local.get 9))
    (global.set 10
      (local.get 10))
    (global.set 11
      (local.get 11))
    (global.set 12
      (local.get 12))
    (return
      (call 12
        (i32.const 0)))
  )
  (func (;33;) (type 3)
    (call 10
      (call 17
        (call 15)
        (i64.const 5)))
  )
  (func (;34;) (type 3)
    (local i32)
    (call 14
      (local.tee 0
        (call 15)))
    (call 10
      (call 17
        (local.get 0)
        (i64.const 7)))
  )
  (func (;35;) (type 3)
    (local i32)
    (call 10
      (call 17
        (local.tee 0
          (call 15))
        (i64.const 5)))
    (call 14
      (local.get 0))
    (call 10
      (call 17
        (local.get 0)
        (i64.const 7)))
    (call 14
      (local.get 0))
    (call 10
      (call 17
        (local.get 0)
        (i64.const 8)))
  )
  (func (;36;) (type 3)
    (call 10
      (call 17
        (call 16)
        (i64.const 2)))
  )
  (func (;37;) (type 3)
    (call 10
      (call 17
        (call 15)
        (i64.const 5)))
  )
)
