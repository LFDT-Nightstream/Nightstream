import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm.Module

/-!
Contract: the Lean-owned WASM module for the 42-times-6 deployment.

Assurance tier: model-level.

Owns: the module structure, exact binary bytes, two instruction batches, and
the result obtained by executing the module semantics.

Does not own: F-prime, physical rows, Rust parsing, Wasmtime, Spartan, WHIR,
or a claim about arbitrary WASM programs.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm

def module : Wasm.Module where
  memoryMinimumPages := 1
  memoryMaximumPages := 1
  dataOffset := 0
  data := [byte 42, byte 0, byte 0, byte 0]
  exportName := [byte 0x6d, byte 0x61, byte 0x69, byte 0x6e]
  instructions :=
    [ .i32Const 0
    , .i32Load 2 0
    , .i32Const 6
    , .i32Mul
    ]

theorem module_valid : module.Valid := by
  decide

def certifiedModule : CertifiedModule where
  module := module
  valid := module_valid

def preparationInstructions : List Instruction :=
  [.i32Const 0, .i32Load 2 0, .i32Const 6]

def multiplicationInstructions : List Instruction :=
  [.i32Mul]

theorem instruction_batches_exact :
    preparationInstructions ++ multiplicationInstructions =
      module.instructions := by
  rfl

theorem module_bytes_exact :
    bytesToNats certifiedModule.bytes =
      [ 0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00
      , 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7f
      , 0x03, 0x02, 0x01, 0x00
      , 0x05, 0x04, 0x01, 0x01, 0x01, 0x01
      , 0x07, 0x08, 0x01, 0x04, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00
      , 0x0a, 0x0c, 0x01, 0x0a, 0x00, 0x41, 0x00, 0x28, 0x02, 0x00
      , 0x41, 0x06, 0x6c, 0x0b
      , 0x0b, 0x0a, 0x01, 0x00, 0x41, 0x00, 0x0b, 0x04, 0x2a, 0x00
      , 0x00, 0x00
      ] := by
  decide

theorem module_hex_exact :
    bytesHex certifiedModule.bytes =
      "0061736d010000000105016000017f03020100050401010101070801046d61696e00000a0c010a00410028020041066c0b0b0a010041000b042a000000" := by
  decide

theorem preparation_stack_exact :
    executeInstructions module.initialMemory preparationInstructions [] =
      some [6, 42] := by
  decide

theorem module_computes_252 :
    module.run = some 252 := by
  decide

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
