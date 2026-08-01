/-!
Contract: a small Lean-owned WASM module format used by proof-carrying
application modules.

Assurance tier: model-level.

Owns: deterministic binary emission and executable semantics for the first
supported instruction subset: `i32.const`, `i32.load`, and `i32.mul`.

Does not own: arbitrary WASM parsing, host imports, control flow, F-prime,
physical rows, Rust, or a security reduction. Unsupported programs need a
larger certified module compiler; they must not be approximated by this one.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm

abbrev Byte := Fin 256

def byte (value : Nat) : Byte :=
  ⟨value % 256, Nat.mod_lt _ (by decide)⟩

inductive Instruction where
  | i32Const (value : Nat)
  | i32Load (alignmentExponent offset : Nat)
  | i32Mul
deriving DecidableEq, Repr

/-- A deliberately small, single-function WASM module. The interface is a
module value, not a protocol constant. A later compiler can add constructors
without changing F-prime or its proof backend. -/
structure Module where
  memoryMinimumPages : Nat
  memoryMaximumPages : Nat
  dataOffset : Nat
  data : List Byte
  exportName : List Byte
  instructions : List Instruction
deriving DecidableEq, Repr

def Instruction.Valid : Instruction -> Prop
  | .i32Const value => value < 64
  | .i32Load alignmentExponent offset =>
      alignmentExponent < 128 ∧ offset < 128
  | .i32Mul => True

instance (instruction : Instruction) : Decidable instruction.Valid := by
  cases instruction <;> simp [Instruction.Valid] <;> infer_instance

/-- Bounds for the one-byte section and immediate encoding used here. -/
def Module.Valid (module : Module) : Prop :=
  module.memoryMinimumPages < 128 ∧
  module.memoryMaximumPages < 128 ∧
  module.memoryMinimumPages ≤ module.memoryMaximumPages ∧
  module.dataOffset < 64 ∧
  module.data.length < 128 ∧
  module.exportName.length < 128 ∧
  module.exportName.all (fun value => decide (value.val < 128)) = true ∧
  module.instructions.length < 64 ∧
  module.instructions.all (fun instruction => decide instruction.Valid) = true

instance (module : Module) : Decidable module.Valid := by
  unfold Module.Valid
  infer_instance

structure CertifiedModule where
  module : Module
  valid : module.Valid

private def encodeUnsignedSmall (value : Nat) : List Byte :=
  [byte value]

private def encodeSignedNonnegativeSmall (value : Nat) : List Byte :=
  [byte value]

def Instruction.encode : Instruction -> List Byte
  | .i32Const value => byte 0x41 :: encodeSignedNonnegativeSmall value
  | .i32Load alignmentExponent offset =>
      byte 0x28 ::
        (encodeUnsignedSmall alignmentExponent ++ encodeUnsignedSmall offset)
  | .i32Mul => [byte 0x6c]

private def encodeSection
    (identifier : Nat) (payload : List Byte) : List Byte :=
  byte identifier :: byte payload.length :: payload

private def typeSection : List Byte :=
  encodeSection 1 [byte 1, byte 0x60, byte 0, byte 1, byte 0x7f]

private def functionSection : List Byte :=
  encodeSection 3 [byte 1, byte 0]

private def memorySection (module : Module) : List Byte :=
  encodeSection 5
    [ byte 1
    , byte 1
    , byte module.memoryMinimumPages
    , byte module.memoryMaximumPages
    ]

private def exportSection (module : Module) : List Byte :=
  encodeSection 7
    ([byte 1, byte module.exportName.length] ++ module.exportName ++
      [byte 0, byte 0])

private def functionBody (module : Module) : List Byte :=
  byte 0 :: module.instructions.flatMap Instruction.encode ++ [byte 0x0b]

private def codeSection (module : Module) : List Byte :=
  let body := functionBody module
  encodeSection 10 ([byte 1, byte body.length] ++ body)

private def dataSection (module : Module) : List Byte :=
  encodeSection 11
    ([ byte 1
     , byte 0
     , byte 0x41
     ] ++ encodeSignedNonnegativeSmall module.dataOffset ++
     [ byte 0x0b
     , byte module.data.length
     ] ++ module.data)

/-- Exact WebAssembly binary for the supported module subset. -/
def Module.encode (module : Module) : List Byte :=
  [ byte 0x00, byte 0x61, byte 0x73, byte 0x6d
  , byte 0x01, byte 0x00, byte 0x00, byte 0x00
  ] ++ typeSection ++ functionSection ++ memorySection module ++
    exportSection module ++ codeSection module ++ dataSection module

def CertifiedModule.bytes (module : CertifiedModule) : List Byte :=
  module.module.encode

def Module.initialMemory (module : Module) : List Nat :=
  List.replicate module.dataOffset 0 ++ module.data.map Fin.val

private def loadByte (memory : List Nat) (address : Nat) : Nat :=
  memory.getD address 0

def loadI32 (memory : List Nat) (address : Nat) : Nat :=
  (loadByte memory address +
      256 * loadByte memory (address + 1) +
      65536 * loadByte memory (address + 2) +
      16777216 * loadByte memory (address + 3)) % 2 ^ 32

def Instruction.execute
    (memory : List Nat) (stack : List Nat) :
    Instruction -> Option (List Nat)
  | .i32Const value => some (value % 2 ^ 32 :: stack)
  | .i32Load _ offset =>
      match stack with
      | address :: rest => some (loadI32 memory (address + offset) :: rest)
      | [] => none
  | .i32Mul =>
      match stack with
      | right :: left :: rest => some ((left * right) % 2 ^ 32 :: rest)
      | _ => none

def executeInstructions
    (memory : List Nat) : List Instruction -> List Nat -> Option (List Nat)
  | [], stack => some stack
  | instruction :: rest, stack =>
      match instruction.execute memory stack with
      | some next => executeInstructions memory rest next
      | none => none

def Module.run (module : Module) : Option Nat := do
  let stack ← executeInstructions module.initialMemory module.instructions []
  stack.head?

def bytesToNats (bytes : List Byte) : List Nat :=
  bytes.map Fin.val

private def hexDigit (value : Nat) : Char :=
  ("0123456789abcdef".toList.getD value '0')

def byteHex (value : Byte) : String :=
  String.ofList [hexDigit (value.val / 16), hexDigit (value.val % 16)]

def bytesHex (bytes : List Byte) : String :=
  String.intercalate "" (bytes.map byteHex)

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.Wasm
