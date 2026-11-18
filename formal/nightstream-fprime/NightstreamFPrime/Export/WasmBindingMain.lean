import NightstreamFPrime.Export.WasmBinding

/-! The executable `checkWasmBinding`: it parses each export file given as an argument, decodes
it, and runs `WasmBinding.check`. -/

namespace NightstreamFPrime.Export.WasmBindingMain

open NightstreamFPrime.Export.Codec

private def isDigit (byte : UInt8) : Bool := 48 ≤ byte && byte ≤ 57

private partial def number (bytes : ByteArray) (pos value : ℕ) : ℕ × ℕ :=
  match bytes[pos]? with
  | some byte =>
    if isDigit byte then number bytes (pos + 1) (value * 10 + (byte - 48).toNat) else (value, pos)
  | none => (value, pos)

mutual

private partial def valueAt (bytes : ByteArray) (pos : ℕ) : Except String (Value × ℕ) :=
  match bytes[pos]? with
  | some 91 =>
    if bytes[pos + 1]? == some 93 then .ok (.array [], pos + 2) else elementsAt bytes (pos + 1) #[]
  | some byte =>
    if !isDigit byte then .error s!"unexpected byte {byte} at {pos}"
    else if byte == 48 && (bytes[pos + 1]?).any isDigit then .error s!"leading zero at {pos}"
    else
      let (n, next) := number bytes pos 0
      .ok (.atom n, next)
  | none => .error s!"unexpected end at {pos}"

private partial def elementsAt (bytes : ByteArray) (pos : ℕ) (done : Array Value) :
    Except String (Value × ℕ) := do
  let (item, next) ← valueAt bytes pos
  match bytes[next]? with
  | some 44 => elementsAt bytes (next + 1) (done.push item)
  | some 93 => .ok (.array (done.push item).toList, next + 1)
  | _ => .error s!"expected ',' or ']' at {next}"

end

/-- The value of an export text (`Value.render` of a numeric array, then one newline). -/
def parse (bytes : ByteArray) : Except String Value := do
  let (value, next) ← valueAt bytes 0
  if next + 1 = bytes.size && bytes[next]? == some 10 then pure value
  else throw s!"expected one final newline at {next}"

/-- Checks one file and prints its verdict. -/
def checkFile (path : String) : IO Bool := do
  match parse (← IO.FS.readBinFile path) >>= WasmBinding.decode with
  | .error reason =>
    IO.println s!"rejected {path}: {reason}"
    return false
  | .ok x =>
    if x.check then
      IO.println s!"accepted {path} size={WasmBinding.sizes.idxOf x.size} \
        slots={x.geometry.bOps} ports={x.plan.flatten.length}"
      return true
    else
      IO.println s!"rejected {path}: {x.reason.getD "no check failed"}"
      return false

end NightstreamFPrime.Export.WasmBindingMain

def main (arguments : List String) : IO UInt32 := do
  if arguments.isEmpty then
    IO.eprintln "usage: checkWasmBinding <file>..."
    return 2
  let mut accepted := true
  for path in arguments do
    let ok ← try NightstreamFPrime.Export.WasmBindingMain.checkFile path catch error => do
      IO.println s!"rejected {path}: {error}"
      pure false
    accepted := accepted && ok
  return if accepted then 0 else 1
