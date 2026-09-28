import NightstreamFPrime.Export.Codec
import NightstreamFPrime.Lifecycle.VerifierContext

/-! Shared input and file boundary for the isolated parity executables. -/

namespace NightstreamFPrime.Export.ParityEmitter

open NightstreamFPrime.Export.Codec
open NightstreamFPrime.Lifecycle.VerifierContext
open NightstreamFPrime.Spec

private def parseVerifierKeyWord (label word : String) : Except String F := do
  if word.isEmpty || word.length > 20 ||
      !(word.toList.all Char.isDigit) then
    throw s!"invalid verifier-key word {label}: expected decimal UInt64"
  let some value := word.toNat?
    | throw s!"invalid verifier-key word {label}: expected decimal UInt64"
  if _ : value < UInt64.size then
    if canonical : value < goldilocksModulus then
      pure ⟨value, canonical⟩
    else
      throw s!"invalid verifier-key word {label}: not a canonical Goldilocks element"
  else
    throw s!"invalid verifier-key word {label}: expected decimal UInt64"

/-- Parse fixture context words; the package conformance check establishes
their equality with the verifier-owned context. -/
def parseVerifierKey (w0 w1 w2 w3 : String) : Except String Digest4 := do
  let c0 ← parseVerifierKeyWord "vk0" w0
  let c1 ← parseVerifierKeyWord "vk1" w1
  let c2 ← parseVerifierKeyWord "vk2" w2
  let c3 ← parseVerifierKeyWord "vk3" w3
  pure { c0, c1, c2, c3 }

def emit (label : String) (value : Value) (path : System.FilePath) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let handle ← IO.FS.Handle.mk path .write
  let _ ← value.writeCanonical handle
  handle.write (ByteArray.empty.push 10)
  handle.flush
  IO.println s!"{label}={path}"

def run (label : String) (value : Value) (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] =>
      emit label value ⟨path⟩
      pure 0
  | ["--", path] =>
      emit label value ⟨path⟩
      pure 0
  | _ =>
      IO.eprintln "expected one output path"
      pure 2

def runIO (label : String) (value : IO Value)
    (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] =>
      emit label (← value) ⟨path⟩
      pure 0
  | ["--", path] =>
      emit label (← value) ⟨path⟩
      pure 0
  | _ =>
      IO.eprintln "expected one output path"
      pure 2

end NightstreamFPrime.Export.ParityEmitter
