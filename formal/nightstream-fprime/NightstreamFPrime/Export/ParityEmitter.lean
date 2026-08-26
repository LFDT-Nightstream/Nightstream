import NightstreamFPrime.Export.Codec

/-! Shared file boundary for the four isolated parity executables. -/

namespace NightstreamFPrime.Export.ParityEmitter

open NightstreamFPrime.Export.Codec

def emit (label : String) (value : Value) (path : System.FilePath) : IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeBinFile path (value.renderBytes.push 10)
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

end NightstreamFPrime.Export.ParityEmitter
