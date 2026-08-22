import NightstreamFPrime.Export.PilotData

/-! Executable entry point for the canonical Stage 1 circuit-package emitter. -/

namespace NightstreamFPrime.Export.Main

open NightstreamFPrime.Export

def progress (message : String) : IO Unit := do
  IO.println message
  IO.getStdout >>= IO.FS.Stream.flush

def emit (path : System.FilePath) : IO Unit := do
  progress "emitter_stage=template"
  progress s!"template_rows={(PilotData.circuitPackage ()).permutation.rows.length}"
  progress "emitter_stage=identifier"
  let identifier := (PilotData.artifact ()).claimedIdentifier
  progress s!"relation_identifier={String.intercalate "," (identifier.map toString)}"
  progress "emitter_stage=render"
  let rendered := Package.Artifact.render (PilotData.artifact ()) ++ "\n"
  progress s!"rendered_bytes={rendered.length}"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path rendered
  IO.println s!"emitted={path}"

def run (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] =>
      emit ⟨path⟩
      pure 0
  | ["--", path] =>
      emit ⟨path⟩
      pure 0
  | _ =>
      IO.eprintln "usage: lake exe emit -- <output-path>"
      pure 2

end NightstreamFPrime.Export.Main

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.Main.run arguments
