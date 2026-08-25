import NightstreamFPrime.Export.Stage1.Data
import NightstreamFPrime.Export.Stage1.PilotParity
import NightstreamFPrime.Export.Stage1.PiCCSParity

/-! Executable entry point for the canonical Stage 1 circuit-package emitter. -/

namespace NightstreamFPrime.Export.Main

open NightstreamFPrime.Export

def progress (message : String) : IO Unit := do
  IO.println message
  IO.getStdout >>= IO.FS.Stream.flush

def emit (path : System.FilePath) : IO Unit := do
  progress "emitter_stage=template"
  progress s!"template_rows={
    (PilotData.permutationTemplate ()).rows.length}"
  let artifact := Stage1.Data.artifact ()
  progress "emitter_stage=render"
  let rendered := Package.Artifact.render artifact ++ "\n"
  progress s!"rendered_bytes={rendered.length}"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path rendered
  IO.println s!"emitted={path}"

def emitPiCcsParity (path : System.FilePath) : IO Unit := do
  let rendered := Stage1.PiCCSParity.render ++ "\n"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path rendered
  IO.println s!"emitted_pi_ccs_parity={path}"

def emitPilotParity (path : System.FilePath) : IO Unit := do
  let rendered := Stage1.PilotParity.render ++ "\n"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  IO.FS.writeFile path rendered
  IO.println s!"emitted_pilot_parity={path}"

def run (arguments : List String) : IO UInt32 := do
  match arguments with
  | [path] =>
      emit ⟨path⟩
      pure 0
  | ["--", path] =>
      emit ⟨path⟩
      pure 0
  | ["--pi-ccs-parity", path] =>
      emitPiCcsParity ⟨path⟩
      pure 0
  | ["--", "--pi-ccs-parity", path] =>
      emitPiCcsParity ⟨path⟩
      pure 0
  | ["--pilot-parity", path] =>
      emitPilotParity ⟨path⟩
      pure 0
  | ["--", "--pilot-parity", path] =>
      emitPilotParity ⟨path⟩
      pure 0
  | _ =>
      IO.eprintln
        "usage: lake exe emit -- <output-path> | --pilot-parity <output-path> | --pi-ccs-parity <output-path>"
      pure 2

end NightstreamFPrime.Export.Main

def main (arguments : List String) : IO UInt32 :=
  NightstreamFPrime.Export.Main.run arguments
