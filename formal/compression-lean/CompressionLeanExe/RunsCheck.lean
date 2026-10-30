import CompressionLean.Checks.Runs
import CompressionLeanExe.Runs

open CompressionLeanExe.Runs in
def main (args : List String) : IO UInt32 := do
  let [runs] := args | IO.eprintln "usage: runsCheck <runs.bin>"; return 2
  let pushes ← IO.ofExcept (readRuns (← IO.FS.readBinFile runs))
  match CompressionLean.Spec.runsFailure pushes with
  | none => IO.println "runs give the matrices"; return 0
  | some failure => IO.println s!"runs fail at {failure}"; return 1
