import CompressionLean.Checks.Runs
import CompressionLean.Spec.OptionBMatrix
import CompressionLeanExe.PackageJson
import CompressionLeanExe.Runs

open CompressionLeanExe.Runs in
/-- The runs of a circuit key against its step package (`Spec.OptionB.keyRunsFailure`). -/
def main (args : List String) : IO UInt32 := do
  let [package, runs] := args | IO.eprintln "usage: runsCheck <package.json> <runs.bin>"; return 2
  let package ← IO.ofExcept (← CompressionLeanExe.PackageJson.readPackage package)
  let pushes ← IO.ofExcept (readRuns (← IO.FS.readBinFile runs))
  match CompressionLean.Spec.OptionB.keyRunsFailure package pushes with
  | none => IO.println "runs give the matrices"; return 0
  | some failure => IO.println s!"runs fail at {failure}"; return 1
