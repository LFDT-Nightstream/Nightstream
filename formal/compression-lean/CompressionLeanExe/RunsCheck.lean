import CompressionLean.Checks.Runs
import CompressionLean.Spec.OptionBMatrix
import CompressionLeanExe.PackageJson
import CompressionLeanExe.Runs

open CompressionLeanExe.Runs in
/-- With `--option-b`, the runs of an option-B circuit key against its step package
(`Spec.OptionB.keyRunsFailure`); otherwise the runs of the Poseidon2 hash chain. -/
def main (args : List String) : IO UInt32 := do
  match args with
  | [runs] =>
      let pushes ← IO.ofExcept (readRuns (← IO.FS.readBinFile runs))
      match CompressionLean.Spec.runsFailure pushes with
      | none => IO.println "runs give the matrices"; return 0
      | some failure => IO.println s!"runs fail at {failure}"; return 1
  | ["--option-b", package, runs] =>
      let package ← IO.ofExcept (← CompressionLeanExe.PackageJson.readPackage package)
      let pushes ← IO.ofExcept (readRuns (← IO.FS.readBinFile runs))
      match CompressionLean.Spec.OptionB.keyRunsFailure package pushes with
      | none => IO.println "runs give the matrices"; return 0
      | some failure => IO.println s!"runs fail at {failure}"; return 1
  | _ => IO.eprintln "usage: runsCheck [--option-b <package.json>] <runs.bin>"; return 2
