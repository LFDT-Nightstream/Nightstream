import CompressionLean.Checks.SetupFacts
import CompressionLeanExe.Parse
import CompressionLeanExe.Runs

/-!
The `setupCheck` executable. It reads the layer-1 key that `neo-spartan`'s
`lowering.rs` writes (`production_layer1.json`) and the runs that
`nightstream`'s `setup_runs.rs` writes (`production_runs.bin`), and runs the
setup check (`setupFailure`) on them. It prints `setup facts hold`, or the
first fact that fails and exits with 1.
-/

open CompressionLeanExe.Runs in
def main (args : List String) : IO UInt32 := do
  let [layer1, runs] := args | IO.eprintln "usage: setupCheck <layer1.json> <runs.bin>"; return 2
  let key ← IO.ofExcept (← CompressionLeanExe.Parse.readJson CompressionLean.Circuit.Layer1Key layer1)
  let pushes ← IO.ofExcept (readRuns (← IO.FS.readBinFile runs))
  match CompressionLean.Spec.setupFailure key pushes with
  | none => IO.println "setup facts hold"; return 0
  | some fact => IO.println s!"setup fact fails: {fact}"; return 1
