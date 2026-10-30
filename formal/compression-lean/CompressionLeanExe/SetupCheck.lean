import CompressionLean.Checks.SetupFacts
import CompressionLeanExe.Parse
import CompressionLeanExe.Runs

open CompressionLeanExe.Runs in
def main (args : List String) : IO UInt32 := do
  let [layer1, runs, columns] := args
    | IO.eprintln "usage: setupCheck <layer1.json> <runs.bin> <setup_columns.bin>"; return 2
  let key ← IO.ofExcept (← CompressionLeanExe.Parse.readJson CompressionLean.Circuit.Layer1Key layer1)
  let pushes ← IO.ofExcept (readRuns (← IO.FS.readBinFile runs))
  if let some fact := CompressionLean.Spec.setupFailure key pushes then
    IO.println s!"setup fact fails: {fact}"; return 1
  IO.println "setup facts hold"
  let cols ← IO.ofExcept (readColumns (← IO.FS.readBinFile columns))
  match CompressionLean.Spec.columnsFailure pushes cols with
  | none => IO.println "setup columns match"; return 0
  | some fact => IO.println s!"setup column fails: {fact}"; return 1
