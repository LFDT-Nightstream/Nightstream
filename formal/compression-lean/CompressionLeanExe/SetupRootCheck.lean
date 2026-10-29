import CompressionLean.Bridge.TableRoot

open Lean CompressionLean.Replay CompressionLean.Bridge

namespace CompressionLeanExe.SetupRootCheck

structure RootInput where
  setupVars : Nat
  tables : Array (Array Word)
  root : Array Word
deriving FromJson

def digestLine (name : String) (digest : Array Word) : String :=
  " ".intercalate (name :: digest.toList.map fun word => toString word.toNat)

end CompressionLeanExe.SetupRootCheck

open CompressionLeanExe.SetupRootCheck in
def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: setupRootCheck <setup-root.json>"; return 2
  let json ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  let input : RootInput ← IO.ofExcept (fromJson? json)
  let root := tableRoot input.setupVars input.tables
  IO.println (digestLine "lean" root)
  IO.println (digestLine "rust" input.root)
  return if root == input.root then 0 else 1
