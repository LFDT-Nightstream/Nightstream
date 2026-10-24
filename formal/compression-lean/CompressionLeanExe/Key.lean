import CompressionLean.Final.KeyFacts

/-!
Owns the report on a layer-1 key file (`layer1.json` of `neo-spartan`'s
`tests/internal/lowering.rs`) that `lowerCheck` and `terminalCheck` print.
-/

namespace CompressionLeanExe.Key

open CompressionLean

/-- Prints whether `key` has `KeyFacts` (`keyFailure`), and returns it. -/
def printFacts (key : Circuit.Layer1Input) : IO Bool := do
  match Final.keyFailure key with
  | some fact => IO.println s!"key fact fails: {fact}"; pure false
  | none => IO.println "key facts hold"; pure true

/-- Prints whether `key` fits the Poseidon2 hash chain (`poseidon2Fits`). -/
def printWidth (key : Circuit.Layer1Input) : IO Unit :=
  IO.println (if Final.poseidon2Fits key then "key width fits the Poseidon2 hash chain"
    else "key width does not fit the Poseidon2 hash chain")

end CompressionLeanExe.Key
