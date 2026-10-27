import CompressionLean.Checks.KeyFacts

/-!
Owns the report on a layer-1 key file (`layer1.json` of `neo-spartan`'s
`tests/internal/lowering.rs`) that `lowerCheck` and `terminalCheck` print.
-/

namespace CompressionLeanExe.Key

open CompressionLean

/-- Prints whether `key` has `KeyFacts` (`keyFailure`), and returns it. -/
def printFacts (key : Circuit.Layer1Key) : IO Bool := do
  match Spec.keyFailure key with
  | some fact => IO.println s!"key fact fails: {fact}"; pure false
  | none => IO.println "key facts hold"; pure true

/-- Prints whether `key` fits the Poseidon2 hash chain (`poseidon2Fits`). -/
def printWidth (key : Circuit.Layer1Key) : IO Unit :=
  IO.println (if Spec.poseidon2Fits key then "key width fits the Poseidon2 hash chain"
    else "key width does not fit the Poseidon2 hash chain")

end CompressionLeanExe.Key
