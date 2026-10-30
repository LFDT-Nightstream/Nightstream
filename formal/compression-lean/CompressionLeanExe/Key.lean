import CompressionLean.Checks.KeyFacts

namespace CompressionLeanExe.Key

open CompressionLean

def printFacts (key : Circuit.Layer1Key) : IO Bool := do
  match Spec.keyFailure key with
  | some fact => IO.println s!"key fact fails: {fact}"; pure false
  | none => IO.println "key facts hold"; pure true

def printWidth (key : Circuit.Layer1Key) : IO Unit :=
  IO.println (if Spec.poseidon2Fits key then "key width fits the Poseidon2 hash chain"
    else "key width does not fit the Poseidon2 hash chain")

end CompressionLeanExe.Key
