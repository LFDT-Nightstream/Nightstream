import CompressionLean.Checks.KeyFacts

namespace CompressionLeanExe.Key

open CompressionLean

def printFacts (key : Circuit.Layer1Key) : IO Bool := do
  match Spec.keyFailure key with
  | some fact => IO.println s!"key fact fails: {fact}"; pure false
  | none => IO.println "key facts hold"; pure true

def printWithin (key : Circuit.Layer1Key) : IO Bool := do
  if Spec.boundedDepthFits key then IO.println "key commits within the bounded-depth scheme's key width"; pure true
  else IO.println "key commits past the bounded-depth scheme's key width"; pure false

end CompressionLeanExe.Key
