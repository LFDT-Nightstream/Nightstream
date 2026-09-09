import NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit

open NightstreamFPrime.Lifecycle.VerifierContext
open NightstreamFPrime.Spec

private def usage : String :=
  "usage: emitPiCCSOwnershipAudit <id0> <id1> <id2> <id3> <output-path>"

private def parseStructuralWord (label word : String) : Except String F := do
  if word.isEmpty || word.length > 20 ||
      !(word.toList.all Char.isDigit) then
    throw s!"invalid structural-identity word {label}: expected decimal UInt64"
  let some value := word.toNat?
    | throw s!"invalid structural-identity word {label}: expected decimal UInt64"
  if _ : value < UInt64.size then
    if canonical : value < goldilocksModulus then
      pure ⟨value, canonical⟩
    else
      throw s!"invalid structural-identity word {label}: not a canonical Goldilocks element"
  else
    throw s!"invalid structural-identity word {label}: expected decimal UInt64"

private def parseStructuralIdentity (w0 w1 w2 w3 : String) :
    Except String Digest4 := do
  let c0 ← parseStructuralWord "id0" w0
  let c1 ← parseStructuralWord "id1" w1
  let c2 ← parseStructuralWord "id2" w2
  let c3 ← parseStructuralWord "id3" w3
  pure { c0, c1, c2, c3 }

private def run (w0 w1 w2 w3 path : String) : IO UInt32 := do
  match parseStructuralIdentity w0 w1 w2 w3 with
  | .error error =>
      IO.eprintln error
      pure 2
  | .ok structural =>
      let output : System.FilePath := ⟨path⟩
      if let some parent := output.parent then
        IO.FS.createDirAll parent
      let handle ← IO.FS.Handle.mk output .write
      handle.putStr
        (NightstreamFPrime.Export.Stage1.PiCCSOwnershipAudit.render structural)
      handle.putStr "\n"
      handle.flush
      IO.println s!"emitted_pi_ccs_ownership_audit={output}"
      pure 0

def main (arguments : List String) : IO UInt32 :=
  match arguments with
  | [w0, w1, w2, w3, path] => run w0 w1 w2 w3 path
  | ["--", w0, w1, w2, w3, path] => run w0 w1 w2 w3 path
  | _ => do
      IO.eprintln usage
      pure 2
