import CompressionLean.Circuit.Terminal
import CompressionLean.Circuit.Verifier

/-!
Owns the whole final program with Lean's layer-1 verifier (`finalViews`):
`finalProgram` with `verifyWith` for a key. Its result is the handles that
layer 1 reads: the claim, the seed and the proof. `TerminalCheck.lean`
compares its instructions with Rust's, and `Final/Shipped.lean` reads a
witness through its handles.
-/

namespace CompressionLean.Circuit

open CompressionLean.Circuit.Package CompressionLean.Circuit.Terminal
open CompressionLean.Replay (Gl)

/-- The terminal claim as the verifier's statement. -/
def claimView (claim : ClaimVars) : ClaimView :=
  { commitment := claim.commitment, publicInput := claim.publicInput, point := claim.point,
    evalK := claim.evalK, evalA := claim.evalA }

/-- The handles of the final program that layer 1 reads. -/
structure FinalViews where
  claim : ClaimView
  seed : DigestVar
  proof : ProofView

/-- `FinalProgram::run` with `verifyWith` for `key`. -/
def finalViews (package : Loaded) (context : Array Gl) (key : Layer1Input) : Build FinalViews :=
  finalProgram package context fun claim state => do
    let (seed, proof) ← verifyWith key state (claimView claim)
    pure { claim := claimView claim, seed, proof }

end CompressionLean.Circuit
