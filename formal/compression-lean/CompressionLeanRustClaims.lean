import CompressionLean.Claims
import CompressionLeanRust.Verifier

/-! The production claim for the Rust shrink verifier. `Claims.poseidon2_bound` bounds the break
event of `shrinkGame`; with `Verifier.Sound`, the same bound holds for `Verifier.RustBreak`, where a
`verify` of the type that Aeneas will give `neo_verified::shrink::verify` accepts a false claim.
This module imports the claims, so the aeneas CI job, which builds the translation and its proofs,
does not build it. -/

namespace CompressionLeanRustClaims

open CompressionLean.Spec
open CompressionLean.Checks
open CompressionLean.Game
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open CompressionLean.Circuit (Layer1Key)
open CompressionLean.Terminal.Package (Loaded load)
open CompressionLean.Final CompressionLean.Setup
open CompressionLeanRust.Verifier (VerifyFn Sound RustBreak rust_bound)

/-- `Claims.poseidon2_bound` for a `Sound` Rust verifier, with the same premises. -/
theorem poseidon2_bound_rust (verify : VerifyFn) (sound : Sound verify CompressionLean.Circuit.Block.poseidonTemplate)
    (key : Layer1Key) (loaded : Loaded) (context : Array CompressionLean.Replay.Word)
    (authority : Lifecycle.VerifierContext.Authority) (pushes : List Push)
    (setup : AjtaiSetupV1.Setup productionProfile.commitmentWidth
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth CompressionLean.Production.poseidon2Width)))
    (facts : KeyFacts key) (width : WidthFits key CompressionLean.Production.poseidon2Width)
    (pass : setupFailure key pushes = none) (runs : runsFailure pushes = none)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : (L0 : ℝ) * L1 * CompressionLean.Layer1.rustDegrees (shapeOf key.relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ ((shapeOf key.relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * CompressionLean.Shrink.rustDegrees
        (CompressionLean.Shrink.Layout.rowVars CompressionLean.Circuit.Block.poseidonTemplate (counts key loaded context))
        (CompressionLean.Shrink.Layout.cellVars CompressionLean.Circuit.Block.poseidonTemplate (counts key loaded context)) /
        2 ^ 191 ≤ 1 / 2 ^ t)
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      CompressionLean.Replay.Word.cast (context.getD lane 0) =
        (Lifecycle.VerifierContext.digest authority).getD lane 0)
    (calls : ℕ) (prover : CompressionLean.Game.Strategy
      (RestoreOutcome CompressionLean.Gl CompressionLean.Cubic (game key (poseidon2Setup setup.seed.bytes pushes) L0 L1))) :
    CompressionLean.Game.prob prover
        (restorePair (CompressionLean.Shrink.shrinkGame (F := CompressionLean.Gl)
          CompressionLean.Circuit.Block.poseidonTemplate (counts key loaded context) (shrinkRun key loaded context) Ls)
          (game key (poseidon2Setup setup.seed.bytes pushes) L0 L1) calls [] [])
        (RustBreak verify (ParentClaim CompressionLean.Production.poseidon2Relation setup.verifierKey authority)
          CompressionLean.Circuit.Block.poseidonTemplate (counts key loaded context) (shrinkRun key loaded context) Ls
          (game key (poseidon2Setup setup.seed.bytes pushes) L0 L1)
          (Reads key loaded context (poseidon2Setup setup.seed.bytes pushes) L0 L1)) ≤
      calls / 2 ^ t :=
  rust_bound sound _ _ _ _ _ _ calls prover _
    (CompressionLean.Claims.poseidon2_bound key loaded context authority pushes setup facts width pass runs L0 L1 h0 h1
      t Ls p0 p1 rust loads valid hctx calls prover)

/-- info: 'CompressionLeanRustClaims.poseidon2_bound_rust' depends on axioms: [propext, Classical.choice, Quot.sound] -/
#guard_msgs in
#print axioms poseidon2_bound_rust

end CompressionLeanRustClaims
