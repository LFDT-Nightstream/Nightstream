import CompressionLean.Final.Encodes
import CompressionLean.Final.ParentClaim
import CompressionLean.Circuit.Valid

/-!
Owns the production bound for the production final program: the bound of
`final_restore_budget_parent` with the program `Encodes.prog`, its Poseidon2
template and its reading `Encodes.Reads`, where `programEncodes` gives the
encoding premise.

The premises that remain are facts about the layer-1 key, the terminal
package and the setup. None is about the witness.
-/

namespace CompressionLean.Final

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open CompressionLean.Bridge (shapeOf barOf queryAt RelationFacts)
open CompressionLean.Circuit (Layer1Input)
open CompressionLean.Circuit.Package (Loaded load)
open CompressionLean.Final.Encodes
open CompressionLean.Circuit.Goldilocks Fin.CommRing

attribute [local instance] Circuit.challengeCubic

/-- Each shrink run of the program names only its own cells. -/
private theorem prog_valid (key : Layer1Input) (loaded : Loaded) (context : Array Replay.Gl) (x : StateWords) :
    (prog key loaded context x).Valid Circuit.Block.poseidonTemplate :=
  Circuit.lower_valid Circuit.Block.poseidonTemplate (Nat.le_of_eq rfl) _ _

/-- **The production bound for the final program** (`CompressionKey::verify`,
`lifecycle/finish.rs:131-145`). Under Fiat–Shamir, with the charged terms of
each commitment within Rust's split, a prover with `calls` queries makes the
final verifier accept a statement whose F′ claim is false, while every
program witness among the shrink candidates has a reading that layer 1 ran,
with probability at most `calls / 2^t`. -/
theorem final_restore_budget_program (key : Layer1Input) (loaded : Loaded) (context : Array Replay.Gl)
    (authority : Lifecycle.VerifierContext.Authority) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation Data.logicalWidth Data.publicFits)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (shape : ParentShape (shapeOf key.relation) Data.logicalWidth)
    (barEq : ∀ p l : Fin ringDegree, barOf p l = gl (nativeBarEntry p l))
    (honest : HonestSetup (shapeOf key.relation) S relation ajtai) (facts : RelationFacts key.relation S)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf key.relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ ((shapeOf key.relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate
        (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts)
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate
        (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts) / 2 ^ 191 ≤ 1 / 2 ^ t)
    (N : ℕ) (sizes : (shapeOf key.relation).Sizes N) (small : N ≤ goldilocksModulus)
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      Circuit.word (context.getD lane 0) = (Lifecycle.VerifierContext.digest authority).getD lane 0)
    (hrun : key.relation.key.layout.runVariables ≠ 0) (hrow : key.relation.key.layout.rowVariables ≠ 0)
    (hslot : key.relation.key.layout.slotVariables + key.relation.matrixVars ≠ 0)
    (kappa : key.relation.kappa = 22) (blocks : key.relation.publicBlocks = 5)
    (point : key.relation.pointVariables = 28) (matrices : key.relation.matrices = 4)
    (root : key.relation.key.root.size = 4) (short : ∀ cl ∈ key.relation.key.layout.classes, cl.len ≤ Replay.D)
    (k1 : key.p1Key.batches.size = key.relation.p1Whir.batches.size)
    (k0 : key.p0Key.batches.size = key.relation.p0Whir.batches.size)
    (hp1 : ∀ b ∈ key.relation.p1Whir.batches, b.size % 3 = 0)
    (hp1count : (key.relation.p1Whir.batches.toList.map fun b => b.size / 3).sum = 2 * key.relation.queries + 4)
    (hp0 : key.relation.p0Whir.batches.toList.map (·.size) =
      [1, 1, 2, 2 * key.relation.matrices, 1, 2 * key.relation.matrices + 1, 1])
    (calls : ℕ) (prover : Layer1.Game.Strategy (RestoreOutcome Gl Cubic (game key.relation S L0 L1))) :
    Layer1.Game.prob prover
        (restorePair (Shrink.shrinkGame (F := Gl) Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context) Ls)
          (game key.relation S L0 L1) calls [] [])
        (FinalRestoreBreak (F := Gl) (ParentClaim relation ajtai authority) Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context)
          (game key.relation S L0 L1) (Reads key loaded context S L0 L1)) ≤
      calls / 2 ^ t :=
  final_restore_budget_parent (shapeOf key.relation) barOf S relation ajtai shape barEq honest facts.fits L0 L1 h0 h1
    (show key.relation.setupVars ≤ 35 by have := facts.high; omega) t Circuit.Block.poseidonTemplate _
    (prog key loaded context) Ls p0 p1 rust (queryAt key.relation) N sizes small
    (Bridge.queryAt_distinct key.relation facts.high) Circuit.Block.poseidonTemplate_wellFormed
    (prog_valid key loaded context) authority (Reads key loaded context S L0 L1)
    (programEncodes S facts L0 L1 loads valid hctx hrun hrow hslot kappa blocks point matrices root short k1 k0 hp1
      hp1count hp0) calls prover

end CompressionLean.Final
