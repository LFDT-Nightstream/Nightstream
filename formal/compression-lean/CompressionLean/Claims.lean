import CompressionLean.Sim.Encodes
import CompressionLean.Final.ParentClaim
import CompressionLean.Checks.KeyFacts
import CompressionLean.Final.RustSetup
import CompressionLean.Checks.SetupFacts
import CompressionLean.Checks.Templates
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-!
Owns the claims of the library: the production bound, and the theorems that
link its premises to a proof or to a check. Each claim is stated in full here;
its proof is in the library.

- `production_bound`: the soundness bound of `CompressionKey::verify` for the
  production final program, for the relation of an application of any logical
  width `lw`. It takes `ParentShape` at `lw`.
- `poseidon2_bound`: `production_bound` for F′'s relation of the Poseidon2
  hash-chain application, the application of the production key. It takes
  `WidthFits` in place of `ParentShape`, and the runs, the Ajtai setup,
  `RunsGive` and the pass of `setupCheck` in place of the setup and its facts.
- `honest_setup_of_build`: the honest setup from Rust's setup build and
  `RunsGive`.
- `key_facts_of_check`, `width_fits_of_check`, `package_templates_of_check`,
  `setup_facts_of_check`: a key, a terminal package or runs that pass the
  check of `lowerCheck`, `terminalCheck` or `setupCheck` have the premises
  that the check decides.

`CompressionLeanTests/Axioms.lean` pins the axioms of exactly these theorems.
`STATUS.md` lists every hypothesis of each claim and how each premise is met;
`TRUST.md` lists what the claims trust.
-/

namespace CompressionLean.Claims

open CompressionLean.Sim
open CompressionLean.Checks
open CompressionLean.Game
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Package (CircuitPackage)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open CompressionLean.Bridge (shapeOf structureOf barOf queryAt)
open CompressionLean.Circuit (Layer1Input)
open CompressionLean.Terminal.Package (Loaded load)
open CompressionLean.Final CompressionLean.Setup

/-- **The production bound** (`CompressionKey::verify`,
`lifecycle/finish.rs:131-145`). Take the production final program for the
layer-1 key `key` (`Sim.prog`), the terminal package with the context
words of `authority`, and an honest setup `S` of an F′ relation of logical
width `lw`. Under Fiat–Shamir, with the charged terms of each commitment
within Rust's split, a prover with `calls` queries makes the final verifier
accept a statement whose F′ claim (`ParentClaim`) is false, while every
program witness among the shrink candidates has a reading that layer 1 ran
(`Sim.Reads`), with probability at most `calls / 2^t`. -/
theorem production_bound {lw : ℕ}
    {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (key : Layer1Input) (loaded : Loaded) (context : Array Replay.Word)
    (authority : Lifecycle.VerifierContext.Authority) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (shape : ParentShape (shapeOf key.relation) lw)
    (honest : HonestSetup (shapeOf key.relation) S relation ajtai) (facts : KeyFacts key)
    (fits : S.Fits (shapeOf key.relation))
    (describes : (structureOf key.relation.key.layout).Describes (shapeOf key.relation) S)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf key.relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ ((shapeOf key.relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate
        (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts)
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate
        (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts) / 2 ^ 191 ≤ 1 / 2 ^ t)
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      Replay.Word.cast (context.getD lane 0) = (Lifecycle.VerifierContext.digest authority).getD lane 0)
    (calls : ℕ) (prover : Game.Strategy (RestoreOutcome Gl Cubic (game key.relation S L0 L1))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame (F := Gl) Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context) Ls)
          (game key.relation S L0 L1) calls [] [])
        (FinalRestoreBreak (F := Gl) (ParentClaim relation ajtai authority) Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context)
          (game key.relation S L0 L1) (Reads key loaded context S L0 L1)) ≤
      calls / 2 ^ t :=
  final_restore_budget_parent (shapeOf key.relation) barOf S relation ajtai shape barOf_eq honest fits L0 L1 h0 h1
    (show key.relation.setupVars ≤ 35 by have := facts.high; omega) t Circuit.Block.poseidonTemplate _
    (prog key loaded context) Ls p0 p1 rust (queryAt key.relation) goldilocksModulus facts.shape le_rfl
    (Bridge.queryAt_distinct key.relation facts.high) Lowering.Block.poseidonTemplate_wellFormed
    (prog_valid key loaded context) authority (Reads key loaded context S L0 L1)
    (programEncodes S facts fits describes L0 L1 loads valid hctx) calls prover

/-- **The honest setup from the build** (`neo_spartan::Setup::build`). A setup
that `build` returns for the seed of an Ajtai setup, from runs that give F′'s
matrices (`RunsGive`), is an honest setup for that setup's key. -/
theorem honest_setup_of_build {lw : ℕ}
    {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (s : Layer1.Shape) (D : s.D = ringDegree) (lanes : s.laneVars = 6)
    (setup : AjtaiSetupV1.Setup productionProfile.commitmentWidth
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth lw)))
    (relation : ProductionKey.LogicalRelation lw pf) (pushes : List Push) {S : Layer1.Setup Gl}
    (built : build setup.seed.bytes (Phi81CarrierLayout.carrierWidth lw) productionProfile.ccsMatrices pushes = some S)
    (gives : RunsGive pushes relation) :
    HonestSetup s S relation setup.verifierKey :=
  honestSetup_of_build s D lanes setup relation pushes built gives

/-- **The key check** of `lowerCheck` and `terminalCheck`: a key that fails no
check has `KeyFacts`. -/
theorem key_facts_of_check {key : Layer1Input} (pass : keyFailure key = none) : KeyFacts key :=
  keyFacts_of_check pass

/-- **The width check** of `lowerCheck` and `terminalCheck`: a key for which
they print that it fits the Poseidon2 hash chain (`poseidon2Fits`) has the
blocks of that application. -/
theorem width_fits_of_check {key : Layer1Input} (pass : poseidon2Fits key = true) :
    WidthFits key (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application) := by
  rw [Poseidon2HashChainV1Package.logicalWidth]
  exact of_decide_eq_true pass

/-- **The package check** of `terminalCheck`: a package whose compact row
invocations all name a template has `CompactTemplatesValid`. -/
theorem package_templates_of_check {package : CircuitPackage}
    (named : Checks.templatesNamed package = true) : package.CompactTemplatesValid :=
  Checks.compactTemplatesValid_of_named named

/-- **The setup check** of `setupCheck`: for a key and runs that fail no fact
of `setupFailure`, `build` returns `poseidon2Setup` at the carrier of the
Poseidon2 hash chain and F′'s matrices, the setup fits the key's shape
(`Setup.Fits`), and the key's structure describes it (`Structure.Describes`). -/
theorem setup_facts_of_check {key : Layer1Input} {pushes : List Push} (seed : List ℕ)
    (pass : setupFailure key pushes = none) :
    build seed (Phi81CarrierLayout.carrierWidth
        (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application))
        productionProfile.ccsMatrices pushes = some (poseidon2Setup seed pushes) ∧
      (poseidon2Setup seed pushes).Fits (shapeOf key.relation) ∧
      (structureOf key.relation.key.layout).Describes (shapeOf key.relation) (poseidon2Setup seed pushes) := by
  rw [Poseidon2HashChainV1Package.logicalWidth]
  exact setupFacts_of_check seed pass

/-- **The production bound for the Poseidon2 hash chain**: `production_bound`
for F′'s relation of `Poseidon2HashChainV1Package.application`, at its
logical width of 43,963,750 columns. The production key is of this
application. The setup is the one that `build` returns for the seed of the
Ajtai setup and the runs (`poseidon2Setup`). The claim takes `WidthFits` in
place of `ParentShape`, and `RunsGive` and the pass of `setupCheck` in place
of the facts of the setup. -/
theorem poseidon2_bound (key : Layer1Input) (loaded : Loaded) (context : Array Replay.Word)
    (authority : Lifecycle.VerifierContext.Authority) (pushes : List Push)
    (setup : AjtaiSetupV1.Setup productionProfile.commitmentWidth
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth
        (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application))))
    (gives : RunsGive pushes
      (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits))
    (facts : KeyFacts key) (width : WidthFits key
      (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application))
    (pass : setupFailure key pushes = none)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf key.relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : (L1 : ℝ) * (1 / 2) ^ ((shapeOf key.relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate
        (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts)
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate
        (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts) / 2 ^ 191 ≤ 1 / 2 ^ t)
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      Replay.Word.cast (context.getD lane 0) = (Lifecycle.VerifierContext.digest authority).getD lane 0)
    (calls : ℕ) (prover : Game.Strategy
      (RestoreOutcome Gl Cubic (game key.relation (poseidon2Setup setup.seed.bytes pushes) L0 L1))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame (F := Gl) Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context) Ls)
          (game key.relation (poseidon2Setup setup.seed.bytes pushes) L0 L1) calls [] [])
        (FinalRestoreBreak (F := Gl)
          (ParentClaim (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
            Poseidon2HashChainV1Package.fits) setup.verifierKey authority)
          Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context)
          (game key.relation (poseidon2Setup setup.seed.bytes pushes) L0 L1)
          (Reads key loaded context (poseidon2Setup setup.seed.bytes pushes) L0 L1)) ≤
      calls / 2 ^ t := by
  obtain ⟨built, fits, describes⟩ := setup_facts_of_check setup.seed.bytes pass
  exact production_bound key loaded context authority _
    (PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits)
    setup.verifierKey (parentShape_of _ _ rfl rfl facts.kappa facts.matrices facts.blocks width facts.norm)
    (honest_setup_of_build _ rfl rfl setup _ pushes built gives) facts fits describes L0 L1 h0 h1 t Ls p0 p1 rust
    loads valid hctx calls prover

end CompressionLean.Claims
