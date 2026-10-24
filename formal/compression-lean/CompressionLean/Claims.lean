import CompressionLean.Final.Encodes
import CompressionLean.Final.ParentClaim
import CompressionLean.Final.KeyFacts
import CompressionLean.Final.RustSetup

/-!
Owns the claims of the library: the production bound, and the theorems that
link its premises to a proof or to a check. Each claim is stated in full here;
its proof is in the library.

- `production_bound`: the soundness bound of `CompressionKey::verify` for the
  production final program, for the relation of an application of any logical
  width `lw`. It takes `ParentShape` at `lw`.
- `honest_setup_of_build`: the honest setup from Rust's setup build and
  `RunsGive`.
- `key_facts_of_check`, `package_templates_of_check`: a key or a terminal
  package that passes the check of `lowerCheck` and `terminalCheck` has the
  premise that the check decides.

`CompressionLeanTests/Axioms.lean` pins the axioms of exactly these theorems.
`STATUS.md` lists the named premises of each claim and how each one is met;
`TRUST.md` lists what the claims trust.
-/

namespace CompressionLean.Claims

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Export.Package (CircuitPackage)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open CompressionLean.Bridge (shapeOf structureOf barOf queryAt)
open CompressionLean.Circuit (Layer1Input)
open CompressionLean.Circuit.Package (Loaded load)
open CompressionLean.Final
open CompressionLean.Final.Encodes
open CompressionLean.Circuit.Goldilocks Fin.CommRing

attribute [local instance] Circuit.challengeCubic

/-- **The production bound** (`CompressionKey::verify`,
`lifecycle/finish.rs:131-145`). Take the production final program for the
layer-1 key `key` (`Encodes.prog`), the terminal package with the context
words of `authority`, and an honest setup `S` of an F′ relation of logical
width `lw`. Under Fiat–Shamir, with the charged terms of each commitment
within Rust's split, a prover with `calls` queries makes the final verifier
accept a statement whose F′ claim (`ParentClaim`) is false, while every
program witness among the shrink candidates has a reading that layer 1 ran
(`Encodes.Reads`), with probability at most `calls / 2^t`. -/
theorem production_bound {lw : ℕ}
    {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (key : Layer1Input) (loaded : Loaded) (context : Array Replay.Gl)
    (authority : Lifecycle.VerifierContext.Authority) (S : Layer1.Setup Gl)
    (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (shape : ParentShape (shapeOf key.relation) lw)
    (barEq : ∀ p l : Fin ringDegree, barOf p l = gl (nativeBarEntry p l))
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
    (N : ℕ) (sizes : (shapeOf key.relation).Sizes N) (small : N ≤ goldilocksModulus)
    (loads : load (CompressionLean.Terminal.circuitPackage ()) CompressionLean.Terminal.outputs = .ok loaded)
    (valid : (CompressionLean.Terminal.circuitPackage ()).CompactTemplatesValid)
    (hctx : ∀ lane : Fin 4,
      Circuit.word (context.getD lane 0) = (Lifecycle.VerifierContext.digest authority).getD lane 0)
    (calls : ℕ) (prover : Layer1.Game.Strategy (RestoreOutcome Gl Cubic (game key.relation S L0 L1))) :
    Layer1.Game.prob prover
        (restorePair (Shrink.shrinkGame (F := Gl) Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context) Ls)
          (game key.relation S L0 L1) calls [] [])
        (FinalRestoreBreak (F := Gl) (ParentClaim relation ajtai authority) Circuit.Block.poseidonTemplate
          (Circuit.lower (program key loaded context).program ([] : List Spec.F)).counts (prog key loaded context)
          (game key.relation S L0 L1) (Reads key loaded context S L0 L1)) ≤
      calls / 2 ^ t :=
  final_restore_budget_parent (shapeOf key.relation) barOf S relation ajtai shape barEq honest fits L0 L1 h0 h1
    (show key.relation.setupVars ≤ 35 by have := facts.high; omega) t Circuit.Block.poseidonTemplate _
    (prog key loaded context) Ls p0 p1 rust (queryAt key.relation) N sizes small
    (Bridge.queryAt_distinct key.relation facts.high) Circuit.Block.poseidonTemplate_wellFormed
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

/-- **The package check** of `terminalCheck`: a package whose compact row
invocations all name a template has `CompactTemplatesValid`. -/
theorem package_templates_of_check {package : CircuitPackage}
    (named : Circuit.Package.templatesNamed package = true) : package.CompactTemplatesValid :=
  Circuit.Package.compactTemplatesValid_of_named named

end CompressionLean.Claims
