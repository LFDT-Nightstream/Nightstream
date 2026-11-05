import CompressionLean.Sim.OptionBEncodes
import CompressionLean.Final.OptionBParent
import CompressionLean.Checks.OptionBMatrix
import CompressionLean.Checks.OptionBCover
import CompressionLean.Checks.KeyFacts
import CompressionLean.Final.RustSetup
import CompressionLean.Checks.SetupFacts
import CompressionLean.Checks.Runs
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import CompressionLean.Final.OptionBIvc

namespace CompressionLean.Claims

open CompressionLean.Spec
open CompressionLean.Sim
open CompressionLean.Checks
open CompressionLean.Game
open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Lifecycle NightstreamFPrime.Export.Stage1
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CoefficientKernel
open CompressionLean.Circuit (Layer1Key)
open CompressionLean.Final CompressionLean.Setup
open NightstreamFPrime.Spec.HyperNova.ReverseStages (NifsAdversary NifsExtractor Stage Assumption1 Closed
  reverseStages visitAfter)
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.RingFPolynomial (modulus modulus_monic)

theorem key_facts_of_check {key : Layer1Key} (pass : keyFailure key = none) : KeyFacts key :=
  Checks.keyFailure_sound pass

theorem option_b_shape_of_check {key : Layer1Key} (facts : KeyFacts key) (pass : optionBFits key = true) :
    ParentShape (shapeOf key.relation) Export.OptionB.StepRelation.keyWidth :=
  parentShape_of _ _ rfl rfl facts.constants.kappa facts.constants.matrices facts.constants.blocks
    (of_decide_eq_true pass) facts.constants.norm

theorem option_b_setup_facts_of_check {key : Layer1Key} {pushes : List Push} (seed : List ℕ)
    (pass : setupFailureAt (keyColumns key) key pushes = none) :
    build seed (keyColumns key) productionProfile.ccsMatrices pushes = some (setupAt seed (keyColumns key) pushes) ∧
      (setupAt seed (keyColumns key) pushes).Fits (shapeOf key.relation) ∧
      (structureOf key.relation.key.layout).Describes (shapeOf key.relation) (setupAt seed (keyColumns key) pushes) :=
  Checks.setupFailureAt_sound seed pass

theorem option_b_runs_of_check {p : Export.OptionB.StepPackage.Package} {pushes : List Push}
    (pass : OptionB.keyRunsFailure p pushes = none) {pieces : List Export.Generic.Pieces.Piece} {next : ℕ}
    (fits : Export.Generic.CircuitRelation.Fits Export.Generic.CircuitRelation.publicColumns pieces next p.width)
    (packed : Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns pieces = true)
    (rows : p.relationRows =
      Export.Generic.CircuitRelation.relationRows Export.Generic.CircuitRelation.publicColumns pieces next p.width)
    (blocks : p.blocks = Export.Generic.CircuitRelation.blocksOf pieces)
    (start : p.sboxStart = Export.Generic.CircuitRelation.sboxStart pieces next)
    (logical : Layout.ProductionAssignment.logicalWidth (OptionB.slotsOf p) ≤ Export.OptionB.StepRelation.keyWidth) :
    RunsGive pushes (OptionB.keyRelation pieces next p.width fits (OptionB.keyLayout p logical)) :=
  Checks.OptionB.keyRunsFailure_sound pass fits packed rows blocks start logical

theorem option_b_logical_of_check {p : Export.OptionB.StepPackage.Package} {pushes : List Push}
    (pass : OptionB.keyRunsFailure p pushes = none) :
    Layout.ProductionAssignment.logicalWidth (OptionB.slotsOf p) ≤ Export.OptionB.StepRelation.keyWidth :=
  Checks.OptionB.logical_of_check pass

theorem option_b_fits_of_check {p : Export.OptionB.StepPackage.Package} {pushes : List Push}
    (pass : OptionB.keyRunsFailure p pushes = none) {pieces : List Export.Generic.Pieces.Piece} {next : ℕ}
    (rows : p.relationRows =
      Export.Generic.CircuitRelation.relationRows Export.Generic.CircuitRelation.publicColumns pieces next p.width)
    (blocks : p.blocks = Export.Generic.CircuitRelation.blocksOf pieces)
    (start : p.sboxStart = Export.Generic.CircuitRelation.sboxStart pieces next) :
    Export.Generic.CircuitRelation.Fits Export.Generic.CircuitRelation.publicColumns pieces next p.width :=
  Checks.OptionB.fits_of_check pass rows blocks start

/-- A pass of `coverCheck` on a step package and the layer-1 key of its circuit (CW-10.6): the key's
norm bound is at least the accumulator bound of the light cap, and its blocks hold the package's
logical width. -/
theorem option_b_covers_of_check {p : Export.OptionB.StepPackage.Package} {key : Layer1Key}
    (pass : OptionB.coverFailure p key = none) :
    productionGlobalParams.accumulatorBound Folding.Schedule.productionLightFoldCap ≤
        (shapeOf key.relation).normBound ∧
      Layout.ProductionAssignment.logicalWidth (OptionB.slotsOf p) ≤ ringDegree * (shapeOf key.relation).blocks :=
  ⟨(Checks.OptionB.coverFailure_sound pass).normCovers, (Checks.OptionB.coverFailure_sound pass).blocksCover⟩

theorem option_b_honest_of_check (key : Layer1Key)
    (setup : AjtaiSetupV1.Setup productionProfile.commitmentWidth
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth Export.OptionB.StepRelation.keyWidth)))
    (relation : ProductionKey.LogicalRelation Export.OptionB.StepRelation.keyWidth
      Export.OptionB.StepRelation.keyPublicFits)
    (pushes : List Push) (pass : setupFailureAt (keyColumns key) key pushes = none) (gives : RunsGive pushes relation) :
    HonestSetup (shapeOf key.relation) (setupAt setup.seed.bytes (keyColumns key) pushes) relation setup.verifierKey :=
  Final.build_honest _ rfl rfl setup relation pushes (option_b_setup_facts_of_check _ pass).1
    (fun _ inside _ => by rw [Checks.keyColumns_div]; exact inside) gives

/-- Soundness of `lifecycle/option_b/compression.rs::OptionB::verify_final` at the layer-1 level: the
shrink layer accepts a statement while no finish columns satisfy the final program's checks with
both next slots' claim words in their layer-1 relations, with probability at most `calls / 2^t`.
Fiat–Shamir inside the program is assumed (`TRUST.md` §3). -/
theorem option_b_bound (finish : NightstreamFPrime.Export.OptionB.StepPackage.Package)
    (stateKeys : List Replay.Word) (keys : Fin 2 → Layer1Key) (S : Fin 2 → Layer1.Setup Gl)
    (facts : ∀ slot, KeyFacts (keys slot)) (fits : ∀ slot, (S slot).Fits (shapeOf (keys slot).relation))
    (describes : ∀ slot,
      (structureOf (keys slot).relation.key.layout).Describes (shapeOf (keys slot).relation) (S slot))
    (fit : OptionB.fits finish = true) (keysLength : stateKeys.length ≤ 8)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : ∀ slot, (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf (keys slot).relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : ∀ slot, (L1 : ℝ) * (1 / 2) ^ ((shapeOf (keys slot).relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees
        (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate (OptionB.counts finish stateKeys keys))
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate (OptionB.counts finish stateKeys keys)) / 2 ^ 191 ≤
        1 / 2 ^ t)
    (calls : ℕ)
    (prover : Game.Strategy (OptionB.Outcome Gl Cubic (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame (F := Gl) Circuit.Block.poseidonTemplate (OptionB.counts finish stateKeys keys)
            (OptionB.shrinkRun finish stateKeys keys) Ls)
          (choice (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)) calls [] [])
        (OptionB.RestoreBreak (F := Gl)
          (OptionB.Holds (shapeOf (keys 0).relation) (shapeOf (keys 1).relation) modulus barOf (S 0) (S 1)
            (OptionB.FinishTerminal finish stateKeys) (OptionB.slotClaim finish 0) (OptionB.slotClaim finish 1))
          Circuit.Block.poseidonTemplate (OptionB.counts finish stateKeys keys) (OptionB.shrinkRun finish stateKeys keys)
          (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)
          (OptionB.Reads finish stateKeys keys 0 (S 0) L0 L1) (OptionB.Reads finish stateKeys keys 1 (S 1) L0 L1)) ≤
      calls / 2 ^ t :=
  Final.OptionB.restore_budget (shapeOf (keys 0).relation) (shapeOf (keys 1).relation) modulus barOf (S 0) (S 1)
    L0 L1 h0 h1 cubic_big (show (keys 0).relation.setupVars ≤ 35 by have := (facts 0).relation.high; omega)
    (show (keys 1).relation.setupVars ≤ 35 by have := (facts 1).relation.high; omega) t
    Circuit.Block.poseidonTemplate _ (OptionB.shrinkRun finish stateKeys keys) Ls (p0 0) (p1 0) (p0 1) (p1 1) rust
    (queryAt (keys 0).relation) (queryAt (keys 1).relation) goldilocksModulus goldilocksModulus (facts 0).shape
    (facts 1).shape (cubic_natCast_injective _ le_rfl) (cubic_natCast_injective _ le_rfl)
    (Bridge.queryAt_distinct (keys 0).relation (facts 0).relation.high)
    (Bridge.queryAt_distinct (keys 1).relation (facts 1).relation.high) (fits 0) (fits 1) modulus_monic
    (by rw [modulus_natDegree]; rfl) (by rw [modulus_natDegree]; rfl) (show 2 ≤ Replay.D by decide)
    Lowering.Block.poseidonTemplate_wellFormed (Sim.OptionB.shrinkRun_valid finish stateKeys keys)
    (cubic_natCast_injective 9 (by norm_num [goldilocksModulus])) _ _ _ _ _
    (Sim.OptionB.optionBEncodes S facts fits describes L0 L1 fit keysLength) calls prover

/-- Soundness of `lifecycle/option_b/compression.rs::OptionB::verify_final`: the shrink layer accepts
a statement while no refresh step input with its counter and states has the finish checks and
both next slots holding under their circuits' relations (`OptionB.ParentHolds`, SuperNeo v1.2
Definition 21 for the last fold), with probability at most `calls / 2^t`. The finish package is the
one that F′'s emitter writes (`OptionB.finishPackage`). Fiat–Shamir inside the program is assumed
(`TRUST.md` §3). -/
theorem option_b_parent_bound {lw : ℕ}
    {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (program : Stage1.Application.Program) (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (next width : ℕ)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ)
    (packed : Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (starts : Export.Generic.CircuitRelation.startsFrom
      (Layout.OptionB.RefreshAssembler.sharedEnd Export.OptionB.StepRelation.base program.witnessWordCount)
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (stateKeys : List Replay.Word) (keys : Fin 2 → Layer1Key) (S : Fin 2 → Layer1.Setup Gl)
    (facts : ∀ slot, KeyFacts (keys slot)) (fits : ∀ slot, (S slot).Fits (shapeOf (keys slot).relation))
    (describes : ∀ slot,
      (structureOf (keys slot).relation.key.layout).Describes (shapeOf (keys slot).relation) (S slot))
    (shape : ∀ slot, ParentShape (shapeOf (keys slot).relation) lw)
    (honest : ∀ slot, HonestSetup (shapeOf (keys slot).relation) (S slot) (circuitKeys slot).relation ajtai)
    (fit : OptionB.fits (OptionB.finishPackage program relation next width batches rest) = true)
    (keysLength : stateKeys.length ≤ 8)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : ∀ slot, (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf (keys slot).relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : ∀ slot, (L1 : ℝ) * (1 / 2) ^ ((shapeOf (keys slot).relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees
        (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage program relation next width batches rest) stateKeys keys))
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage program relation next width batches rest) stateKeys keys))
          / 2 ^ 191 ≤ 1 / 2 ^ t)
    (calls : ℕ)
    (prover : Game.Strategy (OptionB.Outcome Gl Cubic (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame (F := Gl) Circuit.Block.poseidonTemplate
            (OptionB.counts (OptionB.finishPackage program relation next width batches rest) stateKeys keys)
            (OptionB.shrinkRun (OptionB.finishPackage program relation next width batches rest) stateKeys keys) Ls)
          (choice (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)) calls [] [])
        (OptionB.RestoreBreak (F := Gl) (OptionB.ParentHolds program ajtai circuitKeys)
          Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage program relation next width batches rest) stateKeys keys)
          (OptionB.shrinkRun (OptionB.finishPackage program relation next width batches rest) stateKeys keys)
          (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)
          (OptionB.Reads (OptionB.finishPackage program relation next width batches rest) stateKeys keys 0
            (S 0) L0 L1)
          (OptionB.Reads (OptionB.finishPackage program relation next width batches rest) stateKeys keys 1
            (S 1) L0 L1)) ≤
      calls / 2 ^ t :=
  (prob_mono prover _ fun r => Final.OptionB.restoreBreak_mono (fun _ holds =>
      Final.OptionB.parentHolds_of_holds program relation ajtai circuitKeys next width batches rest packed starts
        (fun slot => shapeOf (keys slot).relation) S barOf barOf_eq shape (fun slot => (facts slot).shape.blocks_le) honest holds) _ _ _ _ _ _ _ r).trans
    (option_b_bound _ stateKeys keys S facts fits describes fit keysLength L0 L1 h0 h1 t Ls p0 p1 rust calls prover)

/-- `option_b_parent_bound` in the terms of option B's IVC walk (CW-10.1): the shrink layer accepts a
statement while no last fold exists (`OptionB.LastFold`), with probability at most `calls / 2^t`. A
last fold is a refresh step input with the statement's iteration and states and a nonzero
iteration, the finish checks under the setup's keys, and witnesses of both next slots under their
circuits' relations. The final program's state keys are the setup's key digests
(`OptionB.KeysAre`). -/
theorem option_b_last_fold_bound {lw : ℕ}
    {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    (program : Stage1.Application.Program) (relation : ProductionKey.LogicalRelation lw pf)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf))
    (circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf) (next width : ℕ)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ)
    (packed : Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (starts : Export.Generic.CircuitRelation.startsFrom
      (Layout.OptionB.RefreshAssembler.sharedEnd Export.OptionB.StepRelation.base program.witnessWordCount)
      (Export.OptionB.FinishRelation.pieces program relation) = true)
    (stateKeys : List Replay.Word) (keys : Fin 2 → Layer1Key) (S : Fin 2 → Layer1.Setup Gl)
    (facts : ∀ slot, KeyFacts (keys slot)) (fits : ∀ slot, (S slot).Fits (shapeOf (keys slot).relation))
    (describes : ∀ slot,
      (structureOf (keys slot).relation.key.layout).Describes (shapeOf (keys slot).relation) (S slot))
    (shape : ∀ slot, ParentShape (shapeOf (keys slot).relation) lw)
    (honest : ∀ slot, HonestSetup (shapeOf (keys slot).relation) (S slot) (circuitKeys slot).relation ajtai)
    (fit : OptionB.fits (OptionB.finishPackage program relation next width batches rest) = true)
    (keysLength : stateKeys.length ≤ 8) (keyWords : OptionB.KeysAre stateKeys circuitKeys)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : ∀ slot, (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf (keys slot).relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : ∀ slot, (L1 : ℝ) * (1 / 2) ^ ((shapeOf (keys slot).relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees
        (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage program relation next width batches rest) stateKeys keys))
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage program relation next width batches rest) stateKeys keys))
          / 2 ^ 191 ≤ 1 / 2 ^ t)
    (calls : ℕ)
    (prover : Game.Strategy (OptionB.Outcome Gl Cubic (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1))) :
    Game.prob prover (OptionB.finalGame (OptionB.finishPackage program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)
        (OptionB.RestoreBreak (F := Gl) (OptionB.LastFold program ajtai circuitKeys)
          Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage program relation next width batches rest) stateKeys keys)
          (OptionB.shrinkRun (OptionB.finishPackage program relation next width batches rest) stateKeys keys)
          (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)
          (OptionB.Reads (OptionB.finishPackage program relation next width batches rest) stateKeys keys 0
            (S 0) L0 L1)
          (OptionB.Reads (OptionB.finishPackage program relation next width batches rest) stateKeys keys 1
            (S 1) L0 L1)) ≤
      calls / 2 ^ t :=
  (prob_mono prover _ fun r => Final.OptionB.restoreBreak_mono (fun _ holds =>
      Final.OptionB.lastFold_of_holds program relation ajtai circuitKeys next width batches rest packed starts
        (fun slot => shapeOf (keys slot).relation) S barOf barOf_eq shape (fun slot => (facts slot).shape.blocks_le)
        honest keyWords keysShaped holds) _ _ _ _ _ _ _ r).trans
    (option_b_bound _ stateKeys keys S facts fits describes fit keysLength L0 L1 h0 h1 t Ls p0 p1 rust calls prover)

/-- Option B's final verifier and its IVC bound in one experiment (CW-10.1). The experiment draws an
outcome of the final verifier's state-restoration experiment (`Game.law`), then the coins of the
finish stage's extractor, then the coins of the reverse extractor's stages. The finish stage's NIFS
adversary outputs the finish fold of the accepted statement's last fold
(`OptionB.finishAdversary`), and Assumption 1 gives its extractor. The chain's IVC adversary
outputs the terminal that the extractor's answer gives (`OptionB.chainAdversary`). When both are
admitted, the final verifier accepts a statement whose iteration is at most `depth` and that no
valid advice reaches with probability at most `calls / 2^t`, plus the finish stage's Assumption 1
error, plus the terms of `Export.OptionB.IvcSecurity.probability_bound`. Both adversaries choose the
last fold with `Classical.epsilon`: the compression claims prove that it exists, not that the final
proof gives it. -/
theorem option_b_ivc_bound {lw : ℕ}
    {pf : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth lw}
    {relation : ProductionKey.LogicalRelation lw pf}
    {ajtai : PaperAlgebra.AjtaiKey (logicalWidth := lw) (publicFits := pf)}
    {circuitKeys : Fin 2 → Lifecycle.OptionB.CircuitKey lw pf} {step : AppState → AppWitness → AppState}
    {cubeFits : Phi81CarrierLayout.carrierWidth lw ≤ 2 ^ cubeVariables}
    (exports : Export.OptionB.TerminalBridge.Exports relation ajtai circuitKeys step cubeFits) (next width : ℕ)
    (batches : List NightstreamFPrime.Circuit.WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ)
    (packed : Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns
      (Export.OptionB.FinishRelation.pieces exports.program relation) = true)
    (starts : Export.Generic.CircuitRelation.startsFrom
      (Layout.OptionB.RefreshAssembler.sharedEnd Export.OptionB.StepRelation.base exports.program.witnessWordCount)
      (Export.OptionB.FinishRelation.pieces exports.program relation) = true)
    (stateKeys : List Replay.Word) (keys : Fin 2 → Layer1Key) (S : Fin 2 → Layer1.Setup Gl)
    (facts : ∀ slot, KeyFacts (keys slot)) (fits : ∀ slot, (S slot).Fits (shapeOf (keys slot).relation))
    (describes : ∀ slot,
      (structureOf (keys slot).relation.key.layout).Describes (shapeOf (keys slot).relation) (S slot))
    (shape : ∀ slot, ParentShape (shapeOf (keys slot).relation) lw)
    (honest : ∀ slot, HonestSetup (shapeOf (keys slot).relation) (S slot) (circuitKeys slot).relation ajtai)
    (fit : OptionB.fits (OptionB.finishPackage exports.program relation next width batches rest) = true)
    (keysLength : stateKeys.length ≤ 8) (keyWords : OptionB.KeysAre stateKeys circuitKeys)
    (keysShaped : ∀ key, (circuitKeys key).digest.length = 4)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : ∀ slot, (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf (keys slot).relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : ∀ slot, (L1 : ℝ) * (1 / 2) ^ ((shapeOf (keys slot).relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees
        (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys))
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys))
          / 2 ^ 191 ≤ 1 / 2 ^ t)
    (calls : ℕ)
    (prover : Game.Strategy (OptionB.Outcome Gl Cubic (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)))
    {Admitted : NifsAdversary (Export.OptionB.IvcWalk.WalkAttempt lw pf) → Prop}
    {StageAdmitted : Stage (Export.OptionB.IvcWalk.Visit lw pf) (Export.OptionB.IvcWalk.Answer lw pf) → Prop}
    {Efficient : (adversary : NifsAdversary (Export.OptionB.IvcWalk.WalkAttempt lw pf)) →
      NifsExtractor (Export.OptionB.IvcWalk.Answer lw pf) adversary → Prop}
    {error : NifsAdversary (Export.OptionB.IvcWalk.WalkAttempt lw pf) → ℝ}
    (assumption : Assumption1 (Export.OptionB.IvcWalk.WalkSucceeds ajtai)
      (Export.OptionB.IvcWalk.WalkExtracted ajtai) Admitted Efficient error)
    (closed : Closed (Export.OptionB.IvcWalk.attemptOf ajtai circuitKeys exports)
      (Export.OptionB.IvcWalk.back ajtai circuitKeys exports) Admitted StageAdmitted Efficient)
    (finishAdmitted : Admitted (OptionB.finishAdversary exports.program ajtai circuitKeys
        (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) fun r => OptionB.acceptedOf r.1))
    (control : AppState → AppWitness → Fin 2) (cap depth : ℕ) :
    ∃ extractor : NifsExtractor (Export.OptionB.IvcWalk.Answer lw pf) (OptionB.finishAdversary exports.program ajtai circuitKeys
        (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) fun r => OptionB.acceptedOf r.1),
      Efficient _ extractor ∧
      ∀ admitted : StageAdmitted (Stage.start (Export.OptionB.IvcSecurity.walkAdversary
        (OptionB.chainAdversary exports.program ajtai circuitKeys (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls))
          (fun r => OptionB.acceptedOf r.1) extractor depth))),
      Game.prob prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)
          (OptionB.RestoreBreak (F := Gl) (fun x => depth < (x 0).val ∨ Export.OptionB.IvcWalk.Reaches exports.program (OptionB.terminalOf x))
          Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys)
          (OptionB.shrinkRun (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys)
          (game (keys 0) (S 0) L0 L1) (game (keys 1) (S 1) L0 L1)
          (OptionB.Reads (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys 0
            (S 0) L0 L1)
          (OptionB.Reads (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys 1
            (S 1) L0 L1)) ≤
        calls / 2 ^ t + error (OptionB.finishAdversary exports.program ajtai circuitKeys
        (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) fun r => OptionB.acceptedOf r.1) +
          ∑ j : Fin depth,
            (((reverseStages assumption closed (Export.OptionB.IvcSecurity.walkAdversary
              (OptionB.chainAdversary exports.program ajtai circuitKeys (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) (fun r => OptionB.acceptedOf r.1)
                extractor depth)) admitted j.val).1.tape.toOuterMeasure
                {tape | Export.OptionB.IvcWalk.WalkCollision ajtai circuitKeys exports
                  (visitAfter (Export.OptionB.IvcWalk.back ajtai circuitKeys exports)
                    ((reverseStages assumption closed (Export.OptionB.IvcSecurity.walkAdversary
              (OptionB.chainAdversary exports.program ajtai circuitKeys (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) (fun r => OptionB.acceptedOf r.1)
                extractor depth)) admitted j.val).1.input tape) ((reverseStages assumption closed (Export.OptionB.IvcSecurity.walkAdversary
              (OptionB.chainAdversary exports.program ajtai circuitKeys (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) (fun r => OptionB.acceptedOf r.1)
                extractor depth)) admitted j.val).1.results tape))}).toReal +
              error ((reverseStages assumption closed (Export.OptionB.IvcSecurity.walkAdversary
              (OptionB.chainAdversary exports.program ajtai circuitKeys (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) (fun r => OptionB.acceptedOf r.1)
                extractor depth)) admitted j.val).1.nifs (Export.OptionB.IvcWalk.attemptOf ajtai circuitKeys exports)
                (Export.OptionB.IvcWalk.back ajtai circuitKeys exports))) := by
  obtain ⟨extractor, efficient, extractionError⟩ := assumption _ finishAdmitted
  refine ⟨extractor, efficient, fun admitted => ?_⟩
  have compression := option_b_last_fold_bound exports.program relation ajtai circuitKeys next width batches rest
    packed starts stateKeys keys S facts fits describes shape honest fit keysLength keyWords keysShaped L0 L1 h0 h1 t
    Ls p0 p1 rust calls prover
  have ivc := Export.OptionB.IvcSecurity.probability_bound ajtai circuitKeys exports assumption closed
    (OptionB.chainAdversary exports.program ajtai circuitKeys (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) (fun r => OptionB.acceptedOf r.1) extractor
      depth) admitted keysShaped control cap depth
    (fun tape _ => Final.OptionB.chainOutput_iteration_le exports.program ajtai circuitKeys depth _ _)
  rw [Game.prob_eq_law] at compression ⊢
  refine (Final.OptionB.chain_le (Succeeds := Export.OptionB.IvcWalk.WalkSucceeds ajtai)
    (Extracted := Export.OptionB.IvcWalk.WalkExtracted ajtai) _ extractor
    (OptionB.chainAdversary exports.program ajtai circuitKeys (Game.law prover (OptionB.finalGame (OptionB.finishPackage exports.program relation next width batches rest) stateKeys keys S L0 L1 Ls
          calls)) (fun r => OptionB.acceptedOf r.1) extractor
      depth).output _ _ (Export.OptionB.IvcSecurity.FalseAcceptance ajtai circuitKeys exports control cap)
    fun draw hit => Final.OptionB.chain_event exports keysShaped control cap depth _ _ extractor _ _
      (fun r broken => ?_) draw hit).trans (add_le_add (add_le_add compression extractionError) ivc)
  obtain ⟨x, acceptedEq, notClaim, weaker⟩ := Final.OptionB.restoreBreak_accepted
    (Claim' := OptionB.LastFold exports.program ajtai circuitKeys) _ _ _ _ _ _ _ r broken
  exact ⟨x, acceptedEq, Nat.le_of_not_lt fun deep => notClaim (Or.inl deep),
    fun reached => notClaim (Or.inr reached), weaker⟩

/-- Soundness of `OptionB::verify_final` for the Poseidon2 hash chain (`option_b_parent_bound`). Each
circuit key's relation is that of the circuit whose pieces its emitted step package holds
(`OptionB.circuitKey`), each slot's layer-1 setup is the one that Rust builds from its key's runs
(`OptionB.keySetup`), and each premise about a key, a setup or the runs is a check of Rust's files
(`STATUS.md`). -/
theorem option_b_poseidon2_bound (packages : Fin 2 → Export.OptionB.StepPackage.Package)
    (pieces : Fin 2 → List Export.Generic.Pieces.Piece) (nexts : Fin 2 → ℕ) (digests : Fin 2 → KeyDigest)
    (keys : Fin 2 → Layer1Key) (pushes : Fin 2 → List Push)
    (setup : AjtaiSetupV1.Setup productionProfile.commitmentWidth
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth Export.OptionB.StepRelation.keyWidth)))
    (facts : ∀ slot, KeyFacts (keys slot)) (within : ∀ slot, optionBFits (keys slot) = true)
    (setups : ∀ slot, setupFailureAt (keyColumns (keys slot)) (keys slot) (pushes slot) = none)
    (runs : ∀ slot, OptionB.keyRunsFailure (packages slot) (pushes slot) = none)
    (fits : ∀ slot, Export.Generic.CircuitRelation.Fits Export.Generic.CircuitRelation.publicColumns (pieces slot)
      (nexts slot) (packages slot).width)
    (logical : ∀ slot,
      Layout.ProductionAssignment.logicalWidth (OptionB.slotsOf (packages slot)) ≤ Export.OptionB.StepRelation.keyWidth)
    (packed : ∀ slot,
      Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns (pieces slot) = true)
    (rows : ∀ slot, (packages slot).relationRows = Export.Generic.CircuitRelation.relationRows
      Export.Generic.CircuitRelation.publicColumns (pieces slot) (nexts slot) (packages slot).width)
    (blocks : ∀ slot, (packages slot).blocks = Export.Generic.CircuitRelation.blocksOf (pieces slot))
    (starts : ∀ slot, (packages slot).sboxStart = Export.Generic.CircuitRelation.sboxStart (pieces slot) (nexts slot))
    (next width : ℕ) (batches : List NightstreamFPrime.Circuit.WitnessBatch) (rest : ℕ × ℕ × ℕ × ℕ)
    (finishPacked : Export.Generic.CircuitRelation.packable Export.Generic.CircuitRelation.publicColumns
      (Export.OptionB.FinishRelation.pieces Stage1.Poseidon2HashChainV1.program
        Export.OptionB.StepRelation.keyRelation) = true)
    (finishStarts : Export.Generic.CircuitRelation.startsFrom
      (Layout.OptionB.RefreshAssembler.sharedEnd Export.OptionB.StepRelation.base
        Stage1.Poseidon2HashChainV1.program.witnessWordCount)
      (Export.OptionB.FinishRelation.pieces Stage1.Poseidon2HashChainV1.program
        Export.OptionB.StepRelation.keyRelation) = true)
    (stateKeys : List Replay.Word)
    (fit : OptionB.fits (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
      Export.OptionB.StepRelation.keyRelation next width batches rest) = true)
    (keysLength : stateKeys.length ≤ 8)
    (L0 L1 : ℕ) (h0 : 1 ≤ L0) (h1 : 1 ≤ L1) (t Ls : ℕ)
    (p0 : ∀ slot, (L0 : ℝ) * L1 * Layer1.rustDegrees (shapeOf (keys slot).relation) / 2 ^ 191 ≤ 1 / 2 ^ (t + 1))
    (p1 : ∀ slot, (L1 : ℝ) * (1 / 2) ^ ((shapeOf (keys slot).relation).queries - 1) ≤ 1 / 2 ^ (t + 1))
    (rust : (Ls : ℝ) * Shrink.rustDegrees
        (Shrink.Layout.rowVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
            Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys))
        (Shrink.Layout.cellVars Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
            Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys))
          / 2 ^ 191 ≤ 1 / 2 ^ t)
    (calls : ℕ)
    (prover : Game.Strategy (OptionB.Outcome Gl Cubic
      (game (keys 0) (OptionB.keySetup setup.seed.bytes (keys 0) (pushes 0)) L0 L1)
      (game (keys 1) (OptionB.keySetup setup.seed.bytes (keys 1) (pushes 1)) L0 L1))) :
    Game.prob prover
        (restorePair (Shrink.shrinkGame (F := Gl) Circuit.Block.poseidonTemplate
            (OptionB.counts (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
              Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys)
            (OptionB.shrinkRun (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
              Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys) Ls)
          (choice (game (keys 0) (OptionB.keySetup setup.seed.bytes (keys 0) (pushes 0)) L0 L1)
            (game (keys 1) (OptionB.keySetup setup.seed.bytes (keys 1) (pushes 1)) L0 L1)) calls [] [])
        (OptionB.RestoreBreak (F := Gl)
          (OptionB.ParentHolds Stage1.Poseidon2HashChainV1.program setup.verifierKey fun slot =>
            OptionB.circuitKey (digests slot) (packages slot) (pieces slot) (nexts slot) (fits slot) (logical slot))
          Circuit.Block.poseidonTemplate
          (OptionB.counts (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
            Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys)
          (OptionB.shrinkRun (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
            Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys)
          (game (keys 0) (OptionB.keySetup setup.seed.bytes (keys 0) (pushes 0)) L0 L1)
          (game (keys 1) (OptionB.keySetup setup.seed.bytes (keys 1) (pushes 1)) L0 L1)
          (OptionB.Reads (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
            Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys 0
              (OptionB.keySetup setup.seed.bytes (keys 0) (pushes 0)) L0 L1)
          (OptionB.Reads (OptionB.finishPackage Stage1.Poseidon2HashChainV1.program
            Export.OptionB.StepRelation.keyRelation next width batches rest) stateKeys keys 1
              (OptionB.keySetup setup.seed.bytes (keys 1) (pushes 1)) L0 L1)) ≤
      calls / 2 ^ t :=
  option_b_parent_bound Stage1.Poseidon2HashChainV1.program Export.OptionB.StepRelation.keyRelation setup.verifierKey
    (fun slot => OptionB.circuitKey (digests slot) (packages slot) (pieces slot) (nexts slot) (fits slot) (logical slot))
    next width batches rest finishPacked finishStarts stateKeys keys
    (fun slot => OptionB.keySetup setup.seed.bytes (keys slot) (pushes slot)) facts
    (fun slot => (option_b_setup_facts_of_check _ (setups slot)).2.1)
    (fun slot => (option_b_setup_facts_of_check _ (setups slot)).2.2)
    (fun slot => option_b_shape_of_check (facts slot) (within slot))
    (fun slot => option_b_honest_of_check (keys slot) setup _ (pushes slot) (setups slot)
      (option_b_runs_of_check (runs slot) (fits slot) (packed slot) (rows slot) (blocks slot) (starts slot)
        (logical slot)))
    fit keysLength L0 L1 h0 h1 t Ls p0 p1 rust calls prover

end CompressionLean.Claims
