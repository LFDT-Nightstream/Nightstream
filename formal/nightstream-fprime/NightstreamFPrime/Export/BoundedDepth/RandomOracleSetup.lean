import NightstreamFPrime.Export.BoundedDepth.StepRelation
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightSolver
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshSolver
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.HashSolver
import NightstreamFPrime.Spec.AjtaiSetupV1.Programming

/-! The bounded-depth knowledge bounds of the light and refresh folds, averaged over the Ajtai key
setup (premise P1: SHAKE128 is a random oracle, `AjtaiSetupV1.Programming`): the bounded-depth form
of `RandomOracleSetup.knowledge_error_le_setup`.

The setup reads one 256-bit chunk per key coefficient (`SetupIndex`) and reduces it to a
field element (`setupKey`). The adversary and the circuit key may depend on the chunks. The
kernel chance of each fold (`LightUniqueness.kernelChance`, `RefreshUniqueness.kernelChance`) is
at most 1, so `expect_le_programmed` replaces the chunk average of the kernel chance by the
average over a uniform key matrix, with the chunks a uniform preimage of the matrix
(`lightMsisAdvantage`, `refreshMsisAdvantage`), plus the programming error
`|SetupIndex| · 2 · (2^32 − 1) / 2^256`.

* `light_knowledge_error_le_setup`, `refresh_knowledge_error_le_setup`: the average success is
  at most the average extraction and prior-state collision chances, the fold's statistical
  error, the MSIS advantage and the programming error.
* `light_msisAdvantage_le_solver`, `refresh_msisAdvantage_le_solver`: the MSIS advantage is at
  most the advantage of the fold's MSIS solver (`LightSolver.solve`, `RefreshSolver.solve`) on a
  uniform key matrix at `msisChainBound p cap`, plus `(3Q + 61)/budget` (light) or
  `(35Q + 653)/budget` (refresh).
* `programmingError_lt`: at the bounded-depth scheme's key width `2^26` (`StepRelation.keyWidth`), the setup has
  `22 · 1,242,757 · 54` chunks and the programming error is below `2^-192`.
* `light_production_knowledge_error_lt`, `refresh_production_knowledge_error_lt`: at the key
  width, the average success is below the average extraction and collision chances, the
  statistical error, the MSIS advantage and `2^-192`. With `Q = 2^64` the statistical error is
  below `2^-54` (light) and `2^-51` (refresh) (`light_production_knowledge_error_two_pow_64`,
  `refresh_production_knowledge_error_two_pow_64`).
* `light_hashCollisions_le_finder`, `refresh_hashCollisions_le_finder`: the average collision
  chances are at most `2` (light) or `18` (refresh) times the Poseidon2 collision finder's
  average chance (`HashSolver.lightFinderChance`, `HashSolver.refreshFinderChance`) plus
  `(4Q + 63)/budget` or `(52Q + 959)/budget`.
* `light_production_knowledge_error_lt_solvers`, `refresh_production_knowledge_error_lt_solvers`:
  at the key width, the average success is below the average extraction chance, the statistical
  error, the MSIS solver's advantage, the collision finder's advantage times `2` or `18`, the
  work over `budget` (`(7Q + 124)/budget` light, `(87Q + 1612)/budget` refresh) and `2^-192`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.RandomOracleSetup

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.AjtaiSetupV1.Programming
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)

attribute [local instance low] Classical.propDecidable

/-- One setup chunk per coefficient of the Ajtai key: row, block and lane. -/
abbrev SetupIndex (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  KeyIndex productionProfile.commitmentWidth
    (Phi81ColumnLayout.blockCount (FullShape logicalWidth publicFits).carrierWidth)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The Ajtai key that the setup chunks give (`AjtaiSetupV1.Programming.verifierKey_eq`). -/
def setupKey (chunks : SetupIndex logicalWidth publicFits → Chunk) :
    AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  keyOf (residues chunks)

/-- The programming error of `expect_le_programmed`. -/
noncomputable def programmingError (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :
    ℝ :=
  Fintype.card (SetupIndex logicalWidth publicFits) * (2 * 4294967295 / 2 ^ 256 : ℝ)

/-- A chance in `[0, 1]` that depends on the Ajtai key: its chunk average is at most its
average over a uniform matrix and a uniform preimage, plus the programming error. -/
private theorem expect_le_matrix (value : AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits) → (SetupIndex logicalWidth publicFits → Chunk) → ℝ)
    (nonnegative : ∀ ajtai chunks, 0 ≤ value ajtai chunks)
    (le_one : ∀ ajtai chunks, value ajtai chunks ≤ 1) :
    𝔼 chunks, value (setupKey chunks) chunks ≤
      (𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
        value (keyOf matrix) chunks) + programmingError logicalWidth publicFits := by
  have bound := expect_le_programmed (fun chunks => value (setupKey chunks) chunks)
    (fun chunks => nonnegative _ chunks) (fun chunks => le_one _ chunks)
  have programmed : (𝔼 matrix : SetupIndex logicalWidth publicFits → F,
      𝔼 chunks ∈ preimages matrix, value (setupKey chunks) chunks) =
        𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
          value (keyOf matrix) chunks := by
    refine Finset.expect_congr rfl fun matrix _ => Finset.expect_congr rfl fun chunks member => ?_
    rw [setupKey, mem_preimages.mp member]
  rw [programmed] at bound
  exact bound

/-- A per-key bound `value ≤ solver + loss` averages over the matrix and its preimages. -/
private theorem average_le (value solver : AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits) → (SetupIndex logicalWidth publicFits → Chunk) → ℝ)
    {loss : ℝ} (lossNonnegative : 0 ≤ loss)
    (each : ∀ ajtai chunks, value ajtai chunks ≤ solver ajtai chunks + loss) :
    (𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
        value (keyOf matrix) chunks) ≤
      (𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
        solver (keyOf matrix) chunks) + loss := by
  calc
    _ ≤ 𝔼 matrix : SetupIndex logicalWidth publicFits → F, ((𝔼 chunks ∈ preimages matrix,
          solver (keyOf matrix) chunks) + loss) :=
      Finset.expect_le_expect fun matrix _ => by
        rcases (preimages matrix).eq_empty_or_nonempty with empty | nonempty
        · simp [empty, lossNonnegative]
        · calc
            _ ≤ 𝔼 chunks ∈ preimages matrix, (solver (keyOf matrix) chunks + loss) :=
              Finset.expect_le_expect fun chunks _ => each _ chunks
            _ = _ := by rw [Finset.expect_add_distrib, Finset.expect_const nonempty]
    _ = _ := by rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty]

/-- A per-key bound `budget · (value − solver) ≤ work` gives `value ≤ solver + work/budget`. -/
private theorem le_add_div {value solver work : ℝ} {budget : ℕ} (positive : 0 < budget)
    (bound : budget * (value - solver) ≤ work) : value ≤ solver + work / budget := by
  have positiveReal : (0 : ℝ) < budget := by exact_mod_cast positive
  rw [← sub_le_iff_le_add', le_div_iff₀ positiveReal, mul_comm]
  exact bound

section Light

variable (key : (SetupIndex logicalWidth publicFits → Chunk) → CircuitKey logicalWidth publicFits)
  (adversary : (SetupIndex logicalWidth publicFits → Chunk) →
    OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
      (Attempt logicalWidth publicFits))

/-- The chance that the light attempt succeeds, for the key of `chunks`. -/
noncomputable def lightSuccess (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, if Fork.Succeeds (LightFork.game (setupKey chunks) (key chunks)) (adversary chunks)
    oracle then 1 else 0

/-- The chance that a successful light run extracts, weighted by the retry law. -/
noncomputable def lightExtraction (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, ∑ retries,
    Uniqueness.weight (LightFork.game (setupKey chunks) (key chunks)) (adversary chunks) oracle
        retries *
      (if LightKnowledge.Extracts (setupKey chunks) (key chunks) (adversary chunks) oracle retries
        then 1 else 0)

/-- The prior-state collision chances of the light fork and uniqueness steps. -/
noncomputable def lightHashCollisions (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, (if Fork.Succeeds (LightFork.game (setupKey chunks) (key chunks)) (adversary chunks)
      oracle then
      ∑ index, LightFork.collisionChance (setupKey chunks) (key chunks) (adversary chunks) index
        oracle else 0) +
    LightKnowledge.stateChance (setupKey chunks) (key chunks) (adversary chunks)

/-- The light kernel chance on a uniform key matrix, with the adversary and the circuit key
given a uniform preimage of the matrix. -/
noncomputable def lightMsisAdvantage : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
    LightUniqueness.kernelChance (keyOf matrix) (key chunks) (adversary chunks)

/-- The light MSIS solver's chance on a uniform key matrix, with `budget` adversary runs. -/
noncomputable def lightSolverAdvantage (budget : ℕ) : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
    LightSolver.solverChance (keyOf matrix) (key chunks) (adversary chunks) budget

/-- The light knowledge bound averaged over the setup. -/
theorem light_knowledge_error_le_setup {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, lightSuccess key adversary chunks ≤
      𝔼 chunks, (lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
        LightKnowledge.statisticalError queries + lightMsisAdvantage key adversary +
        programmingError logicalWidth publicFits := by
  have perKey (chunks : SetupIndex logicalWidth publicFits → Chunk) :
      lightSuccess key adversary chunks ≤
        (lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
          LightKnowledge.statisticalError queries +
          LightUniqueness.kernelChance (setupKey chunks) (key chunks) (adversary chunks) := by
    have bound := LightKnowledge.knowledge_error_le (setupKey chunks) (key chunks)
      (adversary chunks) (bounded chunks)
    unfold LightKnowledge.knowledgeError at bound
    unfold lightSuccess lightExtraction lightHashCollisions
    linarith
  have binding := expect_le_matrix
    (fun ajtai chunks => LightUniqueness.kernelChance ajtai (key chunks) (adversary chunks))
    (fun ajtai chunks => Uniqueness.reductionChance_nonnegative _ _ _)
    (fun ajtai chunks => Uniqueness.reductionChance_le_one _ _ _)
  calc _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk,
          ((lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
            LightKnowledge.statisticalError queries +
            LightUniqueness.kernelChance (setupKey chunks) (key chunks) (adversary chunks)) :=
        Finset.expect_le_expect fun chunks _ => perKey chunks
    _ = _ := by
        rw [Finset.expect_add_distrib, Finset.expect_add_distrib,
          Finset.expect_const Finset.univ_nonempty]
    _ ≤ _ := by
        unfold lightMsisAdvantage
        linarith

/-- The light MSIS advantage is at most the light solver's advantage plus `(3Q + 61)/budget`. -/
theorem light_msisAdvantage_le_solver {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    lightMsisAdvantage key adversary ≤
      lightSolverAdvantage key adversary budget + (3 * queries + 61) / budget :=
  average_le
    (fun ajtai chunks => LightUniqueness.kernelChance ajtai (key chunks) (adversary chunks))
    (fun ajtai chunks => LightSolver.solverChance ajtai (key chunks) (adversary chunks) budget)
    (by positivity) fun ajtai chunks => le_add_div positive
      (LightSolver.kernelChance_le_solver ajtai (key chunks) (adversary chunks)
        (bounded chunks) budget)

/-- The light Poseidon2 collision finder's chance, averaged over the setup. -/
noncomputable def lightHashAdvantage (budget : ℕ) : ℝ :=
  𝔼 chunks, HashSolver.lightFinderChance (setupKey chunks) (key chunks) (adversary chunks) budget

/-- The light collision chances averaged over the setup are at most twice the finder's
advantage plus `(4Q + 63)/budget`. -/
theorem light_hashCollisions_le_finder {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, lightHashCollisions key adversary chunks ≤
      2 * lightHashAdvantage key adversary budget + (4 * queries + 63) / budget := by
  calc
    _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk,
          (2 * HashSolver.lightFinderChance (setupKey chunks) (key chunks) (adversary chunks)
              budget + (4 * queries + 63) / budget) :=
      Finset.expect_le_expect fun chunks _ =>
        HashSolver.light_hashCollisions_le_finder (setupKey chunks) (key chunks)
          (adversary chunks) (bounded chunks) positive
    _ = _ := by
      rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty,
        ← Finset.mul_expect]
      rfl

end Light

section Refresh

variable (key : (SetupIndex logicalWidth publicFits → Chunk) → CircuitKey logicalWidth publicFits)
  (adversary : (SetupIndex logicalWidth publicFits → Chunk) →
    OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
      (Attempt logicalWidth publicFits))

/-- The chance that the refresh attempt succeeds, for the key of `chunks`. -/
noncomputable def refreshSuccess (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, if Fork.Succeeds (RefreshFork.game (setupKey chunks) (key chunks)) (adversary chunks)
    oracle then 1 else 0

/-- The chance that a successful refresh run extracts, weighted by the retry law. -/
noncomputable def refreshExtraction (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, ∑ retries,
    Uniqueness.weight (RefreshFork.game (setupKey chunks) (key chunks)) (adversary chunks) oracle
        retries *
      (if RefreshKnowledge.Extracts (setupKey chunks) (key chunks) (adversary chunks) oracle
          retries then 1 else 0)

/-- The prior-state collision chances of the refresh fork and uniqueness steps. -/
noncomputable def refreshHashCollisions (chunks : SetupIndex logicalWidth publicFits → Chunk) :
    ℝ :=
  𝔼 oracle, (if Fork.Succeeds (RefreshFork.game (setupKey chunks) (key chunks))
      (adversary chunks) oracle then
      ∑ index, RefreshFork.collisionChance (setupKey chunks) (key chunks) (adversary chunks) index
        oracle else 0) +
    RefreshKnowledge.stateChance (setupKey chunks) (key chunks) (adversary chunks)

/-- The refresh kernel chance on a uniform key matrix, with the adversary and the circuit key
given a uniform preimage of the matrix. -/
noncomputable def refreshMsisAdvantage : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
    RefreshUniqueness.kernelChance (keyOf matrix) (key chunks) (adversary chunks)

/-- The refresh MSIS solver's chance on a uniform key matrix, with `budget` adversary runs. -/
noncomputable def refreshSolverAdvantage (budget : ℕ) : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
    RefreshSolver.solverChance (keyOf matrix) (key chunks) (adversary chunks) budget

/-- The refresh knowledge bound averaged over the setup. -/
theorem refresh_knowledge_error_le_setup {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, refreshSuccess key adversary chunks ≤
      𝔼 chunks, (refreshExtraction key adversary chunks +
          refreshHashCollisions key adversary chunks) +
        RefreshKnowledge.statisticalError queries + refreshMsisAdvantage key adversary +
        programmingError logicalWidth publicFits := by
  have perKey (chunks : SetupIndex logicalWidth publicFits → Chunk) :
      refreshSuccess key adversary chunks ≤
        (refreshExtraction key adversary chunks + refreshHashCollisions key adversary chunks) +
          RefreshKnowledge.statisticalError queries +
          RefreshUniqueness.kernelChance (setupKey chunks) (key chunks) (adversary chunks) := by
    have bound := RefreshKnowledge.knowledge_error_le (setupKey chunks) (key chunks)
      (adversary chunks) (bounded chunks)
    unfold RefreshKnowledge.knowledgeError at bound
    unfold refreshSuccess refreshExtraction refreshHashCollisions
    linarith
  have binding := expect_le_matrix
    (fun ajtai chunks => RefreshUniqueness.kernelChance ajtai (key chunks) (adversary chunks))
    (fun ajtai chunks => Uniqueness.reductionChance_nonnegative _ _ _)
    (fun ajtai chunks => Uniqueness.reductionChance_le_one _ _ _)
  calc _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk,
          ((refreshExtraction key adversary chunks + refreshHashCollisions key adversary chunks) +
            RefreshKnowledge.statisticalError queries +
            RefreshUniqueness.kernelChance (setupKey chunks) (key chunks) (adversary chunks)) :=
        Finset.expect_le_expect fun chunks _ => perKey chunks
    _ = _ := by
        rw [Finset.expect_add_distrib, Finset.expect_add_distrib,
          Finset.expect_const Finset.univ_nonempty]
    _ ≤ _ := by
        unfold refreshMsisAdvantage
        linarith

/-- The refresh MSIS advantage is at most the refresh solver's advantage plus
`(35Q + 653)/budget`. -/
theorem refresh_msisAdvantage_le_solver {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    refreshMsisAdvantage key adversary ≤
      refreshSolverAdvantage key adversary budget + (35 * queries + 653) / budget :=
  average_le
    (fun ajtai chunks => RefreshUniqueness.kernelChance ajtai (key chunks) (adversary chunks))
    (fun ajtai chunks => RefreshSolver.solverChance ajtai (key chunks) (adversary chunks) budget)
    (by positivity) fun ajtai chunks => le_add_div positive
      (RefreshSolver.kernelChance_le_solver ajtai (key chunks) (adversary chunks)
        (bounded chunks) budget)

/-- The refresh Poseidon2 collision finder's chance, averaged over the setup. -/
noncomputable def refreshHashAdvantage (budget : ℕ) : ℝ :=
  𝔼 chunks, HashSolver.refreshFinderChance (setupKey chunks) (key chunks) (adversary chunks) budget

/-- The refresh collision chances averaged over the setup are at most 18 times the finder's
advantage plus `(52Q + 959)/budget`. -/
theorem refresh_hashCollisions_le_finder {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, refreshHashCollisions key adversary chunks ≤
      18 * refreshHashAdvantage key adversary budget + (52 * queries + 959) / budget := by
  calc
    _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk,
          (18 * HashSolver.refreshFinderChance (setupKey chunks) (key chunks) (adversary chunks)
              budget + (52 * queries + 959) / budget) :=
      Finset.expect_le_expect fun chunks _ =>
        HashSolver.refresh_hashCollisions_le_finder (setupKey chunks) (key chunks)
          (adversary chunks) (bounded chunks) positive
    _ = _ := by
      rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty,
        ← Finset.mul_expect]
      rfl

end Refresh

section Production

local notation "Width" => StepRelation.keyWidth
local notation "Fits" => StepRelation.keyPublicFits

/-- At the bounded-depth scheme's key width the setup has `22 · 1,242,757 · 54` chunks. -/
theorem setupIndex_card :
    Fintype.card (SetupIndex Width Fits) = 22 * 1242757 * 54 := by
  show Fintype.card (KeyIndex productionProfile.commitmentWidth
    (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth StepRelation.keyWidth))) = _
  rw [Fintype.card_prod, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, Fintype.card_fin]
  rfl

/-- At the bounded-depth scheme's key width the programming error is below `2^-192`. -/
theorem programmingError_lt : programmingError Width Fits < 1 / 2 ^ 192 := by
  unfold programmingError
  rw [setupIndex_card]
  norm_num

variable (key : (SetupIndex Width Fits → Chunk) → CircuitKey Width Fits)

/-- The light knowledge bound at the bounded-depth scheme's key width. -/
theorem light_production_knowledge_error_lt
    (adversary : (SetupIndex Width Fits → Chunk) →
      OracleComp (Point (LightSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, lightSuccess key adversary chunks <
      𝔼 chunks, (lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
        LightKnowledge.statisticalError queries + lightMsisAdvantage key adversary +
        1 / 2 ^ 192 := by
  have bound := light_knowledge_error_le_setup key adversary bounded
  have small := programmingError_lt
  linarith

/-- The refresh knowledge bound at the bounded-depth scheme's key width. -/
theorem refresh_production_knowledge_error_lt
    (adversary : (SetupIndex Width Fits → Chunk) →
      OracleComp (Point (RefreshSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, refreshSuccess key adversary chunks <
      𝔼 chunks, (refreshExtraction key adversary chunks +
          refreshHashCollisions key adversary chunks) +
        RefreshKnowledge.statisticalError queries + refreshMsisAdvantage key adversary +
        1 / 2 ^ 192 := by
  have bound := refresh_knowledge_error_le_setup key adversary bounded
  have small := programmingError_lt
  linarith

/-- The light knowledge bound at the bounded-depth scheme's key width with `2^64` queries. -/
theorem light_production_knowledge_error_two_pow_64
    (adversary : (SetupIndex Width Fits → Chunk) →
      OracleComp (Point (LightSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    (bounded : ∀ chunks, (adversary chunks).QueryBound (2 ^ 64)) :
    𝔼 chunks, lightSuccess key adversary chunks <
      𝔼 chunks, (lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
        lightMsisAdvantage key adversary + 1 / 2 ^ 54 + 1 / 2 ^ 192 := by
  have bound := light_production_knowledge_error_lt key adversary bounded
  have statistical := LightKnowledge.statisticalError_two_pow_64
  linarith

/-- The refresh knowledge bound at the bounded-depth scheme's key width with `2^64` queries. -/
theorem refresh_production_knowledge_error_two_pow_64
    (adversary : (SetupIndex Width Fits → Chunk) →
      OracleComp (Point (RefreshSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    (bounded : ∀ chunks, (adversary chunks).QueryBound (2 ^ 64)) :
    𝔼 chunks, refreshSuccess key adversary chunks <
      𝔼 chunks, (refreshExtraction key adversary chunks +
          refreshHashCollisions key adversary chunks) +
        refreshMsisAdvantage key adversary + 1 / 2 ^ 51 + 1 / 2 ^ 192 := by
  have bound := refresh_production_knowledge_error_lt key adversary bounded
  have statistical := RefreshKnowledge.statisticalError_two_pow_64
  linarith

/-- The light knowledge bound at the bounded-depth scheme's key width with both terms of named problems as the
advantage of an explicit algorithm: the MSIS solver and the Poseidon2 collision finder, each
with `budget` adversary runs. -/
theorem light_production_knowledge_error_lt_solvers
    (adversary : (SetupIndex Width Fits → Chunk) →
      OracleComp (Point (LightSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, lightSuccess key adversary chunks <
      𝔼 chunks, lightExtraction key adversary chunks + LightKnowledge.statisticalError queries +
        lightSolverAdvantage key adversary budget + 2 * lightHashAdvantage key adversary budget +
        (7 * queries + 124) / budget + 1 / 2 ^ 192 := by
  have bound := light_production_knowledge_error_lt key adversary bounded
  have msis := light_msisAdvantage_le_solver key adversary bounded positive
  have hash := light_hashCollisions_le_finder key adversary bounded positive
  have work : ((3 * queries + 61 : ℝ)) / budget + (4 * queries + 63) / budget =
      (7 * queries + 124) / budget := by ring
  rw [Finset.expect_add_distrib] at bound
  linarith

/-- The refresh knowledge bound at the bounded-depth scheme's key width with both terms of named problems as
the advantage of an explicit algorithm. -/
theorem refresh_production_knowledge_error_lt_solvers
    (adversary : (SetupIndex Width Fits → Chunk) →
      OracleComp (Point (RefreshSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, refreshSuccess key adversary chunks <
      𝔼 chunks, refreshExtraction key adversary chunks +
        RefreshKnowledge.statisticalError queries + refreshSolverAdvantage key adversary budget +
        18 * refreshHashAdvantage key adversary budget + (87 * queries + 1612) / budget +
        1 / 2 ^ 192 := by
  have bound := refresh_production_knowledge_error_lt key adversary bounded
  have msis := refresh_msisAdvantage_le_solver key adversary bounded positive
  have hash := refresh_hashCollisions_le_finder key adversary bounded positive
  have work : ((35 * queries + 653 : ℝ)) / budget + (52 * queries + 959) / budget =
      (87 * queries + 1612) / budget := by ring
  rw [Finset.expect_add_distrib] at bound
  linarith

end Production

end NightstreamFPrime.Export.BoundedDepth.RandomOracleSetup
