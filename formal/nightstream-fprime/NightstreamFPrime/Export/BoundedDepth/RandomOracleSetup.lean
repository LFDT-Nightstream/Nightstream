import NightstreamFPrime.Export.BoundedDepth.StepRelation
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.LightSolver
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.RefreshSolver
import NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.HashSolver
import NightstreamFPrime.Spec.AjtaiSetupV1.Programming
import NightstreamFPrime.Spec.Sponge

/-! The bounded-depth knowledge bounds of the light and refresh folds, averaged over the Ajtai key
setup (premise P1: SHAKE128 is a random oracle, `AjtaiSetupV1.Programming`): the bounded-depth form
of `RandomOracleSetup.knowledge_error_le_setup`.

The setup reads one stream of 84 SHAKE128 words per key element (`SetupIndex`) and keeps the
words below `q` (`setupKey`). The adversary and the circuit key may depend on the streams. The
kernel chance of each fold (`LightUniqueness.kernelChance`, `RefreshUniqueness.kernelChance`) is
at most 1, so `expect_le_programmed` replaces the stream average of the kernel chance by the
average over a uniform key matrix, with the streams a uniform preimage of the matrix
(`lightMsisAdvantage`, `refreshMsisAdvantage`), plus the programming error
`|SetupIndex| · 2 · Rejection.fallbackBound`: a key element is exactly uniform unless 21 or more
of its 84 words are rejected, and `fallbackBound = C(84, 21) · 59^21 / 2^1344` bounds that
chance.

* `light_knowledge_error_le_setup`, `refresh_knowledge_error_le_setup`: the average success is
  at most the average extraction and prior-state collision chances, the fold's statistical
  error, the MSIS advantage and the programming error.
* `light_msisAdvantage_le_solver`, `refresh_msisAdvantage_le_solver`: the MSIS advantage is at
  most the advantage of the fold's MSIS solver (`LightSolver.solve`, `RefreshSolver.solve`) on a
  uniform key matrix at `msisChainBound p cap`, plus `(3Q + 61)/budget` (light) or
  `(31Q + 523)/budget` (refresh).
* `programmingError_lt`: at the bounded-depth scheme's key width `2^26` (`StepRelation.keyWidth`), the setup has
  `18 · 1,048,576` streams and the programming error is below `2^-1129`.
* `light_production_knowledge_error_lt`, `refresh_production_knowledge_error_lt`: at the key
  width, the average success is below the average extraction and collision chances, the
  statistical error, the MSIS advantage and `2^-1129`. With `Q = 2^64` the statistical error is
  below `2^-54` (light) and `2^-51` (refresh) (`light_production_knowledge_error_two_pow_64`,
  `refresh_production_knowledge_error_two_pow_64`).
* `light_hashCollisions_le_finder`, `refresh_hashCollisions_le_finder`: the average collision
  chances are at most `2` (light) or `16` (refresh) times the Poseidon2 collision finder's
  average chance (`HashSolver.lightFinderChance`, `HashSolver.refreshFinderChance`) plus
  `(4Q + 63)/budget` or `(46Q + 763)/budget`.
* `light_production_knowledge_error_lt_solvers`, `refresh_production_knowledge_error_lt_solvers`:
  at the key width, the average success is below the average extraction chance, the statistical
  error, the MSIS solver's advantage, the collision finder's advantage times `2` or `16`, the
  work over `budget` (`(7Q + 124)/budget` light, `(77Q + 1286)/budget` refresh) and `2^-1129`.
* `light_hashAdvantage_le`, `refresh_hashAdvantage_le`: under `[P2-PERM]` and `[SPONGE]`, the
  finder's advantage is at most `C(calls + 2, 2)/q⁴ + spongeError calls`
  (`Spec.Sponge.collision_le`). The finder is one algorithm with one output. Its tape holds the
  setup streams, the target and the `budget` oracles on which it runs the adversary, so its
  reruns of the adversary are inside the one run that `[SPONGE]` moves. `calls` counts the
  permutation calls of that run, the adversary's in each rerun included. The adversary is a Lean
  function, and Lean has no count of its calls: `calls` is a parameter of the adversary class.
* `light_production_knowledge_error_lt_numeric`, `refresh_production_knowledge_error_lt_numeric`:
  the bounds above with this number and the statistical error at most `(Q + 1) · 575/2^128`
  (light) or `(Q + 1) · 4748/2^128` (refresh). -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.RandomOracleSetup

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.AjtaiSetupV1.Programming
open NightstreamFPrime.Spec.AjtaiSetupV1.Rejection (Stream element fallbackBound fallbackBound_lt
  fallbackBound_nonneg)
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.BoundedDepth
open NightstreamFPrime.Lifecycle.RandomOracle
open NightstreamFPrime.Lifecycle.RandomOracle.Test (Answer Point)
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle
open NightstreamFPrime.Lifecycle.BoundedDepth.RandomOracle.ForkLink (Attempt)

attribute [local instance low] Classical.propDecidable

/-- One setup stream per element of the Ajtai key: row and block. -/
abbrev SetupIndex (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth) :=
  KeyIndex productionProfile.commitmentWidth
    (ConcreteRingColumnLayout.blockCount (FullShape logicalWidth publicFits).carrierWidth)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

/-- The Ajtai key that the setup streams give (`AjtaiSetupV1.Programming.verifierKey_eq`). -/
def setupKey (chunks : SetupIndex logicalWidth publicFits → Stream) :
    AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  keyOf (residues element chunks)

/-- The programming error of `expect_le_programmed`. -/
noncomputable def programmingError (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth) :
    ℝ :=
  Fintype.card (SetupIndex logicalWidth publicFits) * ((2 * fallbackBound : ℚ) : ℝ)

/-- A chance in `[0, 1]` that depends on the Ajtai key: its chunk average is at most its
average over a uniform matrix and a uniform preimage, plus the programming error. -/
private theorem expect_le_matrix (value : AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits) → (SetupIndex logicalWidth publicFits → Stream) → ℝ)
    (nonnegative : ∀ ajtai chunks, 0 ≤ value ajtai chunks)
    (le_one : ∀ ajtai chunks, value ajtai chunks ≤ 1) :
    𝔼 chunks, value (setupKey chunks) chunks ≤
      (𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
        value (keyOf matrix) chunks) + programmingError logicalWidth publicFits := by
  have bound := expect_le_programmed element element_onto element_bias_le
    (fun chunks => value (setupKey chunks) chunks)
    (fun chunks => nonnegative _ chunks) (fun chunks => le_one _ chunks)
  have programmed : (𝔼 matrix : SetupIndex logicalWidth publicFits → RingF,
      𝔼 chunks ∈ preimages element matrix, value (setupKey chunks) chunks) =
        𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
          value (keyOf matrix) chunks := by
    refine Finset.expect_congr rfl fun matrix _ => Finset.expect_congr rfl fun chunks member => ?_
    rw [setupKey, (mem_preimages element).mp member]
  rw [programmed] at bound
  exact bound

/-- A per-key bound `value ≤ solver + loss` averages over the matrix and its preimages. -/
private theorem average_le (value solver : AjtaiKey (logicalWidth := logicalWidth)
      (publicFits := publicFits) → (SetupIndex logicalWidth publicFits → Stream) → ℝ)
    {loss : ℝ} (lossNonnegative : 0 ≤ loss)
    (each : ∀ ajtai chunks, value ajtai chunks ≤ solver ajtai chunks + loss) :
    (𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
        value (keyOf matrix) chunks) ≤
      (𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
        solver (keyOf matrix) chunks) + loss := by
  calc
    _ ≤ 𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, ((𝔼 chunks ∈ preimages element matrix,
          solver (keyOf matrix) chunks) + loss) :=
      Finset.expect_le_expect fun matrix _ => by
        rcases (preimages element matrix).eq_empty_or_nonempty with empty | nonempty
        · simp [empty, lossNonnegative]
        · calc
            _ ≤ 𝔼 chunks ∈ preimages element matrix, (solver (keyOf matrix) chunks + loss) :=
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

variable (key : (SetupIndex logicalWidth publicFits → Stream) → CircuitKey logicalWidth publicFits)
  (adversary : (SetupIndex logicalWidth publicFits → Stream) →
    OracleComp (Point (LightSchedule.schedule logicalWidth publicFits) 8) Answer
      (Attempt logicalWidth publicFits))

/-- The chance that the light attempt succeeds, for the key of `chunks`. -/
noncomputable def lightSuccess (chunks : SetupIndex logicalWidth publicFits → Stream) : ℝ :=
  𝔼 oracle, if Fork.Succeeds (LightFork.game (setupKey chunks) (key chunks)) (adversary chunks)
    oracle then 1 else 0

/-- The chance that a successful light run extracts, weighted by the retry law. -/
noncomputable def lightExtraction (chunks : SetupIndex logicalWidth publicFits → Stream) : ℝ :=
  𝔼 oracle, ∑ retries,
    Uniqueness.weight (LightFork.game (setupKey chunks) (key chunks)) (adversary chunks) oracle
        retries *
      (if LightKnowledge.Extracts (setupKey chunks) (key chunks) (adversary chunks) oracle retries
        then 1 else 0)

/-- The prior-state collision chances of the light fork and uniqueness steps. -/
noncomputable def lightHashCollisions (chunks : SetupIndex logicalWidth publicFits → Stream) : ℝ :=
  𝔼 oracle, (if Fork.Succeeds (LightFork.game (setupKey chunks) (key chunks)) (adversary chunks)
      oracle then
      ∑ index, LightFork.collisionChance (setupKey chunks) (key chunks) (adversary chunks) index
        oracle else 0) +
    LightKnowledge.stateChance (setupKey chunks) (key chunks) (adversary chunks)

/-- The light kernel chance on a uniform key matrix, with the adversary and the circuit key
given a uniform preimage of the matrix. -/
noncomputable def lightMsisAdvantage : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
    LightUniqueness.kernelChance (keyOf matrix) (key chunks) (adversary chunks)

/-- The light MSIS solver's chance on a uniform key matrix, with `budget` adversary runs. -/
noncomputable def lightSolverAdvantage (budget : ℕ) : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
    LightSolver.solverChance (keyOf matrix) (key chunks) (adversary chunks) budget

/-- The light knowledge bound averaged over the setup. -/
theorem light_knowledge_error_le_setup {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, lightSuccess key adversary chunks ≤
      𝔼 chunks, (lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
        LightKnowledge.statisticalError queries + lightMsisAdvantage key adversary +
        programmingError logicalWidth publicFits := by
  have perKey (chunks : SetupIndex logicalWidth publicFits → Stream) :
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
  calc _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Stream,
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
    _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Stream,
          (2 * HashSolver.lightFinderChance (setupKey chunks) (key chunks) (adversary chunks)
              budget + (4 * queries + 63) / budget) :=
      Finset.expect_le_expect fun chunks _ =>
        HashSolver.light_hashCollisions_le_finder (setupKey chunks) (key chunks)
          (adversary chunks) (bounded chunks) positive
    _ = _ := by
      rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty,
        ← Finset.mul_expect]
      rfl

/-- `[SPONGE]` for the light finder: under `Spec.Sponge.Transfer` of its collision event with
`calls` permutation calls, its advantage is at most `C(calls + 2, 2)/q⁴ + spongeError calls`. -/
theorem light_hashAdvantage_le (budget blocks calls : ℕ)
    (finder : Sponge.Transfer Sponge.CollisionConditions
      (fun _ : Unit => lightHashAdvantage key adversary budget)
      (fun model : Sponge.CollisionAdversary blocks calls => Sponge.collisionChance model) calls) :
    lightHashAdvantage key adversary budget ≤
      Sponge.birthdayBound calls + Sponge.spongeError calls :=
  Sponge.collision_le finder ()

end Light

section Refresh

variable (key : (SetupIndex logicalWidth publicFits → Stream) → CircuitKey logicalWidth publicFits)
  (adversary : (SetupIndex logicalWidth publicFits → Stream) →
    OracleComp (Point (RefreshSchedule.schedule logicalWidth publicFits) 8) Answer
      (Attempt logicalWidth publicFits))

/-- The chance that the refresh attempt succeeds, for the key of `chunks`. -/
noncomputable def refreshSuccess (chunks : SetupIndex logicalWidth publicFits → Stream) : ℝ :=
  𝔼 oracle, if Fork.Succeeds (RefreshFork.game (setupKey chunks) (key chunks)) (adversary chunks)
    oracle then 1 else 0

/-- The chance that a successful refresh run extracts, weighted by the retry law. -/
noncomputable def refreshExtraction (chunks : SetupIndex logicalWidth publicFits → Stream) : ℝ :=
  𝔼 oracle, ∑ retries,
    Uniqueness.weight (RefreshFork.game (setupKey chunks) (key chunks)) (adversary chunks) oracle
        retries *
      (if RefreshKnowledge.Extracts (setupKey chunks) (key chunks) (adversary chunks) oracle
          retries then 1 else 0)

/-- The prior-state collision chances of the refresh fork and uniqueness steps. -/
noncomputable def refreshHashCollisions (chunks : SetupIndex logicalWidth publicFits → Stream) :
    ℝ :=
  𝔼 oracle, (if Fork.Succeeds (RefreshFork.game (setupKey chunks) (key chunks))
      (adversary chunks) oracle then
      ∑ index, RefreshFork.collisionChance (setupKey chunks) (key chunks) (adversary chunks) index
        oracle else 0) +
    RefreshKnowledge.stateChance (setupKey chunks) (key chunks) (adversary chunks)

/-- The refresh kernel chance on a uniform key matrix, with the adversary and the circuit key
given a uniform preimage of the matrix. -/
noncomputable def refreshMsisAdvantage : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
    RefreshUniqueness.kernelChance (keyOf matrix) (key chunks) (adversary chunks)

/-- The refresh MSIS solver's chance on a uniform key matrix, with `budget` adversary runs. -/
noncomputable def refreshSolverAdvantage (budget : ℕ) : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → RingF, 𝔼 chunks ∈ preimages element matrix,
    RefreshSolver.solverChance (keyOf matrix) (key chunks) (adversary chunks) budget

/-- The refresh knowledge bound averaged over the setup. -/
theorem refresh_knowledge_error_le_setup {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, refreshSuccess key adversary chunks ≤
      𝔼 chunks, (refreshExtraction key adversary chunks +
          refreshHashCollisions key adversary chunks) +
        RefreshKnowledge.statisticalError queries + refreshMsisAdvantage key adversary +
        programmingError logicalWidth publicFits := by
  have perKey (chunks : SetupIndex logicalWidth publicFits → Stream) :
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
  calc _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Stream,
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
`(31Q + 523)/budget`. -/
theorem refresh_msisAdvantage_le_solver {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    refreshMsisAdvantage key adversary ≤
      refreshSolverAdvantage key adversary budget + (31 * queries + 523) / budget :=
  average_le
    (fun ajtai chunks => RefreshUniqueness.kernelChance ajtai (key chunks) (adversary chunks))
    (fun ajtai chunks => RefreshSolver.solverChance ajtai (key chunks) (adversary chunks) budget)
    (by positivity) fun ajtai chunks => le_add_div positive
      (RefreshSolver.kernelChance_le_solver ajtai (key chunks) (adversary chunks)
        (bounded chunks) budget)

/-- The refresh Poseidon2 collision finder's chance, averaged over the setup. -/
noncomputable def refreshHashAdvantage (budget : ℕ) : ℝ :=
  𝔼 chunks, HashSolver.refreshFinderChance (setupKey chunks) (key chunks) (adversary chunks) budget

/-- The refresh collision chances averaged over the setup are at most 16 times the finder's
advantage plus `(46Q + 763)/budget`. -/
theorem refresh_hashCollisions_le_finder {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, refreshHashCollisions key adversary chunks ≤
      16 * refreshHashAdvantage key adversary budget + (46 * queries + 763) / budget := by
  calc
    _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Stream,
          (16 * HashSolver.refreshFinderChance (setupKey chunks) (key chunks) (adversary chunks)
              budget + (46 * queries + 763) / budget) :=
      Finset.expect_le_expect fun chunks _ =>
        HashSolver.refresh_hashCollisions_le_finder (setupKey chunks) (key chunks)
          (adversary chunks) (bounded chunks) positive
    _ = _ := by
      rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty,
        ← Finset.mul_expect]
      rfl

/-- `[SPONGE]` for the refresh finder: under `Spec.Sponge.Transfer` of its collision event with
`calls` permutation calls, its advantage is at most `C(calls + 2, 2)/q⁴ + spongeError calls`. -/
theorem refresh_hashAdvantage_le (budget blocks calls : ℕ)
    (finder : Sponge.Transfer Sponge.CollisionConditions
      (fun _ : Unit => refreshHashAdvantage key adversary budget)
      (fun model : Sponge.CollisionAdversary blocks calls => Sponge.collisionChance model) calls) :
    refreshHashAdvantage key adversary budget ≤
      Sponge.birthdayBound calls + Sponge.spongeError calls :=
  Sponge.collision_le finder ()

end Refresh

section Production

attribute [local irreducible] Real.instLT Real.instLE

local notation "Width" => StepRelation.keyWidth
local notation "Fits" => StepRelation.keyPublicFits

/-- At the bounded-depth scheme's key width the setup has `18 · 1,048,576` streams. -/
theorem setupIndex_card :
    Fintype.card (SetupIndex Width Fits) = 18 * 1048576 := by
  show Fintype.card (KeyIndex productionProfile.commitmentWidth
    (ConcreteRingColumnLayout.blockCount (RingCarrierLayout.carrierWidth StepRelation.keyWidth))) = _
  rw [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
  rfl

/-- At the bounded-depth scheme's key width the programming error is below `2^-1129`. -/
theorem programmingError_lt : programmingError Width Fits < 1 / 2 ^ 1129 := by
  have element := (Rat.cast_lt (K := ℝ)).mpr fallbackBound_lt
  rw [Rat.cast_div, Rat.cast_one, Rat.cast_pow, Rat.cast_ofNat] at element
  have nonnegative : (0 : ℝ) ≤ ((fallbackBound : ℚ) : ℝ) := by
    exact_mod_cast fallbackBound_nonneg
  have split : (2 : ℝ) ^ 26 * (1 / 2 ^ 1155) = 1 / 2 ^ 1129 := by
    rw [mul_one_div, show 1155 = 26 + 1129 from rfl, pow_add,
      div_mul_cancel_left₀ (pow_ne_zero _ two_ne_zero), one_div]
  unfold programmingError
  rw [setupIndex_card, ← split]
  calc _ ≤ 2 ^ 26 * ((fallbackBound : ℚ) : ℝ) := by
        push_cast
        nlinarith
    _ < 2 ^ 26 * (1 / 2 ^ 1155) := by gcongr

variable (key : (SetupIndex Width Fits → Stream) → CircuitKey Width Fits)

/-- The light knowledge bound at the bounded-depth scheme's key width. -/
theorem light_production_knowledge_error_lt
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (LightSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, lightSuccess key adversary chunks <
      𝔼 chunks, (lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
        LightKnowledge.statisticalError queries + lightMsisAdvantage key adversary +
        1 / 2 ^ 1129 := by
  have bound := light_knowledge_error_le_setup key adversary bounded
  have small := programmingError_lt
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at small ⊢
  linarith

/-- The refresh knowledge bound at the bounded-depth scheme's key width. -/
theorem refresh_production_knowledge_error_lt
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (RefreshSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks, refreshSuccess key adversary chunks <
      𝔼 chunks, (refreshExtraction key adversary chunks +
          refreshHashCollisions key adversary chunks) +
        RefreshKnowledge.statisticalError queries + refreshMsisAdvantage key adversary +
        1 / 2 ^ 1129 := by
  have bound := refresh_knowledge_error_le_setup key adversary bounded
  have small := programmingError_lt
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at small ⊢
  linarith

/-- The light knowledge bound at the bounded-depth scheme's key width with `2^64` queries. -/
theorem light_production_knowledge_error_two_pow_64
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (LightSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    (bounded : ∀ chunks, (adversary chunks).QueryBound (2 ^ 64)) :
    𝔼 chunks, lightSuccess key adversary chunks <
      𝔼 chunks, (lightExtraction key adversary chunks + lightHashCollisions key adversary chunks) +
        lightMsisAdvantage key adversary + 1 / 2 ^ 54 + 1 / 2 ^ 1129 := by
  have bound := light_production_knowledge_error_lt key adversary bounded
  have statistical := LightKnowledge.statisticalError_two_pow_64
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at bound ⊢
  generalize LightKnowledge.statisticalError (2 ^ 64) = statisticalError at bound statistical
  linarith

/-- The refresh knowledge bound at the bounded-depth scheme's key width with `2^64` queries. -/
theorem refresh_production_knowledge_error_two_pow_64
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (RefreshSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    (bounded : ∀ chunks, (adversary chunks).QueryBound (2 ^ 64)) :
    𝔼 chunks, refreshSuccess key adversary chunks <
      𝔼 chunks, (refreshExtraction key adversary chunks +
          refreshHashCollisions key adversary chunks) +
        refreshMsisAdvantage key adversary + 1 / 2 ^ 51 + 1 / 2 ^ 1129 := by
  have bound := refresh_production_knowledge_error_lt key adversary bounded
  have statistical := RefreshKnowledge.statisticalError_two_pow_64
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at bound ⊢
  generalize RefreshKnowledge.statisticalError (2 ^ 64) = statisticalError at bound statistical
  linarith

/-- The light knowledge bound at the bounded-depth scheme's key width with both terms of named problems as the
advantage of an explicit algorithm: the MSIS solver and the Poseidon2 collision finder, each
with `budget` adversary runs. -/
theorem light_production_knowledge_error_lt_solvers
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (LightSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, lightSuccess key adversary chunks <
      𝔼 chunks, lightExtraction key adversary chunks + LightKnowledge.statisticalError queries +
        lightSolverAdvantage key adversary budget + 2 * lightHashAdvantage key adversary budget +
        (7 * queries + 124) / budget + 1 / 2 ^ 1129 := by
  have bound := light_production_knowledge_error_lt key adversary bounded
  have msis := light_msisAdvantage_le_solver key adversary bounded positive
  have hash := light_hashCollisions_le_finder key adversary bounded positive
  have work : ((3 * queries + 61 : ℝ)) / budget + (4 * queries + 63) / budget =
      (7 * queries + 124) / budget := by ring
  rw [Finset.expect_add_distrib] at bound
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at bound ⊢
  linarith

/-- The refresh knowledge bound at the bounded-depth scheme's key width with both terms of named problems as
the advantage of an explicit algorithm. -/
theorem refresh_production_knowledge_error_lt_solvers
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (RefreshSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) :
    𝔼 chunks, refreshSuccess key adversary chunks <
      𝔼 chunks, refreshExtraction key adversary chunks +
        RefreshKnowledge.statisticalError queries + refreshSolverAdvantage key adversary budget +
        16 * refreshHashAdvantage key adversary budget + (77 * queries + 1286) / budget +
        1 / 2 ^ 1129 := by
  have bound := refresh_production_knowledge_error_lt key adversary bounded
  have msis := refresh_msisAdvantage_le_solver key adversary bounded positive
  have hash := refresh_hashCollisions_le_finder key adversary bounded positive
  have work : ((31 * queries + 523 : ℝ)) / budget + (46 * queries + 763) / budget =
      (77 * queries + 1286) / budget := by ring
  rw [Finset.expect_add_distrib] at bound
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at bound ⊢
  linarith

/-- The light knowledge bound at the bounded-depth scheme's key width with numbers for the
statistical term and, under `[P2-PERM]` and `[SPONGE]`, for the collision finder. -/
theorem light_production_knowledge_error_lt_numeric
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (LightSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) (blocks calls : ℕ)
    (finder : Sponge.Transfer Sponge.CollisionConditions
      (fun _ : Unit => lightHashAdvantage key adversary budget)
      (fun model : Sponge.CollisionAdversary blocks calls => Sponge.collisionChance model) calls) :
    𝔼 chunks, lightSuccess key adversary chunks <
      𝔼 chunks, lightExtraction key adversary chunks + ((queries : ℝ) + 1) * (575 / 2 ^ 128) +
        lightSolverAdvantage key adversary budget +
        2 * (Sponge.birthdayBound calls + Sponge.spongeError calls) +
        (7 * queries + 124) / budget + 1 / 2 ^ 1129 := by
  have bound := light_production_knowledge_error_lt_solvers key adversary bounded positive
  have statistical := LightKnowledge.statisticalError_le queries
  have hash := light_hashAdvantage_le key adversary budget blocks calls finder
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at bound ⊢
  linarith

/-- The refresh knowledge bound at the bounded-depth scheme's key width with numbers for the
statistical term and, under `[P2-PERM]` and `[SPONGE]`, for the collision finder. -/
theorem refresh_production_knowledge_error_lt_numeric
    (adversary : (SetupIndex Width Fits → Stream) →
      OracleComp (Point (RefreshSchedule.schedule Width Fits) 8) Answer (Attempt Width Fits))
    {queries : Nat} (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ}
    (positive : 0 < budget) (blocks calls : ℕ)
    (finder : Sponge.Transfer Sponge.CollisionConditions
      (fun _ : Unit => refreshHashAdvantage key adversary budget)
      (fun model : Sponge.CollisionAdversary blocks calls => Sponge.collisionChance model) calls) :
    𝔼 chunks, refreshSuccess key adversary chunks <
      𝔼 chunks, refreshExtraction key adversary chunks + ((queries : ℝ) + 1) * (4748 / 2 ^ 128) +
        refreshSolverAdvantage key adversary budget +
        16 * (Sponge.birthdayBound calls + Sponge.spongeError calls) +
        (77 * queries + 1286) / budget + 1 / 2 ^ 1129 := by
  have bound := refresh_production_knowledge_error_lt_solvers key adversary bounded positive
  have statistical := RefreshKnowledge.statisticalError_le queries
  have hash := refresh_hashAdvantage_le key adversary budget blocks calls finder
  generalize (1 : ℝ) / 2 ^ 1129 = setupError at bound ⊢
  linarith

end Production

end NightstreamFPrime.Export.BoundedDepth.RandomOracleSetup
