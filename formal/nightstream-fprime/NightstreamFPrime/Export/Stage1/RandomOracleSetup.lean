import NightstreamFPrime.Export.Stage1.RandomOracleLink
import NightstreamFPrime.Lifecycle.RandomOracleSolver
import NightstreamFPrime.Export.Stage1.PerApplicationFixedPoint
import NightstreamFPrime.Spec.KnowledgeContract

/-!
Owns the one-fold knowledge theorem with the Ajtai key drawn inside the game:
the key's coefficients are the reductions of uniform 256-bit setup chunks
(`AjtaiSetupV1.Programming`, premise P1: SHAKE128 as a random oracle), and the
adversary is chosen before the chunks, so it may read the whole key but cannot
contain a kernel vector of it.

Inputs: an adversary for each value of the setup chunks, its claim and prior
preimage, and the verifier-owned context digest. The digest is computed from
the public setup authority (seed and dimensions), so it does not depend on the
chunks.

Outputs: `setupKey`, `msisAdvantage`, `knowledge_error_le_setup`,
`solverAdvantage`, `msisAdvantage_le_solver`, `production_knowledge_error_lt`,
`randomized_knowledge_error_lt` and `contract`.

Does not own: MSIS hardness, the hardness of the state hash, machine work
beyond counted adversary runs, or SHAKE128 itself.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.RandomOracleSetup

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.RandomOracle
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.AjtaiSetupV1.Programming
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.TranscriptCoverage
open NightstreamFPrime.Lifecycle.RandomOracleTest
open NightstreamFPrime.Lifecycle.RandomOracleExtraction
open NightstreamFPrime.Lifecycle.RandomOracleUniqueness
open NightstreamFPrime.Lifecycle.RandomOracleKnowledge
open NightstreamFPrime.Lifecycle.RandomOracleBinding
open NightstreamFPrime.Export.Stage1.RandomOracleLink
open StrongReduction ConcreteCarrier

attribute [local instance low] Classical.propDecidable

abbrev SetupIndex (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  KeyIndex productionProfile.commitmentWidth
    (Phi81ColumnLayout.blockCount (FullShape logicalWidth publicFits).carrierWidth)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

def setupKey (chunks : SetupIndex logicalWidth publicFits → Chunk) :
    AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  keyOf (residues chunks)

variable (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  {Output : Type}
  (adversary : (SetupIndex logicalWidth publicFits → Chunk) →
    OracleComp (Point logicalWidth publicFits (ProductionKey.degreeBound relation)) Answer Output)
  (claim : Output → Claim relation)
  (prior : Output → Lifecycle.HashPreimage (logicalWidth := logicalWidth) (publicFits := publicFits))
  (contextDigest : KeyDigest)

local notation "Linked" => linkedClaim relation claim prior contextDigest
local notation "Oracle" => Point logicalWidth publicFits (ProductionKey.degreeBound relation) → Answer
local notation "Retries" => Fin Nifs.PaperProfile.arity.total → Answer

noncomputable def msisAdvantage : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
    kernelChance relation (keyOf matrix) (adversary chunks) Linked

noncomputable def extraction (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, ∑ retries, weight relation (setupKey chunks) (adversary chunks) Linked oracle retries *
    (if Extracts relation (setupKey chunks) (adversary chunks) Linked oracle retries then 1 else 0)

noncomputable def hashCollisions (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, (if Succeeds relation (setupKey chunks) (adversary chunks) Linked oracle then
      ∑ index, hashMismatchChance relation (setupKey chunks) (adversary chunks) claim prior
        contextDigest index oracle else 0) +
    hashRunningChance relation (setupKey chunks) (adversary chunks) claim prior contextDigest

theorem knowledge_error_le_setup {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk, 𝔼 oracle,
        (if Succeeds relation (setupKey chunks) (adversary chunks) Linked oracle then (1 : ℝ) else 0) ≤
      𝔼 chunks, (extraction relation adversary claim prior contextDigest chunks +
          hashCollisions relation adversary claim prior contextDigest chunks) +
        statisticalError queries + msisAdvantage relation adversary claim prior contextDigest +
        Fintype.card (SetupIndex logicalWidth publicFits) * (2 * 4294967295 / 2 ^ 256 : ℝ) := by
  have perKey (chunks : SetupIndex logicalWidth publicFits → Chunk) :
      𝔼 oracle, (if Succeeds relation (setupKey chunks) (adversary chunks) Linked oracle then (1 : ℝ)
          else 0) ≤
        (extraction relation adversary claim prior contextDigest chunks +
            hashCollisions relation adversary claim prior contextDigest chunks) +
          statisticalError queries + kernelChance relation (setupKey chunks) (adversary chunks) Linked := by
    have linked := knowledge_error_le_linked relation (setupKey chunks) (adversary chunks) claim prior
      contextDigest (bounded chunks)
    have kernel := collisionChance_le_kernelChance relation (setupKey chunks) (adversary chunks) Linked
    unfold linkedKnowledgeError at linked
    unfold extraction hashCollisions
    linarith
  have binding := expect_le_programmed
    (fun chunks => kernelChance relation (setupKey chunks) (adversary chunks) Linked)
    (fun chunks => reductionChance_nonnegative relation (setupKey chunks) (adversary chunks) Linked _)
    (fun chunks => reductionChance_le_one relation (setupKey chunks) (adversary chunks) Linked _)
  have programmed : (𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
      kernelChance relation (setupKey chunks) (adversary chunks) Linked) =
        msisAdvantage relation adversary claim prior contextDigest := by
    unfold msisAdvantage
    refine Finset.expect_congr rfl fun matrix _ => Finset.expect_congr rfl fun chunks member => ?_
    rw [setupKey, mem_preimages.mp member]
  rw [programmed] at binding
  calc _ ≤ 𝔼 chunks : SetupIndex logicalWidth publicFits → Chunk,
          ((extraction relation adversary claim prior contextDigest chunks +
              hashCollisions relation adversary claim prior contextDigest chunks) +
            statisticalError queries + kernelChance relation (setupKey chunks) (adversary chunks) Linked) :=
        Finset.expect_le_expect fun chunks _ => perKey chunks
    _ = _ := by
        rw [Finset.expect_add_distrib, Finset.expect_add_distrib,
          Finset.expect_const Finset.univ_nonempty]
    _ ≤ _ := by linarith

noncomputable def solverAdvantage (budget : ℕ) : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
    RandomOracleSolver.solverChance relation (keyOf matrix) (adversary chunks) Linked budget

theorem msisAdvantage_le_solver {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) {budget : ℕ} (positive : 0 < budget) :
    msisAdvantage relation adversary claim prior contextDigest ≤
      solverAdvantage relation adversary claim prior contextDigest budget +
        (35 * queries + 653) / budget := by
  have positiveReal : (0 : ℝ) < budget := by exact_mod_cast positive
  have loss : 0 ≤ ((35 * queries + 653 : ℝ)) / budget := by positivity
  have each (matrix : SetupIndex logicalWidth publicFits → F) (chunks : SetupIndex logicalWidth publicFits → Chunk) :
      kernelChance relation (keyOf matrix) (adversary chunks) Linked ≤
        RandomOracleSolver.solverChance relation (keyOf matrix) (adversary chunks) Linked budget +
          (35 * queries + 653) / budget := by
    have bound := RandomOracleSolver.kernelChance_le_solver relation (keyOf matrix) (adversary chunks) Linked
      (bounded chunks) budget
    rw [← sub_le_iff_le_add', le_div_iff₀ positiveReal, mul_comm]
    exact bound
  unfold msisAdvantage solverAdvantage
  calc
    _ ≤ 𝔼 matrix : SetupIndex logicalWidth publicFits → F, ((𝔼 chunks ∈ preimages matrix,
          RandomOracleSolver.solverChance relation (keyOf matrix) (adversary chunks) Linked budget) +
            (35 * queries + 653) / budget) :=
      Finset.expect_le_expect fun matrix _ => by
        rcases (preimages matrix).eq_empty_or_nonempty with empty | nonempty
        · simp [empty, loss]
        · calc
            _ ≤ 𝔼 chunks ∈ preimages matrix,
                  (RandomOracleSolver.solverChance relation (keyOf matrix) (adversary chunks) Linked budget +
                    (35 * queries + 653) / budget) :=
              Finset.expect_le_expect fun chunks _ => each matrix chunks
            _ = _ := by rw [Finset.expect_add_distrib, Finset.expect_const nonempty]
    _ = _ := by rw [Finset.expect_add_distrib, Finset.expect_const Finset.univ_nonempty]

local notation "Chunks" => SetupIndex logicalWidth publicFits → Chunk

private theorem contract_weight_sum :
    ∑ run : Chunks × (Oracle × Retries),
      runWeight relation (setupKey run.1) (adversary run.1) Linked run.2 /
        Fintype.card Chunks = 1 := by
  have positive : (0 : ℝ) < Fintype.card Chunks := by exact_mod_cast Fintype.card_pos
  have each (chunks : Chunks) :
      ∑ run : Oracle × Retries,
          runWeight relation (setupKey chunks) (adversary chunks) Linked run / Fintype.card Chunks =
        1 / Fintype.card Chunks := by
    rw [← Finset.sum_div, runWeight_sum]
  rw [Fintype.sum_prod_type]
  simp only [each]
  rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one_div_cancel positive.ne']

private theorem contract_failure_sum :
    ∑ run : Chunks × (Oracle × Retries),
        runWeight relation (setupKey run.1) (adversary run.1) Linked run.2 / Fintype.card Chunks *
          (if Succeeds relation (setupKey run.1) (adversary run.1) Linked run.2.1 ∧
              extract relation (setupKey run.1) (adversary run.1) Linked run.2.1 run.2.2 = none
            then 1 else 0) =
      𝔼 chunks : Chunks, (𝔼 oracle, (if Succeeds relation (setupKey chunks) (adversary chunks)
          Linked oracle then (1 : ℝ) else 0) -
        extraction relation adversary claim prior contextDigest chunks) := by
  have perChunks (chunks : Chunks) :
      ∑ run : Oracle × Retries,
        runWeight relation (setupKey chunks) (adversary chunks) Linked run / Fintype.card Chunks *
          (if Succeeds relation (setupKey chunks) (adversary chunks) Linked run.1 ∧
              extract relation (setupKey chunks) (adversary chunks) Linked run.1 run.2 = none
            then 1 else 0) =
        (𝔼 oracle, (if Succeeds relation (setupKey chunks) (adversary chunks) Linked oracle
            then (1 : ℝ) else 0) -
          extraction relation adversary claim prior contextDigest chunks) / Fintype.card Chunks := by
    unfold extraction
    rw [← failure_eq relation (setupKey chunks) (adversary chunks) Linked, Finset.sum_div]
    refine Finset.sum_congr rfl fun run _ => ?_
    ring
  calc _ = ∑ chunks : Chunks, ∑ run : Oracle × Retries,
          runWeight relation (setupKey chunks) (adversary chunks) Linked run / Fintype.card Chunks *
            (if Succeeds relation (setupKey chunks) (adversary chunks) Linked run.1 ∧
                extract relation (setupKey chunks) (adversary chunks) Linked run.1 run.2 = none
              then 1 else 0) :=
        Fintype.sum_prod_type _
    _ = _ := by
      rw [Finset.sum_congr rfl fun chunks _ => perChunks chunks, Finset.expect_eq_sum_div_card,
        Finset.card_univ, Finset.sum_div]

noncomputable def contract {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) : KnowledgeContract where
  Run := Chunks × (Oracle × Retries)
  weight run := runWeight relation (setupKey run.1) (adversary run.1) Linked run.2 /
    Fintype.card Chunks
  weight_nonnegative run :=
    div_nonneg (runWeight_nonnegative relation (setupKey run.1) (adversary run.1) Linked run.2)
      (Nat.cast_nonneg _)
  weight_sum := contract_weight_sum relation adversary claim prior contextDigest
  Statement := AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits) ×
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) ×
    Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)
  statement run := (setupKey run.1,
    (claimed relation (adversary run.1) Linked run.2.1).running,
    (claimed relation (adversary run.1) Linked run.2.1).fresh)
  accepts run := Succeeds relation (setupKey run.1) (adversary run.1) Linked run.2.1
  Witness := OutputWitness productionShape (Phi81CarrierLayout.carrierWidth logicalWidth)
  extract run := extract relation (setupKey run.1) (adversary run.1) Linked run.2.1 run.2.2
  Holds statement witness := SourceHolds extensionOps K.embed (PaperAlgebra.openingMaps statement.1)
    productionGlobalParams ((ProductionKey.key relation statement.1).statement statement.2.1 statement.2.2)
    witness
  witness_statement := by
    rintro ⟨chunks, oracle, retries⟩ witness returned
    exact extract_holds relation (setupKey chunks) (adversary chunks) Linked returned
  error := 𝔼 chunks : Chunks, hashCollisions relation adversary claim prior contextDigest chunks +
    statisticalError queries + msisAdvantage relation adversary claim prior contextDigest +
    Fintype.card (SetupIndex logicalWidth publicFits) * (2 * 4294967295 / 2 ^ 256 : ℝ)
  knowledge_sound := by
    have setup := knowledge_error_le_setup relation adversary claim prior contextDigest bounded
    rw [Finset.expect_add_distrib] at setup
    rw [contract_failure_sum, Finset.expect_sub_distrib]
    linarith

section Production

variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)

local notation "Width" => PerApplicationFixedPoint.logicalWidth application
local notation "Fits" => PerApplicationFixedPoint.publicFits application
local notation "Selected" => PerApplicationFixedPoint.relation application fits

theorem programmingError_lt_of_fits (fits : PerApplicationFixedPoint.FitsTwoPow28 application) :
    (Fintype.card (SetupIndex Width Fits) : ℝ) * (2 * 4294967295 / 2 ^ 256) < 1 / 2 ^ 190 := by
  have blocks : Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth Width) ≤ 4971027 := by
    have carrier := fits.carrier
    unfold Phi81ColumnLayout.blockCount
    calc _ ≤ (2 ^ Lifecycle.cubeVariables + ringDegree - 1) / ringDegree :=
          Nat.div_le_div_right (by omega)
      _ = 4971027 := by norm_num [Lifecycle.cubeVariables, ringDegree]
  have card : (Fintype.card (SetupIndex Width Fits) : ℝ) ≤ 22 * (4971027 * 54) := by
    show (Fintype.card (KeyIndex productionProfile.commitmentWidth
      (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth Width))) : ℝ) ≤ _
    rw [Fintype.card_prod, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin, Fintype.card_fin]
    have rows : productionProfile.commitmentWidth = 22 := rfl
    rw [rows, show ringDegree = 54 from rfl]
    exact_mod_cast Nat.mul_le_mul_left 22 (Nat.mul_le_mul_right 54 blocks)
  calc _ ≤ (22 * (4971027 * 54) : ℝ) * (2 * 4294967295 / 2 ^ 256) :=
        mul_le_mul_of_nonneg_right card (by positivity)
    _ < 1 / 2 ^ 190 := by norm_num

theorem production_knowledge_error_lt {Output : Type}
    (adversary : (SetupIndex Width Fits → Chunk) →
      OracleComp (Point Width Fits (ProductionKey.degreeBound Selected)) Answer Output)
    (claim : Output → Claim Selected)
    (prior : Output → Lifecycle.HashPreimage (logicalWidth := Width) (publicFits := Fits))
    (contextDigest : KeyDigest) {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) :
    𝔼 chunks : SetupIndex Width Fits → Chunk, 𝔼 oracle,
        (if Succeeds Selected (setupKey chunks) (adversary chunks)
            (linkedClaim Selected claim prior contextDigest) oracle then (1 : ℝ) else 0) <
      𝔼 chunks, (extraction Selected adversary claim prior contextDigest chunks +
          hashCollisions Selected adversary claim prior contextDigest chunks) +
        statisticalError queries +
        msisAdvantage Selected adversary claim prior contextDigest + 1 / 2 ^ 190 := by
  have bound := knowledge_error_le_setup Selected adversary claim prior contextDigest bounded
  have small := programmingError_lt_of_fits application fits
  linarith

theorem randomized_knowledge_error_lt {Coins Output : Type} [Fintype Coins] [Nonempty Coins]
    (adversary : Coins → (SetupIndex Width Fits → Chunk) →
      OracleComp (Point Width Fits (ProductionKey.degreeBound Selected)) Answer Output)
    (claim : Output → Claim Selected)
    (prior : Output → Lifecycle.HashPreimage (logicalWidth := Width) (publicFits := Fits))
    (contextDigest : KeyDigest) {queries : Nat}
    (bounded : ∀ coins chunks, (adversary coins chunks).QueryBound queries) :
    𝔼 coins, 𝔼 chunks : SetupIndex Width Fits → Chunk, 𝔼 oracle,
        (if Succeeds Selected (setupKey chunks) (adversary coins chunks)
            (linkedClaim Selected claim prior contextDigest) oracle then (1 : ℝ) else 0) <
      𝔼 coins, 𝔼 chunks, (extraction Selected (adversary coins) claim prior contextDigest chunks +
          hashCollisions Selected (adversary coins) claim prior contextDigest chunks) +
        statisticalError queries +
        𝔼 coins, msisAdvantage Selected (adversary coins) claim prior contextDigest + 1 / 2 ^ 190 := by
  have each (coins : Coins) := production_knowledge_error_lt application fits (adversary coins) claim
    prior contextDigest (bounded coins)
  calc _ < 𝔼 coins, (𝔼 chunks, (extraction Selected (adversary coins) claim prior contextDigest chunks +
          hashCollisions Selected (adversary coins) claim prior contextDigest chunks) +
        statisticalError queries + msisAdvantage Selected (adversary coins) claim prior contextDigest +
        1 / 2 ^ 190) :=
        Finset.expect_lt_expect (fun coins _ => (each coins).le)
          (Finset.univ_nonempty.elim fun coins member => ⟨coins, member, each coins⟩)
    _ = _ := by
        rw [Finset.expect_add_distrib, Finset.expect_add_distrib, Finset.expect_add_distrib,
          Finset.expect_const Finset.univ_nonempty, Finset.expect_const Finset.univ_nonempty]

end Production

end NightstreamFPrime.Export.Stage1.RandomOracleSetup
