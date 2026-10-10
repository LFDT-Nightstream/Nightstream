import NightstreamFPrime.Export.Stage1.RandomOracleLink
import NightstreamFPrime.Export.Stage1.SetupDistribution
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

Outputs:
- `setupKey`: the key of the chunks;
- `msisAdvantage`: the success of an MSIS solver for a uniform matrix: it draws
  chunks uniformly among the matrix's preimages, and runs the binding reduction
  (`RandomOracleBinding.rerunKernel`) on the adversary for those chunks, under
  the matrix's key. Its output is a short nonzero kernel vector of that key;
- `knowledge_error_le_setup`: averaged over the chunks, the linked knowledge
  bound with the binding term replaced by `msisAdvantage` plus the programming
  error;
- `production_knowledge_error_lt`: the same for every application that fits
  the `2 ^ 28` profile, at its own relation and key size, where the
  programming error is below `2 ^ -190` (`programmingError_lt_of_fits`);
- `contract`: the theorem as a `Spec.KnowledgeContract`, whose runs are the
  chunks, the oracle and the extractor's retries.

Does not own: MSIS hardness, the hardness of the state hash, the running time
of the solver, or SHAKE128 itself.
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

/-- The key coordinates `(row, block, lane)`: one setup chunk for each. -/
abbrev SetupIndex (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth) :=
  KeyIndex productionProfile.commitmentWidth
    (Phi81ColumnLayout.blockCount (FullShape logicalWidth publicFits).carrierWidth)

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}

/-- The key whose coefficients are the residues of the chunks. -/
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

/-- The MSIS solver's success on a uniform matrix: chunks uniform among the
matrix's preimages, then the binding reduction's chance on the adversary for
those chunks under the matrix's key (`RandomOracleBinding.kernelChance`). -/
noncomputable def msisAdvantage : ℝ :=
  𝔼 matrix : SetupIndex logicalWidth publicFits → F, 𝔼 chunks ∈ preimages matrix,
    kernelChance relation (keyOf matrix) (adversary chunks) Linked

/-- The extractor's success for the chunks' key. -/
noncomputable def extraction (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, ∑ retries, weight relation (setupKey chunks) (adversary chunks) Linked oracle retries *
    (if Extracts relation (setupKey chunks) (adversary chunks) Linked oracle retries then 1 else 0)

/-- The state-hash collision chances of the retries and reruns for the chunks'
key (`RandomOracleLink.hashMismatchChance`, `RandomOracleLink.hashRunningChance`). -/
noncomputable def hashCollisions (chunks : SetupIndex logicalWidth publicFits → Chunk) : ℝ :=
  𝔼 oracle, (if Succeeds relation (setupKey chunks) (adversary chunks) Linked oracle then
      ∑ index, hashMismatchChance relation (setupKey chunks) (adversary chunks) claim prior
        contextDigest index oracle else 0) +
    hashRunningChance relation (setupKey chunks) (adversary chunks) claim prior contextDigest

/-- **One-fold knowledge soundness with the key in the game.** Over uniform
setup chunks and the Fiat–Shamir oracle, the verifier accepts the linked claim
at most as often as the extractor succeeds, plus the state-hash collision
chances, the statistical error, the MSIS solver's success and the programming
error. -/
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
    (fun chunks => kernelChance_nonnegative relation (setupKey chunks) (adversary chunks) Linked)
    (fun chunks => kernelChance_le_one relation (setupKey chunks) (adversary chunks) Linked)
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

/-! ## The knowledge contract -/

local notation "Chunks" => SetupIndex logicalWidth publicFits → Chunk

/-- The knowledge contract of the production NIFS in the random-oracle model,
with the key drawn from the setup chunks and the prior-state link, for an
adversary with at most `queries` oracle queries. A run is the chunks, the
oracle and the extractor's retries; the statement includes the chunks' key. -/
noncomputable def contract {queries : Nat}
    (bounded : ∀ chunks, (adversary chunks).QueryBound queries) : KnowledgeContract where
  Run := Chunks × (Oracle × Retries)
  weight run := runWeight relation (setupKey run.1) (adversary run.1) Linked run.2 /
    Fintype.card Chunks
  weight_nonnegative run :=
    div_nonneg (runWeight_nonnegative relation (setupKey run.1) (adversary run.1) Linked run.2)
      (Nat.cast_nonneg _)
  weight_sum := by
    have positive : (0 : ℝ) < Fintype.card Chunks := by exact_mod_cast Fintype.card_pos
    have each (chunks : Chunks) :
        ∑ run : Oracle × Retries,
            runWeight relation (setupKey chunks) (adversary chunks) Linked run / Fintype.card Chunks =
          1 / Fintype.card Chunks := by
      rw [← Finset.sum_div, runWeight_sum]
    calc _ = ∑ chunks : Chunks, ∑ run : Oracle × Retries,
            runWeight relation (setupKey chunks) (adversary chunks) Linked run / Fintype.card Chunks :=
          Fintype.sum_prod_type _
      _ = ∑ _chunks : Chunks, 1 / (Fintype.card Chunks : ℝ) := Finset.sum_congr rfl fun chunks _ => each chunks
      _ = 1 := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
        field_simp
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
    have average :
        ∑ chunks : Chunks, (𝔼 oracle, (if Succeeds relation (setupKey chunks) (adversary chunks) Linked
            oracle then (1 : ℝ) else 0) -
          extraction relation adversary claim prior contextDigest chunks) / Fintype.card Chunks =
        𝔼 chunks : Chunks, (𝔼 oracle, (if Succeeds relation (setupKey chunks) (adversary chunks)
            Linked oracle then (1 : ℝ) else 0) -
          extraction relation adversary claim prior contextDigest chunks) := by
      rw [Finset.expect_eq_sum_div_card, Finset.card_univ, Finset.sum_div]
    rw [Finset.expect_add_distrib] at setup
    calc _ = ∑ chunks : Chunks, ∑ run : Oracle × Retries,
            runWeight relation (setupKey chunks) (adversary chunks) Linked run / Fintype.card Chunks *
              (if Succeeds relation (setupKey chunks) (adversary chunks) Linked run.1 ∧
                  extract relation (setupKey chunks) (adversary chunks) Linked run.1 run.2 = none
                then 1 else 0) :=
          Fintype.sum_prod_type _
      _ = _ := (Finset.sum_congr rfl fun chunks _ => perChunks chunks).trans average
      _ ≤ _ := by
        rw [Finset.expect_sub_distrib]
        linarith

/-! ## Every application of the profile -/

section Production

variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)

local notation "Width" => PerApplicationFixedPoint.logicalWidth application
local notation "Fits" => PerApplicationFixedPoint.publicFits application
local notation "Selected" => PerApplicationFixedPoint.relation application fits

/-- The `2 ^ 28` profile bounds every key by `22 · 54 · ⌈2 ^ 28 / 54⌉`
coefficients, so its programming error is below `2 ^ -190`. The exponent is
derived from that bound; it is not a target for cryptographic security. -/
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

/-- The setup-game theorem for every application that fits the `2 ^ 28`
profile, at its own relation and key size: the programming error is below
`2 ^ -190` (`programmingError_lt_of_fits`). -/
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

end Production

end NightstreamFPrime.Export.Stage1.RandomOracleSetup
