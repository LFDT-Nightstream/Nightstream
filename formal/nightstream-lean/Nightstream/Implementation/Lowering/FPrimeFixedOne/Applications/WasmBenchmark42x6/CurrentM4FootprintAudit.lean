import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost

/-!
Contract: exact component census for the selected 42-times-6 benchmark NIFS
footprint.

Assurance tier: model-level.

Owns: the field counts absorbed by the ΠCCS transcript, its exact Poseidon2
permutation count, the intrinsic verifier component costs, and the activation
overhead that produces the final selected-NIFS footprint.

Does not own: a production budget, an MSIS security judgment, Rust equality,
or a claim that this reduced benchmark is a production deployment.

Emits constraints: none. It decomposes the cost of existing rows.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4Cost
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier

private theorem cost_ext
    {left right : Cost}
    (rows : left.recurringRows = right.recurringRows)
    (committed : left.committedColumns = right.committedColumns)
    (publicEq : left.publicColumns = right.publicColumns)
    (auxiliary : left.auxiliaryColumns = right.auxiliaryColumns) :
    left = right := by
  cases left
  cases right
  simp_all

/-- Degree of the selected FE row polynomial. -/
def rowDegree : Nat :=
  SumCheck.Fe.Drow
    (KSplitNcStaticInput.layoutInput
      (shape := shape) Semantics.polynomial)

/-- Number of authoritative field coordinates absorbed before ΠCCS. -/
noncomputable def statementFields : Nat :=
  ConcreteNifsStaticFootprint.statementFieldCount
    shape publicRingColumns commitmentRows publicFits

/-- Number of selected output field coordinates absorbed after ΠCCS. -/
def outputFields : Nat :=
  ConcreteNifsStaticFootprint.outputFieldCount shape

/-- Compact transcript control state after the selected output is absorbed. -/
noncomputable def transcriptControl : SymbolicDuplexCount.Control :=
  KSplitNcTranscriptCount.afterOutput shape.rowVariables rowDegree
    PiCcsDomains.production.laneVariables
    PiCcsDomains.production.blockVariables
    0 statementFields outputFields

/-- Intrinsic Poseidon2 cost of the complete ΠCCS transcript. -/
noncomputable def transcriptCost : Cost :=
  KSplitNcTranscriptCount.cost shape.rowVariables rowDegree
    PiCcsDomains.production.laneVariables
    PiCcsDomains.production.blockVariables
    0 statementFields outputFields

/-- Intrinsic cost of the block/lane numeric bridge. -/
noncomputable def blockLaneCost : Cost :=
  KSplitNcBlockLaneRows.cost
    (KSplitNcTranscript.numericColumns
      (ConcreteNifsStaticFootprint.compactTranscriptInput
        shape Semantics.polynomial publicRingColumns commitmentRows publicFits))

/-- Intrinsic cost of binding the selected ΠCCS endpoint values. -/
noncomputable def endpointCost : Cost :=
  KSplitNcOperationalRows.endpointCost
    (ConcreteNifsStaticFootprint.compactOperationalInput
      shape Semantics.polynomial publicRingColumns commitmentRows publicFits)

/-- Intrinsic cost of the canonical ΠRLC action. -/
def piRlcActionCost : Cost :=
  ConcreteNifsPiRlcActionRows.cost shape publicRingColumns commitmentRows

theorem rowDegree_exact : rowDegree = 9 := by
  rfl

theorem statementFields_exact : statementFields = 61_398 := by
  rfl

theorem outputFields_exact : outputFields = 22_683 := by
  rfl

theorem transcriptPermutationCount_exact :
    transcriptControl.entries = 21_349 := by
  rfl

theorem transcriptCost_exact :
    transcriptCost = ⟨7_514_848, 0, 0, 7_514_848⟩ := by
  apply cost_ext <;> rfl

theorem blockLaneCost_exact :
    blockLaneCost = ⟨1_129, 0, 0, 1_011⟩ := by
  apply cost_ext <;> rfl

theorem endpointCost_exact :
    endpointCost = ⟨74_389, 0, 0, 74_381⟩ := by
  apply cost_ext <;> rfl

theorem compactOperationalCost_exact :
    ConcreteNifsStaticFootprint.compactOperationalCost
        shape Semantics.polynomial publicRingColumns commitmentRows
          publicFits =
      ⟨7_590_366, 0, 0, 7_590_240⟩ := by
  unfold ConcreteNifsStaticFootprint.compactOperationalCost
  change transcriptCost + blockLaneCost + endpointCost =
    ⟨7_590_366, 0, 0, 7_590_240⟩
  rw [transcriptCost_exact, blockLaneCost_exact, endpointCost_exact]
  rfl

theorem piRlcSamplerCost_exact :
    PiRlcCanonicalSamplerProgram.cost =
      ⟨105_930, 0, 0, 99_885⟩ := by
  apply cost_ext <;> rfl

theorem piRlcChallengeCost_exact :
    ConcreteNifsOperationalSampler.challengeCost =
      ⟨810, 0, 0, 0⟩ := by
  apply cost_ext <;> rfl

theorem samplerCost_exact :
    ConcreteNifsStaticFootprint.samplerCost
        shape Semantics.polynomial publicRingColumns commitmentRows
          publicFits =
      ⟨7_697_106, 0, 0, 7_690_125⟩ := by
  unfold ConcreteNifsStaticFootprint.samplerCost
  rw [compactOperationalCost_exact, piRlcSamplerCost_exact,
    piRlcChallengeCost_exact]
  rfl

theorem piRlcActionCost_exact :
    piRlcActionCost = ⟨2_145_906, 0, 0, 2_143_260⟩ := by
  apply cost_ext <;> rfl

theorem intrinsicCost_exact :
    ConcreteNifsStaticFootprint.intrinsicCost
        shape Semantics.polynomial publicRingColumns commitmentRows
          publicFits =
      ⟨9_886_806, 0, 0, 9_833_395⟩ := by
  unfold ConcreteNifsStaticFootprint.intrinsicCost
  rw [samplerCost_exact]
  apply cost_ext <;> rfl

/-- Activation adds one residual row and one residual auxiliary column for
each intrinsic verifier row. -/
theorem activationOverhead_exact :
    ActivatedRawProgram.overheadCost 9_886_806 =
      ⟨9_886_806, 0, 0, 9_886_806⟩ := by
  rfl

theorem selectedNifsCost_exact :
    ConcreteNifsStaticFootprint.cost
        shape Semantics.polynomial publicRingColumns commitmentRows
          publicFits =
      ⟨19_773_612, 0, 0, 19_720_201⟩ := by
  unfold ConcreteNifsStaticFootprint.cost
  rw [intrinsicCost_exact]
  rfl

/-- The transcript is the largest intrinsic row family. -/
theorem transcript_dominates_piRlcAction :
    piRlcActionCost.recurringRows < transcriptCost.recurringRows := by
  rw [piRlcActionCost_exact, transcriptCost_exact]
  decide

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4FootprintAudit
