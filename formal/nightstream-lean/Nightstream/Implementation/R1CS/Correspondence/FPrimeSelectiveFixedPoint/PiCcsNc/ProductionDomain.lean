import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsDomains

/-!
Exact stabilized fixed-point candidate dimensions and canonical `Pi_CCS` domains.

Assurance tier: artifact-checked for the compiler candidate's row, column, and
committed-width accounting;
model-level for the stated source arities and the arithmetic consequences.

Owns: the intended production semantic shape, equality of its logical width
with the stable fixed-point compiler candidate, exact carrier/block sizes,
coverage by the canonical flat and block×lane domains, and minimality of each
binary cube.

Does not own: materialization of the candidate as a production
relation, assignment decoding, generated delayed-projection rows, NC
acceptance, transcript scheduling, commitment binding, Rust dataflow, or
permission to raise the constructor guard or remove rows.

Emits constraints: no.

The width certificate comes from the exact stabilized compiler audit without
allocating the complete matrices. The source arities below are the explicit
intended production semantic profile; this leaf does not infer them from a
digest or from a carried claim.

| Stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.fixed_point.pi_ccs.domain.artifact` | stabilized candidate has 14,944,219 rows and 11,437,038 aligned logical columns | checked |
| `f_prime.fixed_point.pi_ccs.width.artifact` | 311-coordinate prefix plus the 11,436,699-coordinate maximum arm rounds from 11,437,010 to 11,437,038 | checked |
| `f_prime.fixed_point.pi_ccs.domain.shape` | production uses row arity 24, one fresh source, fourteen running sources, and thirteen matrices | computed profile |
| `f_prime.fixed_point.pi_ccs.domain.flat` | 24 column variables and 6 lane variables cover the completed carrier | derived |
| `f_prime.fixed_point.pi_ccs.domain.block_lane` | the canonical 19 block variables and 6 lane variables cover 211,797 live blocks by 54 live lanes; 18 block variables are sufficient and minimal for this artifact | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

namespace FixedArtifact

def relationRows : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.G.relationRows

def relationColumns : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.G.physicalCoordinates

def unpaddedCoordinates : Nat :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.G.unpaddedCoordinates

end FixedArtifact

/-- Binary row cube of the fixed-point relation. -/
def rowVariables : Nat := 24

/-- One fresh source is folded against the running batch. -/
def freshCount : Nat := 1

/-- Fixed active fan-in of the running batch. -/
def runningCount : Nat := 14

/-- Number of coefficient-matrix ports in the fixed-point relation. -/
def matrixCount : Nat := 13

/-- Exact production semantic shape. Its logical width is read from the stable
artifact facade rather than duplicated as an independent semantic literal. -/
def semanticShape : SemanticShape where
  rowVariables := rowVariables
  logicalWidth := FixedArtifact.relationColumns
  freshCount := freshCount
  runningCount := runningCount
  matrixCount := matrixCount

/-! ## Artifact and shape identities -/

@[simp] theorem artifact_relationRows : FixedArtifact.relationRows = 14944219 := by
  rfl

@[simp] theorem artifact_relationColumns :
    FixedArtifact.relationColumns = 11437038 := by
  rfl

@[simp] theorem artifact_unpaddedCoordinates :
    FixedArtifact.unpaddedCoordinates = 11437010 := by
  rfl

/-- Exact generated prefix/max/round-up identity. This is candidate accounting,
not evidence that the guarded constructor materialized the complete relation. -/
theorem artifact_width_accounting :
    FixedArtifact.unpaddedCoordinates = 311 + 11436699 /\
      FixedArtifact.relationColumns =
        FixedArtifact.unpaddedCoordinates + 28 := by
  exact ⟨
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.unpadded_accounting,
    Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.physical_eq_unpadded_add_padding⟩

/-- The stabilized candidate fits the current guarded materializer's width
budget. This accounting theorem does not claim that complete matrices were
successfully emitted. -/
theorem artifact_fits_current_constructor_guard :
    FixedArtifact.relationColumns <= 16000000 /\
      16000000 - FixedArtifact.relationColumns = 4562962 := by
  exact Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain.WidthCensus.fits_current_constructor_guard

@[simp] theorem semanticShape_rowVariables :
    semanticShape.rowVariables = 24 := by
  rfl

@[simp] theorem semanticShape_logicalWidth :
    semanticShape.logicalWidth = FixedArtifact.relationColumns := by
  rfl

theorem semanticShape_logicalWidth_exact :
    semanticShape.logicalWidth = 11437038 := by
  rw [semanticShape_logicalWidth, artifact_relationColumns]

@[simp] theorem semanticShape_freshCount :
    semanticShape.freshCount = 1 := by
  rfl

@[simp] theorem semanticShape_runningCount :
    semanticShape.runningCount = 14 := by
  rfl

@[simp] theorem semanticShape_matrixCount :
    semanticShape.matrixCount = 13 := by
  rfl

@[simp] theorem semanticShape_sourceCount :
    semanticShape.sourceCount = 15 := by
  rfl

/-- The selected row cube covers the complete artifact row interval. -/
theorem rowCube_covers :
    FixedArtifact.relationRows <= 2 ^ semanticShape.rowVariables := by
  decide

/-- The artifact width is already a whole number of Phi81 blocks, so carrier
completion adds no columns. -/
@[simp] theorem semanticShape_carrierWidth :
    semanticShape.carrierWidth = 11437038 := by
  decide

theorem carrierWidth_eq_artifact_relationColumns :
    semanticShape.carrierWidth = FixedArtifact.relationColumns := by
  rw [semanticShape_carrierWidth, artifact_relationColumns]

/-- Exact number of live Phi81 blocks in the completed production carrier. -/
@[simp] theorem semanticShape_blockCount :
    Phi81ColumnLayout.blockCount semanticShape.carrierWidth = 211797 := by
  decide

/-! ## Canonical domain coverage and round counts -/

/-- The legacy flat-column view covers every completed carrier coordinate and
every live Phi81 lane. -/
theorem flatDomain_covers : PiCcsDomains.production.fe.Covers semanticShape := by
  constructor <;> decide

/-- The canonical block×lane view covers every live Phi81 block and lane. -/
theorem blockLaneDomain_covers : PiCcsDomains.production.nc.Covers semanticShape := by
  constructor <;> decide

/-- Canonical block×lane NC uses 19 block rounds followed by 6 lane rounds. -/
theorem blockLaneRoundCount :
    PiCcsDomains.production.nc.blockVariables +
        PiCcsDomains.production.nc.laneVariables = 25 := by
  exact PiCcsDomains.fixedPointProduction_blockRoundCount

/-- The legacy flat arithmetization would use 24 column rounds followed by 6
lane rounds. This is a dimension comparison, not an implementation claim. -/
theorem legacyFlatRoundCount :
    PiCcsDomains.production.fe.columnVariables +
        PiCcsDomains.production.fe.laneVariables = 30 := by
  exact PiCcsDomains.fixedPointProduction_flatRoundCount

/-- Number of semantically live lanes in each Phi81 block. -/
def liveLaneCount : Nat := ringDegree

/-- Number of verifier-computed virtual zero lanes in the 64-lane cube. -/
def virtualLaneCount : Nat :=
  PiCcsDomains.production.nc.laneCount - liveLaneCount

@[simp] theorem liveLaneCount_exact : liveLaneCount = 54 := by
  rfl

@[simp] theorem virtualLaneCount_exact : virtualLaneCount = 10 := by
  decide

theorem live_add_virtual_lanes :
    liveLaneCount + virtualLaneCount =
      PiCcsDomains.production.nc.laneCount := by
  decide

/-! ## Minimality -/

/-- Twenty-four row variables are necessary to cover every artifact row. -/
theorem rowVariables_minimal
    {variables : Nat}
    (covers : FixedArtifact.relationRows <= 2 ^ variables) :
    semanticShape.rowVariables <= variables := by
  rw [semanticShape_rowVariables]
  rcases Nat.lt_or_ge variables 24 with smaller | enough
  · have variablesLe : variables <= 23 := by omega
    have powerLe : 2 ^ variables <= 8388608 := by
      calc
        2 ^ variables <= 2 ^ 23 :=
          Nat.pow_le_pow_of_le (by decide) variablesLe
        _ = 8388608 := by decide
    rw [artifact_relationRows] at covers
    omega
  · exact enough

/-- Twenty-four flat-column variables are necessary to cover the complete
artifact-backed carrier. -/
theorem flatColumnVariables_minimal
    {variables : Nat}
    (covers : semanticShape.carrierWidth <= 2 ^ variables) :
    PiCcsDomains.production.fe.columnVariables <= variables := by
  change 24 <= variables
  rcases Nat.lt_or_ge variables 24 with smaller | enough
  · have variablesLe : variables <= 23 := by omega
    have powerLe : 2 ^ variables <= 8388608 := by
      calc
        2 ^ variables <= 2 ^ 23 :=
          Nat.pow_le_pow_of_le (by decide) variablesLe
        _ = 8388608 := by decide
    rw [semanticShape_carrierWidth] at covers
    omega
  · exact enough

/-- Eighteen block variables suffice to cover the 211,797 live Phi81 blocks. -/
theorem eighteenBlockVariables_cover :
    Phi81ColumnLayout.blockCount semanticShape.carrierWidth <= 2 ^ 18 := by
  rw [semanticShape_blockCount]
  decide

/-- Eighteen block variables are necessary to cover the 211,797 live Phi81
blocks. Together with `eighteenBlockVariables_cover`, this proves the exact
artifact minimum. The canonical production transcript intentionally retains
its 19-variable capacity. -/
theorem blockVariables_minimal
    {variables : Nat}
    (covers :
      Phi81ColumnLayout.blockCount semanticShape.carrierWidth <=
        2 ^ variables) :
    18 <= variables := by
  rcases Nat.lt_or_ge variables 18 with smaller | enough
  · have variablesLe : variables <= 17 := by omega
    have powerLe : 2 ^ variables <= 131072 := by
      calc
        2 ^ variables <= 2 ^ 17 :=
          Nat.pow_le_pow_of_le (by decide) variablesLe
        _ = 131072 := by decide
    rw [semanticShape_blockCount] at covers
    omega
  · exact enough

/-- Six lane variables are necessary to cover all 54 live coefficients. -/
theorem laneVariables_minimal
    {variables : Nat}
    (covers : ringDegree <= 2 ^ variables) :
    PiCcsDomains.production.nc.laneVariables <= variables := by
  change 6 <= variables
  rcases Nat.lt_or_ge variables 6 with smaller | enough
  · have variablesLe : variables <= 5 := by omega
    have powerLe : 2 ^ variables <= 32 := by
      calc
        2 ^ variables <= 2 ^ 5 :=
          Nat.pow_le_pow_of_le (by decide) variablesLe
        _ = 32 := by decide
    have degree : ringDegree = 54 := by rfl
    rw [degree] at covers
    omega
  · exact enough

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.ProductionDomain
