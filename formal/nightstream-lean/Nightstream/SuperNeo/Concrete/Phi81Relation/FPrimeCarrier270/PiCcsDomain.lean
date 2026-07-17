import Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.Assignment
import Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Parameters

/-!
Verifier domain for the plain five-ring F-prime `Pi_CCS` carrier.

Assurance tier: model-level.

Owns: the plain semantic shape whose complete logical carrier is exactly the
five public Phi81 rings; the least binary widths for both the existing flat
column/lane NC domain and the canonical block/lane NC domain; exact coverage;
and their resulting variable counts.

Does not own: a Rust structure, application-private suffixes, FE dimensions,
transcript messages, generated rows, costs, or artifact conformance.

Emits constraints: no.

Authority boundary: the dimensions are derived from the typed public carrier
(`257 + 13 = 270 = 5 * 54`) and Phi81 degree, not inferred from an observed
round list or generated artifact.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.plain_carrier.width` | the repaired public carrier has 270 fields and no private suffix in this profile | computed | `plainShape_carrierWidth` |
| `nifs.pi_ccs.nc_domain.columns` | the least binary cube covering 270 columns has 9 variables | derived | `domain_columnVariables`, `columnVariables_minimal` |
| `nifs.pi_ccs.nc_domain.lanes` | the least binary cube covering 54 lanes has 6 variables | derived | `domain_laneVariables`, `laneVariables_minimal` |
| `nifs.pi_ccs.nc_domain.coverage` | the 512-by-64 product covers every live carrier coordinate | derived | `domain_covers` |
| `nifs.pi_ccs.nc_domain.rounds` | NC has exactly `9 + 6 = 15` variables | derived | `domain_variableCount` |
| `nifs.pi_ccs.block_nc_domain.blocks` | the least binary cube covering five complete rings has 3 variables | derived | `blockDomain_blockVariables`, `blockVariables_minimal` |
| `nifs.pi_ccs.block_nc_domain.lanes` | the same minimal 6-variable lane cube covers every Phi81 coefficient | derived | `blockDomain_laneVariables`, `blockDomain_laneVariables_minimal` |
| `nifs.pi_ccs.block_nc_domain.coverage` | the 8-by-64 product covers every live block/lane cell | derived | `blockDomain_covers` |
| `nifs.pi_ccs.block_nc_domain.rounds` | canonical block/lane NC has exactly `3 + 6 = 9` variables | derived | `blockDomain_variableCount` |
-/

namespace Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270

/-- Plain F-prime carrier profile: all 270 aligned coordinates are public.

Row and source arities stay explicit because they do not affect the NC
column/lane domain. -/
def plainShape
    (rowVariables freshCount runningCount matrixCount : Nat) :
    SemanticShape where
  rowVariables := rowVariables
  logicalWidth := alignedPublicWidth
  freshCount := freshCount
  runningCount := runningCount
  matrixCount := matrixCount

/-- Candidate binary product domain for the complete public carrier and Phi81
coefficient degree. Minimality is proved below rather than trusted. -/
def domain : FlatNcDomain where
  columnVariables := 9
  laneVariables := 6

theorem alignedWidth_eq_legacy_add_padding :
    alignedPublicWidth = legacyPublicWidth + fixedPaddingWidth := by
  decide

@[simp] theorem plainShape_carrierWidth
    (rowVariables freshCount runningCount matrixCount : Nat) :
    (plainShape rowVariables freshCount runningCount matrixCount).carrierWidth =
      alignedPublicWidth := by
  change Phi81CarrierLayout.carrierWidth alignedPublicWidth =
    alignedPublicWidth
  decide

@[simp] theorem domain_columnVariables :
    domain.columnVariables = 9 := by
  rfl

@[simp] theorem domain_laneVariables :
    domain.laneVariables = 6 := by
  rfl

@[simp] theorem domain_columnCount :
    domain.columnCount = 512 := by
  decide

@[simp] theorem domain_laneCount :
    domain.laneCount = 64 := by
  decide

theorem domain_covers
    (rowVariables freshCount runningCount matrixCount : Nat) :
    domain.Covers
      (plainShape rowVariables freshCount runningCount matrixCount) := by
  constructor
  · rw [plainShape_carrierWidth, domain_columnCount]
    decide
  · rw [domain_laneCount]
    decide

/-- No smaller binary column cube can cover all 270 live columns. -/
theorem columnVariables_minimal
    {variables : Nat}
    (covers : alignedPublicWidth <= 2 ^ variables) :
    domain.columnVariables <= variables := by
  rw [domain_columnVariables]
  rcases Nat.lt_or_ge variables 9 with smaller | enough
  · have variablesLe : variables <= 8 := by
      omega
    have powerLe : 2 ^ variables <= 256 := by
      calc
        2 ^ variables <= 2 ^ 8 :=
          Nat.pow_le_pow_of_le (by decide) variablesLe
        _ = 256 := by decide
    have width : alignedPublicWidth = 270 := by
      decide
    rw [width] at covers
    omega
  · exact enough

/-- No smaller binary lane cube can cover all 54 Phi81 coefficients. -/
theorem laneVariables_minimal
    {variables : Nat}
    (covers : ringDegree <= 2 ^ variables) :
    domain.laneVariables <= variables := by
  rw [domain_laneVariables]
  rcases Nat.lt_or_ge variables 6 with smaller | enough
  · have variablesLe : variables <= 5 := by
      omega
    have powerLe : 2 ^ variables <= 32 := by
      calc
        2 ^ variables <= 2 ^ 5 :=
          Nat.pow_le_pow_of_le (by decide) variablesLe
        _ = 32 := by decide
    have degree : ringDegree = 54 := by
      decide
    rw [degree] at covers
    omega
  · exact enough

theorem domain_variableCount :
    domain.columnVariables + domain.laneVariables = 15 := by
  rw [domain_columnVariables, domain_laneVariables]

/-! ## Canonical block/lane domain -/

/-- Least binary product domain covering the five live Phi81 blocks and all
54 coefficient lanes. This is a semantic domain candidate only; it does not
assert that the active transcript or verifier implements block/lane NC. -/
def blockDomain : BlockNcDomain where
  blockVariables := 3
  laneVariables := 6

@[simp] theorem blockDomain_blockVariables :
    blockDomain.blockVariables = 3 := by
  rfl

@[simp] theorem blockDomain_laneVariables :
    blockDomain.laneVariables = 6 := by
  rfl

@[simp] theorem blockDomain_blockCount :
    blockDomain.blockCount = 8 := by
  decide

@[simp] theorem blockDomain_laneCount :
    blockDomain.laneCount = 64 := by
  decide

/-- The canonical block/lane product covers all five complete assignment
blocks and every real Phi81 coefficient. -/
theorem blockDomain_covers
    (rowVariables freshCount runningCount matrixCount : Nat) :
    blockDomain.Covers
      (plainShape rowVariables freshCount runningCount matrixCount) := by
  constructor
  · rw [plainShape_carrierWidth, blockDomain_blockCount]
    decide
  · rw [blockDomain_laneCount]
    decide

/-- Three variables are necessary to cover all five live Phi81 blocks. -/
theorem blockVariables_minimal
    {variables : Nat}
    (covers :
      Phi81ColumnLayout.blockCount alignedPublicWidth <= 2 ^ variables) :
    blockDomain.blockVariables <= variables := by
  rw [blockDomain_blockVariables]
  rcases Nat.lt_or_ge variables 3 with smaller | enough
  · have variablesLe : variables <= 2 := by omega
    have powerLe : 2 ^ variables <= 4 := by
      calc
        2 ^ variables <= 2 ^ 2 :=
          Nat.pow_le_pow_of_le (by decide) variablesLe
        _ = 4 := by decide
    have liveBlocks :
        Phi81ColumnLayout.blockCount alignedPublicWidth = 5 := by
      decide
    rw [liveBlocks] at covers
    omega
  · exact enough

/-- Six variables remain necessary on the lane axis of the block domain. -/
theorem blockDomain_laneVariables_minimal
    {variables : Nat}
    (covers : ringDegree <= 2 ^ variables) :
    blockDomain.laneVariables <= variables := by
  simpa only [blockDomain_laneVariables, domain_laneVariables] using
    laneVariables_minimal covers

/-- The canonical block/lane arithmetization has nine variables in total. -/
theorem blockDomain_variableCount :
    blockDomain.blockVariables + blockDomain.laneVariables = 9 := by
  rw [blockDomain_blockVariables, blockDomain_laneVariables]

end Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270.PiCcsDomain
