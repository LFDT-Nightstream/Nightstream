import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutManifest
import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.FreshAssignmentPacking

/-!
Contract: model-level refinement boundary for lowering one ordinary private
Goldilocks field from a canonical source residue to 41 centered coordinates.

Owns: the exact deterministic 41-coordinate materializer, the three-symbol
width boundary, a conservative logical lowering that retains every centered
gate obligation, and the exact authority hypotheses required before those
obligations may be discharged to the outer fresh CCS opening.

Does not own: generated fixed-F-prime placement data, Rust materialization,
production row indices, a materialized fixed F-prime assignment, or
authorization to remove any production row.

Emits constraints: no. `PrivateCoordinatesGateWord` is a logical obligation
schedule. Production packs two centered residuals per physical row; that
separate residual-pair refinement must be composed before any row-count claim.

Authority boundary: `SafeAccepts` is the generic safe lowering. The
norm-discharged path requires `FreshCcsNormDischargeAuthority`, which assumes
semantic `CCS.Holds` for the same-index assignment with verifier-owned `b = 2`.
It does not derive that premise from production Pi_CCS acceptance. Existing
`BindsEncodedAssignment` binds only eligible 41-coordinate projections and is
strictly weaker than the whole-vector hypothesis here.

| Branch | Mathematical obligation | Result | Tier |
|---|---|---|---|
| materializer | `target=(source+shift)%p`, little-endian trits, `0/1/2 -> -1/0/1` | `materializeWord_coordinate` | model-level |
| decoder | 41-coordinate word decodes to the canonical source | `decode_materializeWord` | model-level |
| width | `3^40 < p < 3^41` for a three-symbol alphabet | `threeSymbol_width_boundary` | model-level |
| local gates | exactly `fieldCount * 41` logical `d^3-d=0` obligations | `safeAccepts_iff` | conditional model-level |
| norm discharge | outer strict norm implies every eligible coordinate is centered | `normDischargedLowering_sound` | conditional model-level |
| exact assignment | equal lengths and same-index values, with no permutation | `ExactFreshAssignmentBinding` | refinement boundary |
| fresh CCS premise | whole-vector identity plus assumed semantic `CCS.Holds`, `b=2` | `freshCcsAuthority_privateNorm` | conditional model-level |
-/

namespace Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.CenteredTernaryField
open Nightstream.Implementation.R1CS.CenteredTernaryNormDischarged
open Nightstream.Implementation.R1CS.CenteredTernaryLinearCompiler
open Nightstream.Implementation.R1CS.FPrimeFieldLayout
open Nightstream.Implementation.R1CS.ShiftedTernaryCompiler
open Nightstream.SuperNeo

set_option maxRecDepth 262144

/-! ## Exact deterministic 41-coordinate materializer -/

/-- The exact materializer future Rust conformance must implement. The result
type carries the 41-coordinate length in the kernel. -/
def materializeWord (source : Nat) : FiniteWord :=
  finiteEncode source

/-- The deterministic materializer formula, exposed without helper aliases:
shift modulo Goldilocks, extract a little-endian trit, then center it. -/
theorem materializeWord_coordinate (source : Nat) (index : Fin digitCount) :
    materializeWord source index =
      match ((source + shift) % goldilocksP / 3 ^ index.val) % 3 with
      | 0 => goldilocksP - 1
      | 1 => 0
      | _ => 1 := by
  rfl

theorem materializeWord_accepted (source : Nat) :
    FiniteAlphabetWord (materializeWord source) := by
  exact finiteEncode_alphabet source

/-- Honest materialization is complete for every canonical source residue. -/
theorem decode_materializeWord
    {source : Nat} (canonical : source < goldilocksP) :
    decodeFiniteWord (materializeWord source) = source := by
  exact decodeFiniteWord_finiteEncode canonical

theorem materializeWord_represents
    {source : Nat} (canonical : source < goldilocksP) :
    decodeFiniteWord (materializeWord source) = source ∧
      FiniteAlphabetWord (materializeWord source) := by
  exact ⟨decode_materializeWord canonical, materializeWord_accepted source⟩

/-- Capacity boundary relative to exactly three choices per coordinate.
Forty coordinates cannot name every Goldilocks residue; forty-one have enough
raw words. This is a width statement, not a uniqueness statement. -/
theorem threeSymbol_width_boundary :
    3 ^ 40 < goldilocksP ∧ goldilocksP < 3 ^ 41 := by
  simpa [digitCount] using width_floor

theorem forty_threeSymbolCoordinates_insufficient :
    3 ^ 40 < goldilocksP :=
  threeSymbol_width_boundary.1

theorem fortyOne_threeSymbolCoordinates_have_capacity :
    goldilocksP < 3 ^ 41 :=
  threeSymbol_width_boundary.2

/-! ## Generic safe lowering with logical centered obligations -/

/-- One exact `d^3-d=0` obligation for each of the 41 coordinates of every
ordinary private field. This is the logical leaf schedule. The production
degree-seven residual-pair family packs two such leaves into one physical row
and is intentionally proved in the separate SuperNeo project. -/
def PrivateCoordinatesGateWord {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (encoded : Nat → Nat) : Prop :=
  ∀ field : Fin fieldCount,
    GateWord fun index => encoded (layout.encodedColumn field index)

/-- Logical leaf count only. It is not a physical row count because production
residual-pair rows pack two leaves. -/
def centeredLogicalObligationCount (fieldCount : Nat) : Nat :=
  fieldCount * digitCount

theorem centeredLogicalObligationCount_eq (fieldCount : Nat) :
    centeredLogicalObligationCount fieldCount = fieldCount * 41 := by
  simp [centeredLogicalObligationCount, digitCount]

/-- Equivalent semantic alphabet form of the logical gate schedule. -/
def PrivateCoordinatesCentered {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (encoded : Nat → Nat) : Prop :=
  ∀ (field : Fin fieldCount) (digit : Fin digitCount),
    CenteredResidue (encoded (layout.encodedColumn field digit.val))

theorem privateCoordinatesGateWord_iff_centered
    (prime : EuclidPrime goldilocksP)
    {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (encoded : Nat → Nat)
    (canonical : ∀ column, encoded column < goldilocksP) :
    PrivateCoordinatesGateWord layout encoded ↔
      PrivateCoordinatesCentered layout encoded := by
  constructor
  · intro gates field digit
    exact (centeredUnitGate_iff prime
      (canonical (layout.encodedColumn field digit.val))).mp
      (gates field digit.val digit.isLt)
  · intro centered field index indexLt
    exact (centeredUnitGate_iff prime
      (canonical (layout.encodedColumn field index))).mpr
      (centered field ⟨index, indexLt⟩)

/-- Conservative logical acceptance: arbitrary source R1CS rows are linearly
substituted and every centered-unit obligation remains explicit. -/
def SafeAccepts {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (sourceRows : List Row) (encoded : Nat → Nat) : Prop :=
  Satisfies (loweredRows layout sourceRows) encoded ∧
    PrivateCoordinatesGateWord layout encoded

theorem safeAccepts_iff
    (prime : EuclidPrime goldilocksP)
    {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (sourceRows : List Row) (encoded : Nat → Nat)
    (canonical : ∀ column, encoded column < goldilocksP) :
    SafeAccepts layout sourceRows encoded ↔
      Satisfies sourceRows (decodedAssignment layout encoded) ∧
        PrivateCoordinatesCentered layout encoded := by
  unfold SafeAccepts
  rw [loweredRows_iff_sourceRows layout sourceRows encoded,
    privateCoordinatesGateWord_iff_centered prime layout encoded canonical]

/-- Generic honest completeness uses the exact deterministic word supplied by
`LinearCompiler.HonestMaterializer`. This remains model-level until generated
Rust evidence instantiates that materializer. -/
theorem honest_safe_complete
    (prime : EuclidPrime goldilocksP)
    {fieldCount : Nat}
    {layout : CenteredTernaryLinearCompiler.Layout fieldCount}
    (materializer : CenteredTernaryLinearCompiler.HonestMaterializer layout)
    (sourceRows : List Row) {source : Nat → Nat}
    (canonical : ∀ column, source column < goldilocksP)
    (accepted : Satisfies sourceRows source) :
    ∃ encoded,
      SafeAccepts layout sourceRows encoded ∧
        decodedAssignment layout encoded = source := by
  rcases honest_complete materializer sourceRows canonical accepted with
    ⟨encoded, norm, lowered, decoded⟩
  refine ⟨encoded, ⟨lowered, ?_⟩, decoded⟩
  intro field index indexLt
  let digit : Fin digitCount := ⟨index, indexLt⟩
  exact (centeredUnitGate_iff prime (norm field digit).1).mpr
    (normBoundTwo_iff_centeredResidue.mp (norm field digit))

/-! ## Conditional norm-discharged lowering -/

theorem privateCoordinatesCentered_of_norm
    {fieldCount : Nat}
    {layout : CenteredTernaryLinearCompiler.Layout fieldCount}
    {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundTwo layout encoded) :
    PrivateCoordinatesCentered layout encoded := by
  intro field digit
  exact normBoundTwo_iff_centeredResidue.mp (norm field digit)

theorem privateCoordinatesGateWord_of_norm
    (prime : EuclidPrime goldilocksP)
    {fieldCount : Nat}
    {layout : CenteredTernaryLinearCompiler.Layout fieldCount}
    {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundTwo layout encoded) :
    PrivateCoordinatesGateWord layout encoded := by
  intro field index indexLt
  let digit : Fin digitCount := ⟨index, indexLt⟩
  exact (centeredUnitGate_iff prime (norm field digit).1).mpr
    (normBoundTwo_iff_centeredResidue.mp (norm field digit))

/-- Semantic soundness of the norm-discharged logical path. This theorem does
not assert that a production acceptance path supplies `norm` or may remove
physical rows. -/
theorem normDischargedLowering_sound
    {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (sourceRows : List Row) {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundTwo layout encoded)
    (satisfies : Satisfies (loweredRows layout sourceRows) encoded) :
    Satisfies sourceRows (decodedAssignment layout encoded) ∧
      PrivateCoordinatesCentered layout encoded := by
  exact ⟨(loweredRows_iff_sourceRows
      layout sourceRows encoded).mp satisfies,
    privateCoordinatesCentered_of_norm norm⟩

/-- At the model level, an externally authorized norm supplies exactly the
logical gate obligations retained by `SafeAccepts`. Physical row deletion
still requires the generated CE and residual-pair correspondence bridges. -/
theorem normDischargedLowering_recovers_safe
    (prime : EuclidPrime goldilocksP)
    {fieldCount : Nat}
    (layout : CenteredTernaryLinearCompiler.Layout fieldCount)
    (sourceRows : List Row) {encoded : Nat → Nat}
    (norm : PrivateCoordinatesNormBoundTwo layout encoded)
    (satisfies : Satisfies (loweredRows layout sourceRows) encoded) :
    SafeAccepts layout sourceRows encoded :=
  ⟨satisfies, privateCoordinatesGateWord_of_norm prime norm⟩

/-! ## Exact fresh-CCS authority hypotheses

`Manifest.BindsEncodedAssignment` quantifies only over eligible fields and 41
digits. The structure below is deliberately stronger: it binds every finite
encoded coordinate to the same outer fresh-CCS assignment index. Production
performs no permutation, so a general coordinate-bijection API would hide the
actual ABI and create proof surface without a protocol need. -/

/-- Whole-vector same-index equality required before using the fresh CCS norm
to discharge ordinary-private centered gate obligations. -/
structure ExactFreshAssignmentBinding
    (manifest : Manifest)
    (encodedValues : List Nat)
    (assignment : Nightstream.SuperNeo.Concrete.Assignment) : Prop where
  encodedLength : encodedValues.length = manifest.encodedColumnCount
  assignmentLength : assignment.length = manifest.ceAssignmentLength
  universeIdentity : manifest.encodedColumnCount = manifest.ceAssignmentLength
  pointwise : ∀ coordinate, coordinate < encodedValues.length →
    encodedValues.getD coordinate 0 =
      (assignment.getD coordinate 0).val

theorem ExactFreshAssignmentBinding.sameLength
    {manifest : Manifest}
    {encodedValues : List Nat}
    {assignment : Nightstream.SuperNeo.Concrete.Assignment}
    (binding : ExactFreshAssignmentBinding manifest encodedValues assignment) :
    encodedValues.length = assignment.length := by
  rw [binding.encodedLength, binding.assignmentLength,
    binding.universeIdentity]

/-- Exact semantic authority required from the outer fresh CCS relation.
The current fixed-selector estimator has no combined assignment materializer,
so no generated artifact yet instantiates this complete structure. -/
structure FreshCcsNormDischargeAuthority
    (artifact : GeneratedArtifact)
    (context : Nightstream.SuperNeo.Concrete.Context)
    (params : GlobalParams)
    (statement : Nightstream.SuperNeo.Concrete.CCSStatement)
    (encodedValues : List Nat)
    (assignment : Nightstream.SuperNeo.Concrete.Assignment) : Prop where
  whole : ExactFreshAssignmentBinding artifact.manifest
    encodedValues assignment
  verifierOwnsBoundTwo : params.b = 2
  statementFresh : statement.stage = .fresh
  ccsHolds : CCS.Holds
    (Nightstream.SuperNeo.Concrete.relationSemantics context)
    params statement assignment

private theorem getD_mem_of_lt
    {values : List Nightstream.SuperNeo.Concrete.F}
    {index : Nat} (indexLt : index < values.length) :
    values.getD index 0 ∈ values := by
  have member := List.getElem_mem (l := values) indexLt
  rwa [List.getElem_eq_getD 0] at member

/-- Conditional derivation of the private-coordinate norm from the exact
fresh-CCS authority boundary. No fixed-selector materializer currently
instantiates this structure, so this is not a production row-removal theorem. -/
theorem freshCcsAuthority_privateNorm
    (artifact : GeneratedArtifact)
    (compiler : CompilerBinding artifact)
    (context : Nightstream.SuperNeo.Concrete.Context)
    (params : GlobalParams)
    (statement : Nightstream.SuperNeo.Concrete.CCSStatement)
    (encodedValues : List Nat)
    (assignment : Nightstream.SuperNeo.Concrete.Assignment)
    (authority : FreshCcsNormDischargeAuthority artifact
      context params statement encodedValues assignment) :
    PrivateCoordinatesNormBoundTwo compiler.layout
      (assignmentOf encodedValues) := by
  have expanded :=
    (Nightstream.SuperNeo.Concrete.ccsMembership_iff
      context params statement assignment).mp authority.ccsHolds
  have normTwo : Nightstream.SuperNeo.Concrete.normBounded 2 assignment := by
    have normAtStage := expanded.2.2.1
    simpa [NormStage.bound, authority.statementFresh,
      authority.verifierOwnsBoundTwo] using normAtStage
  intro field digit
  have within := compiler.placementWithin field
  have coordinateLtManifest :
      compiler.layout.encodedColumn field digit.val <
        artifact.manifest.encodedColumnCount := by
    rw [compiler.encodedColumn_eq field digit.val digit.isLt]
    change
      (compiler.placement field).encodedStart + digitCount ≤
          artifact.manifest.encodedColumnCount ∧
        (compiler.placement field).ceStart + digitCount ≤
          artifact.manifest.ceAssignmentLength at within
    omega
  have coordinateLtEncoded :
      compiler.layout.encodedColumn field digit.val < encodedValues.length := by
    rw [authority.whole.encodedLength]
    exact coordinateLtManifest
  have coordinateLtAssignment :
      compiler.layout.encodedColumn field digit.val < assignment.length := by
    rw [authority.whole.assignmentLength,
      ← authority.whole.universeIdentity]
    exact coordinateLtManifest
  have member := getD_mem_of_lt coordinateLtAssignment
  have centered := concrete_normBounded_two_implies_centered normTwo member
  apply normBoundTwo_iff_centeredResidue.mpr
  unfold assignmentOf
  rw [authority.whole.pointwise _ coordinateLtEncoded]
  exact centered

end Nightstream.Implementation.R1CS.OrdinaryPrivateFieldRefinement
