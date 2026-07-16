import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.FixedCarrierNifs
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.NifsPaper.PublicInputBoundary
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.CarrierCoverageRefinement
import Nightstream.SuperNeo.Folding.PiCCS

/-!
Production Π_CCS public-carrier boundary for the fixed F' NIFS profiles.

Owns: a typed, order-sensitive schema for fresh CCS columns, running/output CE
columns, and non-CE transcript/NC sidecars; decoding those columns without
conflating the 257-field CCS input with the 270-coefficient packed CE carrier;
the exact value equations still required from generated binding rows; the
width-only bridge to a generic optimized Π_CCS carrier counterexample; and the
exact bridge to the current fixed-carrier Π_CCS, NIFS, and recursive F' fixture.

Does not own: a `PiCCS.Attempt`, `PiCCS.Shape`, `PiCCS.Accepted`, private
CCS/CE openings, SumCheck truth, extraction, general NIFS refinement, or row removal.
The first counterexample uses a minimal zero-relation CCS structure. The second
executes the exact `1 x 257` all-zero carrier fixture used by the current full
F' tests with a canonical fixed-k zero accumulator through the complete native
NIFS stack. A third, honestly linked public input with the same hidden tail is
also replayed by the recursive F' circuit.

| Component | Production representation | Mathematical role | Status |
|---|---|---|---|
| fresh public input | 257 scalar columns | Definition-12 CCS input | decoded only |
| running/output public input | 270 active `X` columns | five complete ring columns used by Π_RLC/Π_DEC | decoded only |
| output point | `r` pairs | paper CE evaluation point | decoded only |
| output evaluations | 3 rows × first 108 limbs | three `RingK` evaluations | decoded only |
| `s_col`, `fold_digest` | separate columns | transcript/NC context, not paper `Point` | explicit sidecar |
| `BatchColumnShape` | fixed generated layout | exact 257/270/3×128 carrier cardinalities | open artifact boundary |
| `BatchWiring` | missing generated facade and row theorem | fresh projection binding, full running binding, shared `r` | open artifact boundary |
| NC carrier coverage | generic optimized Π_CCS execution at widths 257/270 | accepted tail violates independently specified full-carrier NC truth | artifact-checked width match |
| fixed carrier fixture | exact 1×257 R1CS-to-CCS structure, seed 41, canonical running zero | a hidden tail value 2 survives Π_CCS, complete fixed NIFS, and the linked recursive F' circuit | artifact-checked end-to-end execution; general refinement open |

The generic paper `PiCCS.InputProduct` uses one `PublicInput` type for both CCS
and CE and `PiCCS.Shape.samePublicInput` requires literal equality. Production
does not have that shape: fresh CCS exposes 257 fields, while its CE output
carries 270 coefficients and only 257 distinguished positions are bound to the
fresh input. The other 13 coefficients are not zero padding after ring-linear
folding. `PublicInputBoundary.publicProjection_not_injective` proves that the
257-field projection cannot recover the full carrier, and
`PublicInputBoundary.ringAction_enters_extra_coefficient` proves that the
selected scalar region is not closed under the actual Φ81 ring action.

Consequently this module intentionally exports no paper acceptance theorem.
Closing the gap requires either a paper-grounded heterogeneous CCS-to-CE
embedding with relation and folding proofs, or a semantics-preserving change to
an aligned production relation. Merely embedding 257 values into 270 slots or
truncating a CE claim is not such a proof.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper

/-! ## Typed production column schema -/

/-- Verifier-visible columns of one fresh Definition-12 CCS statement. -/
structure FreshColumns where
  commitmentData : List Nat
  scalarPublicInput : List Nat
deriving DecidableEq

/-- Verifier-visible paper CE fields. `packedPublicInput` contains every active
coefficient; `evaluationRows` contains three 128-limb implementation rows whose
first 108 limbs decode to paper evaluations. -/
structure CeColumns where
  commitmentData : List Nat
  packedPublicInput : List Nat
  r : List (Nat × Nat)
  evaluationRows : List (List Nat)
deriving DecidableEq

/-- Implementation transcript/NC context carried beside a CE claim. These
columns are deliberately not members of the paper CE point. -/
structure TranscriptNcColumns where
  sCol : List (Nat × Nat)
  foldDigest : List Nat
deriving DecidableEq

/-- Pure column grouping still missing from the generated affine maps.

Fresh entries precede running entries exactly as in the verifier's `K + k`
order. Source and output columns are distinct; equality must come from rows. -/
structure BatchColumns
    (params : GlobalParams) (arity : BatchArity params) where
  fresh : Fin arity.freshCount → FreshColumns
  running : Fin (arity.mode.count params) → CeColumns
  outputs : Fin arity.total → CeColumns
  runningContext : Fin (arity.mode.count params) → TranscriptNcColumns
  outputContext : Fin arity.total → TranscriptNcColumns

/-- Fixed cardinalities of one fresh CCS column group. -/
structure FreshColumnShape (columns : FreshColumns) : Prop where
  commitment : columns.commitmentData.length =
    FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length
  publicInput : columns.scalarPublicInput.length =
    PublicInputBoundary.productionPublicWidth

/-- Fixed cardinalities of one packed CE column group. Keeping these obligations
explicit prevents `getD`-based decoding from silently accepting short rows. -/
structure CeColumnShape (columns : CeColumns) : Prop where
  commitment : columns.commitmentData.length =
    FPrimeFullHistoryPiDec.layout.parent.commitment.dataCols.length
  packedPublicInput : columns.packedPublicInput.length =
    PublicInputBoundary.productionPackedWidth
  point : columns.r.length = FPrimeFullHistoryPiDec.layout.parent.rCols.length
  evaluationCount : columns.evaluationRows.length =
    FPrimeFullHistoryPiDec.layout.parent.yRingCols.length
  evaluationWidth : ∀ row, row < columns.evaluationRows.length →
    (columns.evaluationRows.getD row []).length =
      (FPrimeFullHistoryPiDec.layout.parent.yRingCols.getD row []).length

/-- Fixed cardinalities of the non-CE transcript/NC context. -/
structure TranscriptNcColumnShape (columns : TranscriptNcColumns) : Prop where
  sCol : columns.sCol.length =
    FPrimeFullHistoryPiDec.layout.parent.sColCols.length
  foldDigest : columns.foldDigest.length =
    FPrimeFullHistoryPiDec.layout.parent.foldDigestCols.length

/-- Shape obligations for every member of the heterogeneous production batch. -/
structure BatchColumnShape
    {params : GlobalParams} {arity : BatchArity params}
    (columns : BatchColumns params arity) : Prop where
  fresh : ∀ index, FreshColumnShape (columns.fresh index)
  running : ∀ index, CeColumnShape (columns.running index)
  outputs : ∀ index, CeColumnShape (columns.outputs index)
  runningContext : ∀ index,
    TranscriptNcColumnShape (columns.runningContext index)
  outputContext : ∀ index,
    TranscriptNcColumnShape (columns.outputContext index)

/-! ## Heterogeneous decoding -/

def decodeFreshCommitment (assignment : Nat → Nat)
    (columns : FreshColumns) : PackedCommitment :=
  ⟨values assignment columns.commitmentData⟩

def decodeFreshPublicInput (assignment : Nat → Nat)
    (columns : FreshColumns) : Concrete.PublicInput :=
  values assignment columns.scalarPublicInput

def decodeFreshCcs (assignment : Nat → Nat) (columns : FreshColumns) :
    CCS.Instance Unit Concrete.PublicInput PackedCommitment where
  constraintSystem := ()
  commitment := decodeFreshCommitment assignment columns
  publicInput := decodeFreshPublicInput assignment columns
  stage := .fresh

def decodeCeCommitment (assignment : Nat → Nat)
    (columns : CeColumns) : PackedCommitment :=
  ⟨values assignment columns.commitmentData⟩

def decodeCePublicInput (assignment : Nat → Nat)
    (columns : CeColumns) : PackedPublicInput :=
  ⟨values assignment columns.packedPublicInput⟩

def decodeCePoint (assignment : Nat → Nat) (columns : CeColumns) : Point :=
  extensionValues assignment columns.r

def decodeCeEvaluations (assignment : Nat → Nat)
    (columns : CeColumns) : Array Evaluation :=
  (columns.evaluationRows.map (decodedEvaluation assignment)).toArray

def decodeCe (assignment : Nat → Nat) (columns : CeColumns) :
    CE.Instance Unit PackedPublicInput Point Evaluation PackedCommitment where
  constraintSystem := ()
  commitment := decodeCeCommitment assignment columns
  publicInput := decodeCePublicInput assignment columns
  point := decodeCePoint assignment columns
  evaluations := decodeCeEvaluations assignment columns
  stage := .fresh

def decodeTranscriptNc (assignment : Nat → Nat)
    (columns : TranscriptNcColumns) : List (Scalar × Scalar) × List Scalar :=
  (pairValues assignment columns.sCol, values assignment columns.foldDigest)

/-! ## Exact row obligations, not paper acceptance -/

/-- Production fresh-output binding: commitment equality and equality of only
the 257 distinguished scalar positions. It intentionally does not claim that
the fresh statement determines all 270 packed output coefficients. -/
structure FreshOutputBinding (assignment : Nat → Nat)
    (fresh : FreshColumns) (output : CeColumns) : Prop where
  commitment :
    decodeCeCommitment assignment output = decodeFreshCommitment assignment fresh
  distinguishedPublicInput :
    unpackPublicInput (decodeCePublicInput assignment output) =
      decodeFreshPublicInput assignment fresh

/-- Production running-output binding preserves the complete packed carrier. -/
structure RunningOutputBinding (assignment : Nat → Nat)
    (running output : CeColumns) : Prop where
  commitment :
    decodeCeCommitment assignment output = decodeCeCommitment assignment running
  packedPublicInput :
    decodeCePublicInput assignment output = decodeCePublicInput assignment running

/-- Assignment-indexed equations that the generated Π_CCS binding artifact must
derive. This deliberately stops before `PiCCS.Shape`: its fresh and CE public
inputs have different types and different production cardinalities. -/
structure BatchWiring
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat → Nat) (columns : BatchColumns params arity) : Prop where
  fresh : ∀ index,
    FreshOutputBinding assignment (columns.fresh index)
      (columns.outputs (Fin.castAdd (arity.mode.count params) index))
  running : ∀ index,
    RunningOutputBinding assignment (columns.running index)
      (columns.outputs (Fin.natAdd arity.freshCount index))
  sharedOutputPoint : ∀ left right,
    decodeCePoint assignment (columns.outputs left) =
      decodeCePoint assignment (columns.outputs right)

/-- Separate implementation-sidecar equations. They are not used to manufacture
`PiCCS.Shape.sharedOutputPoint`. -/
structure SidecarWiring
    {params : GlobalParams} {arity : BatchArity params}
    (assignment : Nat → Nat) (columns : BatchColumns params arity) : Prop where
  sharedOutputContext : ∀ left right,
    decodeTranscriptNc assignment (columns.outputContext left) =
      decodeTranscriptNc assignment (columns.outputContext right)

/-! ## Fixed arities and explicit blocker -/

def recursiveArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.bootstrap Concrete.productionGlobalParams 1 (by decide) (by decide)

def terminalArity : BatchArity Concrete.productionGlobalParams :=
  BatchArity.active Concrete.productionGlobalParams 1 (by decide) (by decide)

/-- Carrier-level witness that distinguished fresh equality cannot be promoted
to full packed equality without an additional relation theorem. -/
theorem distinguishedProjection_does_not_determine_packedInput :
    PublicInputBoundary.zeroPackedInput ≠ PublicInputBoundary.lastTailPackedInput ∧
      unpackPublicInput PublicInputBoundary.zeroPackedInput =
        unpackPublicInput PublicInputBoundary.lastTailPackedInput :=
  PublicInputBoundary.publicProjection_not_injective

/-! ## Width-matched executable NC counterexample -/

/-- The lower-level optimized Π_CCS counterexample uses exactly the scalar and
packed carrier widths exposed by the fixed F' profile. This is a dimension
bridge only; it does not identify the counterexample's minimal CCS structure
with the fixed F' structure. -/
theorem carrierCoverageArtifact_matches_fixedWidths :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.logicalWidth =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.counterexampleShape.carrierWidth =
        PublicInputBoundary.productionPackedWidth := by
  decide

/-- At the exact fixed F' scalar/carrier widths, the artifact-checked optimized
Π_CCS API accepts a packed witness whose independently specified complete
carrier violates NC truth.

This theorem does not claim acceptance under the fixed F' CCS structure or by
NIFS/F'. Those remaining refinement steps must be proved separately. -/
theorem fixedWidth_pi_ccs_artifact_accepts_nc_false_carrier :
    Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.logicalWidth =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.counterexampleShape.carrierWidth =
        PublicInputBoundary.productionPackedWidth ∧
      Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailPiCcsAccepted = true ∧
      ¬ Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.AssignmentTruth
        Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tailSemanticAssignment := by
  exact ⟨carrierCoverageArtifact_matches_fixedWidths.1,
    carrierCoverageArtifact_matches_fixedWidths.2,
    Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.artifact_pi_ccs_accepts_pair.2,
    Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tail_semantic_assignment_not_truth⟩

/-! ## Exact current fixed-carrier fixture -/

/-- The drift-checked fixed-carrier export records the exact current fixture
shape and the canonical initial running-claim count. This is implementation
evidence, not a semantic premise for Pi_CCS soundness. -/
theorem fixedCarrierArtifact_exactProfile :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.relationRows = 1 ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.relationColumns =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.relationArity = 3 ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.relationDegree = 2 ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.publicInputLen =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.packedRows =
        Nightstream.SuperNeo.Concrete.ringDegree ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.packedRows *
          Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.packedColumns =
        PublicInputBoundary.productionPackedWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.canonicalRunningCount = 14 := by
  decide

/-- Stage-local outcomes from the exact executable fixture. Keeping Pi_CCS and
the composed fixed-NIFS results separate makes the first accepting boundary
auditable without treating either Boolean as semantic authority. -/
theorem fixedCarrierArtifact_protocolOutcomes :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.zeroPiCcsAccepted = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailPiCcsAccepted = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.zeroNifsProved = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.zeroNifsVerified = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailNifsProved = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailNifsVerified = true := by
  decide

set_option maxRecDepth 4096 in
/-- The fixed-carrier execution uses exactly the packed witness pair already
interpreted by the independent full-carrier NC semantics. Storage equality is
included because the matrix's flat storage index differs from its semantic
block/lane coordinate. -/
theorem fixedCarrierArtifact_sameWitnessPair :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.zeroPackedStorage =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroPackedStorage ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailPackedStorage =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailPackedStorage ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.zeroLogicalDecode =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroLogicalDecode ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailLogicalDecode =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailLogicalDecode ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.zeroFullDecode =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.zeroFullDecode ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailFullDecode =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode := by
  decide

/-- The full decode exported by the fixed-carrier execution is the same decode
whose independently specified semantic assignment violates NC truth. -/
theorem fixedCarrierArtifact_tailDecodeViolatesSemanticTruth :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailFullDecode =
        Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact.tailFullDecode ∧
      ¬ Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.AssignmentTruth
        Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tailSemanticAssignment := by
  exact ⟨fixedCarrierArtifact_sameWitnessPair.2.2.2.2.2,
    Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tail_semantic_assignment_not_truth⟩

/-- Exact artifact-checked counterexample for the carrier fixture used by the
current full-F' tests: optimized Pi_CCS accepts a fresh claim with the same
public input but a different commitment, while the corresponding complete
Phi81 carrier violates the independently specified NC relation.

This does not establish a general Rust refinement theorem or F' acceptance. It
blocks treating the current 257-field fixture as a paper-valid SuperNeo carrier
without an explicit alignment repair. -/
theorem fixedCarrier_pi_ccs_artifact_accepts_nc_false_carrier :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.relationColumns =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.packedRows *
          Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.packedColumns =
        PublicInputBoundary.productionPackedWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.freshPublicInputsEqual = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.commitmentsDiffer = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailPiCcsAccepted = true ∧
      ¬ Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.AssignmentTruth
        Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tailSemanticAssignment := by
  exact ⟨fixedCarrierArtifact_exactProfile.2.1,
    fixedCarrierArtifact_exactProfile.2.2.2.2.2.2.1,
    by decide,
    by decide,
    fixedCarrierArtifact_protocolOutcomes.2.1,
    Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tail_semantic_assignment_not_truth⟩

/-- Exact artifact-checked counterexample at the requested transition boundary:
the complete fixed Construction-2 NIFS prover and verifier accept the same
fresh claim whose full Phi81 carrier violates the independent NC semantics.

This is an executable counterexample for the current fixed carrier fixture, not
a general proof about every CCS relation and not yet an F' circuit execution.
It formally rules out using the current native NIFS as a trusted semantic oracle
for row removal. -/
theorem fixedCarrier_nifs_artifact_accepts_nc_false_carrier :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.relationColumns =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.freshPublicInputsEqual = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.commitmentsDiffer = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailNifsProved = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.tailNifsVerified = true ∧
      ¬ Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Semantics.Nc.AssignmentTruth
        Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tailSemanticAssignment := by
  exact ⟨fixedCarrierArtifact_exactProfile.2.1,
    by decide,
    by decide,
    fixedCarrierArtifact_protocolOutcomes.2.2.2.2.1,
    fixedCarrierArtifact_protocolOutcomes.2.2.2.2.2,
    Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.tail_semantic_assignment_not_truth⟩

set_option maxRecDepth 4096 in
/-- Exact shape and execution facts for the linked fresh input consumed by the
recursive F' relation. Coordinate 257 is the first coefficient outside the
257-field public view but inside the completed 270-coefficient Phi81 carrier. -/
theorem fixedCarrierArtifact_linkedRecursiveProfile :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedPublicInput.length =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedPublicInput.getD 0 0 = 1 ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailFullDecode.length =
        PublicInputBoundary.productionPackedWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailFullDecode.getD
          PublicInputBoundary.productionPublicWidth (0, 0) = (2, 0) ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailNifsProved = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailNifsVerified = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedRecursiveFPrimeBuilt = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedRecursiveFPrimeSatisfied = true := by
  decide

set_option maxRecDepth 4096 in
/-- Independent strict-norm interpretation of the exact linked carrier rejects
the exported tail value. The execution Boolean above is not used to derive this
fact; it is checked against the semantic `normBounded` predicate separately. -/
theorem fixedCarrierArtifact_linkedTail_not_normBounded :
    ¬ normBounded 2
      (Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.baseFields
        Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailFullDecode) := by
  intro bounded
  have member :
      Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.baseFieldOfPair (2, 0) ∈
        Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.baseFields
          Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailFullDecode := by
    decide
  have bad := bounded
    (Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.baseFieldOfPair (2, 0))
    member
  exact (by decide :
    ¬ centeredMagnitude
        (Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.baseFieldOfPair (2, 0)) < 2) bad

/-- End-to-end artifact-checked counterexample for the current recursive F'
implementation: the linked fresh input has the required 257-field public shape,
the fixed NIFS prover and verifier accept it, and the complete recursive F'
relation is satisfied, while Lean independently rejects its 270-coefficient
carrier under the paper's strict `b = 2` norm.

This theorem is exact execution evidence plus kernel-checked semantics. It is
not a general Rust/R1CS refinement theorem, and it authorizes no row removal.
It establishes that the current implementation cannot serve as the trusted
semantic oracle until the carrier alignment is repaired. -/
theorem fixedCarrier_recursive_f_prime_artifact_accepts_non_norm_carrier :
    Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedPublicInput.length =
        PublicInputBoundary.productionPublicWidth ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailNifsProved = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailNifsVerified = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedRecursiveFPrimeBuilt = true ∧
      Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedRecursiveFPrimeSatisfied = true ∧
      ¬ normBounded 2
        (Nightstream.Implementation.R1CS.PiCcsNc.CarrierCoverageRefinement.baseFields
          Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact.linkedTailFullDecode) := by
  exact ⟨fixedCarrierArtifact_linkedRecursiveProfile.1,
    fixedCarrierArtifact_linkedRecursiveProfile.2.2.2.2.1,
    fixedCarrierArtifact_linkedRecursiveProfile.2.2.2.2.2.1,
    fixedCarrierArtifact_linkedRecursiveProfile.2.2.2.2.2.2.1,
    fixedCarrierArtifact_linkedRecursiveProfile.2.2.2.2.2.2.2,
    fixedCarrierArtifact_linkedTail_not_normBounded⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiCcs
