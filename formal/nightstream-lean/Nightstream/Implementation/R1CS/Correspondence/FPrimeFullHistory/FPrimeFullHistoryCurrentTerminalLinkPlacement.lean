import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryCurrentTerminalLinkPlacement
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeProductionFreshPublicSingletonRows
import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkCanonicalRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryProductionDigestCodec
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter

/-!
Contract: artifact-checked placement of the complete current
`terminal.latest_link` owner in one honest two-step full-history synthesis.

Assurance tier: artifact-checked bounded placement.

Owns:
- the exact local-to-full-history column map for all 527 isolated columns;
- exact equality between the 270 mapped isolated rows and the generated
  current row range `[9673389, 9673659)`;
- the exact producer-bit alignment with the recursive output encoder;
- exact equivalence with the singleton production source program and compact
  fixed-one prior-public equality, including the selected typed Terminal
  program's `priorLinkAccepted` Boolean, at the output-derived digest;
- reduction of satisfaction of the generated current range to the typed
  plain-carrier checker and frozen logical public-input equality;
- one selected typed digest shared by the recursive output wires, the current
  terminal carrier, and the selected production codec.

Does not own: a generated artifact for every current full-history row,
inclusion of this range in the stale captured aggregate, Rust-source or
compiled-Rust semantics, host shape checks, terminal NIFS semantics, or
Poseidon2 collision resistance. It also does not identify these rows with the
distinct selected-lowering `freshCheck` call or with physical recipes for the
three direct calls `freshPublic`, `encodeInstance`, and `encodedEqual`.

Emits constraints: no; Rust emits the bounded range and its generator records
the exact affine-pin receipt imported above.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

/-- Exact isolated-to-current column schedule. The recursive producer bits
remain at columns `16766..17021`; the fresh plain carrier occupies
`4090877..4091146`. -/
def columnMap : List Nat :=
  [0] ++
    (List.range 256).map (fun bit => 16766 + bit) ++
    (List.range 270).map (fun coordinate => 4090877 + coordinate)

abbrev Pulled (assignment : Nat -> Nat) : Nat -> Nat :=
  Relabel.assignment columnMap assignment

theorem columnMap_length :
    columnMap.length = FPrimeTerminalLink.colCount := by
  native_decide

theorem mapsOne :
    Relabel.column columnMap 0 = 0 := by
  native_decide

/-- Every producer-side local bit lands on the same physical column as the
already generated recursive-output encoding owner. -/
theorem producerBitColumnMap :
    forall (lane : Fin 4) (bit : Fin 64),
      Relabel.column columnMap
          (FPrimeTerminalLink.lastXOutBitCol
            (lane.val * 64 + bit.val)) =
        Relabel.column FPrimeFullHistoryOutputEncoding.columnMap
          (FPrimeEncoding.publicBitCol lane.val bit.val) := by
  native_decide

theorem freshOneColumnMap :
    Relabel.column columnMap FPrimeTerminalLink.freshOneCol =
      4090877 := by
  native_decide

theorem freshBitColumnMap :
    forall bit : Fin 256,
      Relabel.column columnMap
          (FPrimeTerminalLink.freshBitCol bit.val) =
        4090878 + bit.val := by
  native_decide

theorem freshPaddingColumnMap :
    forall padding : Fin 13,
      Relabel.column columnMap
          (FPrimeTerminalLink.freshPaddingCol padding.val) =
        4091134 + padding.val := by
  native_decide

/-- The bounded Rust-generated range is byte-for-byte the selected isolated
owner after the exact column relabeling. -/
theorem mapped_rows_eq_generated :
    FPrimeTerminalLink.rows.map (Relabel.row columnMap) =
      FPrimeFullHistoryCurrentTerminalLinkPlacement.rows := by
  native_decide

/-- Satisfaction transports in both directions; the placement certificate
adds no semantic premise. -/
theorem generatedRows_iff_localRows
    (assignment : Nat -> Nat) :
    Satisfies FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment ↔
      Satisfies FPrimeTerminalLink.rows (Pulled assignment) := by
  rw [← mapped_rows_eq_generated]
  simpa using
    (Relabel.satisfies_mapped_iff
      FPrimeTerminalLink.rows columnMap assignment)

/-- Concrete producer-column alignment between the two generated ownership
blocks. No digest is used as column-placement authority. -/
theorem producerColumnsAligned
    (assignment : Nat -> Nat) :
    FPrimeTerminalLinkCanonicalRefinement.ProducerColumnsAligned
      (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)
      (Pulled assignment) := by
  intro lane bit
  change
    assignment
        (Relabel.column columnMap
          (FPrimeTerminalLink.lastXOutBitCol
            (lane.val * 64 + bit.val))) =
      assignment
        (Relabel.column FPrimeFullHistoryOutputEncoding.columnMap
          (FPrimeEncoding.publicBitCol lane.val bit.val))
  rw [producerBitColumnMap lane bit]

/-- Exact output-encoding semantics discharge the producer-alignment premise
for the current placed owner. -/
theorem producerAligned
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (encoding :
      FPrimeEncodingSound.Holds
        (FPrimeFullHistoryOutputEncodingSound.Pulled assignment)) :
    FPrimeTerminalLinkCanonicalRefinement.ProducerAligned
      (FPrimeFullHistoryProductionDigestCodec.decodedDigest
        assignment encoding)
      (Pulled assignment) := by
  intro lane bit
  have columns := producerColumnsAligned assignment lane bit
  have encoded :=
    FPrimeEncodingCanonicalBits.publicBit_eq_encodedBit
      (Relabel.canonical canonical) encoding lane bit
  rw [
    FPrimeFullHistoryProductionDigestCodec.decodedDigest_eq_logicalLinkDigest
      assignment canonical encoding
  ]
  exact columns.trans encoded

/-- The exact placed range is equivalent to the production-shaped singleton
prior-link program. The digest is reconstructed from the recursive output
owner rather than accepted as a carried authority. -/
theorem generatedRows_iff_sourceProgram
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputRows :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment) :
    Satisfies FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment ↔
      CanonicalPublicInputLinkProgram.run
          CanonicalPublicInputLinkProgram.plain
          (FPrimeFullHistoryProductionDigestCodec.decodedDigest
            assignment
            (FPrimeFullHistoryRecursiveOutputSound.sound
              goldilocksPrime canonical one outputRows).encoding)
          CanonicalPlainCarrierLink.carrierWidth
          (FPrimeProductionFreshPublicSingletonRows.rawClaimOfAssignment
            (Pulled assignment)) =
        true := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one outputRows
  rw [generatedRows_iff_localRows]
  rw [← FPrimeTerminalLinkBatch.rows_one_eq_artifact]
  exact
    FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_sourceProgram
      (FPrimeFullHistoryProductionDigestCodec.decodedDigest
        assignment facts.encoding)
      (Relabel.canonical canonical)
      (Relabel.constantOne mapsOne one)
      (producerAligned canonical facts.encoding)

universe uParams uStructure uHeader uRunning uNifsProof uNebulaDigest
  uNebulaOpen

section PriorLinkAdapter

variable
  {Params : Type uParams}
  {StructureDigest : Type uStructure}
  {Header : Type uHeader}
  {Running : Type uRunning}
  {NifsProof : Type uNifsProof}
  {Nebula : Type}
  {NebulaDigest : Type uNebulaDigest}
  {NebulaOpen : Type uNebulaOpen}

local notation "AdapterParameters" =>
  FixedOneCanonicalAdapter.Parameters
    Params StructureDigest Header ProductionDigest Running
      CanonicalPlainCarrierLink.RawClaim NifsProof Nebula NebulaDigest
      NebulaOpen

local notation "AdapterFresh" =>
  FixedOneCanonicalAdapter.FreshInput
    ProductionDigest CanonicalPlainCarrierLink.RawClaim Nebula

/-- Artifact-checked prior-link refinement at the current production
placement. Under the exact output owner and source-link semantics, the placed
rows accept exactly when the compact fixed-one adapter's singleton fresh
public value equals `encodeInstance` at the digest derived from the output
wires. -/
theorem generatedRows_iff_freshPublic_eq_encodeInstance
    (adapter : AdapterParameters)
    (fresh : AdapterFresh)
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputRows :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (claimedDigest :
      fresh.claimedDigest =
        FPrimeFullHistoryProductionDigestCodec.decodedDigest
          assignment
          (FPrimeFullHistoryRecursiveOutputSound.sound
            goldilocksPrime canonical one outputRows).encoding)
    (ordered :
      fresh.ordered =
        [FPrimeProductionFreshPublicSingletonRows.rawClaimOfAssignment
          (Pulled assignment)])
    (linkSemantics :
      adapter.step.freshLink =
        CanonicalPlainCarrierSource.sourceCheck) :
    Satisfies FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment ↔
      FixedOneCanonicalAdapter.freshPublic adapter fresh =
        FixedOneCanonicalAdapter.encodeInstance
          (some
            (FPrimeFullHistoryProductionDigestCodec.decodedDigest
              assignment
              (FPrimeFullHistoryRecursiveOutputSound.sound
                goldilocksPrime canonical one outputRows).encoding)) := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one outputRows
  rw [generatedRows_iff_localRows]
  exact
    FPrimeProductionFreshPublicSingletonRows.artifactRows_iff_freshPublic_eq_encodeInstance
      adapter fresh
      (FPrimeFullHistoryProductionDigestCodec.decodedDigest
        assignment facts.encoding)
      (Relabel.canonical canonical)
      (Relabel.constantOne mapsOne one)
      (producerAligned canonical facts.encoding)
      claimedDigest ordered linkSemantics

end PriorLinkAdapter

section LoweringPriorLink

variable
  {Params : Type}
  {StructureDigest : Type}
  {Header : Type}
  {Running : Type}
  {NifsProof : Type}
  {Nebula : Type}
  {NebulaDigest : Type}
  {NebulaOpen : Type}

local notation "AdapterParameters" =>
  FixedOneCanonicalAdapter.Parameters
    Params StructureDigest Header ProductionDigest Running
      CanonicalPlainCarrierLink.RawClaim NifsProof Nebula NebulaDigest
      NebulaOpen

local notation "AdapterFresh" =>
  FixedOneCanonicalAdapter.FreshInput
    ProductionDigest CanonicalPlainCarrierLink.RawClaim Nebula

local notation "DirectState" =>
  Nightstream.HyperNova.Construction2.State
    ProductionDigest Running CanonicalPlainCarrierLink.RawClaim Nebula

/-- The bounded current placement reaches the exact prior-link Boolean used
by the selected typed Terminal program. The proof aligns the terminal fresh
input and its verifier-computed prior digest explicitly; it does not identify
this relation with the later terminal `freshCheck`. -/
theorem generatedRows_iff_loweringPriorLinkAccepted
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq CanonicalPlainCarrierLink.RawClaim]
    [DecidableEq NifsProof]
    [DecidableEq Nebula]
    [DecidableEq NebulaOpen]
    (adapter : AdapterParameters)
    (configuration :
      FixedOneLoweringAdapter.Configuration adapter)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
        (Option DirectState))
    (proof :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne.Proof
        Running configuration.RunningWitness AdapterFresh
          configuration.FreshWitness)
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputRows :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (priorDigest :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorDigest
          (FixedOneLoweringAdapter.parameters adapter configuration)
          statement proof =
        some
          (FPrimeFullHistoryProductionDigestCodec.decodedDigest
            assignment
            (FPrimeFullHistoryRecursiveOutputSound.sound
              goldilocksPrime canonical one outputRows).encoding))
    (claimedDigest :
      proof.fresh.claimedDigest =
        FPrimeFullHistoryProductionDigestCodec.decodedDigest
          assignment
          (FPrimeFullHistoryRecursiveOutputSound.sound
            goldilocksPrime canonical one outputRows).encoding)
    (ordered :
      proof.fresh.ordered =
        [FPrimeProductionFreshPublicSingletonRows.rawClaimOfAssignment
          (Pulled assignment)])
    (linkSemantics :
      adapter.step.freshLink =
        CanonicalPlainCarrierSource.sourceCheck) :
    Satisfies FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment ↔
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted
          (FixedOneLoweringAdapter.parameters adapter configuration)
          statement proof =
        true := by
  rw [
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted,
    priorDigest
  ]
  simpa [
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.encodedEqual
  ] using
    (generatedRows_iff_freshPublic_eq_encodeInstance
      adapter proof.fresh goldilocksPrime canonical one outputRows
      claimedDigest ordered linkSemantics)

end LoweringPriorLink

/-- Headline bounded-placement refinement. Once the exact recursive output
owner is satisfied, the generated current range is satisfied exactly when its
typed carrier is the zero completion of a logical input accepted by the
frozen paper equality. -/
theorem generatedRows_iff_logicalPaperLink
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputRows :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment) :
    Satisfies FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment ↔
      exists logical,
        CanonicalPublicInputLink.check
          (FPrimeFullHistoryProductionDigestCodec.decodedDigest
            assignment
            (FPrimeFullHistoryRecursiveOutputSound.sound
              goldilocksPrime canonical one outputRows).encoding)
          logical = true /\
        FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
            (Pulled assignment) =
          CanonicalPlainCarrierLink.completeClaim logical := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one outputRows
  rw [generatedRows_iff_localRows]
  exact
    FPrimeTerminalLinkCanonicalRefinement.satisfies_iff_logicalPaperLink
      (FPrimeFullHistoryProductionDigestCodec.decodedDigest
        assignment facts.encoding)
      (Relabel.canonical canonical)
      (Relabel.constantOne mapsOne one)
      (producerAligned canonical facts.encoding)

/-- Constructive current-placement bridge. The selected digest is reconstructed
from the recursive output owner, and the current placed rows force the exact
typed plain carrier for that digest. -/
theorem output_and_generated_rows_construct_currentPlainOwner
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (outputRows :
      Satisfies FPrimeFullHistoryRecursiveOutput.rows assignment)
    (currentRows :
      Satisfies
        FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment) :
    exists digest : ProductionDigest,
      Satisfies FPrimeTerminalLink.rows (Pulled assignment) /\
        FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
            (Pulled assignment) =
          CanonicalPlainCarrierLink.encodeClaim digest /\
        (digestCodec.encode digest).map (fun field => field.val) =
          FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment /\
        digestCodec.decode (digestCodec.encode digest) = some digest /\
        exists logical,
          CanonicalPublicInputLink.check digest logical = true /\
            FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
                (Pulled assignment) =
              CanonicalPlainCarrierLink.completeClaim logical := by
  let facts :=
    FPrimeFullHistoryRecursiveOutputSound.sound
      goldilocksPrime canonical one outputRows
  let digest :=
    FPrimeFullHistoryProductionDigestCodec.decodedDigest
      assignment facts.encoding
  have localRows :=
    (generatedRows_iff_localRows assignment).mp currentRows
  have accepted :=
    FPrimeTerminalLinkCanonicalRefinement.check_of_satisfies
      digest
      (Relabel.canonical canonical)
      (Relabel.constantOne mapsOne one)
      localRows
      (producerAligned canonical facts.encoding)
  have claim :
      FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
          (Pulled assignment) =
        CanonicalPlainCarrierLink.encodeClaim digest :=
    (CanonicalPlainCarrierLink.check_eq_true_iff
      digest
      (FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
        (Pulled assignment))).mp accepted
  have codecXOut :
      (digestCodec.encode digest).map (fun field => field.val) =
        FPrimeFullHistoryRecursiveOutput.xOutColumns.map assignment := by
    rw [
      FPrimeFullHistoryProductionDigestCodec.codec_values_eq_outputDigest
        assignment facts.encoding
    ]
    exact
      FPrimeFullHistoryRecursiveOutputSound.outputDigest_eq_xOutColumns
        assignment
  exact
    ⟨digest, localRows, claim, codecXOut,
      FPrimeFullHistoryProductionDigestCodec.codec_roundtrip
        assignment facts.encoding,
      (CanonicalPlainCarrierLink.check_reduces_to_logicalPaperLink
        digest
        (FPrimeTerminalLinkCanonicalRefinement.claimOfAssignment
          (Pulled assignment))).mp accepted⟩

end Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound
