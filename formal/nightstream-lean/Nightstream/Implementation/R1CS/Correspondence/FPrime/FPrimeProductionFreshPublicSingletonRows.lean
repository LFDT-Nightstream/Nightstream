import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeTerminalLinkBatch
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPlainCarrierSerialization
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge

/-!
Contract: exact singleton composition from the compact fixed-one fresh-public
equality through the production source predicate to the selected terminal-link
rows.

Assurance tier: artifact-checked relation refinement.

Owns:
- the source-shaped raw claim serialized from the one typed row assignment;
- bidirectional equivalence between the selected 270-row singleton block and
  the 273-obligation source program;
- bidirectional equivalence between both representations and the compact
  fixed-one adapter equality;
- specialization of that result to the current exact isolated artifact.

Does not own: proof that the host performs the three source shape checks,
producer output-encoding rows, a surrounding full-history placement, Rust
syntax or compiled semantics, a multi-fresh batch reduction, NIFS, or the
optional application suffix.

The source program costs three more obligations because it checks expected
length, `m_in`, and vector length. The physical relation receives an already
typed `Fin 1` claim and therefore owns only the 270 coordinate rows.

Emits constraints: no; it composes an existing receipt-owned row block.
-/

namespace Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open CanonicalPlainCarrierLink
open CanonicalPlainCarrierSerialization
open CanonicalPlainCarrierSource
open CanonicalPublicInputLinkProgram
open FixedOneCanonicalAdapter
open ProductionFreshPublicSingletonBridge

universe uParams uStructure uHeader uRunning uNifsProof uNebulaDigest
  uNebulaOpen

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

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
    Params StructureDigest Header ProductionDigest Running RawClaim NifsProof
      Nebula NebulaDigest NebulaOpen

local notation "AdapterFresh" =>
  FixedOneCanonicalAdapter.FreshInput ProductionDigest RawClaim Nebula

def singletonClaim : Fin 1 :=
  ⟨0, by decide⟩

/-- The exact raw list presented to the source checker, reconstructed without
padding, truncation, or a free digest from the one typed physical claim. -/
def rawClaimOfAssignment
    (z : Nat -> Nat) : RawClaim :=
  serializeClaim
    (FPrimeTerminalLinkBatch.claimOfAssignment z singletonClaim)

theorem rawClaimOfAssignment_length
    (z : Nat -> Nat) :
    (rawClaimOfAssignment z).x.length = carrierWidth := by
  exact Carrier.coordinates_length _

/-- The source-level shape checks plus 270 coordinate obligations differ
from the already-typed physical block by exactly three obligations. -/
theorem sourceProgram_cost_eq_rows_plus_shape :
    CanonicalPublicInputLinkProgram.cost
        CanonicalPublicInputLinkProgram.plain =
      FPrimeTerminalLinkBatch.rowCount 1 + 3 := by
  decide

/-- Selected singleton rows are equivalent to the raw production checker.
No one-way serialization loss remains. -/
theorem selectedRows_iff_sourceCheck
    (digest : ProductionDigest)
    {z : Nat -> Nat}
    (canonical : forall column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned :
      FPrimeTerminalLinkBatch.ProducerAligned digest z) :
    Satisfies (FPrimeTerminalLinkBatch.rows 1) z <->
      CanonicalPlainCarrierSource.sourceCheck
        digest (rawClaimOfAssignment z) = true := by
  have rowsIff :=
    FPrimeTerminalLinkBatch.satisfies_iff_checks
      (batchSize := 1) digest canonical one producerAligned
  constructor
  · intro satisfies
    have typedAccepted :=
      rowsIff.mp satisfies singletonClaim
    exact
      (sourceCheck_serializeClaim_iff_check
        digest
        (FPrimeTerminalLinkBatch.claimOfAssignment
          z singletonClaim)).mpr typedAccepted
  · intro sourceAccepted
    apply rowsIff.mpr
    intro claim
    have typedAccepted :=
      (sourceCheck_serializeClaim_iff_check
        digest
        (FPrimeTerminalLinkBatch.claimOfAssignment
          z singletonClaim)).mp sourceAccepted
    have claimEqual : claim = singletonClaim :=
      Subsingleton.elim _ _
    subst claim
    exact typedAccepted

/-- The same exact relation stated against the selected six-phase source
program. -/
theorem selectedRows_iff_sourceProgram
    (digest : ProductionDigest)
    {z : Nat -> Nat}
    (canonical : forall column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned :
      FPrimeTerminalLinkBatch.ProducerAligned digest z) :
    Satisfies (FPrimeTerminalLinkBatch.rows 1) z <->
      CanonicalPublicInputLinkProgram.run
          CanonicalPublicInputLinkProgram.plain
          digest carrierWidth (rawClaimOfAssignment z) =
        true := by
  rw [CanonicalPublicInputLinkProgram.run_plain_eq_sourceCheck]
  exact selectedRows_iff_sourceCheck
    digest canonical one producerAligned

/-- The selected physical relation accepts exactly when the compact native
adapter's singleton fresh-public value equals `encodeInstance`. -/
theorem selectedRows_iff_freshPublic_eq_encodeInstance
    (adapter : AdapterParameters)
    (fresh : AdapterFresh)
    (digest : ProductionDigest)
    {z : Nat -> Nat}
    (canonical : forall column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned :
      FPrimeTerminalLinkBatch.ProducerAligned digest z)
    (claimedDigest : fresh.claimedDigest = digest)
    (ordered : fresh.ordered = [rawClaimOfAssignment z])
    (linkSemantics :
      adapter.step.freshLink =
        CanonicalPlainCarrierSource.sourceCheck) :
    Satisfies (FPrimeTerminalLinkBatch.rows 1) z <->
      FixedOneCanonicalAdapter.freshPublic adapter fresh =
        FixedOneCanonicalAdapter.encodeInstance (some digest) := by
  exact
    (selectedRows_iff_sourceCheck
      digest canonical one producerAligned).trans
      (freshPublic_eq_encodeInstance_iff_sourceCheck
        adapter fresh digest (rawClaimOfAssignment z)
        claimedDigest ordered linkSemantics).symm

/-- Current isolated physical artifact specialization. The row-list equality
is kernel-checked by the arbitrary-batch owner. -/
theorem artifactRows_iff_freshPublic_eq_encodeInstance
    (adapter : AdapterParameters)
    (fresh : AdapterFresh)
    (digest : ProductionDigest)
    {z : Nat -> Nat}
    (canonical : forall column, z column < goldilocksP)
    (one : z 0 = 1)
    (producerAligned :
      FPrimeTerminalLinkBatch.ProducerAligned digest z)
    (claimedDigest : fresh.claimedDigest = digest)
    (ordered : fresh.ordered = [rawClaimOfAssignment z])
    (linkSemantics :
      adapter.step.freshLink =
        CanonicalPlainCarrierSource.sourceCheck) :
    Satisfies FPrimeTerminalLink.rows z <->
      FixedOneCanonicalAdapter.freshPublic adapter fresh =
        FixedOneCanonicalAdapter.encodeInstance (some digest) := by
  rw [← FPrimeTerminalLinkBatch.rows_one_eq_artifact]
  exact selectedRows_iff_freshPublic_eq_encodeInstance
    adapter fresh digest canonical one producerAligned
    claimedDigest ordered linkSemantics

end Nightstream.Implementation.R1CS.FPrimeProductionFreshPublicSingletonRows
