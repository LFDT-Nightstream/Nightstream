import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.CanonicalPublicInputLinkProgram
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneCanonicalAdapter

/-!
Contract: exact singleton reduction from the compact fixed-one fresh-public
equality to the production plain-carrier link program.

Assurance tier: model-level.

Owns:
- the HyperNova one-fresh-instance specialization of the native adapter;
- exact reduction of compact `freshPublic = encodeInstance` to the canonical
  270-coordinate source check;
- exact reduction to the six-phase, 273-obligation typed source program and
  the logical paper public-input equality.

Does not own: a multi-fresh batch reduction, Rust-source or compiled-Rust
semantics, R1CS rows, Poseidon2, NIFS, or an encoding profile.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge

open Nightstream.Implementation.Encoding.FPrime
open CanonicalPlainCarrierLink
open CanonicalPlainCarrierSource
open CanonicalPublicInputLinkProgram
open FixedOneCanonicalAdapter

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

/-- On the paper's singleton fresh input, compact adapter equality is exactly
the production source-shaped 270-coordinate public-link check. -/
theorem freshPublic_eq_encodeInstance_iff_sourceCheck
    (adapter : AdapterParameters)
    (fresh : AdapterFresh)
    (digest : ProductionDigest)
    (raw : RawClaim)
    (claimedDigest : fresh.claimedDigest = digest)
    (ordered : fresh.ordered = [raw])
    (linkSemantics :
      adapter.step.freshLink = CanonicalPlainCarrierSource.sourceCheck) :
    FixedOneCanonicalAdapter.freshPublic adapter fresh =
        FixedOneCanonicalAdapter.encodeInstance (some digest) ↔
      CanonicalPlainCarrierSource.sourceCheck digest raw = true := by
  cases fresh with
  | mk claimedDigestValue nifsContext orderedValues =>
      simp only at claimedDigest ordered
      subst claimedDigestValue
      subst orderedValues
      simp [FixedOneCanonicalAdapter.freshPublic,
        FixedOneCanonicalAdapter.encodeInstance, linkSemantics]

/-- The compact singleton equality is exactly the selected six-phase source
program, whose definitional cost is 273 scalar obligations. -/
theorem freshPublic_eq_encodeInstance_iff_program
    (adapter : AdapterParameters)
    (fresh : AdapterFresh)
    (digest : ProductionDigest)
    (raw : RawClaim)
    (claimedDigest : fresh.claimedDigest = digest)
    (ordered : fresh.ordered = [raw])
    (linkSemantics :
      adapter.step.freshLink = CanonicalPlainCarrierSource.sourceCheck) :
    FixedOneCanonicalAdapter.freshPublic adapter fresh =
        FixedOneCanonicalAdapter.encodeInstance (some digest) ↔
      CanonicalPublicInputLinkProgram.run
          CanonicalPublicInputLinkProgram.plain digest carrierWidth raw =
        true := by
  rw [freshPublic_eq_encodeInstance_iff_sourceCheck
    adapter fresh digest raw claimedDigest ordered linkSemantics]
  rw [CanonicalPublicInputLinkProgram.run_plain_eq_sourceCheck]

/-- Full singleton refinement chain from compact adapter equality to the
typed and logical HyperNova public-input relations. -/
theorem freshPublic_eq_encodeInstance_reduces_to_logicalPaperLink
    (adapter : AdapterParameters)
    (fresh : AdapterFresh)
    (digest : ProductionDigest)
    (raw : RawClaim)
    (claimedDigest : fresh.claimedDigest = digest)
    (ordered : fresh.ordered = [raw])
    (linkSemantics :
      adapter.step.freshLink = CanonicalPlainCarrierSource.sourceCheck) :
    FixedOneCanonicalAdapter.freshPublic adapter fresh =
        FixedOneCanonicalAdapter.encodeInstance (some digest) ↔
      ∃ typed logical,
        CanonicalPlainCarrierLink.check digest typed = true ∧
          CanonicalPublicInputLink.check digest logical = true ∧
          raw.mIn = typed.mIn ∧
          raw.x = typed.x.coordinates ∧
          typed = CanonicalPlainCarrierLink.completeClaim logical := by
  rw [freshPublic_eq_encodeInstance_iff_sourceCheck
    adapter fresh digest raw claimedDigest ordered linkSemantics]
  exact
    CanonicalPlainCarrierSource.sourceCheck_reduces_to_logicalPaperLink
      digest raw

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ProductionFreshPublicSingletonBridge
