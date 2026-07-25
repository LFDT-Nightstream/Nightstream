import Nightstream.Implementation.R1CS.Correspondence.FPrime.FPrimeProductionFreshPublicSingletonRows
import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.FixedOneLoweringAdapter
import Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

/-!
Contract: semantic refinement of the Rust-emitted fused terminal prior-link
program to the exact Boolean consumed by the selected typed Terminal lowering.

Assurance tier: artifact-checked relation refinement.

Owns:
- acceptance of the generated three-instruction program through its checked
  receipt compiler;
- exact equivalence with `Terminal.priorLinkAccepted` for the singleton
  production fresh-public claim;
- explicit alignment of the verifier-computed prior digest, fresh claim
  ordering, producer columns, and production source-link semantics.

Does not own: derivation of the digest from producer output rows, compiled Rust
semantics, full-history column placement, `runningCheck`, terminal `freshCheck`,
or the surrounding terminal verifier.

The fused 270-row relation implements the prior-public equality
`freshPublic = encodeInstance(priorDigest)`. It is not the independent
terminal `freshCheck`.

Emits constraints: no; it interprets the checked program compiler.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.LoweringRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

variable
  {Params : Type}
  {StructureDigest : Type}
  {Header : Type}
  {Running : Type}
  {NifsProof : Type}
  {Nebula : Type}
  {NebulaDigest : Type}
  {NebulaOpen : Type}

local notation "RawClaim" =>
  CanonicalPlainCarrierLink.RawClaim

local notation "AdapterParameters" =>
  FixedOneCanonicalAdapter.Parameters
    Params StructureDigest Header ProductionDigest Running RawClaim NifsProof
      Nebula NebulaDigest NebulaOpen

local notation "AdapterFresh" =>
  FixedOneCanonicalAdapter.FreshInput ProductionDigest RawClaim Nebula

local notation "DirectState" =>
  Nightstream.HyperNova.Construction2.State
    ProductionDigest Running RawClaim Nebula

/-- The exact Rust-emitted source schedule accepts exactly when the selected
typed Terminal program's prior-public equality accepts. The digest is explicit
here; a surrounding output-owner theorem must derive it from producer rows. -/
theorem generatedPlain_accepts_iff_priorLinkAccepted
    [DecidableEq ProductionDigest]
    [DecidableEq Running]
    [DecidableEq RawClaim]
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
    (digest : ProductionDigest)
    {assignment : Nat -> Nat}
    (canonical :
      forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (producerAligned :
      FPrimeTerminalLinkBatch.ProducerAligned digest assignment)
    (priorDigest :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorDigest
          (FixedOneLoweringAdapter.parameters adapter configuration)
          statement proof =
        some digest)
    (claimedDigest :
      proof.fresh.claimedDigest = digest)
    (ordered :
      proof.fresh.ordered =
        [FPrimeProductionFreshPublicSingletonRows.rawClaimOfAssignment
          assignment])
    (linkSemantics :
      adapter.step.freshLink =
        CanonicalPlainCarrierSource.sourceCheck) :
    Accepts generatedPlain 1 assignment <->
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted
          (FixedOneLoweringAdapter.parameters adapter configuration)
          statement proof =
        true := by
  rw [generated_plain_accepts_iff_selectedRows]
  rw [
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted,
    priorDigest
  ]
  simpa [
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.encodedEqual
  ] using
    (FPrimeProductionFreshPublicSingletonRows.selectedRows_iff_freshPublic_eq_encodeInstance
      adapter proof.fresh digest canonical one producerAligned
      claimedDigest ordered linkSemantics)

end Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.LoweringRefinement
