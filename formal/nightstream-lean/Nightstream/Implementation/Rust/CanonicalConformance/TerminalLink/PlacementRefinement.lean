import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.FPrimeFullHistoryCurrentTerminalLinkPlacement
import Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

/-!
Contract: place the checked Rust-emitted terminal-link source program in the
current bounded full-history row range.

Assurance tier: artifact-checked bounded placement refinement.

Owns:
- exact equality between the relabeled singleton compiler output and the
  generated current `terminal.latest_link` range;
- equivalence between acceptance of the Rust-emitted source program on the
  pulled assignment and satisfaction of that generated range;
- refinement of that placed program to the selected typed Terminal
  `priorLinkAccepted` Boolean at the digest derived from recursive output rows.

Does not own: compiled Rust semantics, a whole-program full-history artifact,
host shape checks, Poseidon2 collision resistance, `runningCheck`, terminal
`freshCheck`, or the surrounding terminal verifier.

The placed 270-row program is the fused prior-public equality. It is not the
independent terminal `freshCheck`.

Emits constraints: no; it composes the checked source compiler with the
artifact-checked column placement.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeTerminalLinkBatch
open Nightstream.Implementation.R1CS.FPrimeFullHistoryCurrentTerminalLinkPlacementSound
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.Program
open Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.ProgramRefinement

abbrev ProductionDigest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

/-- Relabeling the checked singleton compiler output yields exactly the
bounded rows emitted at the current full-history placement. -/
theorem generatedPlain_compile_eq_currentPlacement :
    Option.map (List.map (Relabel.row columnMap))
        (compile generatedPlain 1) =
      some FPrimeFullHistoryCurrentTerminalLinkPlacement.rows := by
  rw [generated_plain_compile]
  simp only [Option.map_some]
  rw [rows_one_eq_artifact, mapped_rows_eq_generated]

/-- Acceptance of the checked source program on the pulled assignment is
exactly satisfaction of its generated current placement. -/
theorem generatedPlain_accepts_pulled_iff_generatedRows
    (assignment : Nat -> Nat) :
    Accepts generatedPlain 1 (Pulled assignment) <->
      Satisfies
        FPrimeFullHistoryCurrentTerminalLinkPlacement.rows assignment := by
  rw [generated_plain_accepts_iff_selectedRows, rows_one_eq_artifact]
  exact (generatedRows_iff_localRows assignment).symm

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

/-- The checked Rust-emitted program, after exact current placement, accepts
exactly the prior-public equality Boolean consumed by the selected typed
Terminal program. The verifier-computed digest is derived from the recursive
output owner; no carried digest is used as authority. -/
theorem generatedPlain_accepts_pulled_iff_loweringPriorLinkAccepted
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
    Accepts generatedPlain 1 (Pulled assignment) <->
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted
          (FixedOneLoweringAdapter.parameters adapter configuration)
          statement proof =
        true := by
  rw [generatedPlain_accepts_pulled_iff_generatedRows]
  exact
    generatedRows_iff_loweringPriorLinkAccepted
      adapter configuration statement proof goldilocksPrime canonical one
      outputRows priorDigest claimedDigest ordered linkSemantics

end Nightstream.Implementation.Rust.CanonicalConformance.TerminalLink.PlacementRefinement
