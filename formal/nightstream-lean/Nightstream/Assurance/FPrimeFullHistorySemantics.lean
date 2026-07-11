import Nightstream.Assurance.FPrimeConcreteNifs
import Nightstream.Implementation.R1CS.FPrimeFullHistoryBaseGenericSound
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursivePreludeHashes
import Nightstream.Implementation.R1CS.FPrimeFullHistoryRecursiveShellSound

/-!
Contract: fixed executable services for the exact plain/stateless `[1,1]`
full-history profile.

The minimal supported carrier has one fresh claim per producer.  Its chunk
shape is fixed at preprocessing time, while the fresh public projection is
the four-lane `x_out` value carried by the claim.  Unsupported starts or batch
cardinalities receive an inert digest and cannot be confused with either
supported transition.
-/

namespace Nightstream.Assurance.FPrimeFullHistorySemantics

open Nightstream.Implementation.R1CS

abbrev Digest := FPrimeConcreteNifs.Digest
abbrev Fresh := FPrimeConcreteNifs.Fresh
abbrev Accumulator := FPrimeConcreteNifs.Accumulator
abbrev Proof := FPrimeConcreteNifs.Proof

/-- Verifier-owned digest schedule for the only two producer starts in the
supported artifact. -/
def chunkDigest (start : Nat) (fresh : List Fresh) : Digest :=
  if fresh.length != 1 then []
  else if start = 0 then
    FPrimeFullHistoryBaseStepSound.chunkDigestValue
  else if start = 1 then
    FPrimeFullHistoryRecursivePreludeHashes.chunkDigestValue
  else []

/-- The minimal carrier's delayed public link. -/
def freshLink (digest : Digest) (fresh : Fresh) : Bool :=
  decide (digest = fresh.publicXOut)

/-- Stateful application semantics is outside this stateless profile. -/
def applicationStep (_prior : Digest) (_fresh : List Fresh)
    (_next : Digest) : Bool := false

/-- Fully fixed M4 step semantics for the supported profile. -/
def semantics :=
  FPrimeConcreteNifs.stepSemantics chunkDigest freshLink applicationStep

theorem base_chunk_digest (fresh : Fresh) :
    chunkDigest 0 [fresh] =
      FPrimeFullHistoryBaseStepSound.chunkDigestValue := by
  simp [chunkDigest]

theorem recursive_chunk_digest (fresh : Fresh) :
    chunkDigest 1 [fresh] =
      FPrimeFullHistoryRecursivePreludeHashes.chunkDigestValue := by
  simp [chunkDigest]

/-- Exact recursive-prelude rows compute the fixed start-one chunk digest.
The start-column equality is kept explicit here so this lemma can be reused
at the narrow owner boundary. -/
theorem recursiveCoreLaws_of_start
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (start : assignment
      FPrimeFullHistoryRecursivePreludeHashes.startColumn = 1)
    (preludeSatisfies :
      Satisfies FPrimeFullHistoryRecursivePrelude.rows assignment)
    (nextFresh : Fresh) :
    FPrimeFullHistoryRecursiveShellSound.CoreLaws assignment semantics
      nextFresh := by
  have lanes := FPrimeFullHistoryRecursivePreludeHashes.next_chunk_digest_fixed
    canonical one start preludeSatisfies
  have lane0 := lanes 0 (by decide)
  have lane1 := lanes 1 (by decide)
  have lane2 := lanes 2 (by decide)
  have lane3 := lanes 3 (by decide)
  have outputValues :
      FPrimeFullHistoryRecursivePreludeHashes.nextChunkDigestColumns.map
          assignment =
        FPrimeFullHistoryRecursivePreludeHashes.chunkDigestValue := by
    simpa [FPrimeFullHistoryRecursivePreludeHashes.nextChunkDigestColumns,
      FPrimeFullHistoryRecursivePreludeHashes.chunkDigestValue,
      show List.range 4 = [0, 1, 2, 3] by decide] using
        And.intro lane0 (And.intro lane1 (And.intro lane2 lane3))
  refine ⟨?_⟩
  change chunkDigest 1 [nextFresh] =
    FPrimeFullHistoryBaseStepSound.digestAt
      (FPrimeFullHistoryRecursiveShellSound.nextValues assignment) 14
  rw [recursive_chunk_digest]
  symm
  calc
    FPrimeFullHistoryBaseStepSound.digestAt
        (FPrimeFullHistoryRecursiveShellSound.nextValues assignment) 14 =
      FPrimeFullHistoryRecursivePreludeHashes.nextChunkDigestColumns.map
        assignment := by
          simp [FPrimeFullHistoryBaseStepSound.digestAt,
            FPrimeFullHistoryRecursiveShellSound.nextValues,
            FPrimeFullHistoryRecursiveOutput.stateOutColumns,
            FPrimeFullHistoryRecursivePreludeHashes.nextChunkDigestColumns,
            show List.range 4 = [0, 1, 2, 3] by decide]
    _ = FPrimeFullHistoryRecursivePreludeHashes.chunkDigestValue := outputValues

/-- Fixed-semantics `CoreLaws` require no caller assertion.  The authoritative
start value is derived from the exact base owner and adjacent state-link rows;
the exact recursive prelude then computes the four digest lanes. -/
theorem recursiveCoreLaws
    (goldilocksPrime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (baseSatisfies : Satisfies FPrimeFullHistoryBase.rows assignment)
    (stateLinkSatisfies :
      Satisfies FPrimeFullHistoryStateLink.rows assignment)
    (preludeSatisfies :
      Satisfies FPrimeFullHistoryRecursivePrelude.rows assignment)
    (nextFresh : Fresh) :
    FPrimeFullHistoryRecursiveShellSound.CoreLaws assignment semantics
      nextFresh := by
  have baseFacts := FPrimeFullHistoryBaseFacts.sound goldilocksPrime canonical
    one baseSatisfies
  have start :=
    (FPrimeFullHistoryRecursiveShellSound.counterInputs_one baseFacts canonical
      one stateLinkSatisfies).2
  rw [FPrimeFullHistoryCounterSound.concreteColumns.2.1] at start
  change assignment 10843 = 1 at start
  exact recursiveCoreLaws_of_start canonical one start preludeSatisfies
    nextFresh

/-- Primitive base laws are consequences of the fixed executable semantics,
not assumptions supplied to CIR-SOUND. -/
theorem baseLaws (fresh : Fresh) :
    FPrimeFullHistoryBaseGenericSound.BaseLaws semantics fresh := by
  exact {
    emptyRunningDigest := rfl
    initialNebula := rfl
    chunkDigest := base_chunk_digest fresh
    nebulaNone := rfl
  }

theorem freshLinkLaws :
    FPrimeFullHistoryBaseGenericSound.FreshLinkLaws semantics
      (fun fresh => fresh.publicXOut) := by
  exact {
    freshLink_eq := by
      intro digest fresh
      rfl
  }

end Nightstream.Assurance.FPrimeFullHistorySemantics
