import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Sampler.Refinement.OneScalarRows

/-!
Semantic composition of all sixteen production lanes for one recursive-profile
`Pi_RLC` scalar challenge.

Owns: the protocol -> digest block -> lane hierarchy; the exact accepted-count
chain from zero through all 64 candidates; and simultaneous refinement of every
candidate leaf to the independent transcript machine and verifier-owned sampler
decision.

Does not own: the 54-of-64 selection-tail semantics, coefficient-vector
assembly, subsequent scalar challenges, Rust trace conformance, row removal, or
aggregate circuit costs.

Emits constraints: no.

Authority boundary: the generated owner supplies accepted equations only.
Candidate values come from the independent transcript machine, accept/symbol
decisions come from `ProductionAlphabet.verifier`, and every counter bound is
derived from those Boolean decisions rather than trusted witness values.

| Protocol | Phase | Child | Mathematical obligation | Lean result |
|---|---|---|---|---|
| `Pi_RLC` | sampler/block 0 | lanes 0..3 | candidates 0..15 refine transcript and sampler semantics | `Refines.block0` |
| `Pi_RLC` | sampler/block 1 | lanes 0..3 | candidates 16..31 refine transcript and sampler semantics | `Refines.block1` |
| `Pi_RLC` | sampler/block 2 | lanes 0..3 | candidates 32..47 refine transcript and sampler semantics | `Refines.block2` |
| `Pi_RLC` | sampler/block 3 | lanes 0..3 | candidates 48..63 refine transcript and sampler semantics | `Refines.block3` |
| `Pi_RLC` | sampler/count chain | all sixteen lanes | exact prefix starts at zero and never exceeds processed candidates | `Refines.finalCountLe` |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalar

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript

def index0 : Fin 4 := ⟨0, by decide⟩
def index1 : Fin 4 := ⟨1, by decide⟩
def index2 : Fin 4 := ⟨2, by decide⟩
def index3 : Fin 4 := ⟨3, by decide⟩

abbrev LaneResult
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (block lane : Fin 4) : Prop :=
  LaneRows.RefinesMachineLane assignment canonical block lane
    (OneScalarRows.cumPrev block lane)
    (ChunkOrder.accepted_refines_lane prime canonical one accepted block lane)

structure BlockRefines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment)
    (block : Fin 4) : Prop where
  lane0 : LaneResult prime canonical one accepted block index0
  lane1 : LaneResult prime canonical one accepted block index1
  lane2 : LaneResult prime canonical one accepted block index2
  lane3 : LaneResult prime canonical one accepted block index3

structure Refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) : Prop where
  initialCountZero : assignment OneScalarRows.initialCountColumn = 0
  block0 : BlockRefines prime canonical one accepted index0
  block1 : BlockRefines prime canonical one accepted index1
  block2 : BlockRefines prime canonical one accepted index2
  block3 : BlockRefines prime canonical one accepted index3
  finalCountLe : assignment 356193 <=
    Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound

/-- All sixteen exact lane pieces refine the independent transcript/sampler
semantics. The proof carries an explicit integer bound after every lane, so a
forged cumulative witness cannot authorize a later lane. -/
theorem accepted_refines
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat -> Nat}
    (canonical : ChunkOrder.CanonicalAssignment assignment)
    (one : assignment 0 = 1)
    (accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment) :
    Refines prime canonical one accepted := by
  have initialZero : assignment 350649 = 0 := by
    simpa [OneScalarRows.initialCountColumn] using
      OneScalarRows.accepted_initialCount_zero canonical one accepted

  have lane00 : LaneResult prime canonical one accepted index0 index0 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index0 index0 (OneScalarRows.cumPrev index0 index0)
      (by change assignment 350649 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index0 index0)
  have count00Raw := lane00.production.finalCount_le_add_four
  have count00 : assignment 352011 <= 4 := by
    change assignment 352011 <= assignment 350649 + 4 at count00Raw
    omega

  have lane01 : LaneResult prime canonical one accepted index0 index1 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index0 index1 (OneScalarRows.cumPrev index0 index1)
      (by change assignment 352011 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index0 index1)
  have count01Raw := lane01.production.finalCount_le_add_four
  have count01 : assignment 352169 <= 8 := by
    change assignment 352169 <= assignment 352011 + 4 at count01Raw
    omega

  have lane02 : LaneResult prime canonical one accepted index0 index2 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index0 index2 (OneScalarRows.cumPrev index0 index2)
      (by change assignment 352169 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index0 index2)
  have count02Raw := lane02.production.finalCount_le_add_four
  have count02 : assignment 352327 <= 12 := by
    change assignment 352327 <= assignment 352169 + 4 at count02Raw
    omega

  have lane03 : LaneResult prime canonical one accepted index0 index3 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index0 index3 (OneScalarRows.cumPrev index0 index3)
      (by change assignment 352327 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index0 index3)
  have count03Raw := lane03.production.finalCount_le_add_four
  have count03 : assignment 352485 <= 16 := by
    change assignment 352485 <= assignment 352327 + 4 at count03Raw
    omega

  have lane10 : LaneResult prime canonical one accepted index1 index0 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index1 index0 (OneScalarRows.cumPrev index1 index0)
      (by change assignment 352485 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index1 index0)
  have count10Raw := lane10.production.finalCount_le_add_four
  have count10 : assignment 353247 <= 20 := by
    change assignment 353247 <= assignment 352485 + 4 at count10Raw
    omega

  have lane11 : LaneResult prime canonical one accepted index1 index1 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index1 index1 (OneScalarRows.cumPrev index1 index1)
      (by change assignment 353247 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index1 index1)
  have count11Raw := lane11.production.finalCount_le_add_four
  have count11 : assignment 353405 <= 24 := by
    change assignment 353405 <= assignment 353247 + 4 at count11Raw
    omega

  have lane12 : LaneResult prime canonical one accepted index1 index2 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index1 index2 (OneScalarRows.cumPrev index1 index2)
      (by change assignment 353405 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index1 index2)
  have count12Raw := lane12.production.finalCount_le_add_four
  have count12 : assignment 353563 <= 28 := by
    change assignment 353563 <= assignment 353405 + 4 at count12Raw
    omega

  have lane13 : LaneResult prime canonical one accepted index1 index3 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index1 index3 (OneScalarRows.cumPrev index1 index3)
      (by change assignment 353563 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index1 index3)
  have count13Raw := lane13.production.finalCount_le_add_four
  have count13 : assignment 353721 <= 32 := by
    change assignment 353721 <= assignment 353563 + 4 at count13Raw
    omega

  have lane20 : LaneResult prime canonical one accepted index2 index0 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index2 index0 (OneScalarRows.cumPrev index2 index0)
      (by change assignment 353721 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index2 index0)
  have count20Raw := lane20.production.finalCount_le_add_four
  have count20 : assignment 354483 <= 36 := by
    change assignment 354483 <= assignment 353721 + 4 at count20Raw
    omega

  have lane21 : LaneResult prime canonical one accepted index2 index1 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index2 index1 (OneScalarRows.cumPrev index2 index1)
      (by change assignment 354483 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index2 index1)
  have count21Raw := lane21.production.finalCount_le_add_four
  have count21 : assignment 354641 <= 40 := by
    change assignment 354641 <= assignment 354483 + 4 at count21Raw
    omega

  have lane22 : LaneResult prime canonical one accepted index2 index2 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index2 index2 (OneScalarRows.cumPrev index2 index2)
      (by change assignment 354641 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index2 index2)
  have count22Raw := lane22.production.finalCount_le_add_four
  have count22 : assignment 354799 <= 44 := by
    change assignment 354799 <= assignment 354641 + 4 at count22Raw
    omega

  have lane23 : LaneResult prime canonical one accepted index2 index3 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index2 index3 (OneScalarRows.cumPrev index2 index3)
      (by change assignment 354799 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index2 index3)
  have count23Raw := lane23.production.finalCount_le_add_four
  have count23 : assignment 354957 <= 48 := by
    change assignment 354957 <= assignment 354799 + 4 at count23Raw
    omega

  have lane30 : LaneResult prime canonical one accepted index3 index0 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index3 index0 (OneScalarRows.cumPrev index3 index0)
      (by change assignment 354957 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index3 index0)
  have count30Raw := lane30.production.finalCount_le_add_four
  have count30 : assignment 355719 <= 52 := by
    change assignment 355719 <= assignment 354957 + 4 at count30Raw
    omega

  have lane31 : LaneResult prime canonical one accepted index3 index1 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index3 index1 (OneScalarRows.cumPrev index3 index1)
      (by change assignment 355719 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index3 index1)
  have count31Raw := lane31.production.finalCount_le_add_four
  have count31 : assignment 355877 <= 56 := by
    change assignment 355877 <= assignment 355719 + 4 at count31Raw
    omega

  have lane32 : LaneResult prime canonical one accepted index3 index2 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index3 index2 (OneScalarRows.cumPrev index3 index2)
      (by change assignment 355877 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index3 index2)
  have count32Raw := lane32.production.finalCount_le_add_four
  have count32 : assignment 356035 <= 60 := by
    change assignment 356035 <= assignment 355877 + 4 at count32Raw
    omega

  have lane33 : LaneResult prime canonical one accepted index3 index3 :=
    LaneRows.accepted_refines_machineLane prime canonical one accepted
      index3 index3 (OneScalarRows.cumPrev index3 index3)
      (by change assignment 356035 + 4 <= 64; omega)
      (OneScalarRows.accepted_laneRows accepted index3 index3)
  have count33Raw := lane33.production.finalCount_le_add_four
  have count33 : assignment 356193 <= 64 := by
    change assignment 356193 <= assignment 356035 + 4 at count33Raw
    omega

  exact {
    initialCountZero := by
      simpa [OneScalarRows.initialCountColumn] using initialZero
    block0 := {
      lane0 := lane00
      lane1 := lane01
      lane2 := lane02
      lane3 := lane03 }
    block1 := {
      lane0 := lane10
      lane1 := lane11
      lane2 := lane12
      lane3 := lane13 }
    block2 := {
      lane0 := lane20
      lane1 := lane21
      lane2 := lane22
      lane3 := lane23 }
    block3 := {
      lane0 := lane30
      lane1 := lane31
      lane2 := lane32
      lane3 := lane33 }
    finalCountLe := by
      simpa [Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet.candidateBound]
        using count33
  }

private theorem fin4_cases (index : Fin 4) :
    index = index0 \/ index = index1 \/ index = index2 \/ index = index3 := by
  have indexLt := index.isLt
  have values : index.val = 0 \/ index.val = 1 \/
      index.val = 2 \/ index.val = 3 := by omega
  rcases values with value | value | value | value
  · exact Or.inl (Fin.ext value)
  · exact Or.inr (Or.inl (Fin.ext value))
  · exact Or.inr (Or.inr (Or.inl (Fin.ext value)))
  · exact Or.inr (Or.inr (Or.inr (Fin.ext value)))

/-- Total lookup preserving the protocol -> block -> lane ownership tree. -/
theorem Refines.lane
    {prime : EuclidPrime goldilocksP}
    {assignment : Nat -> Nat}
    {canonical : ChunkOrder.CanonicalAssignment assignment}
    {one : assignment 0 = 1}
    {accepted :
      FPrimeFullHistoryRecursivePiRlcTranscriptRhos.Accepted assignment}
    (refinement : Refines prime canonical one accepted)
    (block lane : Fin 4) :
    LaneResult prime canonical one accepted block lane := by
  rcases fin4_cases block with rfl | rfl | rfl | rfl <;>
    rcases fin4_cases lane with rfl | rfl | rfl | rfl
  all_goals first
    | exact refinement.block0.lane0
    | exact refinement.block0.lane1
    | exact refinement.block0.lane2
    | exact refinement.block0.lane3
    | exact refinement.block1.lane0
    | exact refinement.block1.lane1
    | exact refinement.block1.lane2
    | exact refinement.block1.lane3
    | exact refinement.block2.lane0
    | exact refinement.block2.lane1
    | exact refinement.block2.lane2
    | exact refinement.block2.lane3
    | exact refinement.block3.lane0
    | exact refinement.block3.lane1
    | exact refinement.block3.lane2
    | exact refinement.block3.lane3

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Refinement.OneScalar
