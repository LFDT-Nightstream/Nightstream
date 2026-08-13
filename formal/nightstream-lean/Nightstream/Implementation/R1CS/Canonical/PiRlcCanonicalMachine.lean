import Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
import Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-!
Contract: instantiate the production-shaped `Pi_RLC` coefficient sampler with
the Lean-owned width-8 Poseidon2 duplex.

The older correspondence machine extracts its permutation from a captured
600-row artifact.  This module instead uses
`Poseidon2Reference.referencePermutation` through `Poseidon2Duplex`, so the
semantic machine is selected by the canonical Lean schedule.

Owns:
- overwrite absorption and the raw-pair length word;
- the exact scalar and digest-block domain words;
- two complemented low-word candidates from each of four digest lanes;
- one jointly owned successor state and eight-candidate block; and
- the fixed eight-block/54-of-64 successor state.

Does not own:
- the state handed off by `Pi_CCS`;
- symbolic/R1CS bit decomposition or rejection-selection rows;
- the projection-root challenge used by the quotient identities;
- Rust conformance, probability, or a complete `nifsVerify` recipe.

Assurance tier: model-level.  No generated row or Rust result is imported.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine

open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionAlphabet
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionSchedule
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler.ProductionStrongSet

/-- Rust's counter word modulus.  The selected coordinates are much smaller,
but spelling out the conversion keeps the schedule exact at arbitrary inputs. -/
def u64Modulus : Nat := 2 ^ 64

/-- `usize/u64` counter conversion before field absorption.  `absorbElem`
performs the subsequent Goldilocks reduction. -/
def word (value : Nat) : Nat :=
  value % u64Modulus

/-- Exact `append_fields_raw(&[first, second])` serialization: the length word
is part of the transcript. -/
def appendRawPair
    (constants : Constants) (state : Poseidon2Duplex.State)
    (first second : Nat) : Poseidon2Duplex.State :=
  Poseidon2Duplex.absorbElem constants (word second)
    (Poseidon2Duplex.absorbElem constants (word first)
      (Poseidon2Duplex.absorbElem constants (word 2) state))

/-- One digest call: apply the transcript's pre-squeeze gate and expose the
first four freshly permuted lanes. -/
def digest
    (constants : Constants) (state : Poseidon2Duplex.State) :
    Poseidon2Duplex.State × (Fin 4 → Nat) :=
  let next := Poseidon2Duplex.gate constants state
  (next, fun lane =>
    next.lanes ⟨lane.val, by
      have laneLt := lane.isLt
      change lane.val < width
      simp only [width]
      omega⟩)

/-- The bitwise complement of the `part`th low 16-bit word. -/
def laneChunk (lane : Nat) (part : Fin 2) : Chunk :=
  ⟨((chunkModulus - 1) + chunkModulus -
      ((lane / (2 ^ (16 * part.val))) % chunkModulus)) % chunkModulus,
    Nat.mod_lt _ (by decide)⟩

/-- Two exact candidates per lane and four lanes per digest, in lane-major order. -/
def digestChunks (lanes : Fin 4 → Nat) :
    Fin chunksPerDigest → Chunk :=
  fun position =>
    let lane : Fin 4 :=
      ⟨position.val / 2, by
        have positionLt : position.val < 8 := by
          exact position.isLt
        omega⟩
    let part : Fin 2 :=
      ⟨position.val % 2, Nat.mod_lt _ (by decide)⟩
    laneChunk (lanes lane) part

/-- Per-scalar domain separation. -/
def enterScalar
    (constants : Constants) (state : Poseidon2Duplex.State)
    (coordinate : Nat) : Poseidon2Duplex.State :=
  appendRawPair constants state 0 coordinate

/-- One counter block jointly returns its successor state and all eight
candidates.  The two projections cannot drift independently. -/
def digestBlock
    (constants : Constants) (state : Poseidon2Duplex.State)
    (counter : Nat) :
    Poseidon2Duplex.State × (Fin chunksPerDigest → Chunk) :=
  let result := digest constants (appendRawPair constants state 1 counter)
  (result.1, digestChunks result.2)

/-- Concrete canonical instantiation of the independent block schedule. -/
def machine (constants : Constants) :
    ProductionSchedule.Machine Poseidon2Duplex.State where
  enterScalar := enterScalar constants
  digestBlock := digestBlock constants

/-- Complete coefficient-vector sampler over the canonical machine. -/
def specification (constants : Constants) :
    Specification Poseidon2Duplex.State Chunk Coefficient Scalar :=
  ProductionSchedule.specification (machine constants) assembleCoefficients

@[simp] theorem machine_enterScalar
    (constants : Constants) (state : Poseidon2Duplex.State)
    (coordinate : Nat) :
    (machine constants).enterScalar state coordinate =
      enterScalar constants state coordinate :=
  rfl

@[simp] theorem machine_digestBlock
    (constants : Constants) (state : Poseidon2Duplex.State)
    (counter : Nat) :
    (machine constants).digestBlock state counter =
      digestBlock constants state counter :=
  rfl

/-- The lane/part quotient-remainder address is exact. -/
theorem digestChunks_lane_part
    (lanes : Fin 4 → Nat) (lane : Fin 4) (part : Fin 2) :
    digestChunks lanes
        ⟨lane.val * 2 + part.val, by
          have laneLt := lane.isLt
          have partLt := part.isLt
          change lane.val * 2 + part.val < chunksPerDigest
          simp only [chunksPerDigest]
          omega⟩ =
      laneChunk (lanes lane) part := by
  unfold digestChunks
  dsimp only
  have laneEq :
      (⟨(lane.val * 2 + part.val) / 2, by
        have laneLt := lane.isLt
        have partLt := part.isLt
        omega⟩ : Fin 4) = lane := by
    apply Fin.ext
    change (lane.val * 2 + part.val) / 2 = lane.val
    have partLt := part.isLt
    omega
  have partEq :
      (⟨(lane.val * 2 + part.val) % 2, Nat.mod_lt _ (by decide)⟩ :
        Fin 2) = part := by
    apply Fin.ext
    change (lane.val * 2 + part.val) % 2 = part.val
    have partLt := part.isLt
    omega
  rw [laneEq, partEq]

/-- Every digest block returns a freshly permuted, cursor-zero state. -/
@[simp] theorem digestBlock_absorbed_zero
    (constants : Constants) (state : Poseidon2Duplex.State)
    (counter : Nat) :
    (digestBlock constants state counter).1.absorbed = 0 :=
  rfl

/-- Candidate output and successor state agree with the same eight canonical
digest blocks. -/
theorem fixedSchedule_successorState
    (constants : Constants)
    (initial : Poseidon2Duplex.State)
    (coordinate : Nat) :
    (sourceAt (specification constants) initial coordinate).nextState =
      stateBeforeBlock (machine constants)
        (enterScalar constants
          (stateAt (specification constants) initial coordinate) coordinate)
        coordinate digestRounds := by
  exact source_nextState_eq_fixedBlockState
    (machine constants) assembleCoefficients initial coordinate

/-- The independent strong-set coefficient law applies directly to this
canonical machine. -/
theorem sampledChallenge_valid
    (constants : Constants)
    {challengeCount : Nat}
    (initial : Poseidon2Duplex.State)
    (batch :
      BatchExecution (specification constants) challengeCount
        candidateBound initial)
    (coordinate : Fin challengeCount) :
    ScalarValid (challenge batch coordinate) :=
  ProductionStrongSet.sampledChallenge_valid
    (machine constants) initial batch coordinate

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalMachine
