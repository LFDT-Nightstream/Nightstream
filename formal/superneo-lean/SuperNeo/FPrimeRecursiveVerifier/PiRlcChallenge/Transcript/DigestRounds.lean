import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Sampler.Chunk
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.Cursor

/-!
Owns: digest-lane extraction, the four-round counter schedule, and one rho's
ordered 64-chunk transcript trace.

Does not own: sponge cursor mechanics, rejection/selection arithmetic, the
concrete Poseidon2 round constants, or the authority of the incoming cursor.

Emits constraints: no. This file states executable transcript semantics.

Authority boundary: `rhoDigestTrace` derives chunks only from the supplied
cursor and permutation core; callers must bind both to production.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `laneChunk`, `digestStateChunks` | `challenge.transcript.lane_bit_decomposition` | Four little-endian 16-bit chunks from each of four canonical lanes | Supplied canonical field lanes | No — Rust refinement open |
| `digestRound`, `runDigestRounds` | `challenge.transcript.digest_rounds` | Replays `[1, counter]` with wrapping counters | Authoritative cursor and supplied core | No — concrete Poseidon2/Rust refinement open |
| `rhoDigestTrace` | `challenge.transcript` | Replays `[0, rhoIndex]` followed by four digest rounds | Authoritative cursor and supplied core | No — concrete Poseidon2/Rust refinement open |
| `runDigestRounds_chunks_length` | `challenge.transcript.digest_rounds` | Each round contributes exactly sixteen chunks | Executable model above | No — Rust refinement open |

The sponge cursor mechanics and event order are concrete. `Poseidon2Core` keeps
the width-8 permutation as an explicit dependency so this file does not copy
the round constants already maintained by the Nightstream formal project.
Closing that dependency requires one shared concrete-core refinement; an
arbitrary `Poseidon2Core` is not claimed to be the production permutation.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- One or more digest rounds and their ordered 16-bit chunks. -/
structure DigestRoundsResult where
  cursor : SpongeCursor
  chunks : List Chunk

def digestLanes : Nat := 4
def chunksPerLane : Nat := 4
def fixedDigestRounds : Nat := 4
def u64Modulus : Nat := 2 ^ 64

/-- One 16-bit chunk from a canonical field lane, in little-endian order. -/
def laneChunk (lane : F) (offset : Fin chunksPerLane) : Chunk :=
  ⟨(lane.val / (2 ^ (16 * offset.val))) % chunkModulus,
    Nat.mod_lt _ (by decide : 0 < chunkModulus)⟩

/-- Four low-to-high 16-bit chunks from one canonical 64-bit lane. -/
def laneChunks (lane : F) : List Chunk :=
  List.ofFn (fun offset : Fin chunksPerLane => laneChunk lane offset)

/-- The first four rate lanes, in lane order. -/
def firstDigestLanes (state : SpongeState) : List F :=
  List.ofFn (fun index : Fin digestLanes =>
    state ⟨index.val, Nat.lt_trans index.isLt (by decide)⟩)

/-- Sixteen chunks in the same order as Rust `digest32().chunks_exact(2)`. -/
def digestStateChunks (state : SpongeState) : List Chunk :=
  (firstDigestLanes state).flatMap laneChunks

/-- Wrapping `u64` counter increment used by the native sampler. -/
def nextCounter (counter : Nat) : Nat :=
  (counter + 1) % u64Modulus

/-- One `[1, counter]` absorb followed by `digest32`. -/
def digestRound
    (core : Poseidon2Core) (cursor : SpongeCursor) (counter : Nat) :
    DigestRoundsResult :=
  let appended := appendFieldsRaw core cursor
    [F.ofNat 1, F.ofNat (counter % u64Modulus)]
  let digested := digestCursor core appended
  { cursor := digested
    chunks := digestStateChunks digested.state }

/-- Repeated digest rounds with the native wrapping counter schedule. -/
def runDigestRounds (core : Poseidon2Core) :
    Nat → SpongeCursor → Nat → DigestRoundsResult
  | 0, cursor, _ => { cursor := cursor, chunks := [] }
  | rounds + 1, cursor, counter =>
      let first := digestRound core cursor counter
      let rest := runDigestRounds core rounds first.cursor
        (nextCounter counter)
      { cursor := rest.cursor
        chunks := first.chunks ++ rest.chunks }

/-- Outer `[0, rhoIndex]` separator and exactly four digest rounds. -/
def rhoDigestTrace
    (core : Poseidon2Core) (cursor : SpongeCursor) (rhoIndex : Nat) :
    DigestRoundsResult :=
  let separated := appendFieldsRaw core cursor
    [F.ofNat 0, F.ofNat (rhoIndex % u64Modulus)]
  runDigestRounds core fixedDigestRounds separated
    (rhoIndex % u64Modulus)

@[simp] theorem laneChunk_val (lane : F) (offset : Fin chunksPerLane) :
    (laneChunk lane offset).val =
      (lane.val / (2 ^ (16 * offset.val))) % chunkModulus := rfl

@[simp] theorem laneChunks_length (lane : F) :
    (laneChunks lane).length = chunksPerLane := by
  simp [laneChunks, chunksPerLane]

@[simp] theorem firstDigestLanes_length (state : SpongeState) :
    (firstDigestLanes state).length = digestLanes := by
  simp [firstDigestLanes, digestLanes]

@[simp] theorem digestStateChunks_length (state : SpongeState) :
    (digestStateChunks state).length = 16 := by
  simp [digestStateChunks, firstDigestLanes, laneChunks,
    digestLanes, chunksPerLane]

@[simp] theorem digestRound_chunks_length
    (core : Poseidon2Core) (cursor : SpongeCursor) (counter : Nat) :
    (digestRound core cursor counter).chunks.length = 16 := by
  simp [digestRound]

@[simp] theorem digestRound_absorbed
    (core : Poseidon2Core) (cursor : SpongeCursor) (counter : Nat) :
    (digestRound core cursor counter).cursor.absorbed.val = 0 := by
  simp [digestRound]

theorem runDigestRounds_chunks_length
    (core : Poseidon2Core) (rounds : Nat)
    (cursor : SpongeCursor) (counter : Nat) :
    (runDigestRounds core rounds cursor counter).chunks.length = 16 * rounds := by
  induction rounds generalizing cursor counter with
  | zero => simp [runDigestRounds]
  | succ rounds ih =>
      simp [runDigestRounds, digestRound_chunks_length, ih]
      omega

@[simp] theorem rhoDigestTrace_chunks_length
    (core : Poseidon2Core) (cursor : SpongeCursor) (rhoIndex : Nat) :
    (rhoDigestTrace core cursor rhoIndex).chunks.length = chunksPerSample := by
  simp only [rhoDigestTrace]
  rw [runDigestRounds_chunks_length]
  norm_num [fixedDigestRounds, chunksPerSample]

@[simp] theorem rhoDigestTrace_absorbed
    (core : Poseidon2Core) (cursor : SpongeCursor) (rhoIndex : Nat) :
    (rhoDigestTrace core cursor rhoIndex).cursor.absorbed.val = 0 := by
  simp [rhoDigestTrace, fixedDigestRounds, runDigestRounds, digestRound]

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
