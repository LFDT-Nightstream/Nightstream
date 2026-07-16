import Nightstream.Implementation.R1CS.Core.ChaCha8

/-!
Assignment-free seed derivation for the two `Pi_CCS` output-binding maps.

Assurance tier: executable protocol-profile semantics. This file specifies
the verifier-owned master seeds, row-seed expansion, chunk partition, and
chunk-seed expansion without importing generated blocks, R1CS rows, Rust
artifacts, or commitment outputs.

Owns: little-endian `u32` stream-to-byte conversion; 32-byte seed extraction;
the seeded-PP chunking rule; the public rank/message profiles; and the derived
primary and compression schedules.

Does not own: proof that `ChaCha8.words` agrees with `rand_chacha`; the
production block metadata; coefficient rejection sampling; Phi81 rotation;
SIS security; Poseidon2; transcript authority; row removal; or cost totals.

Emits constraints: no.

Authority boundary: master seeds and dimensions are verifier-owned profile
constants. Derived row/chunk seeds are functions of those constants, never
prover inputs. Rust conformance is deliberately a later refinement theorem.

| Protocol | Phase | Mathematical branch | Definition | Exact obligation |
|---|---|---|---|---|
| `Pi_CCS` | output digest | byte layout | `wordBytes` | one `u32` becomes four little-endian bytes |
| `Pi_CCS` | output digest | row seeds | `seedsFromStream` | every 8 stream words become one 32-byte seed |
| `Pi_CCS` | output digest | chunking | `chunkSize` | `max(min(m, 2^15), 1024)` |
| `Pi_CCS` | output digest | public profile | `primarySchedule` | rank-2, 5,075-column map derived from master byte `0xC3` |
| `Pi_CCS` | output digest | public profile | `compressionSchedule` | rank-1, 82-column map derived from master byte `0xC6` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedDerivation

/-- Four little-endian bytes of one logical `u32` stream word. -/
def wordBytes (word : Nat) : List Nat :=
  [word % 256,
   (word / 256) % 256,
   (word / 65536) % 256,
   (word / 16777216) % 256]

theorem wordBytes_length (word : Nat) : (wordBytes word).length = 4 := by
  rfl

/-- Byte view of a finite logical ChaCha8 word-stream slice. -/
def streamBytes (seed : List Nat) (wordStart wordCount : Nat) : List Nat :=
  (ChaCha8.words seed wordStart wordCount).flatMap wordBytes

/-- Exact behavior of repeated `fill_bytes(&mut [u8; 32])`: each seed
consumes eight consecutive little-endian `u32` words. -/
def seedsFromStream (seed : List Nat) (count : Nat) : List (List Nat) :=
  (List.range count).map fun index =>
    streamBytes seed (8 * index) 8

/-- Seeded-PP chunk width fixed by the verifier profile. -/
def chunkSize (messageCols : Nat) : Nat :=
  Nat.max (Nat.min messageCols (2 ^ 15)) 1024

def chunkCount (messageCols : Nat) : Nat :=
  (messageCols + chunkSize messageCols - 1) / chunkSize messageCols

structure Schedule where
  chunkSize : Nat
  seedsByOutput : List (List (List Nat))
deriving DecidableEq, Repr

/-- Two-stage deterministic expansion used by the seeded public parameters:
master seed to row seeds, then each row seed to its chunk seeds. -/
def derive (masterSeed : List Nat) (kappa messageCols : Nat) : Schedule :=
  { chunkSize := chunkSize messageCols
    seedsByOutput := (seedsFromStream masterSeed kappa).map fun rowSeed =>
      seedsFromStream rowSeed (chunkCount messageCols) }

def primaryMasterSeed : List Nat := List.replicate 32 0xC3
def compressionMasterSeed : List Nat := List.replicate 32 0xC6

def primaryKappa : Nat := 2
def primaryMessageCols : Nat := 5075
def compressionKappa : Nat := 1
def compressionMessageCols : Nat := 82

def primarySchedule : Schedule :=
  derive primaryMasterSeed primaryKappa primaryMessageCols

def compressionSchedule : Schedule :=
  derive compressionMasterSeed compressionKappa compressionMessageCols

theorem primaryChunkSize : primarySchedule.chunkSize = 5075 := by
  decide

theorem primaryChunkCount : chunkCount primaryMessageCols = 1 := by
  decide

theorem compressionChunkSize : compressionSchedule.chunkSize = 1024 := by
  decide

theorem compressionChunkCount : chunkCount compressionMessageCols = 1 := by
  decide

end Nightstream.Implementation.R1CS.PiCcsOutputDigest.Sis.SeedDerivation
