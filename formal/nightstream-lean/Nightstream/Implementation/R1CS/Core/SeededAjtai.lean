import Nightstream.Implementation.R1CS.Core.SeededPhi81Sampler.Schedule

/-!
Contract: derive one exact Ajtai verifier key from a verifier-owned ChaCha8
seed and the canonical Goldilocks rejection sampler.

Owns: master-to-row-to-chunk seed expansion, the setup sampler schedule, and
the exact finite key read by the Lean SuperNeo relation.

Does not own: selection of a deployment seed, Rust conformance, SIS security,
recursive constraints, terminal lowering, or file serialization.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.SeededAjtai

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SeededPhi81Sampler
open Nightstream.SuperNeo.Concrete

/-- Canonical verifier-owned 256-bit setup seed. -/
structure Seed where
  bytes : List Nat
  length_eq : bytes.length = 32
  canonical : forall byte, byte ∈ bytes -> byte < 256

/-- Proof-free setup identity carried by the Rust manifest. -/
structure Identity where
  seed : List Nat
  rejectionFuel : Nat
deriving DecidableEq, Repr

/-- Four little-endian bytes of one ChaCha word. -/
def wordBytes (word : Nat) : List Nat :=
  [word % 256,
   (word / 256) % 256,
   (word / 65536) % 256,
   (word / 16777216) % 256]

/-- Byte view used by `fill_bytes`. -/
def streamBytes (seed : List Nat) (wordStart wordCount : Nat) : List Nat :=
  (ChaCha8.words seed wordStart wordCount).flatMap wordBytes

/-- Exact repeated `fill_bytes(&mut [u8; 32])` seed expansion. -/
def seedsFromStream (seed : List Nat) (count : Nat) : List (List Nat) :=
  (List.range count).map fun index =>
    streamBytes seed (8 * index) 8

/-- Deterministic setup chunk width used by Rust `setup_par`. -/
def chunkSize (messageCols : Nat) : Nat :=
  Nat.max (Nat.min messageCols (2 ^ 15)) 1024

def chunkCount (messageCols : Nat) : Nat :=
  (messageCols + chunkSize messageCols - 1) / chunkSize messageCols

/-- Complete bounded setup sampler. Successful execution has the same first-
accepted meaning for every coefficient as the unbounded sampler. -/
def schedule (masterSeed : List Nat) (verifierRows messageCols fuel : Nat) :
    SeededPhi81Sampler.Schedule where
  chunkSize := chunkSize messageCols
  seedsByOutput :=
    (seedsFromStream masterSeed verifierRows).map fun rowSeed =>
      seedsFromStream rowSeed (chunkCount messageCols)
  rejectionFuel := fuel

/-- A selected setup includes evidence that bounded Lean evaluation never
uses the fail-closed fallback. -/
structure Setup (verifierRows messageCols : Nat) where
  seed : Seed
  rejectionFuel : Nat
  samplingSuccess : exists outputs,
    (schedule seed.bytes verifierRows messageCols rejectionFuel).baseRotations
      pureStream messageCols = some outputs

namespace Setup

def identity {verifierRows messageCols : Nat}
    (setup : Setup verifierRows messageCols) : Identity where
  seed := setup.seed.bytes
  rejectionFuel := setup.rejectionFuel

/-- The exact row-major coefficient tensor selected by the setup. -/
def outputs {verifierRows messageCols : Nat}
    (setup : Setup verifierRows messageCols) : List (List (List Nat)) :=
  ((schedule setup.seed.bytes verifierRows messageCols setup.rejectionFuel
      ).baseRotations pureStream messageCols).getD []

theorem execution_eq_some_outputs {verifierRows messageCols : Nat}
    (setup : Setup verifierRows messageCols) :
    (schedule setup.seed.bytes verifierRows messageCols setup.rejectionFuel
      ).baseRotations pureStream messageCols = some setup.outputs := by
  rcases setup.samplingSuccess with ⟨sampled, success⟩
  simpa [outputs, success] using success

/-- Exact finite Ajtai key. The sampled vector is already the coefficient
representation of one Phi81 ring element. -/
def verifierKey {verifierRows messageCols : Nat}
    (setup : Setup verifierRows messageCols) :
    Fin verifierRows -> Fin messageCols -> RingF :=
  fun row block lane =>
    ⟨(((setup.outputs.getD row.val []).getD block.val []).getD lane.val 0) %
        goldilocksModulus,
      Nat.mod_lt _ (by decide)⟩

@[simp] theorem verifierKey_val {verifierRows messageCols : Nat}
    (setup : Setup verifierRows messageCols)
    (row : Fin verifierRows) (block : Fin messageCols) (lane : Fin ringDegree) :
    (setup.verifierKey row block lane).val =
      (((setup.outputs.getD row.val []).getD block.val []).getD lane.val 0) %
        goldilocksModulus := by
  rfl

end Setup

end Nightstream.Implementation.R1CS.SeededAjtai
