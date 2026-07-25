import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.ObservedTrace

/-!
Contract: typed source program for the native Construction-2 `state_x_out`
preimage.

Owns the exact domain, coordinate order, optional stateful lane, and
present-only Nebula extension. Its interpreter expands into the independent
paper-shaped encoder in `ObservedTrace`; it does not evaluate Poseidon2 or
claim that generated Rust data is authority.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram

open Nightstream.Protocol.FPrime
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

inductive Instruction where
  | domain (value : Nat)
  | verifierDigest
  | piCcsHeader
  | chunkCountHalves
  | stepCountHalves
  | pcHalves
  | currentBoundary
  | semanticState
  | construction2Accumulator
  | nebulaPresentMarker (value : Nat)
  | nebulaDigest
deriving Repr, DecidableEq

abbrev Program := List Instruction

/-- Independently specified implementation source order, parameterized only
by the two optional lane presences. The numeric tags are the production
Poseidon2 domains; `XOutPreimage` remains the protocol-level authority. -/
def canonical
    (semanticPresent nebulaPresent : Bool) : Program :=
  [
    .domain 0x4e460002,
    .verifierDigest,
    .piCcsHeader,
    .chunkCountHalves,
    .stepCountHalves,
    .pcHalves,
    .currentBoundary
  ] ++
  (match semanticPresent with
    | false => []
    | true => [.semanticState]) ++
  [.construction2Accumulator] ++
  (match nebulaPresent with
    | false => []
    | true => [.nebulaPresentMarker 0x4e424c41, .nebulaDigest])

def fieldCost : Instruction → Nat
  | .domain _ | .nebulaPresentMarker _ => 1
  | .chunkCountHalves | .stepCountHalves | .pcHalves => 2
  | .verifierDigest
  | .piCcsHeader
  | .currentBoundary
  | .semanticState
  | .construction2Accumulator
  | .nebulaDigest => 4

def cost (program : Program) : Nat :=
  program.foldl (fun total instruction => total + fieldCost instruction) 0

def instructionFields
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    Instruction → List RawField
  | .domain value => [rawFieldOfNat value]
  | .verifierDigest =>
      lookupRawFields table (.digest preimage.vkFsDigest)
  | .piCcsHeader =>
      lookupRawFields table (.header preimage.piCcsHeader)
  | .chunkCountHalves => u64Halves preimage.chunkCount
  | .stepCountHalves => u64Halves preimage.stepCount
  | .pcHalves => u64Halves preimage.pc
  | .currentBoundary =>
      lookupRawFields table (.digest preimage.currentBoundary)
  | .semanticState =>
      match preimage.semanticState with
      | none => []
      | some digest => lookupRawFields table (.digest digest)
  | .construction2Accumulator =>
      lookupRawFields table
        (.digest preimage.construction2Accumulator)
  | .nebulaPresentMarker value => [rawFieldOfNat value]
  | .nebulaDigest =>
      match preimage.nebula with
      | none => []
      | some digest =>
          lookupRawFields table (.nebulaDigest digest)

def execute
    (program : Program)
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    List RawField :=
  program.flatMap (instructionFields table preimage)

def forPreimage
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    Program :=
  canonical preimage.semanticState.isSome preimage.nebula.isSome

/-- The typed source program expands to exactly the independent canonical
field preimage for every combination of stateless/stateful and plain/Nebula
coordinates. -/
theorem execute_forPreimage
    (table : RawEncodingTable)
    (preimage : XOut.XOutPreimage Digest Header NebulaDigest) :
    execute (forPreimage preimage) table preimage =
      encodeStateXOutPreimage table preimage := by
  rw [encodeStateXOutPreimage_expansion]
  rcases preimage with
    ⟨vkFsDigest, piCcsHeader, chunkCount, stepCount, pc,
      currentBoundary, semanticState, construction2Accumulator, nebula⟩
  cases semanticState <;> cases nebula <;>
    simp [forPreimage, canonical, execute, instructionFields]

theorem statelessPlain_cost :
    cost (canonical false false) = 23 := by
  decide

theorem statelessNebula_cost :
    cost (canonical false true) = 28 := by
  decide

theorem statefulPlain_cost :
    cost (canonical true false) = 27 := by
  decide

theorem statefulNebula_cost :
    cost (canonical true true) = 32 := by
  decide

end Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.StateXOutProgram
