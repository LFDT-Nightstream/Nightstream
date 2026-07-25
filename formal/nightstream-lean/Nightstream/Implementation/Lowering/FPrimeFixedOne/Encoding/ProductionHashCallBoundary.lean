import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionDigestCodecs

/-!
Contract: exact semantic and coordinate boundary between the nonoptional
four-lane XOut core and the totalized fixed-one paper hash.

Assurance tier: model-level.

Owns:
- one canonical rejecting preimage with no current state;
- exact rejection for every absent or misaligned paper hash preimage;
- exact optional-digest coordinates for rejecting and aligned executions;
- an impossibility theorem for replacing the totalized paper hash by an
  always-present four-lane result.

Does not own: state or running codecs, alignment-check rows, XOut preimage
serialization, a typed Poseidon2 recipe, physical placement, `hashPrior` or
`hashNext` `CallRecipe` values, native Poseidon2 parity, or collision
resistance.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Protocol.FPrime
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

universe uParams uStructure uHeader uRunning uFresh uNifsProof
  uNebulaDigest uNebulaOpen

abbrev Digest :=
  Nightstream.Implementation.Encoding.FPrime.Digest

abbrev DirectState
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Nebula : Type) :=
  Nightstream.HyperNova.Construction2.State Digest Running Fresh Nebula

abbrev AdapterParameters
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (NifsProof : Type uNifsProof)
    (Nebula : Type)
    (NebulaDigest : Type uNebulaDigest)
    (NebulaOpen : Type uNebulaOpen) :=
  FixedOneCanonicalAdapter.Parameters
    Params StructureDigest Header Digest Running Fresh NifsProof Nebula
      NebulaDigest NebulaOpen

abbrev Preimage
    (Params : Type uParams)
    (StructureDigest : Type uStructure)
    (Header : Type uHeader)
    (Running : Type uRunning)
    (Fresh : Type uFresh)
    (Nebula : Type) :=
  HashPreimage
    (XOut.Context Params StructureDigest Header Digest)
    (Option (DirectState Running Fresh Nebula))
    Running
    1

section

variable
  {Params : Type uParams}
  {StructureDigest : Type uStructure}
  {Header : Type uHeader}
  {Running : Type uRunning}
  {Fresh : Type uFresh}
  {NifsProof : Type uNifsProof}
  {Nebula : Type}
  {NebulaDigest : Type uNebulaDigest}
  {NebulaOpen : Type uNebulaOpen}

local notation "Parameters" =>
  AdapterParameters Params StructureDigest Header Running Fresh NifsProof
    Nebula NebulaDigest NebulaOpen

local notation "PaperPreimage" =>
  Preimage Params StructureDigest Header Running Fresh Nebula

variable
  [DecidableEq (DirectState Running Fresh Nebula)]
  [DecidableEq Running]

/-- A concrete preimage at which the paper hash must reject. No arbitrary
digest, failure callback, or implementation value is needed to construct it. -/
def absentCurrentPreimage (parameters : Parameters) : PaperPreimage where
  verifierKeys := fun _ => parameters.context
  iteration := 0
  z0 := none
  current := none
  running := fun _ => parameters.step.emptyRunning
  pc := 1

@[simp] theorem absentCurrentPreimage_not_aligned
    (parameters : Parameters) :
    FixedOneCanonicalAdapter.alignedHashPreimage parameters
        (absentCurrentPreimage parameters) =
      false :=
  rfl

/-- The totalized paper hash rejects the canonical absent-current preimage. -/
@[simp] theorem paperHash_absentCurrent
    (parameters : Parameters) :
    FixedOneCanonicalAdapter.paperHash parameters
        (absentCurrentPreimage parameters) =
      none :=
  rfl

/-- Every failed alignment check reaches the exact rejecting digest value. -/
theorem paperHash_eq_none_of_not_aligned
    (parameters : Parameters)
    (preimage : PaperPreimage)
    (notAligned :
      FixedOneCanonicalAdapter.alignedHashPreimage parameters preimage =
        false) :
    FixedOneCanonicalAdapter.paperHash parameters preimage = none := by
  cases currentEq : preimage.current with
  | none =>
      simp [FixedOneCanonicalAdapter.paperHash, currentEq]
  | some current =>
      simp [FixedOneCanonicalAdapter.paperHash, currentEq, notAligned]

/-- Rejection is exactly absence of the current state or failure of one of
the duplicated-carrier alignment checks. -/
theorem paperHash_eq_none_iff
    (parameters : Parameters)
    (preimage : PaperPreimage) :
    FixedOneCanonicalAdapter.paperHash parameters preimage = none ↔
      preimage.current = none ∨
        FixedOneCanonicalAdapter.alignedHashPreimage parameters preimage =
          false := by
  cases currentEq : preimage.current with
  | none =>
      simp [FixedOneCanonicalAdapter.paperHash, currentEq]
  | some current =>
      cases alignedEq :
          FixedOneCanonicalAdapter.alignedHashPreimage parameters preimage with
      | false =>
          simp [FixedOneCanonicalAdapter.paperHash, currentEq, alignedEq]
      | true =>
          simp [FixedOneCanonicalAdapter.paperHash, currentEq, alignedEq]

/-- Exact five-coordinate encoding of the canonical rejecting preimage. -/
theorem absentCurrent_encoding_exact
    (parameters : Parameters) :
    ProductionDigestCodecs.optionalDigestCodec.encode
        (FixedOneCanonicalAdapter.paperHash parameters
          (absentCurrentPreimage parameters)) =
      [0, 0, 0, 0, 0] := by
  rw [paperHash_absentCurrent]
  exact ProductionDigestCodecs.optionalDigestCodec_encode_none

/-- The all-zero optional-digest vector occurs exactly on an absent or
misaligned paper hash preimage. -/
theorem paperHash_encoding_eq_absent_iff
    (parameters : Parameters)
    (preimage : PaperPreimage) :
    ProductionDigestCodecs.optionalDigestCodec.encode
          (FixedOneCanonicalAdapter.paperHash parameters preimage) =
        [0, 0, 0, 0, 0] ↔
      preimage.current = none ∨
        FixedOneCanonicalAdapter.alignedHashPreimage parameters preimage =
          false := by
  rw [← paperHash_eq_none_iff parameters preimage]
  constructor
  · intro encoded
    have encoded' :
        ProductionDigestCodecs.optionalDigestCodec.encode
            (FixedOneCanonicalAdapter.paperHash parameters preimage) =
          ProductionDigestCodecs.optionalDigestCodec.encode none := by
      simpa using encoded
    have decoded :=
      congrArg ProductionDigestCodecs.optionalDigestCodec.decode encoded'
    exact Option.some.inj <| by
      simpa only [ProductionDigestCodecs.optionalDigestCodec_roundtrip] using
        decoded
  · intro rejected
    rw [rejected]
    exact ProductionDigestCodecs.optionalDigestCodec_encode_none

/-- On an aligned present state, the wrapper emits presence one followed by
the exact four-lane XOut core result. -/
theorem alignedCurrent_encoding_exact
    (parameters : Parameters)
    (preimage : PaperPreimage)
    (current : DirectState Running Fresh Nebula)
    (currentEq : preimage.current = some current)
    (aligned :
      FixedOneCanonicalAdapter.alignedHashPreimage parameters preimage =
        true) :
    ProductionDigestCodecs.optionalDigestCodec.encode
        (FixedOneCanonicalAdapter.paperHash parameters preimage) =
      1 ::
        ProductionDigestCodecs.digestCodec.encode
          (XOut.compute parameters.hash parameters.mode parameters.context
            current) := by
  simp [FixedOneCanonicalAdapter.paperHash, currentEq, aligned]

/-- What a digest-only sponge would have to claim in order to replace the
totalized paper hash on every typed preimage. -/
def NonoptionalCoreRefines
    (parameters : Parameters)
    (core : PaperPreimage -> Digest) : Prop :=
  ∀ preimage,
    FixedOneCanonicalAdapter.paperHash parameters preimage =
      some (core preimage)

/-- No always-present four-lane core can extensionally implement the frozen
paper hash on its complete typed domain. The explicit presence/alignment
wrapper is therefore a semantic obligation, not optional serialization. -/
theorem no_nonoptionalCoreRefines
    (parameters : Parameters)
    (core : PaperPreimage -> Digest) :
    ¬ NonoptionalCoreRefines parameters core := by
  intro refines
  have impossible := refines (absentCurrentPreimage parameters)
  simp only [paperHash_absentCurrent] at impossible
  simp at impossible

end

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ProductionHashCallBoundary
