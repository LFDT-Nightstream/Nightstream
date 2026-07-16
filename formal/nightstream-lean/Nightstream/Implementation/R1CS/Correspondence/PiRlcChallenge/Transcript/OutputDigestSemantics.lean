import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachine

/-!
Independent transcript semantics for binding the `Pi_CCS` output digest before
`Pi_RLC` challenge derivation.

Assurance tier: executable implementation semantics. This file specifies the
mathematical byte packing and sponge absorbs without importing a generated
R1CS owner or reading the Rust circuit emitter.

Owns: the ASCII bytes of the protocol label, seven-byte little-endian packing,
the field-count word, the four-field digest sequence, and its deterministic
overwrite-transcript transition.

Does not own: any generated column, row, or count; authority for the digest
contents; the state produced by the preceding `Pi_CCS` verifier; native Rust
conformance; or permission to remove constraints.

Emits constraints: no.

Authority boundary: the digest is an input to this transition, not evidence of
its own correctness. A later theorem must prove that it was recomputed from
accepted `Pi_CCS` outputs before this transition can authorize challenges.

| Protocol | Phase | Mathematical object | Exact obligation |
|---|---|---|---|
| `Pi_RLC` | output bind | label bytes | exact `pi_rlc/input_claims_digest` ASCII sequence |
| `Pi_RLC` | output bind | label limbs | length plus seven-byte little-endian field packing |
| `Pi_RLC` | output bind | digest header | exact field-count word `4` |
| `Pi_RLC` | output bind | transcript state | absorb label, count, and all four digest fields in order |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.OutputDigestSemantics

open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine

/-- Protocol-label bytes, stated independently of Rust string storage. -/
def inputClaimsDigestLabelBytes : List Nat :=
  [112, 105, 95, 114, 108, 99, 47,
   105, 110, 112, 117, 116, 95, 99,
   108, 97, 105, 109, 115, 95, 100,
   105, 103, 101, 115, 116]

/-- One zero-padded seven-byte little-endian limb. -/
def packSevenAt (bytes : List Nat) (start : Nat) : Nat :=
  bytes.getD start 0 +
    256 * bytes.getD (start + 1) 0 +
    256 ^ 2 * bytes.getD (start + 2) 0 +
    256 ^ 3 * bytes.getD (start + 3) 0 +
    256 ^ 4 * bytes.getD (start + 4) 0 +
    256 ^ 5 * bytes.getD (start + 5) 0 +
    256 ^ 6 * bytes.getD (start + 6) 0

/-- Length followed by the four packed limbs of the 26-byte label. -/
def inputClaimsDigestLabelNats : List Nat :=
  [inputClaimsDigestLabelBytes.length,
   packSevenAt inputClaimsDigestLabelBytes 0,
   packSevenAt inputClaimsDigestLabelBytes 7,
   packSevenAt inputClaimsDigestLabelBytes 14,
   packSevenAt inputClaimsDigestLabelBytes 21]

/-- Closed check of the independently computed label encoding. -/
theorem inputClaimsDigestLabelNats_eq :
    inputClaimsDigestLabelNats =
      [26, 13338641331874160, 27970976485502569,
       28252447032566124, 500152231785] := by
  decide

def encodedLabel : List Field :=
  inputClaimsDigestLabelNats.map wordField

def digestList (digest : Fin 4 -> Field) : List Field :=
  [digest ⟨0, by decide⟩, digest ⟨1, by decide⟩,
   digest ⟨2, by decide⟩, digest ⟨3, by decide⟩]

/-- Exact field stream consumed by `append_fields(label, digest)`. -/
def appendSequence (digest : Fin 4 -> Field) : List Field :=
  encodedLabel ++ [wordField 4] ++ digestList digest

/-- Sequential overwrite absorption. Permutations occur only through the
independently specified `absorbElem` cursor transition. -/
def absorbAll : State -> List Field -> State
  | state, [] => state
  | state, value :: rest => absorbAll (absorbElem state value) rest

/-- Pure output-digest binding transition used before `Pi_RLC` sampling. -/
def appendInputClaimsDigest
    (state : State) (digest : Fin 4 -> Field) : State :=
  absorbAll state (appendSequence digest)

/-- State after the first four label fields fill a fresh rate window. -/
def labelPrefix (state : State) : State :=
  absorbElem
    (absorbElem
      (absorbElem
        (absorbElem state (wordField 26))
        (wordField 13338641331874160))
      (wordField 27970976485502569))
    (wordField 28252447032566124)

/-- The fifth label field crosses the first full-rate boundary. -/
def afterFirstBoundary (state : State) : State :=
  absorbElem (labelPrefix state) (wordField 500152231785)

/-- Count and the first two digest fields fill the next rate window. -/
def beforeSecondBoundary
    (state : State) (digest : Fin 4 -> Field) : State :=
  absorbElem
    (absorbElem
      (absorbElem (afterFirstBoundary state) (wordField 4))
      (digest ⟨0, by decide⟩))
    (digest ⟨1, by decide⟩)

/-- The third digest field crosses the second full-rate boundary. -/
def afterSecondBoundary
    (state : State) (digest : Fin 4 -> Field) : State :=
  absorbElem (beforeSecondBoundary state digest) (digest ⟨2, by decide⟩)

/-- The fourth digest field is buffered at cursor two. -/
def completeBinding
    (state : State) (digest : Fin 4 -> Field) : State :=
  absorbElem (afterSecondBoundary state digest) (digest ⟨3, by decide⟩)

theorem encodedLabel_eq :
    encodedLabel =
      [wordField 26, wordField 13338641331874160,
       wordField 27970976485502569, wordField 28252447032566124,
       wordField 500152231785] := by
  rw [encodedLabel, inputClaimsDigestLabelNats_eq]
  rfl

/-- The list specification and the explicit two-boundary phase decomposition
are definitionally the same absorb sequence. -/
theorem appendInputClaimsDigest_eq_completeBinding
    (state : State) (digest : Fin 4 -> Field) :
    appendInputClaimsDigest state digest = completeBinding state digest := by
  rw [appendInputClaimsDigest, appendSequence, encodedLabel_eq]
  rfl

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.OutputDigestSemantics
