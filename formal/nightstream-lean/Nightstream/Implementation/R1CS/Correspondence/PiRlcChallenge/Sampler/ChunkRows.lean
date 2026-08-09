import Nightstream.Implementation.R1CS.Core.Program

/-!
Exact row schema for the four 16-bit candidates in one recursive-profile
`Pi_RLC` sampler lane.

Owns: a readable, equation-derived construction of the 26 rows per candidate
and the four-candidate lane composition.

Does not own: the independent acceptance/decoding semantics, proofs that these
rows imply those semantics, the 64-candidate selection tail, transcript
generation, production column placement, Rust conformance, or cost totals.

Emits constraints: no. This file describes and checks an existing template.

Authority boundary: the generated list is not trusted for its meaning. The
equality below gives it a named row schema; a separate soundness file must prove
that satisfying this schema refines the verifier-owned sampler semantics.

| Protocol | Phase | Constraint family | Rows per candidate | Mathematical equation |
|---|---|---|---:|---|
| `Pi_RLC` | sampler/chunk | acceptance | 4 | `accept = 1 iff chunk != 65535` via a canonical inverse |
| `Pi_RLC` | sampler/chunk | mod-5 range | 4 | `residue * (residue-1) * ... * (residue-4) = 0` |
| `Pi_RLC` | sampler/chunk | quotient range | 15 | 14 Boolean bits and exact radix-2 recomposition |
| `Pi_RLC` | sampler/chunk | decomposition | 1 | `chunk = 5 * quotient + residue` |
| `Pi_RLC` | sampler/chunk | symbol/prefix | 2 | `symbol = residue - 2`; `next = prior + accept` |
| `Pi_RLC` | sampler/lane | four chunks | 104 | concatenate the four 26-row families in source order |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.ChunkRows

open Nightstream.Implementation.R1CS

/-- First local auxiliary column for one of the four candidates. -/
def base (chunk : Nat) : Nat := 66 + 23 * chunk

def acceptCol (chunk : Nat) : Nat := base chunk
def inverseCol (chunk : Nat) : Nat := base chunk + 1
def residueCol (chunk : Nat) : Nat := base chunk + 2
def quotientCol (chunk : Nat) : Nat := base chunk + 3
def residueProductCol (chunk stage : Nat) : Nat := base chunk + 4 + stage
def quotientBitCol (chunk offset : Nat) : Nat := base chunk + 7 + offset
def symbolCol (chunk : Nat) : Nat := base chunk + 21
def cumulativeCol (chunk : Nat) : Nat := base chunk + 22

/-- Chunk zero starts from the lane input count at local column 65; every later
chunk starts from its immediate predecessor's cumulative output. -/
def priorCumulativeCol (chunk : Nat) : Nat :=
  if chunk = 0 then 65 else base chunk - 1

def sourceBitCol (chunk offset : Nat) : Nat := 1 + 16 * chunk + offset

def chunkTerms (chunk : Nat) : List (Nat × Nat) :=
  [(0, 65535)] ++
    (List.range 16).map fun offset =>
      (sourceBitCol chunk offset, goldilocksP - 2 ^ offset)

def quotientTerms (chunk : Nat) : List (Nat × Nat) :=
  (List.range 14).map fun offset => (quotientBitCol chunk offset, 2 ^ offset)

def differenceTerms (chunk : Nat) : List (Nat × Nat) :=
  chunkTerms chunk ++ [(0, goldilocksP - 65535)]

def oneMinusAcceptTerms (chunk : Nat) : List (Nat × Nat) :=
  [(acceptCol chunk, goldilocksP - 1), (0, 1)]

def zeroEqualityRow (terms : List (Nat × Nat)) : Row :=
  ⟨terms, [(0, 1)], []⟩

def acceptanceRows (chunk : Nat) : List Row :=
  [ bitRow (acceptCol chunk),
    ⟨oneMinusAcceptTerms chunk, differenceTerms chunk, []⟩,
    ⟨differenceTerms chunk, [(inverseCol chunk, 1)],
      [(acceptCol chunk, 1)]⟩,
    ⟨oneMinusAcceptTerms chunk, [(inverseCol chunk, 1)], []⟩ ]

def residueRangeRows (chunk : Nat) : List Row :=
  [ ⟨[(residueCol chunk, 1)],
      [(residueCol chunk, 1), (0, goldilocksP - 1)],
      [(residueProductCol chunk 0, 1)]⟩,
    ⟨[(residueProductCol chunk 0, 1)],
      [(residueCol chunk, 1), (0, goldilocksP - 2)],
      [(residueProductCol chunk 1, 1)]⟩,
    ⟨[(residueProductCol chunk 1, 1)],
      [(residueCol chunk, 1), (0, goldilocksP - 3)],
      [(residueProductCol chunk 2, 1)]⟩,
    ⟨[(residueProductCol chunk 2, 1)],
      [(residueCol chunk, 1), (0, goldilocksP - 4)], []⟩ ]

def quotientRangeRows (chunk : Nat) : List Row :=
  (List.range 14).map fun offset => bitRow (quotientBitCol chunk offset)

def quotientRecompositionRow (chunk : Nat) : Row :=
  zeroEqualityRow
    ([(quotientCol chunk, 1)] ++
      (List.range 14).map fun offset =>
        (quotientBitCol chunk offset, goldilocksP - 2 ^ offset))

def decompositionRow (chunk : Nat) : Row :=
  zeroEqualityRow
    (chunkTerms chunk ++
      [(quotientCol chunk, goldilocksP - 5),
       (residueCol chunk, goldilocksP - 1)])

def symbolRow (chunk : Nat) : Row :=
  zeroEqualityRow
    [(symbolCol chunk, 1),
     (residueCol chunk, goldilocksP - 1),
     (0, 2)]

def cumulativeRow (chunk : Nat) : Row :=
  zeroEqualityRow
    [(cumulativeCol chunk, 1),
     (priorCumulativeCol chunk, goldilocksP - 1),
     (acceptCol chunk, goldilocksP - 1)]

/-- The meaningful constraint-family grouping for one candidate. -/
def chunkRows (chunk : Nat) : List Row :=
  acceptanceRows chunk ++
    residueRangeRows chunk ++
    quotientRangeRows chunk ++
    [quotientRecompositionRow chunk, decompositionRow chunk,
      symbolRow chunk, cumulativeRow chunk]

/-- Protocol → sampler lane → candidate family → concrete rows. -/
def rows : List Row :=
  (List.range 4).flatMap chunkRows

theorem chunkRows_length (chunk : Nat) : (chunkRows chunk).length = 26 := by
  simp [chunkRows, acceptanceRows, residueRangeRows, quotientRangeRows]

theorem rows_length : rows.length = 104 := by
  decide

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.ChunkRows
