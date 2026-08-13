import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.AggregateAcceptanceSchema

/-! Generated exact aggregate-acceptance leaf data; do not hand-edit.

Owns: the production gate arity, role-to-matrix bindings, nine normalized
active rows, and exact sparse-polynomial specialization.

Does not own: singleton fixture geometry, source-bit decoding, selectors,
inactive rows, or the fixed-F' 960-chunk physical outer image.

Emits constraints: no.

Authority boundary: this is artifact evidence only. Handwritten correspondence
must prove that these generated equations implement independent semantics.

| Data branch | Exact production evidence | Semantic owner |
|---|---|---|
| `matrixBindings` | forty occupied matrix roles in arity 56 | aggregate artifact refinement |
| `activeRows` | seven bit pairs, one radix-3 aggregate, one root binding | `AggregateAcceptanceRows` |
| `polynomialTerms` | exact 25-term gate specialization | aggregate artifact refinement |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifactData

open AggregateAcceptanceArtifact

def schemaVersion : Nat := 2
def gateArity : Nat := 56
def matrixBindings : List MatrixBinding :=
[
  { role := .selector, index := 0 }
, { role := (.productLeft 0), index := 3 }
, { role := (.productLeft 1), index := 4 }
, { role := (.productLeft 2), index := 5 }
, { role := (.productLeft 3), index := 6 }
, { role := (.productLeft 4), index := 7 }
, { role := (.productLeft 5), index := 8 }
, { role := (.productLeft 6), index := 9 }
, { role := (.productLeft 7), index := 10 }
, { role := (.productLeft 8), index := 11 }
, { role := (.productLeft 9), index := 12 }
, { role := (.productLeft 10), index := 13 }
, { role := (.productLeft 11), index := 14 }
, { role := (.productLeft 12), index := 15 }
, { role := (.productLeft 13), index := 16 }
, { role := (.productLeft 14), index := 17 }
, { role := (.productLeft 15), index := 18 }
, { role := (.productLeft 16), index := 19 }
, { role := (.productLeft 17), index := 20 }
, { role := (.productRight 0), index := 21 }
, { role := (.productRight 1), index := 22 }
, { role := (.productRight 2), index := 23 }
, { role := (.productRight 3), index := 24 }
, { role := (.productRight 4), index := 25 }
, { role := (.productRight 5), index := 26 }
, { role := (.productRight 6), index := 27 }
, { role := (.productRight 7), index := 28 }
, { role := (.productRight 8), index := 29 }
, { role := (.productRight 9), index := 30 }
, { role := (.productRight 10), index := 31 }
, { role := (.productRight 11), index := 32 }
, { role := (.productRight 12), index := 33 }
, { role := (.productRight 13), index := 34 }
, { role := (.productRight 14), index := 35 }
, { role := (.productRight 15), index := 36 }
, { role := (.productRight 16), index := 37 }
, { role := (.productRight 17), index := 38 }
, { role := .productOut, index := 39 }
, { role := .quadraticBitLeft, index := 44 }
, { role := .quadraticBitRight, index := 45 }
]
def activeRows : List ActiveRow :=
[
  [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 0), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 1), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 2), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 3), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 4), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 5), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 6), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 7), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 8), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 9), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 10), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 11), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨.quadraticBitLeft, [⟨(.treeOutput 12), 1⟩]⟩, ⟨.quadraticBitRight, [⟨(.treeOutput 13), 1⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨(.productLeft 0), [⟨.one, 1⟩, ⟨(.chunkBit 0), -1⟩]⟩, ⟨(.productLeft 1), [⟨.one, 3⟩, ⟨(.chunkBit 2), -3⟩]⟩, ⟨(.productLeft 2), [⟨.one, 9⟩, ⟨(.chunkBit 4), -9⟩]⟩, ⟨(.productLeft 3), [⟨.one, 27⟩, ⟨(.chunkBit 6), -27⟩]⟩, ⟨(.productLeft 4), [⟨(.treeOutput 0), 81⟩]⟩, ⟨(.productLeft 5), [⟨(.treeOutput 2), 243⟩]⟩, ⟨(.productLeft 6), [⟨(.treeOutput 4), 729⟩]⟩, ⟨(.productLeft 7), [⟨.one, 2187⟩, ⟨(.chunkBit 8), -2187⟩]⟩, ⟨(.productLeft 8), [⟨.one, 6561⟩, ⟨(.chunkBit 10), -6561⟩]⟩, ⟨(.productLeft 9), [⟨.one, 19683⟩, ⟨(.chunkBit 12), -19683⟩]⟩, ⟨(.productLeft 10), [⟨.one, 59049⟩, ⟨(.chunkBit 14), -59049⟩]⟩, ⟨(.productLeft 11), [⟨(.treeOutput 7), 177147⟩]⟩, ⟨(.productLeft 12), [⟨(.treeOutput 9), 531441⟩]⟩, ⟨(.productLeft 13), [⟨(.treeOutput 11), 1594323⟩]⟩, ⟨(.productRight 0), [⟨.one, 1⟩, ⟨(.chunkBit 1), -1⟩]⟩, ⟨(.productRight 1), [⟨.one, 1⟩, ⟨(.chunkBit 3), -1⟩]⟩, ⟨(.productRight 2), [⟨.one, 1⟩, ⟨(.chunkBit 5), -1⟩]⟩, ⟨(.productRight 3), [⟨.one, 1⟩, ⟨(.chunkBit 7), -1⟩]⟩, ⟨(.productRight 4), [⟨(.treeOutput 1), 1⟩]⟩, ⟨(.productRight 5), [⟨(.treeOutput 3), 1⟩]⟩, ⟨(.productRight 6), [⟨(.treeOutput 5), 1⟩]⟩, ⟨(.productRight 7), [⟨.one, 1⟩, ⟨(.chunkBit 9), -1⟩]⟩, ⟨(.productRight 8), [⟨.one, 1⟩, ⟨(.chunkBit 11), -1⟩]⟩, ⟨(.productRight 9), [⟨.one, 1⟩, ⟨(.chunkBit 13), -1⟩]⟩, ⟨(.productRight 10), [⟨.one, 1⟩, ⟨(.chunkBit 15), -1⟩]⟩, ⟨(.productRight 11), [⟨(.treeOutput 8), 1⟩]⟩, ⟨(.productRight 12), [⟨(.treeOutput 10), 1⟩]⟩, ⟨(.productRight 13), [⟨(.treeOutput 12), 1⟩]⟩, ⟨.productOut, [⟨(.treeOutput 0), 1⟩, ⟨(.treeOutput 1), 3⟩, ⟨(.treeOutput 2), 9⟩, ⟨(.treeOutput 3), 27⟩, ⟨(.treeOutput 4), 81⟩, ⟨(.treeOutput 5), 243⟩, ⟨(.treeOutput 6), 729⟩, ⟨(.treeOutput 7), 2187⟩, ⟨(.treeOutput 8), 6561⟩, ⟨(.treeOutput 9), 19683⟩, ⟨(.treeOutput 10), 59049⟩, ⟨(.treeOutput 11), 177147⟩, ⟨(.treeOutput 12), 531441⟩, ⟨(.treeOutput 13), 1594323⟩]⟩]
, [⟨.selector, [⟨.one, 1⟩]⟩, ⟨(.productLeft 0), [⟨(.treeOutput 6), 1⟩]⟩, ⟨(.productRight 0), [⟨(.treeOutput 13), 1⟩]⟩, ⟨.productOut, [⟨.one, 1⟩, ⟨.accept, -1⟩]⟩]
]
def polynomialTerms : List PolynomialTerm :=
[
  ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 0), 1⟩, ⟨(.productRight 0), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 1), 1⟩, ⟨(.productRight 1), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 2), 1⟩, ⟨(.productRight 2), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 3), 1⟩, ⟨(.productRight 3), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 4), 1⟩, ⟨(.productRight 4), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 5), 1⟩, ⟨(.productRight 5), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 6), 1⟩, ⟨(.productRight 6), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 7), 1⟩, ⟨(.productRight 7), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 8), 1⟩, ⟨(.productRight 8), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 9), 1⟩, ⟨(.productRight 9), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 10), 1⟩, ⟨(.productRight 10), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 11), 1⟩, ⟨(.productRight 11), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 12), 1⟩, ⟨(.productRight 12), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 13), 1⟩, ⟨(.productRight 13), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 14), 1⟩, ⟨(.productRight 14), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 15), 1⟩, ⟨(.productRight 15), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 16), 1⟩, ⟨(.productRight 16), 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨(.productLeft 17), 1⟩, ⟨(.productRight 17), 1⟩]⟩
, ⟨-1, [⟨.selector, 1⟩, ⟨.productOut, 1⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨.quadraticBitLeft, 4⟩]⟩
, ⟨-2, [⟨.selector, 1⟩, ⟨.quadraticBitLeft, 3⟩]⟩
, ⟨1, [⟨.selector, 1⟩, ⟨.quadraticBitLeft, 2⟩]⟩
, ⟨-7, [⟨.selector, 1⟩, ⟨.quadraticBitRight, 4⟩]⟩
, ⟨14, [⟨.selector, 1⟩, ⟨.quadraticBitRight, 3⟩]⟩
, ⟨-7, [⟨.selector, 1⟩, ⟨.quadraticBitRight, 2⟩]⟩
]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Acceptance.AggregateAcceptanceArtifactData
