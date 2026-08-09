import Nightstream.Implementation.R1CS.Artifacts.PiRlcChallenge.PackedMod5Schema

/-! Generated exact active packed-Mod-5 data; do not hand-edit.

Owns: one role-normalized production source block, its projected decoder,
the active row schedule, and the exact Mod-5 polynomial specialization.

Does not own: selector composition, inactive rows, or semantic authority.

Supported profile: isolated one-rho, 64-chunk sampler fixture. Full-F'
placement and outer-image conformance are separate obligations.

Emits constraints: no.

Authority boundary: Rust validates and compares equations directly. No digest
in this file authorizes a row or decoder.

| Data branch | Mathematical obligation | Production check |
|---|---|---|
| `sourceRows` | exact 20-row source language | all 64 trace schemas equal |
| `decoderDefinitions` | exact projected reconstruction | normalized production LCs |
| `activeRows` | exact 6 + 1 + 1 row schedule | materialized CCS matrices |
| `polynomialTerms` | exact packed residual expansion | production sparse polynomial |
-/

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5ArtifactData

open PackedMod5Artifact

def schemaVersion : Nat := 1
def sourceInputOrder : List SourceRole :=
  [.chunkBit 0, .chunkBit 1, .chunkBit 2, .chunkBit 3, .chunkBit 4, .chunkBit 5, .chunkBit 6, .chunkBit 7, .chunkBit 8, .chunkBit 9, .chunkBit 10, .chunkBit 11, .chunkBit 12, .chunkBit 13, .chunkBit 14, .chunkBit 15]
def sourceAllocatedOrder : List SourceRole :=
  [.index, .quotient, .indexProduct 0, .indexProduct 1, .indexProduct 2,
   .quotientBit 0, .quotientBit 1, .quotientBit 2, .quotientBit 3, .quotientBit 4, .quotientBit 5, .quotientBit 6, .quotientBit 7, .quotientBit 8, .quotientBit 9, .quotientBit 10, .quotientBit 11, .quotientBit 12, .quotientBit 13]
def sourceRows : List SourceRow :=
[
  ⟨[⟨.index, 1⟩],
    [⟨.one, -1⟩, ⟨.index, 1⟩],
    [⟨(SourceRole.indexProduct 0), 1⟩]⟩
,   ⟨[⟨(SourceRole.indexProduct 0), 1⟩],
    [⟨.one, -2⟩, ⟨.index, 1⟩],
    [⟨(SourceRole.indexProduct 1), 1⟩]⟩
,   ⟨[⟨(SourceRole.indexProduct 1), 1⟩],
    [⟨.one, -3⟩, ⟨.index, 1⟩],
    [⟨(SourceRole.indexProduct 2), 1⟩]⟩
,   ⟨[⟨(SourceRole.indexProduct 2), 1⟩],
    [⟨.one, -4⟩, ⟨.index, 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 0), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 0), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 1), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 1), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 2), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 2), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 3), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 3), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 4), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 4), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 5), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 5), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 6), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 6), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 7), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 7), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 8), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 8), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 9), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 9), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 10), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 10), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 11), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 11), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 12), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 12), 1⟩],
    []⟩
,   ⟨[⟨(SourceRole.quotientBit 13), 1⟩],
    [⟨.one, -1⟩, ⟨(SourceRole.quotientBit 13), 1⟩],
    []⟩
,   ⟨[⟨.quotient, 1⟩, ⟨(SourceRole.quotientBit 0), -1⟩, ⟨(SourceRole.quotientBit 1), -2⟩, ⟨(SourceRole.quotientBit 2), -4⟩, ⟨(SourceRole.quotientBit 3), -8⟩, ⟨(SourceRole.quotientBit 4), -16⟩, ⟨(SourceRole.quotientBit 5), -32⟩, ⟨(SourceRole.quotientBit 6), -64⟩, ⟨(SourceRole.quotientBit 7), -128⟩, ⟨(SourceRole.quotientBit 8), -256⟩, ⟨(SourceRole.quotientBit 9), -512⟩, ⟨(SourceRole.quotientBit 10), -1024⟩, ⟨(SourceRole.quotientBit 11), -2048⟩, ⟨(SourceRole.quotientBit 12), -4096⟩, ⟨(SourceRole.quotientBit 13), -8192⟩],
    [⟨.one, 1⟩],
    []⟩
,   ⟨[⟨.one, 65535⟩, ⟨(SourceRole.chunkBit 0), -1⟩, ⟨(SourceRole.chunkBit 1), -2⟩, ⟨(SourceRole.chunkBit 2), -4⟩, ⟨(SourceRole.chunkBit 3), -8⟩, ⟨(SourceRole.chunkBit 4), -16⟩, ⟨(SourceRole.chunkBit 5), -32⟩, ⟨(SourceRole.chunkBit 6), -64⟩, ⟨(SourceRole.chunkBit 7), -128⟩, ⟨(SourceRole.chunkBit 8), -256⟩, ⟨(SourceRole.chunkBit 9), -512⟩, ⟨(SourceRole.chunkBit 10), -1024⟩, ⟨(SourceRole.chunkBit 11), -2048⟩, ⟨(SourceRole.chunkBit 12), -4096⟩, ⟨(SourceRole.chunkBit 13), -8192⟩, ⟨(SourceRole.chunkBit 14), -16384⟩, ⟨(SourceRole.chunkBit 15), -32768⟩, ⟨.index, -1⟩, ⟨.quotient, -5⟩],
    [⟨.one, 1⟩],
    []⟩
]
def coordinateOrder : List CoordinateRole :=
  [.quotientLow 0, .quotientLow 1, .quotientLow 2, .quotientLow 3, .quotientLow 4, .quotientLow 5, .quotientLow 6, .quotientLow 7, .quotientLow 8, .quotientLow 9, .quotientLow 10, .quotientLow 11, .quotientLow 12, .residueLeft, .residueRight]
def decoderDefinitions : List DecoderDefinition :=
  [.linear .index [⟨.source .one, 2⟩, ⟨.coordinate .residueLeft, 1⟩, ⟨.coordinate .residueRight, 1⟩],
   .linear (.quotientBit 13) [⟨.source .one, 7380048707653730306⟩, ⟨.source (.chunkBit 0), 450359962632192⟩, ⟨.source (.chunkBit 1), 900719925264384⟩, ⟨.source (.chunkBit 2), 1801439850528768⟩, ⟨.source (.chunkBit 3), 3602879701057536⟩, ⟨.source (.chunkBit 4), 7205759402115072⟩, ⟨.source (.chunkBit 5), 14411518804230144⟩, ⟨.source (.chunkBit 6), 28823037608460288⟩, ⟨.source (.chunkBit 7), 57646075216920576⟩, ⟨.source (.chunkBit 8), 115292150433841152⟩, ⟨.source (.chunkBit 9), 230584300867682304⟩, ⟨.source (.chunkBit 10), 461168601735364608⟩, ⟨.source (.chunkBit 11), 922337203470729216⟩, ⟨.source (.chunkBit 12), 1844674406941458432⟩, ⟨.source (.chunkBit 13), 3689348813882916864⟩, ⟨.source (.chunkBit 14), 7378697627765833728⟩, ⟨.source (.chunkBit 15), -3689348813882916865⟩, ⟨.coordinate (.quotientLow 0), 2251799813160960⟩, ⟨.coordinate (.quotientLow 1), 4503599626321920⟩, ⟨.coordinate (.quotientLow 2), 9007199252643840⟩, ⟨.coordinate (.quotientLow 3), 18014398505287680⟩, ⟨.coordinate (.quotientLow 4), 36028797010575360⟩, ⟨.coordinate (.quotientLow 5), 72057594021150720⟩, ⟨.coordinate (.quotientLow 6), 144115188042301440⟩, ⟨.coordinate (.quotientLow 7), 288230376084602880⟩, ⟨.coordinate (.quotientLow 8), 576460752169205760⟩, ⟨.coordinate (.quotientLow 9), 1152921504338411520⟩, ⟨.coordinate (.quotientLow 10), 2305843008676823040⟩, ⟨.coordinate (.quotientLow 11), 4611686017353646080⟩, ⟨.coordinate (.quotientLow 12), 9223372034707292160⟩, ⟨.coordinate .residueLeft, 450359962632192⟩, ⟨.coordinate .residueRight, 450359962632192⟩],
   .linear .quotient [⟨.source .one, 7378697627765846835⟩, ⟨.source (.chunkBit 0), 3689348813882916864⟩, ⟨.source (.chunkBit 1), 7378697627765833728⟩, ⟨.source (.chunkBit 2), -3689348813882916865⟩, ⟨.source (.chunkBit 3), -7378697627765833730⟩, ⟨.source (.chunkBit 4), 3689348813882916861⟩, ⟨.source (.chunkBit 5), 7378697627765833722⟩, ⟨.source (.chunkBit 6), -3689348813882916877⟩, ⟨.source (.chunkBit 7), -7378697627765833754⟩, ⟨.source (.chunkBit 8), 3689348813882916813⟩, ⟨.source (.chunkBit 9), 7378697627765833626⟩, ⟨.source (.chunkBit 10), -3689348813882917069⟩, ⟨.source (.chunkBit 11), -7378697627765834138⟩, ⟨.source (.chunkBit 12), 3689348813882916045⟩, ⟨.source (.chunkBit 13), 7378697627765832090⟩, ⟨.source (.chunkBit 14), -3689348813882920141⟩, ⟨.source (.chunkBit 15), -7378697627765840282⟩, ⟨.coordinate .residueLeft, 3689348813882916864⟩, ⟨.coordinate .residueRight, 3689348813882916864⟩],
   .product (.indexProduct 0)
     [⟨.source .index, 1⟩]
     [⟨.source .index, 1⟩, ⟨.source .one, -1⟩],
   .product (.indexProduct 1)
     [⟨.source (SourceRole.indexProduct 0), 1⟩]
     [⟨.source .index, 1⟩, ⟨.source .one, -2⟩],
   .product (.indexProduct 2)
     [⟨.source (SourceRole.indexProduct 1), 1⟩]
     [⟨.source .index, 1⟩, ⟨.source .one, -3⟩]]
def gateArity : Nat := 56
def matrixBindings : List MatrixBinding :=
  [ { role := .selector, index := 0 }
, { role := .bitLeft, index := 44 }
, { role := .bitRight, index := 45 }
, { role := .residueLeft, index := 54 }
, { role := .residueRight, index := 55 } ]
def activeRows : List ActiveRow :=
  [ .bitPair (.quotientLow 0) (.quotientLow 1)
, .bitPair (.quotientLow 2) (.quotientLow 3)
, .bitPair (.quotientLow 4) (.quotientLow 5)
, .bitPair (.quotientLow 6) (.quotientLow 7)
, .bitPair (.quotientLow 8) (.quotientLow 9)
, .bitPair (.quotientLow 10) (.quotientLow 11)
, .bitPair (.quotientLow 12) .quotientHigh
, .residuePair ]
def polynomialTerms : List PolynomialTerm :=
[
  ⟨1, [⟨.selector, 1⟩, ⟨.bitLeft, 4⟩]⟩
,   ⟨-2, [⟨.selector, 1⟩, ⟨.bitLeft, 3⟩]⟩
,   ⟨1, [⟨.selector, 1⟩, ⟨.bitLeft, 2⟩]⟩
,   ⟨-7, [⟨.selector, 1⟩, ⟨.bitRight, 4⟩]⟩
,   ⟨14, [⟨.selector, 1⟩, ⟨.bitRight, 3⟩]⟩
,   ⟨-7, [⟨.selector, 1⟩, ⟨.bitRight, 2⟩]⟩
,   ⟨1, [⟨.selector, 1⟩, ⟨.residueLeft, 6⟩]⟩
,   ⟨-2, [⟨.selector, 1⟩, ⟨.residueLeft, 4⟩]⟩
,   ⟨1, [⟨.selector, 1⟩, ⟨.residueLeft, 2⟩]⟩
,   ⟨-7, [⟨.selector, 1⟩, ⟨.residueLeft, 2⟩, ⟨.residueRight, 2⟩]⟩
,   ⟨14, [⟨.selector, 1⟩, ⟨.residueLeft, 1⟩, ⟨.residueRight, 3⟩]⟩
,   ⟨-7, [⟨.selector, 1⟩, ⟨.residueRight, 4⟩]⟩
]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Sampler.Chunk.Mod5.PackedMod5ArtifactData
