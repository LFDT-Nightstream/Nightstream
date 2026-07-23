import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Carrier270.Schema

/-! Generated file: fixed-point public-coordinate decoder chunk.

Owns: exact proof-free coordinate owners exported from the prepared selective
layout used by the bounded fixed-point projected emitter.

Does not own: source semantics, private coordinates, relation satisfaction,
commitment alignment, or row removal. Do not hand-edit.

Emits constraints: no.

| Artifact field | Exact source | Meaning |
|---|---|---|
| `totalColumns` | final projected-emitter width | bounded profile only |
| `rawCoordinates` | validated prepared-layout owners | public decoder data |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk1

open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Wire

def totalColumns : Nat := 11437038
def rawCoordinates : List RawCoordinate := [
  { schemaVersion := 1, column := 256, source := .sourceField 256 }
, { schemaVersion := 1, column := 257, source := .fixedZero }
, { schemaVersion := 1, column := 258, source := .fixedZero }
, { schemaVersion := 1, column := 259, source := .fixedZero }
, { schemaVersion := 1, column := 260, source := .fixedZero }
, { schemaVersion := 1, column := 261, source := .fixedZero }
, { schemaVersion := 1, column := 262, source := .fixedZero }
, { schemaVersion := 1, column := 263, source := .fixedZero }
, { schemaVersion := 1, column := 264, source := .fixedZero }
, { schemaVersion := 1, column := 265, source := .fixedZero }
, { schemaVersion := 1, column := 266, source := .fixedZero }
, { schemaVersion := 1, column := 267, source := .fixedZero }
, { schemaVersion := 1, column := 268, source := .fixedZero }
, { schemaVersion := 1, column := 269, source := .fixedZero }
]

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.Carrier270.Generated.Chunk1
