import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema

/-! Generated file: representative production radix-four centered-domain rows.

Owns: one exact two-coordinate row and the exact fixed-zero tail row from the
final recursive-arm matrices of the production WASM census profile.

Does not own: source-coordinate meaning, all centered rows, selector dispatch,
constraint necessity, security reduction, or permission to remove rows.

Emits constraints: no. Rust materializes both final rows before export.

| Artifact row | Final nonempty ports | Assurance use |
|---|---|---|
| pair | G, E, U, A | exact production coefficient binding |
| tail | G, E, U | exact production fixed-zero binding |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourCenteredDomainRows

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire

def rawPairRow : RawRow where
  schemaVersion := 1
  rows := 8102331
  columns := 12288726
  emittedRow := 45768
  runIndex := 3
  family := .armDomain
  arm := some 1
  ports := [
    { terms := [] }
  , { terms := [{ column := 2431, coefficient := 1 }] }
  , { terms := [{ column := 366188, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 366187, coefficient := 1 }] }
  , { terms := [{ column := 2431, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawTailRow : RawRow where
  schemaVersion := 1
  rows := 8102331
  columns := 12288726
  emittedRow := 4982068
  runIndex := 3
  family := .armDomain
  arm := some 1
  ports := [
    { terms := [] }
  , { terms := [{ column := 2431, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 10672171, coefficient := 1 }] }
  , { terms := [{ column := 2431, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryRadixFourCenteredDomainRows
