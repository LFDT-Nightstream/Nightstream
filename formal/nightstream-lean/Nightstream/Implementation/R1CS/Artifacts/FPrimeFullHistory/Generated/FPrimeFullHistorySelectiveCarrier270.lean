import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveCcsRowSchema

/-! Generated file: exact selective-compiler public-carrier fixture.

Owns: the compiler-produced 257/270 public widths, public-padding, selector,
private-alignment and branch ranges; the exact selector-domain and sum rows;
one representative gated source row; and every public-padding row of the
three-arm F-prime-width fixture.

Does not own: semantic truth of those rows, a full fixed-point F-prime relation,
private branch rows, NIFS soundness, constraint necessity, or row removal.

Emits constraints: no. This file is inert Rust-exported data.

| Artifact family | Exact source | Multiplicity |
|---|---|---:|
| public layout | prepared layout consumed by the selective emitter | 1 |
| selector domain | final matrices joined to the exclusive row ledger | 3 |
| selector total | final matrices joined to the exclusive row ledger | 1 |
| representative arm gate | first retained source row in arm zero | 1 |
| public zero padding | final thirteen-port structure joined to the exclusive row ledger | 13 |
-/

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCarrier270

open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire

def logicalPublicInputLen : Nat := 257
def publicInputLen : Nat := 270
def publicPaddingColumns : List Nat := [257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269]
def selectorColumns : List Nat := [270, 271, 272]
def privateAlignmentPaddingColumns : List Nat := [273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310]
def sharedPrivateStart : Nat := 311
def sharedPrivateEnd : Nat := 311
def branchStart : Nat := 311
def branchEnd : Nat := 311
def ringAlignmentPaddingStart : Nat := 311
def ringAlignmentPaddingEnd : Nat := 324


def rawSelectorRow00 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 0
  runIndex := 0
  family := .selectorDomain
  arm := none
  ports := [
    { terms := [{ column := 270, coefficient := 1 }] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawSelectorRow01 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 1
  runIndex := 0
  family := .selectorDomain
  arm := none
  ports := [
    { terms := [{ column := 271, coefficient := 1 }] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawSelectorRow02 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 2
  runIndex := 0
  family := .selectorDomain
  arm := none
  ports := [
    { terms := [{ column := 272, coefficient := 1 }] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawSelectorRows : List RawRow := [
  rawSelectorRow00
, rawSelectorRow01
, rawSelectorRow02
]

def rawOneHotRow : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 3
  runIndex := 5
  family := .oneHot
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 0, coefficient := 18446744069414584320 }, { column := 270, coefficient := 1 }, { column := 271, coefficient := 1 }, { column := 272, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawGatedRow : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 55
  runIndex := 8
  family := .retained
  arm := some 0
  ports := [
    { terms := [] }
  , { terms := [{ column := 270, coefficient := 1 }] }
  , { terms := [{ column := 1, coefficient := 1 }] }
  , { terms := [{ column := 0, coefficient := 18446744069414584320 }, { column := 1, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow00 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 4
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 257, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow01 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 5
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 258, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow02 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 6
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 259, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow03 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 7
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 260, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow04 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 8
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 261, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow05 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 9
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 262, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow06 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 10
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 263, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow07 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 11
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 264, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow08 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 12
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 265, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow09 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 13
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 266, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow10 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 14
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 267, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow11 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 15
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 268, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRow12 : RawRow where
  schemaVersion := 1
  rows := 836
  columns := 324
  emittedRow := 16
  runIndex := 6
  family := .publicPadding
  arm := none
  ports := [
    { terms := [] }
  , { terms := [{ column := 0, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [{ column := 269, coefficient := 1 }] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
  , { terms := [] }
]

def rawPaddingRows : List RawRow := [
  rawPaddingRow00
, rawPaddingRow01
, rawPaddingRow02
, rawPaddingRow03
, rawPaddingRow04
, rawPaddingRow05
, rawPaddingRow06
, rawPaddingRow07
, rawPaddingRow08
, rawPaddingRow09
, rawPaddingRow10
, rawPaddingRow11
, rawPaddingRow12
]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistorySelectiveCarrier270
