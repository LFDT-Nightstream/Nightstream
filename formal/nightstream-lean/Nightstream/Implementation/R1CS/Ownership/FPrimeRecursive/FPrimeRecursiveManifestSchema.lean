/-! Schema for compact generated recursive-program ownership metadata. -/

namespace Nightstream.Implementation.R1CS.FPrimeRecursiveManifest

structure RowRange where
  name : String
  rowStart : Nat
  rowEnd : Nat
  nonzeroEntries : Nat
  sha256 : String
deriving DecidableEq, Repr, Inhabited

def RowRange.rowCount (range : RowRange) : Nat :=
  range.rowEnd - range.rowStart

def covers : Nat → Nat → List RowRange → Bool
  | cursor, finish, [] => cursor == finish
  | cursor, finish, range :: rest =>
      range.rowStart == cursor &&
      range.rowStart <= range.rowEnd &&
      covers range.rowEnd finish rest

end Nightstream.Implementation.R1CS.FPrimeRecursiveManifest
