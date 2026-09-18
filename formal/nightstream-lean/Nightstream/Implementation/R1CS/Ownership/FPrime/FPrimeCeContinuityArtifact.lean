import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Schema-1 exact one-claim CE child/running continuity artifact.

The full Rust artifact has 1,303 rows: six verifier-owned shape-metadata pins
and 1,297 direct child/running equalities. The equality pairs compress into
eight consecutive runs below without losing any coordinate. The full artifact
hash covers both families; this module's theorem owns the continuity rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeCeContinuity

open Nightstream.Implementation.R1CS

set_option maxRecDepth 32768

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-ce-continuity"
def sourceAnchor : String := "enforce_children_equal_running"
def artifactSha256 : String := "28f0dab3afbac6183eb0804330ff1c23ee2d897dfe1a68fe4715a7e7bf16435b"
def witnessSha256 : String := "11964e484703fe0029d1fa15cd83cc1b5c1af00c62f1b91ba69ea74ad8898f77"

def rowCount : Nat := 1303
def colCount : Nat := 2596
def continuityRowCount : Nat := 1297

structure PairRun where
  left : Nat
  right : Nat
  count : Nat
deriving DecidableEq, Repr

def pairRuns : List PairRun :=
  [⟨1, 1355, 972⟩, ⟨974, 1299, 54⟩, ⟨1294, 1353, 2⟩,
   ⟨1296, 2327, 3⟩, ⟨1158, 2330, 4⟩, ⟨1156, 2462, 2⟩,
   ⟨1028, 2334, 128⟩, ⟨1162, 2464, 132⟩]

def expandRun (run : PairRun) : List (Nat × Nat) :=
  (List.range run.count).map (fun offset =>
    (run.left + offset, run.right + offset))

def columnPairs : List (Nat × Nat) := pairRuns.flatMap expandRun

def equalityRow (columns : Nat × Nat) : Row :=
  ⟨[(columns.1, 1), (columns.2, goldilocksP - 1)], [(0, 1)], []⟩

def continuityRows : List Row := columnPairs.map equalityRow

theorem columnPairs_length : columnPairs.length = continuityRowCount := by decide
theorem continuityRows_length : continuityRows.length = continuityRowCount := by decide

end Nightstream.Implementation.R1CS.FPrimeCeContinuity
