import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Schema-1 exact one-claim CE child/running continuity artifact.

The full Rust artifact has 1,175 rows: six verifier-owned shape-metadata pins
and 1,169 direct child/running CE-core equalities. Child/running `y_zcol` is
absent by the proved PiDEC-child sidecar-erasure boundary; parent authority and
terminal CE validation retain it. The equality pairs compress into eight
consecutive runs below without losing any retained coordinate. The full
artifact hash covers both row families; this module's theorem owns the
continuity rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeCeContinuity

open Nightstream.Implementation.R1CS

set_option maxRecDepth 32768

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-ce-continuity"
def sourceAnchor : String := "enforce_child_core_equal_running"
def artifactSha256 : String := "9395371940047222a2661ac0d5b948a27ca0b743151a01d8e74901c2c59c4506"
def witnessSha256 : String := "17f8eabc453b8543511ce5ca708226298def3c236c1d30d528bdc3acc2f4a470"

def rowCount : Nat := 1175
def colCount : Nat := 2340
def continuityRowCount : Nat := 1169

structure PairRun where
  left : Nat
  right : Nat
  count : Nat
deriving DecidableEq, Repr

def pairRuns : List PairRun :=
  [⟨1, 1227, 972⟩, ⟨974, 1171, 54⟩, ⟨1166, 1225, 2⟩,
   ⟨1168, 2199, 3⟩, ⟨1158, 2202, 4⟩, ⟨1156, 2334, 2⟩,
   ⟨1028, 2206, 128⟩, ⟨1162, 2336, 4⟩]

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
