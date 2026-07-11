import Nightstream.Implementation.R1CS.Semantics

/-!
Exact plain-state row program emitted by Rust `engine::decider::enforce_state_link`.

The two state wire bundles are allocated consecutively, 31 columns each. The
program equates every verifier-key/header lane, both counters, both boundary
digests, pc, semantic/accumulator digests, and public trace. Nebula-enabled
layouts require a separately hashed artifact.
-/

namespace Nightstream.Implementation.R1CS.FPrimeStateLink

open Nightstream.Implementation.R1CS

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-state-link"
def sourceAnchor : String := "enforce_state_link"
def artifactSha256 : String := "cc0411f5a91daedfdb44b4e0bf8572b78a86027517d2b5a6cfe24a68222ba718"
def witnessSha256 : String := "378cf7f9cd0adc46d5d04f3625320e5c95965ae2b96cc5aa5ddf17999d13003d"

def rowCount : Nat := 31
def colCount : Nat := 63
def nextOffset : Nat := 31

def digestPairs (start : Nat) : List (Nat × Nat) :=
  (List.range 4).map (fun lane => (start + lane, start + nextOffset + lane))

def columnPairs : List (Nat × Nat) :=
  digestPairs 1 ++ digestPairs 5 ++ [(9, 40), (10, 41)] ++
  digestPairs 11 ++ digestPairs 15 ++ [(19, 50)] ++
  digestPairs 20 ++ digestPairs 24 ++ digestPairs 28

def equalityRow (columns : Nat × Nat) : Row :=
  ⟨[(columns.1, 1), (columns.2, goldilocksP - 1)], [(0, 1)], []⟩

def rows : List Row :=
  columnPairs.map equalityRow

theorem rows_length : rows.length = rowCount := by decide

end Nightstream.Implementation.R1CS.FPrimeStateLink
