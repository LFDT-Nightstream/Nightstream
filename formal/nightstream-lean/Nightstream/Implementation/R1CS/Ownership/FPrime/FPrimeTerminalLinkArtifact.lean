import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Exact row program emitted by Rust `engine::decider::enforce_terminal_latest_link`
for one trailing fresh claim in the plain 257-field F' public layout.

The host checks nonempty batch and exact lengths before these rows are emitted.
The production helper emits one affine-one equality followed by 256 bitwise
equalities to the last producer step's already-canonical `x_out_bits`.
-/

namespace Nightstream.Implementation.R1CS.FPrimeTerminalLink

open Nightstream.Implementation.R1CS

set_option maxRecDepth 32768

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-terminal-latest-link"
def sourceAnchor : String := "enforce_terminal_latest_link"
def artifactSha256 : String := "e471c6f31577c757e39b2e417f3981b056ed723d55c2d900e89faa440df8b873"
def witnessSha256 : String := "f829ca48cfbc6e1a0270823774e81bdcdc9100dcc0e16674b14ae1ae29d69930"

def rowCount : Nat := 257
def colCount : Nat := 514
def freshOneCol : Nat := 257
def freshBitCol (bit : Nat) : Nat := 258 + bit
def lastXOutBitCol (bit : Nat) : Nat := 1 + bit

def oneRow : Row :=
  ⟨[(freshOneCol, 1), (0, goldilocksP - 1)], [(0, 1)], []⟩

def linkRow (bit : Nat) : Row :=
  ⟨[(freshBitCol bit, 1), (lastXOutBitCol bit, goldilocksP - 1)],
   [(0, 1)], []⟩

def rows : List Row :=
  oneRow :: (List.range 256).map linkRow

theorem rows_length : rows.length = rowCount := by decide

end Nightstream.Implementation.R1CS.FPrimeTerminalLink
