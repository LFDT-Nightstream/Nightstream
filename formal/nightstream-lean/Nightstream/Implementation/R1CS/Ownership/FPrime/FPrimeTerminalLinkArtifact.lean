import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Exact row program emitted by Rust `engine::decider::enforce_terminal_latest_link`
for one trailing fresh claim in the plain 270-field SuperNeo carrier.

The host checks nonempty batch and exact lengths before these rows are emitted.
The production helper emits one affine-one equality followed by 256 bitwise
equalities to the last producer step's already-canonical `x_out_bits`, then
thirteen zero rows for the carrier-completion padding.
-/

namespace Nightstream.Implementation.R1CS.FPrimeTerminalLink

open Nightstream.Implementation.R1CS

set_option maxRecDepth 32768

def schemaVersion : Nat := 1
def artifactKind : String := "r1cs/f-prime-terminal-latest-link"
def sourceAnchor : String := "enforce_terminal_latest_link"
def artifactSha256 : String := "4fb1c29ec13e1743c37c1e1910d5f7a1e130a247eda8f831de31c3a47638b3fe"
def witnessSha256 : String := "0784882569e153e49c9a7dc5e833c625a7afca1895404fceafc845736c993d8b"

def rowCount : Nat := 270
def colCount : Nat := 527
def freshOneCol : Nat := 257
def freshBitCol (bit : Nat) : Nat := 258 + bit
def freshPaddingCol (padding : Nat) : Nat := 514 + padding
def lastXOutBitCol (bit : Nat) : Nat := 1 + bit

def oneRow : Row :=
  ⟨[(freshOneCol, 1), (0, goldilocksP - 1)], [(0, 1)], []⟩

def linkRow (bit : Nat) : Row :=
  ⟨[(freshBitCol bit, 1), (lastXOutBitCol bit, goldilocksP - 1)],
   [(0, 1)], []⟩

def paddingRow (padding : Nat) : Row :=
  ⟨[(freshPaddingCol padding, 1)], [(0, 1)], []⟩

def rows : List Row :=
  oneRow ::
    ((List.range 256).map linkRow ++
      (List.range 13).map paddingRow)

theorem rows_length : rows.length = rowCount := by decide

end Nightstream.Implementation.R1CS.FPrimeTerminalLink
