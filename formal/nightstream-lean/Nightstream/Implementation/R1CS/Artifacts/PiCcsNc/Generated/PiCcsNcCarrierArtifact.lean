import Nightstream.SuperNeo.Concrete.Algebra

/-!
GENERATED FILE - do not edit by hand.

Exact outputs of the optimized production Pi_CCS path for one `m = 257`,
`b = 2` pair that differs only at completed carrier coordinate 257.
Regenerated and
drift-checked by
`cargo test -p neo-reductions --release --test pi_ccs_nc_carrier_lean_artifact`.

This is implementation evidence, not semantic authority. The booleans record
exact optimized Pi_CCS public-API executions; they do not establish a general
Rust refinement theorem or any NIFS/F-prime acceptance claim.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact

def ringDegree : Nat := 54
def logicalWidth : Nat := 257
def ncBound : Nat := 2
def packedColumnCount : Nat := 5
def firstCompletedTail : Nat := 257
def firstTailBlock : Nat := 4
def firstTailLane : Nat := 41
def zeroShapeAccepted : Bool := true
def tailShapeAccepted : Bool := true

def zeroPiCcsAccepted : Bool := true
def tailPiCcsAccepted : Bool := true

def zeroPackedStorage : List Nat :=
  List.replicate 270 0

def tailPackedStorage : List Nat :=
  (List.replicate 270 0).set 209 2

def zeroLogicalDecode : List (Nat × Nat) :=
  List.replicate 257 (0, 0)

def tailLogicalDecode : List (Nat × Nat) :=
  List.replicate 257 (0, 0)

def zeroFullDecode : List (Nat × Nat) :=
  List.replicate 270 (0, 0)

def tailFullDecode : List (Nat × Nat) :=
  (List.replicate 270 (0, 0)).set 257 (2, 0)

def zeroYZcol : List (Nat × Nat) :=
  List.replicate 64 (0, 0)

def tailYZcol : List (Nat × Nat) :=
  List.replicate 64 (0, 0)

end Nightstream.Implementation.R1CS.PiCcsNcCarrierArtifact
