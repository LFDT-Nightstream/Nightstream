import Nightstream.SuperNeo.Concrete.Algebra

/-!
GENERATED FILE - do not edit by hand.

Exact outputs of the current fixed F-prime carrier fixture: the `1 x 257`
all-zero R1CS translated by the direct-CCS frontend, preprocessing seed 41,
and a canonical fixed-k zero accumulator. The tail witness differs only at
completed Phi81 carrier coordinate 257.

Regenerated and drift-checked by
`cargo test -p neo-fold-clean --release --test f_prime_fixed_carrier_nifs_lean_artifact`.

This is implementation evidence, not semantic authority. It does not prove a
general Pi_CCS/NIFS refinement, F-prime acceptance, or permission to remove rows.
-/

namespace Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact

def preprocessSeed : Nat := 41
def ringDegree : Nat := 54
def relationRows : Nat := 1
def relationColumns : Nat := 257
def relationArity : Nat := 3
def relationDegree : Nat := 2
def publicInputLen : Nat := 257
def ncBound : Nat := 2
def packedRows : Nat := 54
def packedColumns : Nat := 5
def firstCompletedTail : Nat := 257
def firstTailBlock : Nat := 4
def firstTailLane : Nat := 41
def tailValue : Nat := 2
def canonicalRunningCount : Nat := 14

def structureDigest : List Nat := [4963443989406736758, 16818727102537945525, 18294972881252195440, 13267265213911833074]

def freshPublicInputsEqual : Bool := true
def commitmentsDiffer : Bool := true
def zeroPiCcsAccepted : Bool := true
def tailPiCcsAccepted : Bool := true

def zeroNifsProved : Bool := true
def zeroNifsVerified : Bool := true
def tailNifsProved : Bool := true
def tailNifsVerified : Bool := true

def linkedTailNifsProved : Bool := true
def linkedTailNifsVerified : Bool := true
def linkedRecursiveFPrimeBuilt : Bool := true
def linkedRecursiveFPrimeSatisfied : Bool := true

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

def linkedPublicInput : List Nat :=
  [1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

def linkedTailFullDecode : List (Nat × Nat) :=
  [(1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0), (1, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (0, 0), (1, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (2, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]

end Nightstream.Implementation.R1CS.FPrimeFixedCarrierNifsArtifact
