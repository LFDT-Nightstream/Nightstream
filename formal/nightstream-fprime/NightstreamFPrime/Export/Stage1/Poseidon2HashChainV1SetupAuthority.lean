import NightstreamFPrime.Spec.AjtaiSetupV1

/-!
Owns the small executable authority for the verifier-selected
`Poseidon2HashChainV1` Ajtai setup. The package setup module proves that these
direct constants equal the dimensions and seed derived from the recursive
fixed point.
-/

namespace NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1SetupAuthority

open NightstreamFPrime.Spec

def verifierRows : Nat := 22

def messageColumns : Nat := 4750596

/-- Exact owner-approved operating-system CSPRNG output, in byte order. -/
def productionSeedBytes : List Nat :=
  [252, 64, 73, 132, 212, 76, 27, 135, 141, 104, 166, 168, 0, 146,
    215, 215, 171, 68, 216, 26, 193, 123, 69, 168, 231, 189, 76, 31,
    30, 55, 23, 2]

def productionSeed : AjtaiSetupV1.Seed where
  bytes := productionSeedBytes
  length_eq := by rfl
  canonical := by
    intro byte member
    simp [productionSeedBytes] at member
    omega

def authorityNats : List Nat :=
  [37] ++ AjtaiSetupV1.setupIdBytes ++
    [verifierRows, messageColumns, 32] ++ productionSeedBytes

end NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1SetupAuthority
