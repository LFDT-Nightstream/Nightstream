import NightstreamFPrime.Spec.Poseidon2

/-!
Lightweight owner of the fixed production-pilot physical values. This module
contains no row list, circuit, lifecycle key, or emitter. Layout proofs and
the package serializer both read these definitions.
-/

namespace NightstreamFPrime.Layout.PilotValues

abbrev digestWords : Nat := 4
abbrev stateHashBaseWords : Nat := 49381
abbrev stateHashWords : Nat :=
  stateHashBaseWords + digestWords + digestWords + digestWords
abbrev priorPublicInputWords : Nat := 270

abbrev priorPreimageStart : Nat := 0
abbrev priorPublicInputStart : Nat := priorPreimageStart + stateHashWords
abbrev outputPreimageStart : Nat :=
  priorPublicInputStart + priorPublicInputWords
abbrev outputDigestStart : Nat := outputPreimageStart + stateHashWords
abbrev externalColumnCount : Nat := outputDigestStart + digestWords

abbrev absorbCount : Nat :=
  (stateHashWords + NightstreamFPrime.Spec.Poseidon2.rate - 1) /
    NightstreamFPrime.Spec.Poseidon2.rate
abbrev permutationRecipeCount : Nat := 592
abbrev permutationOutputLocalStart : Nat := 584
abbrev hashWitnessCount : Nat :=
  (absorbCount + 1) * permutationRecipeCount
abbrev outputHashRowCount : Nat := hashWitnessCount + digestWords
abbrev priorCanonicalPrivateCount : Nat := 4 * 66
abbrev priorCanonicalFreshCount : Nat := 4 * 197
abbrev priorCanonicalRowCount : Nat := 4 * 328
abbrev priorFixedRowCount : Nat := 14
abbrev priorExtraRowCount : Nat :=
  priorCanonicalRowCount + priorFixedRowCount
abbrev priorHashRowCount : Nat := hashWitnessCount + priorExtraRowCount
abbrev hashRowCount : Nat := outputHashRowCount

abbrev priorHashRowStart : Nat := 0
abbrev priorBindingRowCount : Nat := priorExtraRowCount
abbrev priorBindingRowStart : Nat := priorHashRowStart + hashWitnessCount
abbrev outputHashRowStart : Nat := priorBindingRowStart + priorBindingRowCount
abbrev physicalRowCount : Nat :=
  priorHashRowCount + outputHashRowCount

abbrev publicColumnCount : Nat := priorPublicInputWords + digestWords
abbrev logicalColumnCount : Nat :=
  externalColumnCount + 2 * hashWitnessCount + priorCanonicalPrivateCount
abbrev sourceColumnCount : Nat :=
  logicalColumnCount + priorCanonicalFreshCount
abbrev privateColumnCount : Nat := sourceColumnCount - publicColumnCount
abbrev constantColumn : Nat := privateColumnCount
abbrev spartanColumnCount : Nat :=
  privateColumnCount + 1 + publicColumnCount

abbrev secondPrivateStart : Nat := stateHashWords
abbrev witnessPrivateStart : Nat := 2 * stateHashWords
abbrev firstPublicStart : Nat := privateColumnCount + 1
abbrev secondPublicStart : Nat := firstPublicStart + priorPublicInputWords

abbrev witnessPrivateLength : Nat :=
  2 * hashWitnessCount + priorCanonicalPrivateCount +
    priorCanonicalFreshCount
abbrev priorWitnessStart : Nat := witnessPrivateStart
abbrev outputWitnessStart : Nat :=
  witnessPrivateStart + hashWitnessCount + priorCanonicalPrivateCount

end NightstreamFPrime.Layout.PilotValues
