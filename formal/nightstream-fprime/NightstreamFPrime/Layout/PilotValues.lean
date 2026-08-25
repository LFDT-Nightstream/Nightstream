import NightstreamFPrime.Spec.Poseidon2

/-!
Lightweight owner of the fixed production-pilot physical values. This module
contains no row list, circuit, lifecycle key, or emitter. Layout proofs and
the package serializer both read these definitions.
-/

namespace NightstreamFPrime.Layout.PilotValues

abbrev digestWords : Nat := 4
abbrev stateHashBaseWords : Nat := 42463
abbrev stateHashWords : Nat :=
  stateHashBaseWords + digestWords + digestWords + digestWords
abbrev priorPublicInputWords : Nat := 54

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
abbrev hashRowCount : Nat := hashWitnessCount + digestWords

abbrev priorHashRowStart : Nat := 0
abbrev priorBindingRowCount : Nat := 50
abbrev priorBindingRowStart : Nat := priorHashRowStart + hashRowCount
abbrev outputHashRowStart : Nat := priorBindingRowStart + priorBindingRowCount
abbrev physicalRowCount : Nat :=
  hashRowCount + priorBindingRowCount + hashRowCount

abbrev publicColumnCount : Nat := priorPublicInputWords + digestWords
abbrev sourceColumnCount : Nat := externalColumnCount + 2 * hashWitnessCount
abbrev privateColumnCount : Nat := sourceColumnCount - publicColumnCount
abbrev constantColumn : Nat := privateColumnCount
abbrev spartanColumnCount : Nat :=
  privateColumnCount + 1 + publicColumnCount

abbrev secondPrivateStart : Nat := stateHashWords
abbrev witnessPrivateStart : Nat := 2 * stateHashWords
abbrev firstPublicStart : Nat := privateColumnCount + 1
abbrev secondPublicStart : Nat := firstPublicStart + priorPublicInputWords

abbrev witnessPrivateLength : Nat := 2 * hashWitnessCount
abbrev priorWitnessStart : Nat := witnessPrivateStart
abbrev outputWitnessStart : Nat := witnessPrivateStart + hashWitnessCount

end NightstreamFPrime.Layout.PilotValues
