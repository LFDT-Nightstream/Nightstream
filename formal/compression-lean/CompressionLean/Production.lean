import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-! Rust tests `relation_shape_has_the_lean_constants` (`neo-spartan`) and `parent_relation_arguments_are_the_lean_constants` (`nightstream`) pin these constants. -/

namespace CompressionLean.Production

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1

abbrev kappa : ℕ := productionProfile.commitmentWidth

theorem kappa_eq : kappa = 22 := rfl

abbrev matrices : ℕ := productionProfile.ccsMatrices

theorem matrices_eq : matrices = 4 := rfl

abbrev publicBlocks : ℕ := Lifecycle.PaperAlgebra.publicRingColumns

theorem publicBlocks_eq : publicBlocks = 5 := rfl

abbrev pointVariables : ℕ := Lifecycle.cubeVariables

theorem pointVariables_eq : pointVariables = 28 := rfl

abbrev normBound : ℕ := 3672

abbrev poseidon2Width : ℕ := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application

theorem poseidon2Width_eq : poseidon2Width = 43963750 := Poseidon2HashChainV1Package.logicalWidth

noncomputable abbrev poseidon2Relation := PerApplicationFixedPoint.relation Poseidon2HashChainV1Package.application
  Poseidon2HashChainV1Package.fits

end CompressionLean.Production
