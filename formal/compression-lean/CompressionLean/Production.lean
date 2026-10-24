import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Export.Stage1.Poseidon2HashChainV1Package

/-!
Owns the production constants of the layer-1 key, each named once: F′'s
constant where F′ has one, and its value. The key check (`KeyFacts`), the
claim sizes of the final program and `parentShape_of` use these names. The
Rust tests `relation_shape_has_the_lean_constants` (`neo-spartan`) and
`parent_relation_arguments_are_the_lean_constants` (`nightstream`) pin the
same numbers.
-/

namespace CompressionLean.Production

open NightstreamFPrime NightstreamFPrime.Spec NightstreamFPrime.Export.Stage1

/-- `κ`, the commitment rows (`PRODUCTION_VERIFIER_ROWS`): F′'s commitment width. -/
abbrev kappa : ℕ := productionProfile.commitmentWidth

theorem kappa_eq : kappa = 22 := rfl

/-- The CCS matrices: F′'s. -/
abbrev matrices : ℕ := productionProfile.ccsMatrices

theorem matrices_eq : matrices = 4 := rfl

/-- The public blocks (`PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS / D`): F′'s public
ring columns. -/
abbrev publicBlocks : ℕ := Lifecycle.PaperAlgebra.publicRingColumns

theorem publicBlocks_eq : publicBlocks = 5 := rfl

/-- The variables of the point `r`: F′'s cube variables. -/
abbrev pointVariables : ℕ := Lifecycle.cubeVariables

theorem pointVariables_eq : pointVariables = 28 := rfl

/-- PiRLC's guard bound `H` (`compression_bound`, `nightstream/src/lifecycle/finish.rs`).
F′ has no constant for it. -/
abbrev normBound : ℕ := 3672

/-- The logical width of the Poseidon2 hash-chain application, the
application of the production key. -/
abbrev poseidon2Width : ℕ := PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application

theorem poseidon2Width_eq : poseidon2Width = 43963750 := Poseidon2HashChainV1Package.logicalWidth

end CompressionLean.Production
