import NightstreamFPrime.Lifecycle.PaperAlgebra
import NightstreamFPrime.Spec
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.GoldilocksCausal
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetConvention
import NightstreamFPrime.Spec.Folding.PiRLC.PaperForkExtractionWork

/-! Rust tests `relation_shape_has_the_lean_constants` (`neo-spartan`) and `final_proof_verifies` (`nightstream`) pin these constants. -/

namespace CompressionLean.Production

open NightstreamFPrime NightstreamFPrime.Spec

abbrev kappa : ℕ := productionProfile.commitmentWidth

theorem kappa_eq : kappa = 22 := rfl

abbrev matrices : ℕ := productionProfile.ccsMatrices

theorem matrices_eq : matrices = 4 := rfl

abbrev publicBlocks : ℕ := Lifecycle.PaperAlgebra.publicRingColumns

theorem publicBlocks_eq : publicBlocks = 5 := rfl

abbrev pointVariables : ℕ := Lifecycle.cubeVariables

theorem pointVariables_eq : pointVariables = 28 := rfl

/-- `B = 2^16` (`productionGlobalParams.bigB`, `k_rho = 16`). Each slot's honest bound lies below it
(`option_b/compression.rs::compression_bound`). -/
abbrev normLimit : ℕ := 65536

theorem normLimit_eq : normLimit = productionGlobalParams.bigB := by decide

end CompressionLean.Production
