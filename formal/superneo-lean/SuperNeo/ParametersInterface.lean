import SuperNeo.Parameters

/-!
Contract interface for `SuperNeo.Parameters`.

Spec: `specs/Parameters.spec.md`

Paper anchors:
- Appendix B.2, lines 709-727: `b = 2`, `k = 14`, `B = b^k`, `κ = 18`, `K_max = 61`, `T = 216`.
- Definition 1, Section 4, lines 275-282: norm bounds `b, B < q/2`.
-/

namespace SuperNeo

namespace ParametersInterface

/-! ## Core Surfaces (decomposition + security parameters only; η, d, nF live in Dimensions) -/

abbrev kappa := SuperNeo.Parameters.Goldilocks.kappa
abbrev b := SuperNeo.Parameters.Goldilocks.b
abbrev k := SuperNeo.Parameters.Goldilocks.k
abbrev Kmax := SuperNeo.Parameters.Goldilocks.Kmax
abbrev B := SuperNeo.Parameters.Goldilocks.B
abbrev T := SuperNeo.Parameters.Goldilocks.T
abbrev cCoeffMin := SuperNeo.Parameters.Goldilocks.cCoeffMin
abbrev cCoeffMax := SuperNeo.Parameters.Goldilocks.cCoeffMax
abbrev extDegreeK := SuperNeo.Parameters.Goldilocks.extDegreeK
abbrev concreteParameters := SuperNeo.Parameters.Goldilocks.concreteParameters

/-! ## Concrete equalities -/

abbrev b_eq_2 := SuperNeo.Parameters.Goldilocks.b_eq_2
abbrev k_eq_14 := SuperNeo.Parameters.Goldilocks.k_eq_14
abbrev Kmax_eq_61 := SuperNeo.Parameters.Goldilocks.Kmax_eq_61
abbrev T_eq_216 := SuperNeo.Parameters.Goldilocks.T_eq_216
abbrev extDegreeK_eq_2 := SuperNeo.Parameters.Goldilocks.extDegreeK_eq_2
abbrev kappa_eq_18 := SuperNeo.Parameters.Goldilocks.kappa_eq_18
abbrev B_def := SuperNeo.Parameters.Goldilocks.B_def
abbrev B_eq_16384 := SuperNeo.Parameters.Goldilocks.B_eq_16384

/-! ## Bounds -/

abbrev b_lt_modulus_half := SuperNeo.Parameters.Goldilocks.b_lt_modulus_half
abbrev b_lt_modulus := SuperNeo.Parameters.Goldilocks.b_lt_modulus
abbrev B_lt_modulus := SuperNeo.Parameters.Goldilocks.B_lt_modulus

/-! ## Positivity -/

abbrev b_pos := SuperNeo.Parameters.Goldilocks.b_pos
abbrev k_pos := SuperNeo.Parameters.Goldilocks.k_pos
abbrev B_pos := SuperNeo.Parameters.Goldilocks.B_pos
abbrev Kmax_pos := SuperNeo.Parameters.Goldilocks.Kmax_pos
abbrev T_pos := SuperNeo.Parameters.Goldilocks.T_pos
abbrev kappa_pos := SuperNeo.Parameters.Goldilocks.kappa_pos

end ParametersInterface

end SuperNeo
