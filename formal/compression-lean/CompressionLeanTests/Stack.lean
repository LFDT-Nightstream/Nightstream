import CompressionLean.Spec.Checks

/-! `setupCheck` decides `∀ p ∈ pushes, …` over the 38,533,993 production runs. Without
`Spec.decidableBAll_eq_decidableBAllTR`, this evaluation stops with deep recursion. -/

#guard decide (∀ x ∈ List.replicate 38533993 0, x = 0)
