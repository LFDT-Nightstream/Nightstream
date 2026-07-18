import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.Accumulator.Poseidon2Cost

/-! Focused interface gate for the reduced accumulator Poseidon2 cost leaf. -/

open Nightstream.Implementation.R1CS.FPrimeFullHistory.Accumulator.Poseidon2Cost

#check permutation_product_rows_eq
#check permutation_linear_rows_eq
#check sponge_rows_formula
#check sponge_fresh_columns_eq_rows
#check sponge_gate_partition
#check commitment_family_hash_rows_formula
#check canonical_parent_hash_rows_formula

/-! Current Fibonacci compiler diagnostic (`rowVariables = 12`). The ordered
carrier includes its exact ten-field domain message but excludes materializing
those constants. These are raw field-valued R1CS costs, not gadget-native
columns and not a production-conformance claim. -/
#guard commitmentFamilyCarrierFields 12 = 13632
#guard commitmentFamilyPreimageFields 12 = 13642
#guard permutationCount 13642 = 3412
#guard spongeProductRows 13642 = 1173728
#guard spongeLinearRows 13642 = 887116
#guard commitmentFamilyHashRowsFor 12 = 2060844

/-! The canonical-parent candidate still has no approved domain message. -/
#guard canonicalParentCarrierFields 12 = 996
#guard permutationCount 996 = 250
#guard spongeProductRows 996 = 86000
#guard spongeLinearRows 996 = 64998
#guard canonicalParentHashRowsFor 12 0 = 150998
