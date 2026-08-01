import Nightstream.Implementation.Lowering.Nebula.StepPolynomial

set_option autoImplicit false

namespace tests.NebulaStepPolynomial

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.Lowering.Nebula.StepPolynomial

theorem honest_bit_zero : evaluate (bitPoint 1) = 0 := by
  decide

theorem mutated_bit_rejected : evaluate (bitPoint 2) ≠ 0 := by
  decide

theorem honest_product_zero : evaluate (productPoint 0 9) = 0 := by
  decide

theorem mutated_product_rejected : evaluate (productPoint 2 3) ≠ 0 := by
  decide

theorem honest_product_equality_zero :
    evaluate (productEqualityPoint 2 3 6) = 0 := by
  decide

theorem mutated_product_equality_rejected :
    evaluate (productEqualityPoint 2 3 7) ≠ 0 := by
  decide

theorem honest_linear_zero : evaluate (linearPoint 42 42) = 0 := by
  decide

theorem mutated_linear_rejected : evaluate (linearPoint 42 43) ≠ 0 := by
  decide

/-- With `active = 0` and `pad = 1`, the update copies `A` to the output. -/
theorem honest_padding_update_zero :
    evaluate (extensionUpdatePoint 5 5 9 1 0 7 8 10 11 12) = 0 := by
  decide

theorem mutated_padding_update_rejected :
    evaluate (extensionUpdatePoint 6 5 9 1 0 7 8 10 11 12) ≠ 0 := by
  decide

/-- With `pad = 0` and `active = 1`, the update consumes both fingerprint
components and their value factors. -/
theorem honest_active_update_zero :
    evaluate (extensionUpdatePoint 47 2 3 0 1 7 11 0 0 17) = 0 := by
  decide

theorem mutated_active_update_rejected :
    evaluate (extensionUpdatePoint 48 2 3 0 1 7 11 0 0 17) ≠ 0 := by
  decide

end tests.NebulaStepPolynomial
