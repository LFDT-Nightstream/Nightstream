import Mathlib.Tactic

/-!
Closed arithmetic certificates for the exponent-26 production recursive-core
census. The product total includes the complete NIFS output carrier.

This module is intentionally independent of protocol types. Large numeral
normalization occurs once here. Protocol modules consume the resulting
kernel-checked facts.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreArithmetic

theorem facts :
    12127977 + 1710720 + 216270 + 7290 + 4817340 + 5400 + 23812 =
        18908809 /\
      18908809 + 7364831 + 2 * 27469 + 445167 + 34691 + 7349838 + 28 =
        34158302 /\
      18908809 + 7402280 + 2 * 109342 + 445167 + 34691 + 7349838 + 28 =
        34359497 /\
      18908809 + 7452412 + 2 * 218506 + 445167 + 34691 + 7349838 + 28 =
        34627957 /\
      18908809 + 7552676 + 2 * 436834 + 445167 + 34691 + 7349838 + 28 =
        35164877 /\
      2 ^ 25 < 34158302 /\
      2 ^ 25 < 34359497 /\
      2 ^ 25 < 34627957 /\
      2 ^ 25 < 35164877 /\
      34158302 <= 2 ^ 26 /\
      34359497 <= 2 ^ 26 /\
      34627957 <= 2 ^ 26 /\
      35164877 <= 2 ^ 26 := by
  norm_num

theorem product :
    12127977 + 1710720 + 216270 + 7290 + 4817340 + 5400 + 23812 =
      18908809 :=
  facts.1

theorem e1 :
    18908809 + 7364831 + 2 * 27469 + 445167 + 34691 + 7349838 + 28 =
      34158302 :=
  facts.2.1

theorem e4 :
    18908809 + 7402280 + 2 * 109342 + 445167 + 34691 + 7349838 + 28 =
      34359497 :=
  facts.2.2.1

theorem e8 :
    18908809 + 7452412 + 2 * 218506 + 445167 + 34691 + 7349838 + 28 =
      34627957 :=
  facts.2.2.2.1

theorem e16 :
    18908809 + 7552676 + 2 * 436834 + 445167 + 34691 + 7349838 + 28 =
      35164877 :=
  facts.2.2.2.2.1

theorem e1Exceeds25 : 2 ^ 25 < 34158302 :=
  facts.2.2.2.2.2.1

theorem e4Exceeds25 : 2 ^ 25 < 34359497 :=
  facts.2.2.2.2.2.2.1

theorem e8Exceeds25 : 2 ^ 25 < 34627957 :=
  facts.2.2.2.2.2.2.2.1

theorem e16Exceeds25 : 2 ^ 25 < 35164877 :=
  facts.2.2.2.2.2.2.2.2.1

theorem e1Fits26 : 34158302 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.1

theorem e4Fits26 : 34359497 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.2.1

theorem e8Fits26 : 34627957 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.2.2.1

theorem e16Fits26 : 35164877 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.2.2.2

end Nightstream.Implementation.NebulaV2.ProductionRecursiveCoreArithmetic
