import Mathlib.Tactic

/-!
Closed arithmetic certificates for the exponent-26 production recursive-core
census. The product total includes the complete NIFS output carrier.

This module is intentionally independent of protocol types. Large numeral
normalization occurs once here. Protocol modules consume the resulting
kernel-checked facts.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.ProductionRecursiveCoreArithmetic

theorem facts :
    12139373 + 1710720 + 216270 + 7290 + 4817340 + 5400 + 23812 =
        18920205 /\
      18920205 + 7364831 + 2 * 27469 + 445167 + 34691 + 7349838 + 28 =
        34169698 /\
      18920205 + 7402280 + 2 * 109342 + 445167 + 34691 + 7349838 + 28 =
        34370893 /\
      18920205 + 7452412 + 2 * 218506 + 445167 + 34691 + 7349838 + 28 =
        34639353 /\
      18920205 + 7552676 + 2 * 436834 + 445167 + 34691 + 7349838 + 28 =
        35176273 /\
      2 ^ 25 < 34169698 /\
      2 ^ 25 < 34370893 /\
      2 ^ 25 < 34639353 /\
      2 ^ 25 < 35176273 /\
      34169698 <= 2 ^ 26 /\
      34370893 <= 2 ^ 26 /\
      34639353 <= 2 ^ 26 /\
      35176273 <= 2 ^ 26 := by
  norm_num

theorem product :
    12139373 + 1710720 + 216270 + 7290 + 4817340 + 5400 + 23812 =
      18920205 :=
  facts.1

theorem e1 :
    18920205 + 7364831 + 2 * 27469 + 445167 + 34691 + 7349838 + 28 =
      34169698 :=
  facts.2.1

theorem e4 :
    18920205 + 7402280 + 2 * 109342 + 445167 + 34691 + 7349838 + 28 =
      34370893 :=
  facts.2.2.1

theorem e8 :
    18920205 + 7452412 + 2 * 218506 + 445167 + 34691 + 7349838 + 28 =
      34639353 :=
  facts.2.2.2.1

theorem e16 :
    18920205 + 7552676 + 2 * 436834 + 445167 + 34691 + 7349838 + 28 =
      35176273 :=
  facts.2.2.2.2.1

theorem e1Exceeds25 : 2 ^ 25 < 34169698 :=
  facts.2.2.2.2.2.1

theorem e4Exceeds25 : 2 ^ 25 < 34370893 :=
  facts.2.2.2.2.2.2.1

theorem e8Exceeds25 : 2 ^ 25 < 34639353 :=
  facts.2.2.2.2.2.2.2.1

theorem e16Exceeds25 : 2 ^ 25 < 35176273 :=
  facts.2.2.2.2.2.2.2.2.1

theorem e1Fits26 : 34169698 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.1

theorem e4Fits26 : 34370893 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.2.1

theorem e8Fits26 : 34639353 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.2.2.1

theorem e16Fits26 : 35176273 <= 2 ^ 26 :=
  facts.2.2.2.2.2.2.2.2.2.2.2.2

end Nightstream.Implementation.Nebula.ProductionRecursiveCoreArithmetic
