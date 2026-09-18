import TwistShout.OneHotEncoding

open scoped BigOperators
open TwistShout

namespace tests

def sampleDigits : AddressDigits 2 3 :=
  Fin.cases 1 (fun _ => 2)

def otherDigits : AddressDigits 2 3 :=
  Fin.cases 0 (fun _ => 2)

example :
    ∑ k : Fin 4, oneHot (K := Rat) (1 : Fin 4) k = 1 := by
  exact sum_oneHot (K := Rat) (1 : Fin 4)

example :
    productEncoding (K := Rat) (dOneHot (K := Rat) sampleDigits) sampleDigits = 1 := by
  exact productEncoding_eq_one_of_eq (K := Rat) rfl

example :
    productEncoding (K := Rat) (dOneHot (K := Rat) sampleDigits) otherDigits = 0 := by
  exact productEncoding_eq_zero_of_ne (K := Rat) (by decide)

example (i : Fin 2) :
    ∑ k : Fin 3, dOneHot (K := Rat) sampleDigits i k = 1 := by
  exact dOneHot_sum (K := Rat) sampleDigits i

example :
    IsDOneHotEncoding (K := Rat) (dOneHot (K := Rat) sampleDigits) := by
  exact ⟨sampleDigits, rfl⟩

example
    {v : Fin 2 → Fin 3 → Rat}
    (hv : IsDOneHotEncoding (K := Rat) v)
    (hprod : ∀ k, productEncoding (K := Rat) v k = tupleOneHot (K := Rat) sampleDigits k) :
    v = dOneHot (K := Rat) sampleDigits := by
  exact dOneHot_unique (K := Rat) hv hprod

#guard (∑ k : Fin 3, oneHot (K := Rat) (2 : Fin 3) k) = 1

end tests
