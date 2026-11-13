import NightstreamFPrime.Export.Generic.PackedBoolean

set_option autoImplicit false

namespace NightstreamFPrime.Export.Generic.IntegerProduct

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra.Radix
open Fin.CommRing

def base : Nat := 4194304

theorem column_bounds (p0 p1 p2 addend carry digit next : Nat)
    (h0 : p0 < base * base) (h1 : p1 < base * base) (h2 : p2 < base * base)
    (ha : addend < base) (hc : carry < 16777216)
    (hd : digit < base) (hn : next < 16777216) :
    p0 + p1 + p2 + addend + carry < goldilocksModulus ∧
      digit + base * next < goldilocksModulus := by
  dsimp [base, goldilocksModulus] at *
  omega

theorem column_field_iff (left right : Nat)
    (hl : left < goldilocksModulus) (hr : right < goldilocksModulus) :
    fieldOfNat left = fieldOfNat right ↔ left = right := by
  constructor
  · intro equal
    have values := congrArg Fin.val equal
    simpa only [fieldOfNat, Nat.mod_eq_of_lt hl, Nat.mod_eq_of_lt hr] using values
  · exact congrArg fieldOfNat

theorem column_complete (total : Nat) :
    total = total % base + base * (total / base) ∧ total % base < base := by
  exact ⟨(Nat.mod_add_div total base).symm, Nat.mod_lt total (by decide)⟩

theorem radix3_product (b x0 x1 x2 y0 y1 y2 : Nat) :
    (x0 + b * x1 + b^2 * x2) * (y0 + b * y1 + b^2 * y2) =
      x0*y0 + b*(x0*y1 + x1*y0) + b^2*(x0*y2 + x1*y1 + x2*y0) +
        b^3*(x1*y2 + x2*y1) + b^4*(x2*y2) := by
  ring

theorem carry_reconstruct
    (t0 t1 t2 t3 t4 t5 d0 d1 d2 d3 d4 d5 c0 c1 c2 c3 c4 : Nat)
    (h0 : t0 = d0 + base*c0)
    (h1 : t1 + c0 = d1 + base*c1)
    (h2 : t2 + c1 = d2 + base*c2)
    (h3 : t3 + c2 = d3 + base*c3)
    (h4 : t4 + c3 = d4 + base*c4)
    (h5 : t5 + c4 = d5) :
    t0 + base*t1 + base^2*t2 + base^3*t3 + base^4*t4 + base^5*t5 =
      d0 + base*d1 + base^2*d2 + base^3*d3 + base^4*d4 + base^5*d5 := by
  dsimp [base] at *
  omega

theorem truncated_high_zero (x1 x2 y1 y2 : Nat)
    (h1 : x1*y2 = 0) (h2 : x2*(y1+y2) = 0) :
    x1*y2 + x2*y1 = 0 ∧ x2*y2 = 0 := by
  rw [Nat.mul_add] at h2
  omega

theorem division_roles (x y u v : Nat) (equation : x*y + u = v) (bound : u < y) :
    v / y = x ∧ v % y = u := by
  subst v
  constructor
  · simpa [Nat.mul_comm, Nat.add_comm, Nat.div_eq_of_lt bound] using Nat.add_mul_div_left u x (by omega : 0 < y)
  · simp [Nat.add_mod, Nat.mod_eq_of_lt bound]

end NightstreamFPrime.Export.Generic.IntegerProduct
