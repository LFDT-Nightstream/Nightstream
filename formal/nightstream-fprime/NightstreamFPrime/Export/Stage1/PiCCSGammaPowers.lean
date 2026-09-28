import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.TargetPolynomial

/-! A shared gamma-power table. Construction performs one multiplication and
one array push per step. Missing entries use the original exponentiation.
The lookup proof needs no algebra laws: multiplication order is unchanged. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSGammaPowers

open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField
variable {Field : Type uField}

private def build (ops : Ops Field) (gamma : Field) (count : Nat) : Array Field × Field :=
  Nat.fold count (fun _ _ state =>
    (state.1.push state.2, ops.mul gamma state.2)) (#[], ops.one)

private theorem build_succ (ops : Ops Field) (gamma : Field) (count : Nat) :
    build ops gamma (count + 1) =
      ((build ops gamma count).1.push (build ops gamma count).2,
        ops.mul gamma (build ops gamma count).2) := by
  simp only [build, Nat.fold_succ]

private theorem build_size (ops : Ops Field) (gamma : Field) (count : Nat) :
    (build ops gamma count).1.size = count := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [build_succ, Array.size_push, ih]

private theorem build_power (ops : Ops Field) (gamma : Field) (count : Nat) :
    (build ops gamma count).2 = TargetPolynomial.power ops gamma count := by
  induction count with
  | zero => rfl
  | succ count ih => simp only [build_succ, TargetPolynomial.power, ih]

private theorem build_get (ops : Ops Field) (gamma : Field) (count exponent : Nat)
    (inside : exponent < count) :
    (build ops gamma count).1[exponent]? = some (TargetPolynomial.power ops gamma exponent) := by
  induction count generalizing exponent with
  | zero => omega
  | succ count ih =>
      rw [build_succ]
      simp only [Array.getElem?_push, build_size]
      by_cases equal : exponent = count
      · subst exponent
        rw [if_pos rfl, build_power]
      · rw [if_neg equal]
        exact ih exponent (by omega)

/-- Store gamma^0 through gamma^(count-1). Call once and share the result. -/
def prepare (ops : Ops Field) (gamma : Field) (count : Nat) : Array Field :=
  (build ops gamma count).1

theorem prepare_size (ops : Ops Field) (gamma : Field) (count : Nat) :
    (prepare ops gamma count).size = count := build_size ops gamma count

theorem prepare_get (ops : Ops Field) (gamma : Field) (count exponent : Nat)
    (inside : exponent < count) :
    (prepare ops gamma count)[exponent]? = some (TargetPolynomial.power ops gamma exponent) :=
  build_get ops gamma count exponent inside

/-- The match delays fallback evaluation until a requested entry is absent. -/
def lookup (ops : Ops Field) (gamma : Field) (table : Array Field) (exponent : Nat) : Field :=
  match table[exponent]? with
  | some value => value
  | none => TargetPolynomial.power ops gamma exponent

/-- Any prepared extent is correct for every exponent, including fallback.
A chosen cache extent is not an arithmetic or security premise. -/
theorem lookup_prepare (ops : Ops Field) (gamma : Field) (count exponent : Nat) :
    lookup ops gamma (prepare ops gamma count) exponent = TargetPolynomial.power ops gamma exponent := by
  unfold lookup
  by_cases inside : exponent < count
  · rw [prepare_get ops gamma count exponent inside]
  · have outside : (prepare ops gamma count).size ≤ exponent := by
      rw [prepare_size]
      omega
    rw [Array.getElem?_eq_none outside]

end NightstreamFPrime.Export.Stage1.PiCCSGammaPowers
