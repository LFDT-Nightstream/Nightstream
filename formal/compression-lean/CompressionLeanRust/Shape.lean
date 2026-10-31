import NeoVerified.Types
import CompressionLean.Ports.Shrink.BlockPart

namespace CompressionLeanRust

open Aeneas.Std

def Sound {α β γ : Type} (f : α → Result β) (view : β → γ) (copy : α → γ) : Prop :=
  ∀ x y, f x = .ok y → view y = copy x

/-- Without the witness, `H` could be false and the statement empty. Without `Complete`, a
translation that always fails is `Sound`. -/
def Complete {α β γ : Type} (H : α → Prop) (f : α → Result β) (view : β → γ) (copy : α → γ) :
    Prop :=
  (∃ x, H x) ∧ ∀ x, H x → ∃ y, f x = .ok y ∧ view y = copy x

def usizeValue (x : Usize) : ℕ := x.val

def usizeValues {n : Usize} (a : Aeneas.Std.Array Usize n) : List ℕ := a.val.map (·.val)

theorem usizeValue_injective : Function.Injective usizeValue :=
  fun a b h => UScalar.val_eq_imp a b h

theorem usizeValues_injective {n : Usize} : Function.Injective (@usizeValues n) := by
  intro a b h
  exact Aeneas.Std.Array.ext a b ((List.map_injective_iff.mpr usizeValue_injective)
    (show List.map usizeValue a.val = List.map usizeValue b.val from h))

/-- A translated part as the copy's part. -/
def partValue (p : neo_verified.layout.Part) : CompressionLean.Shrink.Part := ⟨p.start.val, p.stride.val⟩

/-- The parts and their end, as `place` returns them. -/
def placeValue (r : Aeneas.Std.Array neo_verified.layout.Part 2#usize × Usize) :
    List CompressionLean.Shrink.Part × ℕ :=
  (r.1.val.map partValue, r.2.val)

theorem partValue_injective : Function.Injective partValue := by
  rintro ⟨a, b⟩ ⟨c, d⟩ h
  simp only [partValue, CompressionLean.Shrink.Part.mk.injEq] at h
  rw [usizeValue_injective h.1, usizeValue_injective h.2]

theorem placeValue_injective : Function.Injective placeValue := by
  rintro ⟨a, x⟩ ⟨b, y⟩ h
  simp only [placeValue, Prod.mk.injEq] at h
  rw [Aeneas.Std.Array.ext a b ((List.map_injective_iff.mpr partValue_injective) h.1),
    usizeValue_injective h.2]

/-- The views that a `Sound` or `Complete` theorem may use. `CompressionLeanRustTests` checks
that each has a theorem `<view>_injective`. -/
def views : List Lean.Name := [``usizeValue, ``usizeValues, ``placeValue]

end CompressionLeanRust
