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

/-- `Sound` and `Complete` from the two halves that each proof shows. -/
theorem complete_of {α β γ : Type} {H : α → Prop} {f : α → Result β} {view : β → γ} {copy : α → γ}
    (sound : Sound f view copy) (witness : ∃ x, H x) (succeeds : ∀ x, H x → ∃ y, f x = .ok y) :
    Complete H f view copy :=
  ⟨witness, fun x hx => let ⟨y, hy⟩ := succeeds x hx; ⟨y, hy, sound x y hy⟩⟩

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

/-- A translated `Gl` as its word, canonical or not. -/
def glValue (x : neo_verified.field.Gl) : ℕ := x.val

/-- A translated `Ext` as its three words. -/
def extValue (x : neo_verified.field.Ext) : ℕ × ℕ × ℕ := (x.c0.val, x.c1.val, x.c2.val)

def optionValue (x : Option neo_verified.field.Gl) : Option ℕ := x.map glValue

theorem glValue_injective : Function.Injective glValue :=
  fun a b h => UScalar.val_eq_imp a b h

theorem extValue_injective : Function.Injective extValue := by
  rintro ⟨a0, a1, a2⟩ ⟨b0, b1, b2⟩ h
  simp only [extValue, Prod.mk.injEq] at h
  rw [glValue_injective h.1, glValue_injective h.2.1, glValue_injective h.2.2]

theorem optionValue_injective : Function.Injective optionValue :=
  Option.map_injective glValue_injective

/-- A translated array of `Gl` as its words, canonical or not. -/
def glValues {n : Usize} (a : Aeneas.Std.Array neo_verified.field.Gl n) : List ℕ := a.val.map glValue

theorem glValues_injective {n : Usize} : Function.Injective (@glValues n) := by
  intro a b h
  exact Aeneas.Std.Array.ext a b ((List.map_injective_iff.mpr glValue_injective)
    (show List.map glValue a.val = List.map glValue b.val from h))

/-- The views that a `Sound` or `Complete` theorem may use. `CompressionLeanRustTests` checks
that each has a theorem `<view>_injective`. -/
def views : List Lean.Name :=
  [``usizeValue, ``usizeValues, ``placeValue, ``glValue, ``extValue, ``optionValue, ``glValues,
    `CompressionLeanRust.built, `CompressionLeanRust.builtVec, `CompressionLeanRust.builtKx, `CompressionLeanRust.builtDigest,
    `CompressionLeanRust.builtNew, `CompressionLeanRust.builtStep, `CompressionLeanRust.builtSample,
    `CompressionLeanRust.builtBits, `CompressionLeanRust.nativeValues, `CompressionLeanRust.nativeExt,
    `CompressionLeanRust.nativeExts, `CompressionLeanRust.extUsizeValue, `CompressionLeanRust.statesValue,
    `CompressionLeanRust.nativeChecked, `CompressionLeanRust.builtChecked]

end CompressionLeanRust
