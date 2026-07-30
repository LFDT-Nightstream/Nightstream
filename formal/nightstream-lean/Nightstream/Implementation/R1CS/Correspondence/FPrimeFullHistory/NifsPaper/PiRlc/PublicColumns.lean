/-!
Contract: artifact-free public-role and coefficient-column vocabulary for the
fixed F' PiRLC batch.

Assurance tier: model-level.

Owns: logical ordering of the public PiRLC roles and the typed family of
coefficient-column lists selected for each role.

Does not own: generated artifacts, row manifests, Rust layouts, transcript
derivation, projection-trace encodings, semantic validity, or constraint
costs.

Emits constraints: no.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_rlc.public.roles` | commitment, public-input, and matrix-evaluation roles occur in paper order | computed | `publicOrder` |
| `nifs.pi_rlc.public.count` | the public role count is `23 + 2 * matrixCount` | model-proved | `public_role_count` |
| `nifs.pi_rlc.public.columns` | every public role selects one coefficient-column list | typed input | `ProjectionColumns.at` |
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc

/-- Matrix-indexed public leaves in paper review order. -/
inductive PublicRole (matrixCount : Nat) where
  | commitment (lane : Fin 18)
  | x (column : Fin 5)
  | yRing (row : Fin matrixCount) (limb : Fin 2)
deriving DecidableEq, Repr

def publicOrder (matrixCount : Nat) : List (PublicRole matrixCount) :=
  (List.ofFn fun lane : Fin 18 => .commitment lane) ++
  (List.ofFn fun column : Fin 5 => .x column) ++
  (List.ofFn fun row : Fin matrixCount =>
    List.ofFn fun limb : Fin 2 => .yRing row limb).flatten

private theorem sum_ofFn_constant (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count induction =>
      rw [List.ofFn_succ]
      rw [List.sum_cons, induction, Nat.succ_mul]
      omega

theorem public_role_count (matrixCount : Nat) :
    (publicOrder matrixCount).length = 23 + 2 * matrixCount := by
  simp only [publicOrder, List.length_append, List.length_ofFn,
    List.length_flatten, List.map_ofFn]
  change 18 + 5 + (List.ofFn fun _ : Fin matrixCount => 2).sum =
    23 + 2 * matrixCount
  rw [sum_ofFn_constant]
  omega

/-- Coefficient columns for every public PiRLC leaf. -/
structure ProjectionColumns (matrixCount : Nat) where
  commitment : Fin 18 → List Nat
  x : Fin 5 → List Nat
  yRing : Fin matrixCount → Fin 2 → List Nat

def ProjectionColumns.at {matrixCount : Nat}
    (columns : ProjectionColumns matrixCount) :
    PublicRole matrixCount → List Nat
  | .commitment lane => columns.commitment lane
  | .x column => columns.x column
  | .yRing row limb => columns.yRing row limb

end Nightstream.Implementation.R1CS.FPrimeFullHistoryNifsPaper.PiRlc
