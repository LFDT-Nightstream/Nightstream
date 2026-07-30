import Nightstream.Implementation.R1CS.Core.Semantics

/-!
Contract: classify `validate_adv_recomposition` into the part that needs the
commitment layer and the part that does not.

Owns: the all-or-nothing presence rule, its three branches, and the proof that
two of them reach no commitment arithmetic at all.

Does not own: the Ajtai fold itself.  That is the third branch and it stays
outside this layer.

## The carried dependency was half right

`PIDEC-*` has said "`adv_recomposition`, needing the Ajtai commitment layer"
since cycle 326.  Reading `recompose_adv` shows the check has **three**
branches, and only one of them calls the mixer:

| branch | `recompose_adv` result | reaches the mixer |
|---|---|---|
| no child carries a sidecar | `Ok(None)` | no |
| some but not all carry one | `Err(AdvPresence)` | no |
| every child carries one | `Ok(Some(combine …))` | yes |

`require_homogeneous`'s own comment names the first branch: "`Ok(None)` means a
plain (non-Nebula) fold".

So in the plain profile the check degenerates to "the parent carries no sidecar
either" — a structural agreement, decidable from the decoded shape, with no
commitment arithmetic reached.  That part was never blocked.

## Consistent with the sidecar finding

`PIDEC-SIDECARS` established that `validate_supported_sidecars` rejects any
claim with a non-empty `aux_openings` or `c_step_coords`, which makes the fourth
recomposition carrier dead in the shipping path.  This is the same profile seen
from another side: the shipping Π_DEC fold is the plain one, and both the
`aux_openings` recomposition and the `adv` commitment fold are inert in it.

Two independent readings agreeing on which profile ships is worth more than
either alone, which is why this is recorded rather than folded into the earlier
entry.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary

/-- How many inputs carry a sidecar. -/
def presence {α : Type} (inputs : List (Option α)) : Nat :=
  (inputs.filter Option.isSome).length

/-- `recompose_adv`'s control flow.  The outer `Option` is the `Result`: `none`
is `Err(AdvPresence)`, `some none` is the plain fold, `some (some _)` is the
mixed commitment. -/
def recomposeAdv {α β : Type} (combine : List α → β)
    (children : List (Option α)) : Option (Option β) :=
  if presence children = 0 then some none
  else if presence children ≠ children.length then none
  else some (some (combine (children.filterMap id)))

/-! ## The three branches -/

/-- **Nothing present: the plain fold.**  No mixer call. -/
theorem recomposeAdv_absent {α β : Type} (combine : List α → β)
    (children : List (Option α)) (absent : presence children = 0) :
    recomposeAdv combine children = some none := by
  unfold recomposeAdv
  rw [if_pos absent]

/-- **Mixed presence is rejected**, and again without reaching the mixer.  This
is the all-or-nothing rule: a fold may not carry a sidecar on some inputs
only. -/
theorem recomposeAdv_mixed {α β : Type} (combine : List α → β)
    (children : List (Option α)) (somePresent : presence children ≠ 0)
    (notAll : presence children ≠ children.length) :
    recomposeAdv combine children = none := by
  unfold recomposeAdv
  rw [if_neg somePresent, if_pos notAll]

/-- **All present: the only branch that calls the mixer.** -/
theorem recomposeAdv_present {α β : Type} (combine : List α → β)
    (children : List (Option α)) (somePresent : presence children ≠ 0)
    (all : presence children = children.length) :
    recomposeAdv combine children
      = some (some (combine (children.filterMap id))) := by
  unfold recomposeAdv
  rw [if_neg somePresent, if_neg (by simp only [all, ne_eq, not_true_eq_false,
    not_false_eq_true])]

/-! ## The plain profile degenerates

`validate_adv_recomposition` compares the recomposition against `parent.adv`.
In the plain profile that comparison is between two absences. -/

/-- **In the plain profile the check is a structural agreement.**

No commitment arithmetic is reached: the recomposition is `none`, so the check
succeeds exactly when the parent carries no sidecar either.  This is the part of
`adv_recomposition` that was never blocked on the Ajtai layer. -/
theorem plain_profile_is_structural {α β : Type} (combine : List α → β)
    (children : List (Option α)) (parent : Option β)
    (absent : presence children = 0) :
    (recomposeAdv combine children = some parent) ↔ parent = none := by
  rw [recomposeAdv_absent combine children absent]
  constructor
  · intro equal
    exact (Option.some_inj.1 equal).symm
  · intro isNone
    rw [isNone]

/-- **Absence is decidable from the decoded shape**, which is why the plain
branch needs no rows: nothing about it is a field constraint. -/
theorem presence_zero_iff {α : Type} (children : List (Option α)) :
    presence children = 0 ↔ ∀ child ∈ children, child = none := by
  unfold presence
  rw [List.length_eq_zero_iff, List.filter_eq_nil_iff]
  constructor
  · intro noneSome child member
    cases child with
    | none => rfl
    | some value => exact absurd rfl (noneSome (some value) member)
  · intro allNone child member
    rw [allNone child member]
    simp

/-- One absent child already breaks the all-present condition. -/
theorem presence_lt_of_absent {α : Type} (children : List (Option α))
    (member : (none : Option α) ∈ children) :
    presence children < children.length := by
  induction children with
  | nil => exact absurd member (by simp)
  | cons head tail inductionHypothesis =>
      have tailBound : presence tail ≤ tail.length := by
        unfold presence
        exact List.length_filter_le _ _
      have headStep : presence (head :: tail) ≤ presence tail + 1 := by
        unfold presence
        cases head <;> simp
      rcases List.mem_cons.1 member with rfl | inTail
      · have dropped : presence (none :: tail) = presence tail := by
          unfold presence
          simp
        simp only [List.length_cons, dropped]
        omega
      · have := inductionHypothesis inTail
        simp only [List.length_cons]
        omega

theorem presence_ne_length_of_absent {α : Type} (children : List (Option α))
    (member : (none : Option α) ∈ children) :
    presence children ≠ children.length :=
  Nat.ne_of_lt (presence_lt_of_absent children member)

/-- **The mixer is reached only when every child carries a sidecar.**

Stated as the contrapositive a reader actually wants: if any child is absent and
the presence count is not zero, the fold is rejected before any commitment
arithmetic happens. -/
theorem mixer_unreached_unless_all_present {α β : Type} (combine : List α → β)
    (children : List (Option α)) (child : Option α) (member : child ∈ children)
    (childAbsent : child = none) (somePresent : presence children ≠ 0) :
    recomposeAdv combine children = none :=
  recomposeAdv_mixed combine children somePresent
    (presence_ne_length_of_absent children (childAbsent ▸ member))

end Nightstream.Implementation.R1CS.Canonical.AdvPresenceBoundary
