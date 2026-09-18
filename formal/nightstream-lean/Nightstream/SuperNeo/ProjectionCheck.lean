/-!
Contract: algebraic soundness boundary for transcript-chosen projection checks.

Production PiRLC replaces full coefficient-wise ring-action materialization
with polynomial identities checked at one challenge `beta`. This module states
the exact deterministic guarantee of that optimization: an accepted bounded
identity is either coefficient-wise exact or the challenge is a root of a
nonzero error polynomial.

The operations are explicit so this theorem does not assume field laws or a
random-oracle model. Probability bounds and Fiat-Shamir unpredictability are
separate M6 obligations. Fixed-width coefficient lists match the production
arrays and prevent trailing-zero representation aliases from being mislabeled
as different polynomials.
-/

namespace Nightstream.SuperNeo.ProjectionCheck

universe uScalar

structure Ops (Scalar : Type uScalar) where
  zero : Scalar
  add : Scalar → Scalar → Scalar
  mul : Scalar → Scalar → Scalar

/-- Constant-first polynomial evaluation by Horner's rule. -/
def eval {Scalar : Type uScalar} (ops : Ops Scalar)
    (coefficients : List Scalar) (point : Scalar) : Scalar :=
  coefficients.foldr (fun coefficient suffix =>
    ops.add coefficient (ops.mul point suffix)) ops.zero

structure Identity (Scalar : Type uScalar) where
  lhs : List Scalar
  rhs : List Scalar
  beta : Scalar
  maxDegree : Nat
deriving DecidableEq, Repr

/-- Production arrays have one shared fixed width and fit the advertised
degree bound. Zero padding is part of the representation. -/
def Identity.WellFormed {Scalar : Type uScalar}
    (identity : Identity Scalar) : Prop :=
  identity.lhs.length = identity.rhs.length ∧
  identity.lhs.length ≤ identity.maxDegree + 1

instance {Scalar : Type uScalar} (identity : Identity Scalar) :
    Decidable identity.WellFormed := by
  unfold Identity.WellFormed
  infer_instance

def Identity.Exact {Scalar : Type uScalar} (identity : Identity Scalar) : Prop :=
  identity.lhs = identity.rhs

instance {Scalar : Type uScalar} [DecidableEq Scalar]
    (identity : Identity Scalar) : Decidable identity.Exact := by
  unfold Identity.Exact
  infer_instance

def Accepted {Scalar : Type uScalar} (ops : Ops Scalar)
    (identity : Identity Scalar) : Prop :=
  identity.WellFormed ∧
  eval ops identity.lhs identity.beta = eval ops identity.rhs identity.beta

instance {Scalar : Type uScalar} [DecidableEq Scalar]
    (ops : Ops Scalar) (identity : Identity Scalar) :
    Decidable (Accepted ops identity) := by
  unfold Accepted
  infer_instance

/-- The precise one-point soundness failure: distinct fixed-width coefficient
vectors collide at the sampled challenge. -/
structure BadRoot {Scalar : Type uScalar} (ops : Ops Scalar)
    (identity : Identity Scalar) : Prop where
  wellFormed : identity.WellFormed
  notExact : ¬ identity.Exact
  collision :
    eval ops identity.lhs identity.beta = eval ops identity.rhs identity.beta

def BatchAccepted {Scalar : Type uScalar} (ops : Ops Scalar)
    (identities : List (Identity Scalar)) : Prop :=
  ∀ identity ∈ identities, Accepted ops identity

def BatchExact {Scalar : Type uScalar}
    (identities : List (Identity Scalar)) : Prop :=
  ∀ identity ∈ identities, identity.Exact

def BatchBadRoot {Scalar : Type uScalar} (ops : Ops Scalar)
    (identities : List (Identity Scalar)) : Prop :=
  ∃ identity ∈ identities, BadRoot ops identity

/-- One accepted projection identity is exact or exposes the named bad root.
There is deliberately no premise that directly asserts the conclusion. -/
theorem accepted_implies_exact_or_badRoot
    {Scalar : Type uScalar} [DecidableEq Scalar]
    (ops : Ops Scalar) (identity : Identity Scalar)
    (accepted : Accepted ops identity) :
    identity.Exact ∨ BadRoot ops identity := by
  by_cases exact : identity.Exact
  · exact Or.inl exact
  · exact Or.inr ⟨accepted.1, exact, accepted.2⟩

/-- Batch form used by PiRLC: every identity is exact, or at least one
transcript challenge is a root of a nonzero bounded error polynomial. -/
theorem batchAccepted_implies_exact_or_badRoot
    {Scalar : Type uScalar} [DecidableEq Scalar]
    (ops : Ops Scalar) (identities : List (Identity Scalar))
    (accepted : BatchAccepted ops identities) :
    BatchExact identities ∨ BatchBadRoot ops identities := by
  induction identities with
  | nil =>
      left
      intro identity member
      simp at member
  | cons head tail inductionHypothesis =>
      have headAccepted : Accepted ops head := accepted head (by simp)
      have tailAccepted : BatchAccepted ops tail := by
        intro identity member
        exact accepted identity (by simp [member])
      rcases accepted_implies_exact_or_badRoot ops head headAccepted with
        headExact | headBad
      · rcases inductionHypothesis tailAccepted with tailExact | tailBad
        · left
          intro identity member
          simp only [List.mem_cons] at member
          rcases member with rfl | member
          · exact headExact
          · exact tailExact identity member
        · right
          rcases tailBad with ⟨identity, member, bad⟩
          exact ⟨identity, by simp [member], bad⟩
      · right
        exact ⟨head, by simp, headBad⟩

theorem exact_is_accepted
    {Scalar : Type uScalar} (ops : Ops Scalar)
    (identity : Identity Scalar) (wellFormed : identity.WellFormed)
    (exact : identity.Exact) : Accepted ops identity := by
  refine ⟨wellFormed, ?_⟩
  rw [exact]

theorem badRoot_is_accepted
    {Scalar : Type uScalar} (ops : Ops Scalar)
    (identity : Identity Scalar) (bad : BadRoot ops identity) :
    Accepted ops identity :=
  ⟨bad.wellFormed, bad.collision⟩

end Nightstream.SuperNeo.ProjectionCheck
