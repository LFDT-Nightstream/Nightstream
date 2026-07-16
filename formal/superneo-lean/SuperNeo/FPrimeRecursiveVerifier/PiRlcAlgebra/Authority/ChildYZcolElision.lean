import SuperNeo.FoldingProtocol.ConstraintSystem.CCS

/-!
Owns: a diagnostic no-read model for legacy Pi_DEC-child and next-running
`y_zcol` sidecars around one paper-level `CE.Statement`.

Does not own: Pi_RLC parent `y_zcol`, Pi_CCS output `y_zcol`, Pi_DEC validity,
or a proof that the production Rust/R1CS verifier factors through this
projection.

Emits constraints: no.

Authority boundary: this file proves only what follows after choosing a
predicate that ignores the sidecars. It does not prove that production may
erase the raw NC projection; the concrete laundering counterexample shows that
such erasure loses an implementation-level obligation.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `LegacyChildRunningCarrier` | child allocation and next-running allocation | Adds two legacy `y_zcol` sidecars to one paper `CE.Statement` | Sidecar type only | No |
| `eraseChildRunningYZcol` | semantic projection | Returns exactly the paper `CE.Statement` | None | No |
| `canonicalChildRunningExtension` | honest legacy encoding | Assigns the same supplied value to both sidecars | Supplied canonical value | No |
| `legacyChildRunningAccepts_projection` | semantic projection | Accepted legacy data implies core acceptance | Legacy pair equality is checked separately | No |
| `erase_child_running_yZcol_sound_complete` | logical projection | Core acceptance is equivalent to existence of an accepted legacy extension | Supplied canonical value | No; existential extension adds no authority |
| `factoredThroughCore_unchanged_by_sidecarMutation` | consumer no-read boundary | A predicate factored through projection is invariant under arbitrary sidecar mutation | Concrete consumer factorization | No; this diagnoses omission rather than justifying it |
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra

universe u v

open SuperNeo.ProofSystem.ConstraintSystem

/--
Legacy recursive carrier for one paper CE statement. The child and
next-running encodings repeat a `y_zcol` value that Definition 13 does not put
in `CE.Statement`.
-/
structure LegacyChildRunningCarrier
    (Commitment : Type u) (YZcol : Type v) where
  statement : CE.Statement Commitment
  childYZcol : YZcol
  nextRunningYZcol : YZcol

/-- Erase both legacy sidecars and retain the exact paper-level CE statement. -/
def eraseChildRunningYZcol
    {Commitment : Type u} {YZcol : Type v}
    (legacy : LegacyChildRunningCarrier Commitment YZcol) :
    CE.Statement Commitment :=
  legacy.statement

/-- Extend a paper statement with one canonical value in both legacy slots. -/
def canonicalChildRunningExtension
    {Commitment : Type u} {YZcol : Type v}
    (statement : CE.Statement Commitment) (yZcol : YZcol) :
    LegacyChildRunningCarrier Commitment YZcol :=
  { statement
    childYZcol := yZcol
    nextRunningYZcol := yZcol }

@[simp] theorem eraseChildRunningYZcol_canonicalExtension
    {Commitment : Type u} {YZcol : Type v}
    (statement : CE.Statement Commitment) (yZcol : YZcol) :
    eraseChildRunningYZcol
        (canonicalChildRunningExtension statement yZcol) = statement := by
  rfl

/--
Legacy acceptance keeps the paper predicate authoritative and separately checks
only that the child and next-running sidecars agree.
-/
def LegacyChildRunningAccepts
    {Commitment : Type u} {YZcol : Type v}
    (coreAccepts : CE.Statement Commitment → Prop)
    (legacy : LegacyChildRunningCarrier Commitment YZcol) : Prop :=
  coreAccepts (eraseChildRunningYZcol legacy) ∧
    legacy.childYZcol = legacy.nextRunningYZcol

/-- Projection soundness: erasing an accepted legacy carrier preserves acceptance. -/
theorem legacyChildRunningAccepts_projection
    {Commitment : Type u} {YZcol : Type v}
    {coreAccepts : CE.Statement Commitment → Prop}
    {legacy : LegacyChildRunningCarrier Commitment YZcol}
    (hLegacy : LegacyChildRunningAccepts coreAccepts legacy) :
    coreAccepts (eraseChildRunningYZcol legacy) :=
  hLegacy.1

/-- Canonical-extension completeness, stated as an exact acceptance equivalence. -/
@[simp] theorem canonicalChildRunningExtension_accepts_iff
    {Commitment : Type u} {YZcol : Type v}
    (coreAccepts : CE.Statement Commitment → Prop)
    (statement : CE.Statement Commitment) (yZcol : YZcol) :
    LegacyChildRunningAccepts coreAccepts
        (canonicalChildRunningExtension statement yZcol) ↔
      coreAccepts statement := by
  constructor
  · intro hLegacy
    exact hLegacy.1
  · intro hCore
    exact ⟨hCore, rfl⟩

/--
At the paper projection only, an accepted core statement is exactly the
projection of some accepted legacy carrier. This existential extension is a
logical fact about a predicate that does not read `y_zcol`; it is not protocol
soundness or permission to omit the production NC authority check. The
explicit `canonicalYZcol` avoids assuming that the sidecar type is inhabited.
-/
theorem erase_child_running_yZcol_sound_complete
    {Commitment : Type u} {YZcol : Type v}
    (coreAccepts : CE.Statement Commitment → Prop)
    (statement : CE.Statement Commitment)
    (canonicalYZcol : YZcol) :
    (∃ legacy : LegacyChildRunningCarrier Commitment YZcol,
        eraseChildRunningYZcol legacy = statement ∧
          LegacyChildRunningAccepts coreAccepts legacy) ↔
      coreAccepts statement := by
  constructor
  · rintro ⟨legacy, hProjection, hLegacy⟩
    rw [← hProjection]
    exact legacyChildRunningAccepts_projection hLegacy
  · intro hCore
    refine ⟨canonicalChildRunningExtension statement canonicalYZcol, rfl, ?_⟩
    exact (canonicalChildRunningExtension_accepts_iff
      coreAccepts statement canonicalYZcol).2 hCore

/-- A legacy predicate factors through the paper carrier when it reads no sidecar. -/
def FactorsThroughCore
    {Commitment : Type u} {YZcol : Type v}
    (legacyPredicate : LegacyChildRunningCarrier Commitment YZcol → Prop)
    (corePredicate : CE.Statement Commitment → Prop) : Prop :=
  ∀ legacy, legacyPredicate legacy ↔
    corePredicate (eraseChildRunningYZcol legacy)

/-- Predicates factored through the core cannot distinguish equal projections. -/
theorem factoredThroughCore_extensional
    {Commitment : Type u} {YZcol : Type v}
    {legacyPredicate : LegacyChildRunningCarrier Commitment YZcol → Prop}
    {corePredicate : CE.Statement Commitment → Prop}
    (hFactors : FactorsThroughCore legacyPredicate corePredicate)
    {left right : LegacyChildRunningCarrier Commitment YZcol}
    (hProjection : eraseChildRunningYZcol left = eraseChildRunningYZcol right) :
    legacyPredicate left ↔ legacyPredicate right := by
  constructor
  · intro hLeft
    have hCoreLeft := (hFactors left).1 hLeft
    have hCoreRight : corePredicate (eraseChildRunningYZcol right) := by
      rw [← hProjection]
      exact hCoreLeft
    exact (hFactors right).2 hCoreRight
  · intro hRight
    have hCoreRight := (hFactors right).1 hRight
    have hCoreLeft : corePredicate (eraseChildRunningYZcol left) := by
      rw [hProjection]
      exact hCoreRight
    exact (hFactors left).2 hCoreLeft

/-- Replace both sidecars while preserving the paper statement. -/
def withChildRunningYZcol
    {Commitment : Type u} {YZcol : Type v}
    (legacy : LegacyChildRunningCarrier Commitment YZcol)
    (childYZcol nextRunningYZcol : YZcol) :
    LegacyChildRunningCarrier Commitment YZcol :=
  { legacy with childYZcol, nextRunningYZcol }

@[simp] theorem eraseChildRunningYZcol_withSidecars
    {Commitment : Type u} {YZcol : Type v}
    (legacy : LegacyChildRunningCarrier Commitment YZcol)
    (childYZcol nextRunningYZcol : YZcol) :
    eraseChildRunningYZcol
        (withChildRunningYZcol legacy childYZcol nextRunningYZcol) =
      eraseChildRunningYZcol legacy := by
  rfl

/--
Generic no-read theorem: arbitrary mutation of either legacy sidecar leaves
every predicate factored through the paper projection unchanged.
-/
theorem factoredThroughCore_unchanged_by_sidecarMutation
    {Commitment : Type u} {YZcol : Type v}
    {legacyPredicate : LegacyChildRunningCarrier Commitment YZcol → Prop}
    {corePredicate : CE.Statement Commitment → Prop}
    (hFactors : FactorsThroughCore legacyPredicate corePredicate)
    (legacy : LegacyChildRunningCarrier Commitment YZcol)
    (childYZcol nextRunningYZcol : YZcol) :
    legacyPredicate
        (withChildRunningYZcol legacy childYZcol nextRunningYZcol) ↔
      legacyPredicate legacy :=
  factoredThroughCore_extensional hFactors
    (eraseChildRunningYZcol_withSidecars
      legacy childYZcol nextRunningYZcol)

end SuperNeo.FPrimeRecursiveVerifier.PiRlcAlgebra
