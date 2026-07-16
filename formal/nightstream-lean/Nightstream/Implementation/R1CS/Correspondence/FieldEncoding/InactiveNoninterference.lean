import Nightstream.Implementation.R1CS.Correspondence.FieldEncoding.LayoutManifest

/-!
Contract: model-level noninterference for branch-relative inactive encoded
coordinates in a selector-composed relation.

Owns: declared read supports for the selector, selected equations, and
authority-visible semantic output; change confinement; the acceptance
soundness/completeness theorem; and authority-output invariance.

Does not own: a generated base/recursive support census, the production fixed
selector materializer, outer-norm refinement, CE/Ajtai commitment
recomputation, or authorization to remove any Rust row.

Emits constraints: no.

Authority boundary: an inactive-coordinate change may alter the exact CE
commitment. The theorem preserves only the declared authority-visible output;
the caller must recompute the commitment from the changed exact assignment and
must independently prove every always-on encoding and norm obligation for both
assignments.

Assurance tier: model-level. Concrete use requires a Rust-generated support
manifest and exact row/matrix refinement.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `ConfinedTo` | fixed selector assignment | Assignments differ only inside one declared inactive support | exact support census | no |
| `selectorComposed_sound` | selected branch rows | Left acceptance implies right acceptance | confinement, selector/selected disjointness, right always-on obligations | only after concrete refinement |
| `selectorComposed_complete` | selected branch rows | Right acceptance implies left acceptance | confinement, selector/selected disjointness, left always-on obligations | only after concrete refinement |
| `selectorComposed_acceptance_iff` | selected branch rows | Acceptance is invariant in both directions | both always-on obligations and the soundness/completeness premises | only after concrete refinement |
| `authorityOutput_invariant` | public/semantic output boundary | Every declared authority-visible semantic output is unchanged | confinement and authority-support disjointness | no commitment-reuse authority |
| `inactiveNoninterference` | complete fixed selector boundary | Selector, acceptance, and authority-visible output are jointly invariant | all preceding premises | only after concrete refinement |
-/

namespace Nightstream.Implementation.R1CS.InactiveFieldNoninterference

/-- A low-norm assignment indexed by the concrete R1CS/CCS column number. -/
abbrev Assignment (Value : Type) := Nat → Value

/-- Column support as a membership predicate. A concrete generated artifact
should define this predicate from compact, exact, nonoverlapping coordinate
runs rather than materializing a million-column list in Lean. -/
abbrev Support := Nat → Prop

/-- Convenience bridge for small regressions and generated exceptional-column
lists. Large production supports should use compact run membership. -/
def Support.ofList (columns : List Nat) : Support :=
  fun column => column ∈ columns

/-- Two assignments agree on every declared member of `support`. -/
def AgreesOn {Value : Type} (support : Support)
    (left right : Assignment Value) : Prop :=
  ∀ column, support column → left column = right column

/-- Every assignment change is confined to `inactive`; columns outside it are
identical. This is stronger and more reviewable than merely asserting that a
particular observer returned the same value. -/
def ConfinedTo {Value : Type} (inactive : Support)
    (left right : Assignment Value) : Prop :=
  ∀ column, ¬ inactive column → left column = right column

/-- A proposition may read only the listed columns. -/
def PredicateReadsOnly {Value : Type} (support : Support)
    (predicate : Assignment Value → Prop) : Prop :=
  ∀ left right, AgreesOn support left right →
    (predicate left ↔ predicate right)

/-- A value-producing observer may read only the listed columns. -/
def ValueReadsOnly {Value Result : Type} (support : Support)
    (observer : Assignment Value → Result) : Prop :=
  ∀ left right, AgreesOn support left right →
    observer left = observer right

/-- Generator-facing support schema. The inactive set is branch-relative;
the selector support is necessarily branch-independent because it determines
which branch is active. -/
structure SupportManifest (Branch : Type) where
  selector : Support
  inactive : Branch → Support
  selectedEquations : Branch → Support
  authorityOutput : Branch → Support

/-- Explicit disjointness used by generated support manifests. The generic
theorem does not assert that a generated run census is exhaustive or
nonoverlapping; those artifact facts remain separate refinement obligations. -/
def SupportsDisjoint (left right : Support) : Prop :=
  ∀ column, left column → right column → False

/-- Semantic boundary consumed by the generic theorem. Read-support proofs are
part of the boundary; disjointness is deliberately not, because it must be
checked separately for each generated branch manifest. -/
structure Boundary (Value Branch Output : Type) where
  supports : SupportManifest Branch
  selector : Assignment Value → Branch
  alwaysOn : Assignment Value → Prop
  selectedEquations : Branch → Assignment Value → Prop
  authorityOutput : Branch → Assignment Value → Output
  selectorReads : ValueReadsOnly supports.selector selector
  selectedEquationsRead : ∀ branch,
    PredicateReadsOnly (supports.selectedEquations branch)
      (selectedEquations branch)
  authorityOutputReads : ∀ branch,
    ValueReadsOnly (supports.authorityOutput branch)
      (authorityOutput branch)

/-- Selector-composed acceptance: always-on encoding/norm obligations are
conjoined with exactly the equations chosen by the constrained selector. -/
def Boundary.Accepts {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    (assignment : Assignment Value) : Prop :=
  boundary.alwaysOn assignment ∧
    boundary.selectedEquations (boundary.selector assignment) assignment

/-- Authority-visible semantic output of the selected branch. This explicitly
excludes the exact CE commitment, which must be recomputed after a witness
change. -/
def Boundary.Output {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    (assignment : Assignment Value) : Output :=
  boundary.authorityOutput (boundary.selector assignment) assignment

theorem agreesOn_of_confinedTo_of_disjoint
    {Value : Type} {inactive support : Support}
    {left right : Assignment Value}
    (confined : ConfinedTo inactive left right)
    (disjoint : SupportsDisjoint inactive support) :
    AgreesOn support left right := by
  intro column columnInSupport
  apply confined column
  intro columnInInactive
  exact disjoint column columnInInactive columnInSupport

theorem selector_invariant
    {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    {inactive : Support} {left right : Assignment Value}
    (confined : ConfinedTo inactive left right)
    (selectorDisjoint : SupportsDisjoint inactive boundary.supports.selector) :
    boundary.selector left = boundary.selector right := by
  apply boundary.selectorReads left right
  exact agreesOn_of_confinedTo_of_disjoint confined selectorDisjoint

private theorem selectedEquations_invariant
    {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    {inactive : Support} {left right : Assignment Value}
    (confined : ConfinedTo inactive left right)
    (selectedDisjoint :
      SupportsDisjoint inactive
        (boundary.supports.selectedEquations (boundary.selector left))) :
    boundary.selectedEquations (boundary.selector left) left ↔
      boundary.selectedEquations (boundary.selector left) right := by
  apply boundary.selectedEquationsRead (boundary.selector left) left right
  exact agreesOn_of_confinedTo_of_disjoint confined selectedDisjoint

/-- Soundness direction for changing only inactive coordinates. The new
assignment must still satisfy all always-on encoding/norm constraints. -/
theorem selectorComposed_sound
    {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    {inactive : Support} {left right : Assignment Value}
    (confined : ConfinedTo inactive left right)
    (selectorDisjoint : SupportsDisjoint inactive boundary.supports.selector)
    (selectedDisjoint :
      SupportsDisjoint inactive
        (boundary.supports.selectedEquations (boundary.selector left)))
    (rightAlwaysOn : boundary.alwaysOn right) :
    boundary.Accepts left → boundary.Accepts right := by
  intro leftAccepts
  have selectorEq := selector_invariant boundary confined selectorDisjoint
  have selectedEq :=
    selectedEquations_invariant boundary confined selectedDisjoint
  refine ⟨rightAlwaysOn, ?_⟩
  rw [← selectorEq]
  exact selectedEq.mp leftAccepts.2

/-- Completeness direction for changing only inactive coordinates. The old
assignment must satisfy all always-on encoding/norm constraints. -/
theorem selectorComposed_complete
    {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    {inactive : Support} {left right : Assignment Value}
    (confined : ConfinedTo inactive left right)
    (selectorDisjoint : SupportsDisjoint inactive boundary.supports.selector)
    (selectedDisjoint :
      SupportsDisjoint inactive
        (boundary.supports.selectedEquations (boundary.selector left)))
    (leftAlwaysOn : boundary.alwaysOn left) :
    boundary.Accepts right → boundary.Accepts left := by
  intro rightAccepts
  have selectorEq := selector_invariant boundary confined selectorDisjoint
  have selectedEq :=
    selectedEquations_invariant boundary confined selectedDisjoint
  refine ⟨leftAlwaysOn, ?_⟩
  apply selectedEq.mpr
  rw [selectorEq]
  exact rightAccepts.2

/-- Acceptance equivalence. Requiring both always-on premises is intentional:
inactive coordinates may change, so norm/alphabet/bitness preservation cannot
be inferred from change confinement alone. -/
theorem selectorComposed_acceptance_iff
    {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    {inactive : Support} {left right : Assignment Value}
    (confined : ConfinedTo inactive left right)
    (selectorDisjoint : SupportsDisjoint inactive boundary.supports.selector)
    (selectedDisjoint :
      SupportsDisjoint inactive
        (boundary.supports.selectedEquations (boundary.selector left)))
    (leftAlwaysOn : boundary.alwaysOn left)
    (rightAlwaysOn : boundary.alwaysOn right) :
    boundary.Accepts left ↔ boundary.Accepts right := by
  constructor
  · exact selectorComposed_sound boundary confined selectorDisjoint
      selectedDisjoint rightAlwaysOn
  · exact selectorComposed_complete boundary confined selectorDisjoint
      selectedDisjoint leftAlwaysOn

/-- Authority-visible semantic outputs cannot change when their exact read
support is disjoint from the inactive support. -/
theorem authorityOutput_invariant
    {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    {inactive : Support} {left right : Assignment Value}
    (confined : ConfinedTo inactive left right)
    (selectorDisjoint : SupportsDisjoint inactive boundary.supports.selector)
    (authorityDisjoint :
      SupportsDisjoint inactive
        (boundary.supports.authorityOutput (boundary.selector left))) :
    boundary.Output left = boundary.Output right := by
  have selectorEq := selector_invariant boundary confined selectorDisjoint
  unfold Boundary.Output
  rw [← selectorEq]
  apply boundary.authorityOutputReads (boundary.selector left) left right
  exact agreesOn_of_confinedTo_of_disjoint confined authorityDisjoint

/-- Joint result needed before any fixed-selector inactive zero row can be
deleted. The exact selected branch chooses the declared inactive support. -/
theorem inactiveNoninterference
    {Value Branch Output : Type}
    (boundary : Boundary Value Branch Output)
    {left right : Assignment Value}
    (confined :
      ConfinedTo (boundary.supports.inactive (boundary.selector left))
        left right)
    (selectorDisjoint :
      SupportsDisjoint (boundary.supports.inactive (boundary.selector left))
        boundary.supports.selector)
    (selectedDisjoint :
      SupportsDisjoint (boundary.supports.inactive (boundary.selector left))
        (boundary.supports.selectedEquations (boundary.selector left)))
    (authorityDisjoint :
      SupportsDisjoint (boundary.supports.inactive (boundary.selector left))
        (boundary.supports.authorityOutput (boundary.selector left)))
    (leftAlwaysOn : boundary.alwaysOn left)
    (rightAlwaysOn : boundary.alwaysOn right) :
    boundary.selector left = boundary.selector right ∧
      (boundary.Accepts left ↔ boundary.Accepts right) ∧
      boundary.Output left = boundary.Output right := by
  exact ⟨selector_invariant boundary confined selectorDisjoint,
    selectorComposed_acceptance_iff boundary confined selectorDisjoint
      selectedDisjoint leftAlwaysOn rightAlwaysOn,
    authorityOutput_invariant boundary confined selectorDisjoint
      authorityDisjoint⟩

end Nightstream.Implementation.R1CS.InactiveFieldNoninterference
