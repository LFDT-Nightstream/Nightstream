import Nightstream.Protocol.Nebula.FullClaim

/-!
Contract: independent V2 terminal semantics.

Assurance tier: protocol model and implementation boundary.

Owns the exact trailing verified full claim, its delayed consumption directly
to a closed carry, one final fold, all sixteen post-PiDEC folded children,
one bounded assignment per child that both opens that child's complete
four-component bundle and participates in the terminal relation, and the
external result check. The structure has no next fresh claim and no segment
continuation.

Does not assert that a concrete NIFS verifier, terminal decider, parser,
cryptographic commitment, generated circuit, Rust implementation, or deployed
verifier satisfies these predicates. Those are separate refinement and
security obligations.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.Nebula.Terminal

open Nightstream.Protocol.Nebula.CommitmentBundle
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.FullClaim

/-- Nightstream's selected `b = 2`, `k_rho = 16` profile exposes sixteen folded CE
children after PiDEC. Terminal verification must check every child. -/
def foldedChildCount : Nat := 16

abbrev FoldedChild := Fin foldedChildCount

theorem foldedChildCount_exact : foldedChildCount = 16 := rfl

/-- Weak, invalid alternative: each component can use a different witness. -/
def OpensSeparately
    {Assignment Commitment : Type}
    (componentMap : Component → Assignment → Commitment)
    (bundle : Bundle Commitment) : Prop :=
  ∀ component, ∃ assignment,
    componentMap component assignment = bundle component

/-- Required terminal authority: one witness opens every component. -/
def HasCommonOpening
    {Assignment Commitment : Type}
    (componentMap : Component → Assignment → Commitment)
    (bundle : Bundle Commitment) : Prop :=
  ∃ assignment, ∀ component,
    componentMap component assignment = bundle component

/-- One final fold. The predicate is supplied by the selected NIFS/terminal
backend. Its inputs include the exact trailing verified claim. -/
structure FinalFold
    {schema : FullClaim.Schema} {Digest Challenge Products : Type}
    (verify : schema.NifsProof →
      FullClaim.Claim schema Digest Challenge Products → Prop)
    (Running FoldProof Folded : Type)
    (folds : Running →
      FullClaim.Claim schema Digest Challenge Products →
      FoldProof → Folded → Prop)
    (running : Running)
    (verified : FullClaim.Verified schema Digest Challenge Products verify) :
    Type where
  proof : FoldProof
  folded : Folded
  accepted : folds running verified.claim proof folded

/-- Exact V2 terminal acceptance. `terminalRelation` and `bundleOf` both read
the same sixteen-child assignment family. For each child, all four bundle
components use that child's one assignment. This rules out an unchecked
PiDEC child, four unrelated lane witnesses, and a terminal relation that reads
another witness family.

The success conclusion is deliberately not an execution theorem. A deployed
soundness theorem must derive this structure from proof bytes and must then
combine it with the independently proved recursive/application refinement. -/
structure Accepted
    {schema : FullClaim.Schema} {Digest Challenge Products : Type}
    (verify : schema.NifsProof →
      FullClaim.Claim schema Digest Challenge Products → Prop)
    (balanced : Products → Prop)
    (Running FoldProof Folded Assignment Commitment Statement : Type)
    (folds : Running →
      FullClaim.Claim schema Digest Challenge Products →
      FoldProof → Folded → Prop)
    (bundleOf : Folded → FoldedChild → Bundle Commitment)
    (commit : Assignment → Bundle Commitment)
    (bounded : Assignment → Prop)
    (terminalRelation : Folded → (FoldedChild → Assignment) → Prop)
    (resultCheck : Statement → ClosedCarry Digest → Prop)
    (running : Running)
    (before : Carry Digest Challenge Products)
    (statement : Statement) : Type where
  trailing : FullClaim.Verified schema Digest Challenge Products verify
  final : ClosedCarry Digest
  consumesTrailing : FullClaim.Transition verify balanced before trailing
    (.closed final)
  finalFold : FinalFold verify Running FoldProof Folded folds running trailing
  assignments : FoldedChild → Assignment
  assignmentsBounded : ∀ child, bounded (assignments child)
  opensCompleteFoldedBundles : ∀ child,
    commit (assignments child) = bundleOf finalFold.folded child
  terminalRelationAccepted : terminalRelation finalFold.folded assignments
  resultAccepted : resultCheck statement final

namespace Accepted

variable
    {schema : FullClaim.Schema} {Digest Challenge Products : Type}
    {verify : schema.NifsProof →
      FullClaim.Claim schema Digest Challenge Products → Prop}
    {balanced : Products → Prop}
    {Running FoldProof Folded Assignment Commitment Statement : Type}
    {folds : Running →
      FullClaim.Claim schema Digest Challenge Products →
      FoldProof → Folded → Prop}
    {bundleOf : Folded → FoldedChild → Bundle Commitment}
    {commit : Assignment → Bundle Commitment}
    {bounded : Assignment → Prop}
    {terminalRelation : Folded → (FoldedChild → Assignment) → Prop}
    {resultCheck : Statement → ClosedCarry Digest → Prop}
    {running : Running}
    {before : Carry Digest Challenge Products}
    {statement : Statement}

/-- The NIFS-accepted full claim and the delayed memory claim are the same
record by construction. -/
theorem consumes_exact_verified_trailing_claim
    (accepted : Accepted verify balanced Running FoldProof Folded Assignment
      Commitment Statement folds bundleOf commit bounded terminalRelation
      resultCheck running before statement) :
    verify accepted.trailing.proof accepted.trailing.claim ∧
      Consumes balanced before accepted.trailing.claim.memory
        (.closed accepted.final) :=
  FullClaim.accepted_claim_is_consumed accepted.consumesTrailing

/-- Every final folded child has one typed assignment that opens all four of
its bundle components, and the terminal relation uses that exact full family. -/
theorem common_witness
    (accepted : Accepted verify balanced Running FoldProof Folded Assignment
      Commitment Statement folds bundleOf commit bounded terminalRelation
      resultCheck running before statement) :
    ∃ assignments : FoldedChild → Assignment,
      (∀ child, bounded (assignments child)) ∧
      (∀ child,
        commit (assignments child) = bundleOf accepted.finalFold.folded child) ∧
      terminalRelation accepted.finalFold.folded assignments :=
  ⟨accepted.assignments, accepted.assignmentsBounded,
    accepted.opensCompleteFoldedBundles,
    accepted.terminalRelationAccepted⟩

/-- The terminal transition cannot be an interior step. -/
theorem trailing_claim_closes_segment
    (accepted : Accepted verify balanced Running FoldProof Folded Assignment
      Commitment Statement folds bundleOf commit bounded terminalRelation
      resultCheck running before statement) :
    ∃ active,
      before = .active active ∧
      accepted.trailing.claim.memory.stepIndex.val + 1 =
        Lifecycle.claimsPerSegment := by
  cases before with
  | closed closed =>
      exact False.elim
        (cannot_consume_from_closed accepted.consumesTrailing.consumes)
  | active active =>
      exact ⟨active, rfl,
        (active_to_closed_requires_all_close_checks
          accepted.consumesTrailing.consumes).1⟩

end Accepted

/-! ## Countermodels for weaker terminal designs -/

namespace Countermodels

/-- A component chooses the only witness that makes that component true. -/
def selectorMap (component assignment : Component) : Bool :=
  decide (assignment = component)

def allTrue : Bundle Bool := fun _ => true

theorem separate_openings_exist :
    OpensSeparately selectorMap allTrue := by
  intro component
  exact ⟨component, by simp [selectorMap, allTrue]⟩

/-- Four valid separate openings do not imply a common witness. -/
theorem no_common_selector_opening :
    ¬ HasCommonOpening selectorMap allTrue := by
  rintro ⟨assignment, every⟩
  have full := every .full
  have operations := every .operations
  simp [selectorMap, allTrue] at full operations
  cases assignment <;> simp_all

/-- Lane-only commitments ignore the authority coordinate represented here
by the Boolean assignment. -/
def laneOnlyMap (_component : Component) (_assignment : Bool) : Bool :=
  false

def fullSensitiveMap (component : Component) (assignment : Bool) : Bool :=
  match component with
  | .full => assignment
  | _ => false

/-- Omitting the full component leaves two different assignments with equal
operations and snapshot components. The full component distinguishes them. -/
theorem lane_components_do_not_bind_full_assignment :
    (∀ component, component ≠ .full →
      fullSensitiveMap component false =
        fullSensitiveMap component true) ∧
    fullSensitiveMap .full false ≠ fullSensitiveMap .full true := by
  constructor
  · intro component notFull
    cases component <;> simp_all [fullSensitiveMap]
  · decide

/-- Another invalid alternative: one witness opens the bundle and an
unrelated witness satisfies the terminal relation. -/
def booleanBundle (value : Bool) : Bundle Bool := fun _ => value

def terminalTrue (_folded : Unit) (assignment : Bool) : Prop :=
  assignment = true

theorem separate_opening_and_terminal_witnesses_exist :
    (∃ openingWitness,
      booleanBundle openingWitness = booleanBundle false) ∧
    (∃ terminalWitness, terminalTrue () terminalWitness) := by
  exact ⟨⟨false, rfl⟩, ⟨true, rfl⟩⟩

/-- The preceding witnesses cannot be merged. This is why `Accepted` stores
one assignment used by both checks. -/
theorem no_common_opening_and_terminal_witness :
    ¬ ∃ witness,
      booleanBundle witness = booleanBundle false ∧
        terminalTrue () witness := by
  rintro ⟨witness, opens, terminal⟩
  have full := congrFun opens Component.full
  simp [booleanBundle, terminalTrue] at full terminal
  simp_all

/-- A terminal that checks only the first post-PiDEC child can accept while a
different folded child has no common four-component opening. -/
def foldedChildBundle (child : FoldedChild) : Bundle Bool :=
  if child.val = 0 then
    fun component => selectorMap component Component.full
  else
    allTrue

theorem checking_only_first_folded_child_is_insufficient :
    (∃ assignment,
      ∀ component,
        selectorMap component assignment =
          foldedChildBundle ⟨0, by decide⟩ component) ∧
      ¬ ∃ assignments : FoldedChild → Component,
        ∀ child component,
          selectorMap component (assignments child) =
            foldedChildBundle child component := by
  constructor
  · exact ⟨Component.full, by simp [foldedChildBundle]⟩
  · rintro ⟨assignments, allChildren⟩
    let second : FoldedChild := ⟨1, by decide⟩
    apply no_common_selector_opening
    refine ⟨assignments second, ?_⟩
    intro component
    simpa [foldedChildBundle, second, allTrue] using
      allChildren second component

end Countermodels

end Nightstream.Protocol.Nebula.Terminal
