/-!
Contract: HyperNova default-instance obligation and Rust's zero-arity base specialization.

Construction 2 populates the running product with a default satisfying pair.
The direct-F' Rust specialization instead uses `RunningInstance::default()`:
empty claims, empty witnesses, and no Π_RLC parent authority. This file states
the deviation precisely. The empty product is relation-valid by zero arity; it
does not pretend that an omitted element is equal to the paper's `u_perp`.

Maps to:
- HyperNova Construction 2 steps 3.2 and prover step 3.
- `construction2::running::RunningInstance::{default,is_empty,shape_ok}`.
- `f_prime::native` base branches.
-/

namespace Nightstream.HyperNova.Construction2.Default

universe uStructure uClaim uWitness uParent

/-- Rust-shaped running product with separate prover-only witnesses. -/
structure RunningProduct
    (Claim : Type uClaim)
    (Witness : Type uWitness)
    (Parent : Type uParent) where
  claims : List Claim
  witnesses : List Witness
  parentAuthority : Option Parent

/-- Exact `#[derive(Default)]` value used by the Rust base branch. -/
def emptyRunning
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    {Parent : Type uParent} :
    RunningProduct Claim Witness Parent where
  claims := []
  witnesses := []
  parentAuthority := none

/-- Rust `shape_ok`: paired prover vectors and parent presence iff nonempty. -/
def ShapeValid
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    {Parent : Type uParent}
    (running : RunningProduct Claim Witness Parent) : Prop :=
  running.claims.length = running.witnesses.length ∧
  (running.claims = [] → running.parentAuthority = none) ∧
  (running.claims ≠ [] → ∃ parent, running.parentAuthority = some parent)

/-- Pointwise relation for Rust's parallel claim/witness vectors. -/
inductive AllPairs
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    (relation : Claim → Witness → Prop) : List Claim → List Witness → Prop where
  | nil : AllPairs relation [] []
  | cons {claim : Claim} {witness : Witness}
      {claims : List Claim} {witnesses : List Witness} :
      relation claim witness → AllPairs relation claims witnesses →
      AllPairs relation (claim :: claims) (witness :: witnesses)

/-- Every paired entry belongs to the running relation. -/
def ProductHolds
    {Structure : Type uStructure}
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    {Parent : Type uParent}
    (relation : Structure → Claim → Witness → Prop)
    (system : Structure)
    (running : RunningProduct Claim Witness Parent) : Prop :=
  ShapeValid running ∧ AllPairs (relation system) running.claims running.witnesses

/-- Paper-level default pair satisfying every compatible structure. -/
structure DefaultPair
    (Structure : Type uStructure)
    (Claim : Type uClaim)
    (Witness : Type uWitness)
    (relation : Structure → Claim → Witness → Prop) where
  claim : Claim
  witness : Witness
  satisfies : ∀ system, relation system claim witness

/-- Exact statement of the implementation specialization. -/
def ZeroAritySpecialization
    {Structure : Type uStructure}
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    {Parent : Type uParent}
    (relation : Structure → Claim → Witness → Prop)
    (system : Structure)
    (running : RunningProduct Claim Witness Parent) : Prop :=
  running.claims.length = 0 ∧
  running.witnesses.length = 0 ∧
  ProductHolds relation system running

/-- Rust's empty running accumulator is a valid zero-arity default specialization. -/
theorem emptyRunning_realizes_default
    {Structure : Type uStructure}
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    {Parent : Type uParent}
    (relation : Structure → Claim → Witness → Prop)
    (paperDefault : DefaultPair Structure Claim Witness relation)
    (system : Structure) :
    ZeroAritySpecialization relation system
      (emptyRunning (Claim := Claim) (Witness := Witness) (Parent := Parent)) := by
  have _paperDefaultExists : relation system paperDefault.claim paperDefault.witness :=
    paperDefault.satisfies system
  exact ⟨rfl, rfl, ⟨⟨rfl, by intro _; rfl, by
    intro nonempty
    exact False.elim (nonempty rfl)⟩, .nil⟩⟩

/-- A prover cannot attach parent authority to the empty base accumulator. -/
theorem empty_claims_with_parent_rejected
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    {Parent : Type uParent}
    (parent : Parent) :
    ¬ ShapeValid ({
      claims := []
      witnesses := []
      parentAuthority := some parent
    } : RunningProduct Claim Witness Parent) := by
  intro shape
  have := shape.2.1 rfl
  cases this

end Nightstream.HyperNova.Construction2.Default
