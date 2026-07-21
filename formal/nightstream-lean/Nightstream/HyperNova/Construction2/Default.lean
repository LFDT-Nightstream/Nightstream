/-!
Contract: HyperNova default-instance obligation and Rust's zero-arity base specialization.

Construction 2 populates the running product with a default satisfying pair.
The direct-F' Rust specialization instead uses `RunningInstance::default()`:
empty claims, empty witnesses, and no Π_RLC parent authority. This file states
both facts separately. Replicating the paper's universally satisfying
claim/witness pair gives a pointwise-valid default vector of any arity. The
empty product is relation-valid by zero arity; it does not pretend that an
omitted element is equal to the paper's `u_perp`.

Owns: the universal paper-default contract, its exact replicated pointwise
installation, and the empty zero-arity implementation deviation.

Does not own: a semantic interpretation of `parentAuthority`, SuperNeo NIFS,
production decoding, Rust control-flow refinement, R1CS rows, costs, or row
removal.

Emits constraints: none.

Assurance tier: model-level.

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

/-- Paper-level default pair satisfying every compatible relation context.
`Structure` may package both the paper public parameters and relation structure,
so the universal field represents HyperNova's quantification over `(pp, s)`. -/
structure DefaultPair
    (Structure : Type uStructure)
    (Claim : Type uClaim)
    (Witness : Type uWitness)
    (relation : Structure → Claim → Witness → Prop) where
  claim : Claim
  witness : Witness
  satisfies : ∀ system, relation system claim witness

/-- Replicating the exact paper default pair gives pointwise relation-valid
claim and witness vectors at every arity and for every structure. Unlike the
zero-arity implementation theorem below, the successor case materially
consumes `DefaultPair.satisfies`. -/
theorem replicatedDefault_allPairs
    {Structure : Type uStructure}
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    (relation : Structure → Claim → Witness → Prop)
    (paperDefault : DefaultPair Structure Claim Witness relation)
    (system : Structure)
    (arity : Nat) :
    AllPairs (relation system)
      (List.replicate arity paperDefault.claim)
      (List.replicate arity paperDefault.witness) := by
  induction arity with
  | zero => exact .nil
  | succ arity inductionHypothesis =>
      exact .cons (paperDefault.satisfies system) inductionHypothesis

/-- Exact statement of the independent implementation specialization. -/
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

/-- Rust's empty running accumulator is a valid zero-arity product. Its proof
does not consume a paper default pair and therefore does not identify omission
with HyperNova's explicit `u_perp`. -/
theorem emptyRunning_zeroArity
    {Structure : Type uStructure}
    {Claim : Type uClaim}
    {Witness : Type uWitness}
    {Parent : Type uParent}
    (relation : Structure → Claim → Witness → Prop)
    (system : Structure) :
    ZeroAritySpecialization relation system
      (emptyRunning (Claim := Claim) (Witness := Witness) (Parent := Parent)) := by
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
