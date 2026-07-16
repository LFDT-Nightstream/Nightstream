import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanTable

/-!
Canonical Boolean-domain ownership for paper-level joint `Pi_CCS` residuals.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: common Boolean-domain indexing before residual-family construction.
Constraint family: shared infrastructure for CCS, norm, and carried-evaluation
tables; this file owns no residual formula itself.

Owns: a typed Boolean vertex, its canonical recursive enumeration, construction
of a `BooleanTable` from a value at every vertex, and the exact leaf-order and
leafwise-zero theorems for that construction.

Does not own: CCS, norm, or carried-evaluation formulas, external integer/bit
serialization, matrix row numbering, alpha/gamma identities, SumCheck,
Fiat--Shamir, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: the paper's Boolean cube is represented once, with the
newly introduced coordinate prepended and the `false` branch before the `true`
branch. Residual families import this order; they must not invent independent
leaf permutations. A later refinement must prove how external numeric row and
bit encodings map into this semantic order.

| Shared owner | Mathematical object | Canonical choice | Proven relation |
|---|---|---|---|
| `BooleanVertex` | `x in {0,1}^ell` | prepended coordinate | typed cube point |
| `BooleanVertex.all` | full Boolean cube | low subtree then high subtree | length `2^ell`, no duplicates, every vertex present |
| `BooleanTable.tabulate` | residual truth table | recurse on the same low/high split | callers cannot permute leaves |
| `BooleanTable.entries_tabulate` | table serialization | `BooleanVertex.all.map values` | exact leaf order |
| `BooleanTable.tabulate_allEntriesZero_iff` | leafwise zero condition | direct pointwise equality | no supplied semantic iff |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

universe uField

/-- A Boolean hypercube vertex in the same recursive coordinate order used by
`BooleanTable`: the newly introduced coordinate is prepended, and `false`
precedes `true`. -/
inductive BooleanVertex : Nat -> Type where
  | nil : BooleanVertex 0
  | cons {variables : Nat} (coordinate : Bool)
      (tail : BooleanVertex variables) : BooleanVertex (variables + 1)
deriving Repr, DecidableEq

namespace BooleanVertex

/-- Canonical low/high enumeration of all Boolean vertices. This enumeration
belongs to the semantic model, not to a prover or production serializer. -/
def all : (variables : Nat) -> List (BooleanVertex variables)
  | 0 => [.nil]
  | variables + 1 =>
      (all variables).map (fun tail => .cons false tail) ++
        (all variables).map (fun tail => .cons true tail)

/-- The explicit enumeration has exactly one slot per Boolean-cube point. -/
theorem all_length (variables : Nat) :
    (all variables).length = 2 ^ variables := by
  induction variables with
  | zero => rfl
  | succ variables inductionHypothesis =>
      simp [all, inductionHypothesis, Nat.pow_succ, Nat.mul_two]

/-- The canonical enumeration contains no duplicate Boolean vertex. -/
theorem all_nodup (variables : Nat) :
    (all variables).Nodup := by
  induction variables with
  | zero => simp [all]
  | succ variables inductionHypothesis =>
      rw [all, List.nodup_append]
      refine ⟨?_, ?_, ?_⟩
      · exact List.Pairwise.map _ (by
          intro left right different equal
          exact different (BooleanVertex.cons.inj equal).2)
          inductionHypothesis
      · exact List.Pairwise.map _ (by
          intro left right different equal
          exact different (BooleanVertex.cons.inj equal).2)
          inductionHypothesis
      · intro value lowMember other highMember
        rcases List.mem_map.mp lowMember with ⟨low, _, rfl⟩
        rcases List.mem_map.mp highMember with ⟨high, _, rfl⟩
        simp

/-- Every typed Boolean vertex occurs in the canonical enumeration. -/
theorem mem_all {variables : Nat} (vertex : BooleanVertex variables) :
    vertex ∈ all variables := by
  induction vertex with
  | nil => simp [all]
  | cons coordinate tail inductionHypothesis =>
      cases coordinate <;> simp [all, inductionHypothesis]

end BooleanVertex

namespace BooleanTable

/-- Construct an explicit Boolean table from a semantic value at every typed
vertex. The recursion fixes low/high placement; callers cannot permute leaves. -/
def tabulate
    {Field : Type uField} :
    {variables : Nat} ->
      (BooleanVertex variables -> Field) -> BooleanTable Field variables
  | 0, values => .leaf (values .nil)
  | _ + 1, values =>
      .branch
        (tabulate (fun tail => values (.cons false tail)))
        (tabulate (fun tail => values (.cons true tail)))

/-- Tabulation leaves are exactly the canonical low/high vertex enumeration. -/
theorem entries_tabulate
    {Field : Type uField}
    {variables : Nat}
    (values : BooleanVertex variables -> Field) :
    (tabulate values).entries = (BooleanVertex.all variables).map values := by
  induction variables with
  | zero => rfl
  | succ variables inductionHypothesis =>
      simp only [tabulate, entries, BooleanVertex.all, List.map_append]
      rw [inductionHypothesis, inductionHypothesis]
      simp [Function.comp_def]

/-- A tabulated residual is leafwise zero exactly when its independently
defined value is zero at every Boolean vertex. -/
theorem tabulate_allEntriesZero_iff
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables : Nat}
    (values : BooleanVertex variables -> Field) :
    (tabulate values).AllEntriesZero ops ↔
      ∀ vertex, values vertex = ops.zero := by
  constructor
  · intro allZero vertex
    apply allZero (values vertex)
    rw [entries_tabulate]
    exact List.mem_map.mpr ⟨vertex, BooleanVertex.mem_all vertex, rfl⟩
  · intro pointwise value member
    rw [entries_tabulate] at member
    rcases List.mem_map.mp member with ⟨vertex, _, rfl⟩
    exact pointwise vertex

end BooleanTable

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
