import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanTable

/-!
Canonical Boolean-domain ownership for paper-level joint `Pi_CCS` residuals.

Protocol: SuperNeo v1.2 `Pi_CCS` (Section 7.3 / Appendix B.2).
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
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField

inductive BooleanVertex : Nat -> Type where
  | nil : BooleanVertex 0
  | cons {variables : Nat} (coordinate : Bool)
      (tail : BooleanVertex variables) : BooleanVertex (variables + 1)
deriving Repr, DecidableEq

namespace BooleanVertex

def all : (variables : Nat) -> List (BooleanVertex variables)
  | 0 => [.nil]
  | variables + 1 =>
      (all variables).map (fun tail => .cons false tail) ++
        (all variables).map (fun tail => .cons true tail)

theorem all_length (variables : Nat) :
    (all variables).length = 2 ^ variables := by
  induction variables with
  | zero => rfl
  | succ variables inductionHypothesis =>
      simp [all, inductionHypothesis, Nat.pow_succ, Nat.mul_two]

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

theorem mem_all {variables : Nat} (vertex : BooleanVertex variables) :
    vertex ∈ all variables := by
  induction vertex with
  | nil => simp [all]
  | cons coordinate tail inductionHypothesis =>
      cases coordinate <;> simp [all, inductionHypothesis]

end BooleanVertex

namespace BooleanTable

def tabulate
    {Field : Type uField} :
    {variables : Nat} ->
      (BooleanVertex variables -> Field) -> BooleanTable Field variables
  | 0, values => .leaf (values .nil)
  | _ + 1, values =>
      .branch
        (tabulate (fun tail => values (.cons false tail)))
        (tabulate (fun tail => values (.cons true tail)))

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

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
