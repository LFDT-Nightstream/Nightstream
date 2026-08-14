import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ConcreteCarrier.Algebra

/-!
Contract: typed port vocabulary for the production selective CCS polynomial.

Owns: the thirteen mathematical matrix-image roles, their exact numeric
indices, and the inverse role/index equivalence. The shared ports at indices
8--12 intentionally have both evaluation and canonical-class meanings;
selector disjointness decides which meaning is active on a row.

Does not own: sparse terms, matrix rows, source-column maps, row schedules,
Rust conformance, or constraint counts.

Emits constraints: no.

| Stage path | Port role | Numeric index | Mathematical use |
|---|---|---:|---|
| `f_prime.selective_ccs.port.bit` | `bit` | 0 | Boolean residual / first evaluation factor |
| `f_prime.selective_ccs.port.general_selector` | `generalSelector` | 1 | Gates ordinary and canonical rows |
| `f_prime.selective_ccs.port.abc` | `a`, `b`, `c` | 2, 3, 4 | Product and result images |
| `f_prime.selective_ccs.port.sbox` | `sboxInput` | 5 | Seventh-power image |
| `f_prime.selective_ccs.port.centered` | `centeredUnit` | 6 | Centered-unit cubic image |
| `f_prime.selective_ccs.port.eval_selector` | `evalSelector` | 7 | Gates five product pairs |
| `f_prime.selective_ccs.port.canonical` | normalized bound classes 0--4 | 8--12 | Select one two-trit transition |
| `f_prime.selective_ccs.port.eval_tail` | `evalTailRight` | 12 | Fifth evaluation pair RHS / class 4 |
| `f_prime.selective_ccs.port.coverage` | all named roles | 0--12 | no unnamed or multiply named physical port |
-/

namespace Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports

/-- Exact matrix arity of the independently specified selective polynomial. -/
def portCount : Nat := 13

/-- Semantic names for the exact thirteen matrix images consumed by the
selective sparse polynomial. -/
inductive Role where
  | bit
  | generalSelector
  | a
  | b
  | c
  | sboxInput
  | centeredUnit
  | evalSelector
  | canonicalDigit
  | canonicalBorrow
  | canonicalNextBorrow
  | canonicalBoundDigit
  | evalTailRight
deriving DecidableEq, Repr

/-- Exact Rust matrix-port number for each semantic role. -/
@[simp] def Role.index : Role -> Fin 13
  | .bit => ⟨0, by decide⟩
  | .generalSelector => ⟨1, by decide⟩
  | .a => ⟨2, by decide⟩
  | .b => ⟨3, by decide⟩
  | .c => ⟨4, by decide⟩
  | .sboxInput => ⟨5, by decide⟩
  | .centeredUnit => ⟨6, by decide⟩
  | .evalSelector => ⟨7, by decide⟩
  | .canonicalDigit => ⟨8, by decide⟩
  | .canonicalBorrow => ⟨9, by decide⟩
  | .canonicalNextBorrow => ⟨10, by decide⟩
  | .canonicalBoundDigit => ⟨11, by decide⟩
  | .evalTailRight => ⟨12, by decide⟩

/-- Distinct semantic roles never alias a numeric matrix port. -/
theorem Role.index_injective : Function.Injective Role.index := by
  intro left right equal
  cases left <;> cases right <;> simp_all

/-- Exact semantic role carried by one physical matrix index. This is the
only numeric-to-semantic conversion for the active selective relation. -/
@[simp] def Role.ofIndex : Fin 13 -> Role
  | ⟨0, _⟩ => .bit
  | ⟨1, _⟩ => .generalSelector
  | ⟨2, _⟩ => .a
  | ⟨3, _⟩ => .b
  | ⟨4, _⟩ => .c
  | ⟨5, _⟩ => .sboxInput
  | ⟨6, _⟩ => .centeredUnit
  | ⟨7, _⟩ => .evalSelector
  | ⟨8, _⟩ => .canonicalDigit
  | ⟨9, _⟩ => .canonicalBorrow
  | ⟨10, _⟩ => .canonicalNextBorrow
  | ⟨11, _⟩ => .canonicalBoundDigit
  | ⟨12, _⟩ => .evalTailRight

@[simp] theorem Role.ofIndex_index (role : Role) :
    Role.ofIndex role.index = role := by
  cases role <;> rfl

@[simp] theorem Role.index_ofIndex :
    forall index : Fin 13, (Role.ofIndex index).index = index
  | ⟨0, _⟩ => rfl
  | ⟨1, _⟩ => rfl
  | ⟨2, _⟩ => rfl
  | ⟨3, _⟩ => rfl
  | ⟨4, _⟩ => rfl
  | ⟨5, _⟩ => rfl
  | ⟨6, _⟩ => rfl
  | ⟨7, _⟩ => rfl
  | ⟨8, _⟩ => rfl
  | ⟨9, _⟩ => rfl
  | ⟨10, _⟩ => rfl
  | ⟨11, _⟩ => rfl
  | ⟨12, _⟩ => rfl

theorem port_count_exact : portCount = 13 := by
  rfl

end Nightstream.Implementation.R1CS.SelectiveCcs.Polynomial.Ports
