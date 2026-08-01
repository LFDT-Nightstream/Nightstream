import Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

/-!
Contract: exact optimizer metrics derived from one proof-free canonical
manifest.

Assurance tier: model-level.

Owns: row, role, sparse-support, and degree metrics, plus the selected
auxiliary-first comparison order.

Does not own: protocol acceptance, a claim that one pass is correct, proving
time, memory use, or Rust measurements.

Emits constraints: none.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.Goldilocks.Optimization

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks.CanonicalManifest

/-- Metrics that remain meaningful before a Rust emitter exists. -/
structure Metrics where
  rows : Nat
  committedColumns : Nat
  publicColumns : Nat
  auxiliaryColumns : Nat
  nonzeroTerms : Nat
  maxRowSupport : Nat
  degree : Nat
deriving DecidableEq, Repr

namespace Metrics

/-- Compute every structural metric from the exact manifest data. -/
def ofManifest
    (program : CanonicalManifest.Program)
    (degree : Nat) : Metrics where
  rows := program.cost.recurringRows
  committedColumns := program.cost.committedColumns
  publicColumns := program.cost.publicColumns
  auxiliaryColumns := program.cost.auxiliaryColumns
  nonzeroTerms := program.statistics.totalNonzeros
  maxRowSupport := program.statistics.maxRowSupport
  degree := degree

/-- The selected optimizer order.

Committed and public columns are authoritative interface roles and must stay
equal. Among valid replacements, auxiliary columns are minimized first, then
rows, nonzero terms, support, and degree. -/
def Better (left right : Metrics) : Prop :=
  left.committedColumns = right.committedColumns /\
  left.publicColumns = right.publicColumns /\
  (left.auxiliaryColumns < right.auxiliaryColumns \/
    (left.auxiliaryColumns = right.auxiliaryColumns /\
      (left.rows < right.rows \/
        (left.rows = right.rows /\
          (left.nonzeroTerms < right.nonzeroTerms \/
            (left.nonzeroTerms = right.nonzeroTerms /\
              (left.maxRowSupport < right.maxRowSupport \/
                (left.maxRowSupport = right.maxRowSupport /\
                  left.degree < right.degree))))))))

theorem better_irreflexive (metrics : Metrics) :
    ¬ Better metrics metrics := by
  intro better
  rcases better with ⟨_, _, auxiliary | auxiliary⟩
  · exact Nat.lt_irrefl _ auxiliary
  · rcases auxiliary with ⟨_, rows | rows⟩
    · exact Nat.lt_irrefl _ rows
    · rcases rows with ⟨_, nonzeros | nonzeros⟩
      · exact Nat.lt_irrefl _ nonzeros
      · rcases nonzeros with ⟨_, support | support⟩
        · exact Nat.lt_irrefl _ support
        · exact Nat.lt_irrefl _ support.2

end Metrics

end Nightstream.Implementation.Lowering.Goldilocks.Optimization
