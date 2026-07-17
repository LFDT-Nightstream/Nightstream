import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.SelectiveSelectorCoverageSchema
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Polynomial.Semantics

/-!
Contract: fail-closed handwritten decoder for compact selector-coverage
artifacts.

Owns: selector-column separation, exact ledger and gate interval partitions,
family-label-to-gate-class decoding, run-wise ledger/gate reconciliation, and
comparison of the exported Rust polynomial with the independent Lean 27-term
syntax.

Does not own: any generated fixture, full matrix rows, selector-port matrix
actions, arithmetic-family identity, an expected zero-cost-node inventory,
branch semantics, production coverage, constraint necessity, or row removal.

Emits constraints: no. Validation is linear in owner runs, gate runs, and the
27 polynomial terms; it never enumerates physical rows.

Authority boundary: a family label becomes usable only as a gate-class claim
after an aligned run from the final selector matrices agrees with the
exhaustive expected-gate map. Labels that share a gate class remain
indistinguishable here. Polynomial factorization additionally requires exact
equality with the independent Lean syntax.

| Validation branch | Mathematical obligation | Cost scale |
|---|---|---|
| selector columns | in bounds, distinct, and disjoint from column zero | selector count |
| owner partition | every interval, including empty intervals, is cursor-anchored | owner runs |
| gate partition | nonempty unit intervals exactly cover all rows | gate runs |
| reconciliation | each nonempty owner has the same interval and expected gate | run count |
| polynomial | ordered canonical terms equal Polynomial.Semantics.terms | 27 terms |
-/

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage

open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CCSResidualTable
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Row.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage.Wire

def supportedSchemaVersion : Nat := 1

def ownerNonempty (run : RawOwnerRun) : Bool :=
  decide (run.start < run.stop)

def ownerCovers (run : RawOwnerRun) (row : Nat) : Prop :=
  run.start ≤ row ∧ row < run.stop

def gateCovers (run : RawGateRun) (row : Nat) : Prop :=
  run.start ≤ row ∧ row < run.stop

private def commonGate (arm : Option Nat) :
    Option (RawGatePort × Nat) :=
  match arm with
  | none => some (.general, 0)
  | some _ => none

private def armGate
    (selectorColumns : List Nat) (port : RawGatePort)
    (arm : Option Nat) : Option (RawGatePort × Nat) :=
  match arm with
  | none => none
  | some index =>
      if bound : index < selectorColumns.length then
        some (port, selectorColumns.get ⟨index, bound⟩)
      else
        none

/-- Exhaustive semantic owner-to-selector map. Empty owner runs pass through
the same map, so zero-cost organizational nodes cannot carry malformed arms. -/
def expectedGate
    (selectorColumns : List Nat) (owner : RawOwnerRun) :
    Option (RawGatePort × Nat) :=
  match owner.family with
  | .selectorDomain => commonGate owner.arm
  | .sharedDomain => commonGate owner.arm
  | .armDomain => armGate selectorColumns .general owner.arm
  | .oneHot => commonGate owner.arm
  | .publicPadding => commonGate owner.arm
  | .privatePadding => commonGate owner.arm
  | .retained => armGate selectorColumns .general owner.arm
  | .poseidon2 => armGate selectorColumns .general owner.arm
  | .centeredUnit => armGate selectorColumns .general owner.arm
  | .shiftedTernaryCanonical =>
      armGate selectorColumns .general owner.arm
  | .polynomialEvaluation =>
      armGate selectorColumns .evaluation owner.arm
  | .productSum => armGate selectorColumns .evaluation owner.arm
  | .ringPadding => commonGate owner.arm

/-- Exact ordered wire image of the independently specified Lean polynomial. -/
def expectedPolynomialTerms : List RawPolynomialTerm :=
  Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Polynomial.Semantics.terms.map fun term => {
    coefficient := term.coefficient.val
    exponents := List.ofFn term.exponents
  }

def Every {Item : Type} (property : Item → Prop) : List Item → Prop
  | [] => True
  | item :: items => property item ∧ Every property items

instance everyDecidable
    {Item : Type} {property : Item → Prop}
    [DecidablePred property] :
    (items : List Item) → Decidable (Every property items)
  | [] => isTrue trivial
  | item :: items =>
      match inferInstanceAs (Decidable (property item)),
          everyDecidable items with
      | isTrue itemValid, isTrue tailValid =>
          isTrue ⟨itemValid, tailValid⟩
      | isFalse itemInvalid, _ =>
          isFalse (fun valid => itemInvalid valid.1)
      | _, isFalse tailInvalid =>
          isFalse (fun valid => tailInvalid valid.2)

def Aligned
    {Left Right : Type} (relation : Left → Right → Prop) :
    List Left → List Right → Prop
  | [], [] => True
  | left :: lefts, right :: rights =>
      relation left right ∧ Aligned relation lefts rights
  | _, _ => False

instance alignedDecidable
    {Left Right : Type} {relation : Left → Right → Prop}
    [DecidableRel relation] :
    (lefts : List Left) → (rights : List Right) →
      Decidable (Aligned relation lefts rights)
  | [], [] => isTrue trivial
  | [], _ :: _ => isFalse (fun valid => valid)
  | _ :: _, [] => isFalse (fun valid => valid)
  | left :: lefts, right :: rights =>
      match inferInstanceAs (Decidable (relation left right)),
          alignedDecidable lefts rights with
      | isTrue headValid, isTrue tailValid =>
          isTrue ⟨headValid, tailValid⟩
      | isFalse headInvalid, _ =>
          isFalse (fun valid => headInvalid valid.1)
      | _, isFalse tailInvalid =>
          isFalse (fun valid => tailInvalid valid.2)

def selectorColumnsValid (raw : RawCoverage) : Prop :=
  raw.selectorColumns.Nodup ∧
    0 ∉ raw.selectorColumns ∧
    Every (fun column => column < raw.columns) raw.selectorColumns

instance selectorColumnsValidDecidable (raw : RawCoverage) :
    Decidable (selectorColumnsValid raw) := by
  unfold selectorColumnsValid
  infer_instance

def ownerLabelsValid (raw : RawCoverage) : Prop :=
  Every (fun owner => expectedGate raw.selectorColumns owner ≠ none)
    raw.ownerRuns

instance ownerLabelsValidDecidable (raw : RawCoverage) :
    Decidable (ownerLabelsValid raw) := by
  unfold ownerLabelsValid
  exact everyDecidable raw.ownerRuns

def ownerPartitionFrom
    (rows cursor : Nat) : List RawOwnerRun → Prop
  | [] => cursor = rows
  | owner :: owners =>
      owner.start = cursor ∧
        owner.start ≤ owner.stop ∧
        owner.stop ≤ rows ∧
        ownerPartitionFrom rows owner.stop owners

instance ownerPartitionFromDecidable (rows cursor : Nat) :
    (owners : List RawOwnerRun) →
      Decidable (ownerPartitionFrom rows cursor owners)
  | [] =>
      if equal : cursor = rows then
        isTrue equal
      else
        isFalse equal
  | owner :: owners =>
      if startAtCursor : owner.start = cursor then
        if ordered : owner.start ≤ owner.stop then
          if bounded : owner.stop ≤ rows then
            match ownerPartitionFromDecidable rows owner.stop owners with
            | isTrue tailValid =>
                isTrue ⟨startAtCursor, ordered, bounded, tailValid⟩
            | isFalse tailInvalid =>
                isFalse (fun valid => tailInvalid valid.2.2.2)
          else
            isFalse (fun valid => bounded valid.2.2.1)
        else
          isFalse (fun valid => ordered valid.2.1)
      else
        isFalse (fun valid => startAtCursor valid.1)

def gatePartitionFrom
    (rows columns cursor : Nat) : List RawGateRun → Prop
  | [] => cursor = rows
  | gate :: gates =>
      gate.start = cursor ∧
        gate.start < gate.stop ∧
        gate.stop ≤ rows ∧
        gate.column < columns ∧
        gate.coefficient = 1 ∧
        gatePartitionFrom rows columns gate.stop gates

instance gatePartitionFromDecidable (rows columns cursor : Nat) :
    (gates : List RawGateRun) →
      Decidable (gatePartitionFrom rows columns cursor gates)
  | [] =>
      if equal : cursor = rows then
        isTrue equal
      else
        isFalse equal
  | gate :: gates =>
      if startAtCursor : gate.start = cursor then
        if nonempty : gate.start < gate.stop then
          if rowBound : gate.stop ≤ rows then
            if columnBound : gate.column < columns then
              if unit : gate.coefficient = 1 then
                match gatePartitionFromDecidable
                    rows columns gate.stop gates with
                | isTrue tailValid =>
                    isTrue ⟨startAtCursor, nonempty, rowBound,
                      columnBound, unit, tailValid⟩
                | isFalse tailInvalid =>
                    isFalse (fun valid => tailInvalid valid.2.2.2.2.2)
              else
                isFalse (fun valid => unit valid.2.2.2.2.1)
            else
              isFalse (fun valid => columnBound valid.2.2.2.1)
          else
            isFalse (fun valid => rowBound valid.2.2.1)
        else
          isFalse (fun valid => nonempty valid.2.1)
      else
        isFalse (fun valid => startAtCursor valid.1)

def ownerGateMatches
    (selectorColumns : List Nat)
    (owner : RawOwnerRun) (gate : RawGateRun) : Prop :=
  owner.start = gate.start ∧
    owner.stop = gate.stop ∧
    gate.coefficient = 1 ∧
    expectedGate selectorColumns owner = some (gate.port, gate.column)

instance ownerGateMatchesDecidable
    (selectorColumns : List Nat) (owner : RawOwnerRun)
    (gate : RawGateRun) :
    Decidable (ownerGateMatches selectorColumns owner gate) := by
  unfold ownerGateMatches
  infer_instance

def runsReconcile (raw : RawCoverage) : Prop :=
  Aligned (ownerGateMatches raw.selectorColumns)
    (raw.ownerRuns.filter ownerNonempty) raw.gateRuns

instance runsReconcileDecidable (raw : RawCoverage) :
    Decidable (runsReconcile raw) := by
  unfold runsReconcile
  exact alignedDecidable
    (raw.ownerRuns.filter ownerNonempty) raw.gateRuns

def polynomialMatches (raw : RawCoverage) : Prop :=
  raw.polynomialArity = 13 ∧
    raw.polynomialTerms = expectedPolynomialTerms

instance polynomialMatchesDecidable (raw : RawCoverage) :
    Decidable (polynomialMatches raw) := by
  unfold polynomialMatches
  infer_instance

def CoverageValid (raw : RawCoverage) : Prop :=
  raw.schemaVersion = supportedSchemaVersion ∧
    (0 < raw.rows ∧ 0 < raw.columns) ∧
    selectorColumnsValid raw ∧
    ownerLabelsValid raw ∧
    ownerPartitionFrom raw.rows 0 raw.ownerRuns ∧
    gatePartitionFrom raw.rows raw.columns 0 raw.gateRuns ∧
    runsReconcile raw ∧
    polynomialMatches raw

instance coverageValidDecidable (raw : RawCoverage) :
    Decidable (CoverageValid raw) := by
  if schema : raw.schemaVersion = supportedSchemaVersion then
    if dimensions : 0 < raw.rows ∧ 0 < raw.columns then
      if selectors : selectorColumnsValid raw then
        if labels : ownerLabelsValid raw then
          if owners : ownerPartitionFrom raw.rows 0 raw.ownerRuns then
            if gates : gatePartitionFrom raw.rows raw.columns 0 raw.gateRuns then
              if reconcile : runsReconcile raw then
                if polynomial : polynomialMatches raw then
                  exact isTrue ⟨schema, dimensions, selectors, labels,
                    owners, gates, reconcile, polynomial⟩
                else
                  exact isFalse (fun valid =>
                    polynomial valid.2.2.2.2.2.2.2)
              else
                exact isFalse (fun valid =>
                  reconcile valid.2.2.2.2.2.2.1)
            else
              exact isFalse (fun valid => gates valid.2.2.2.2.2.1)
          else
            exact isFalse (fun valid => owners valid.2.2.2.2.1)
        else
          exact isFalse (fun valid => labels valid.2.2.2.1)
      else
        exact isFalse (fun valid => selectors valid.2.2.1)
    else
      exact isFalse (fun valid => dimensions valid.2.1)
  else
    exact isFalse (fun valid => schema valid.1)

namespace CoverageValid

theorem ownersPartition {raw : RawCoverage}
    (valid : CoverageValid raw) :
    ownerPartitionFrom raw.rows 0 raw.ownerRuns :=
  valid.2.2.2.2.1

theorem reconcile {raw : RawCoverage}
    (valid : CoverageValid raw) :
    runsReconcile raw :=
  valid.2.2.2.2.2.2.1

theorem polynomial {raw : RawCoverage}
    (valid : CoverageValid raw) :
    polynomialMatches raw :=
  valid.2.2.2.2.2.2.2

end CoverageValid

structure ValidatedCoverage where
  raw : RawCoverage
  valid : CoverageValid raw

def decodeCoverage (raw : RawCoverage) : Option ValidatedCoverage :=
  if valid : CoverageValid raw then
    some ⟨raw, valid⟩
  else
    none

theorem decodeCoverage_isSome_iff (raw : RawCoverage) :
    (decodeCoverage raw).isSome = true ↔ CoverageValid raw := by
  unfold decodeCoverage
  by_cases valid : CoverageValid raw
  · simp [valid]
  · simp [valid]

private theorem aligned_mem_right
    {Left Right : Type}
    {relation : Left → Right → Prop}
    {lefts : List Left} {rights : List Right}
    (related : Aligned relation lefts rights)
    {left : Left} (member : left ∈ lefts) :
    ∃ right, right ∈ rights ∧ relation left right := by
  induction lefts generalizing rights with
  | nil => simp at member
  | cons leftHead leftTail inductionHypothesis =>
      cases rights with
      | nil => simp [Aligned] at related
      | cons rightHead rightTail =>
          simp only [Aligned] at related
          simp only [List.mem_cons] at member
          rcases member with equal | member
          · subst leftHead
            exact ⟨rightHead, by simp, related.1⟩
          · rcases inductionHypothesis related.2 member with
              ⟨right, rightMember, rightRelated⟩
            exact ⟨right, by simp [rightMember], rightRelated⟩

private theorem ownerPartitionFrom_contains
    {rows cursor row : Nat} {owners : List RawOwnerRun}
    (partition : ownerPartitionFrom rows cursor owners)
    (cursorLe : cursor ≤ row) (rowLt : row < rows) :
    ∃ owner, owner ∈ owners ∧ ownerCovers owner row := by
  induction owners generalizing cursor with
  | nil =>
      simp only [ownerPartitionFrom] at partition
      omega
  | cons owner owners inductionHypothesis =>
      simp only [ownerPartitionFrom] at partition
      rcases partition with
        ⟨startAtCursor, ordered, stopBound, tailPartition⟩
      by_cases beforeStop : row < owner.stop
      · refine ⟨owner, by simp, ?_⟩
        exact ⟨by omega, beforeStop⟩
      · rcases inductionHypothesis tailPartition (by omega) with
          ⟨found, member, covers⟩
        exact ⟨found, by simp [member], covers⟩

theorem ValidatedCoverage.nonemptyOwner_has_exactGate
    (coverage : ValidatedCoverage)
    {owner : RawOwnerRun}
    (member : owner ∈ coverage.raw.ownerRuns)
    (nonempty : owner.start < owner.stop) :
    ∃ gate,
      gate ∈ coverage.raw.gateRuns ∧
      gate.start = owner.start ∧
      gate.stop = owner.stop ∧
      gate.coefficient = 1 ∧
      expectedGate coverage.raw.selectorColumns owner =
        some (gate.port, gate.column) := by
  have nonemptyBool : ownerNonempty owner = true := by
    simp [ownerNonempty, nonempty]
  have filtered :
      owner ∈ coverage.raw.ownerRuns.filter ownerNonempty := by
    simp [member, nonemptyBool]
  rcases aligned_mem_right (CoverageValid.reconcile coverage.valid) filtered with
    ⟨gate, gateMember, matched⟩
  exact ⟨gate, gateMember, matched.1.symm, matched.2.1.symm,
    matched.2.2.1, matched.2.2.2⟩

/-- Every row is covered by one reconciled owner/gate pair, derived from the
compact interval partitions rather than checked by row enumeration. -/
theorem ValidatedCoverage.row_reconciles
    (coverage : ValidatedCoverage)
    (row : Fin coverage.raw.rows) :
    ∃ owner gate,
      owner ∈ coverage.raw.ownerRuns ∧
      gate ∈ coverage.raw.gateRuns ∧
      ownerCovers owner row.val ∧
      gateCovers gate row.val ∧
      expectedGate coverage.raw.selectorColumns owner =
        some (gate.port, gate.column) := by
  rcases ownerPartitionFrom_contains
      (CoverageValid.ownersPartition coverage.valid)
      (Nat.zero_le row.val) row.isLt with
    ⟨owner, ownerMember, ownerCovers⟩
  have nonempty : owner.start < owner.stop := by
    exact Nat.lt_of_le_of_lt ownerCovers.1 ownerCovers.2
  rcases coverage.nonemptyOwner_has_exactGate ownerMember nonempty with
    ⟨gate, gateMember, gateStart, gateStop, _coefficient, expected⟩
  refine ⟨owner, gate, ownerMember, gateMember, ownerCovers, ?_, expected⟩
  constructor
  · rw [gateStart]
    exact ownerCovers.1
  · rw [gateStop]
    exact ownerCovers.2

theorem ValidatedCoverage.polynomial_exact
    (coverage : ValidatedCoverage) :
    coverage.raw.polynomialTerms = expectedPolynomialTerms :=
  (CoverageValid.polynomial coverage.valid).2

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.SelectorCoverage
