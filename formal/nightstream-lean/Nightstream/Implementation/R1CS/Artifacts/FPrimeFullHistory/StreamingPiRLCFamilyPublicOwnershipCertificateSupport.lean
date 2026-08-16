import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublicSchema

/-!
Contract: structural cursor semantics for bounded PiRLC public-family owner
certificates.

Owns the generic proof that checked owner prefixes give exact row coverage,
source-object matches, and consecutive indices. It owns no generated data.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOwnershipCertificateSupport

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact

structure OwnerCursor where
  row : Nat
  canonical : Nat
  poseidon2 : Nat
  glue : Nat
  phaseEnvelope : Nat
deriving DecidableEq, Repr

namespace OwnerCursor

def startFor (arm : RawArm) : OwnerCursor :=
  { row := arm.sourceRowCount
    canonical := 0
    poseidon2 := 0
    glue := 0
    phaseEnvelope := 0 }

def index (cursor : OwnerCursor) : OwnerKind → Nat
  | .canonical => cursor.canonical
  | .poseidon2 => cursor.poseidon2
  | .glue => cursor.glue
  | .phaseEnvelope => cursor.phaseEnvelope

def advance (cursor : OwnerCursor) (kind : OwnerKind)
    (rowEnd : Nat) : OwnerCursor :=
  match kind with
  | .canonical =>
      { cursor with row := rowEnd, canonical := cursor.canonical + 1 }
  | .poseidon2 =>
      { cursor with row := rowEnd, poseidon2 := cursor.poseidon2 + 1 }
  | .glue =>
      { cursor with row := rowEnd, glue := cursor.glue + 1 }
  | .phaseEnvelope =>
      { cursor with row := rowEnd, phaseEnvelope := cursor.phaseEnvelope + 1 }

def finalFor (arm : RawArm) : OwnerCursor :=
  { row := arm.rowCount
    canonical := arm.canonicalCalls.length
    poseidon2 := arm.poseidon2Calls.length
    glue := arm.glueRows.length
    phaseEnvelope := 1 }

@[simp] theorem startFor_index (arm : RawArm) (kind : OwnerKind) :
    (startFor arm).index kind = 0 := by
  cases kind <;> rfl

@[simp] theorem advance_index_same
    (cursor : OwnerCursor) (kind : OwnerKind) (rowEnd : Nat) :
    (cursor.advance kind rowEnd).index kind = cursor.index kind + 1 := by
  cases kind <;> rfl

@[simp] theorem advance_row
    (cursor : OwnerCursor) (kind : OwnerKind) (rowEnd : Nat) :
    (cursor.advance kind rowEnd).row = rowEnd := by
  cases kind <;> rfl

theorem advance_index_of_ne
    (cursor : OwnerCursor) (ownerKind kind : OwnerKind) (rowEnd : Nat)
    (different : ownerKind ≠ kind) :
    (cursor.advance ownerKind rowEnd).index kind = cursor.index kind := by
  cases ownerKind <;> cases kind <;> simp_all [advance, index]

end OwnerCursor

def ownerMatchesCheck (arm : RawArm) (owner : Owner) : Bool :=
  match owner.kind with
  | .canonical =>
      match arm.canonicalCalls[owner.index]? with
      | some call =>
          decide (owner.rowStart = call.rowStart) &&
            decide (owner.rowEnd = call.rowEnd)
      | none => false
  | .poseidon2 =>
      match arm.poseidon2Calls[owner.index]? with
      | some call =>
          decide (owner.rowStart = call.rowStart) &&
            decide (owner.rowEnd = call.rowEnd)
      | none => false
  | .glue =>
      match arm.glueRows[owner.index]? with
      | some indexed =>
          decide (owner.rowStart = indexed.index) &&
            decide (owner.rowEnd = indexed.index + 1)
      | none => false
  | .phaseEnvelope =>
      decide (owner.index = 0) &&
        (decide (owner.rowStart = arm.phaseEnvelopeRowStart) &&
          decide (owner.rowEnd = arm.phaseEnvelopeRowEnd))

theorem ownerMatchesCheck_sound
    {arm : RawArm} {owner : Owner}
    (checked : ownerMatchesCheck arm owner = true) :
    owner.Matches arm := by
  cases kindEq : owner.kind with
  | canonical =>
      cases callEq : arm.canonicalCalls[owner.index]? with
      | none => simp [ownerMatchesCheck, kindEq, callEq] at checked
      | some call =>
          have equalities :
              owner.rowStart = call.rowStart ∧
                owner.rowEnd = call.rowEnd := by
            simpa [ownerMatchesCheck, kindEq, callEq, Bool.and_eq_true,
              decide_eq_true_eq] using checked
          unfold Owner.Matches
          rw [kindEq]
          exact ⟨call, callEq, equalities.1, equalities.2⟩
  | poseidon2 =>
      cases callEq : arm.poseidon2Calls[owner.index]? with
      | none => simp [ownerMatchesCheck, kindEq, callEq] at checked
      | some call =>
          have equalities :
              owner.rowStart = call.rowStart ∧
                owner.rowEnd = call.rowEnd := by
            simpa [ownerMatchesCheck, kindEq, callEq, Bool.and_eq_true,
              decide_eq_true_eq] using checked
          unfold Owner.Matches
          rw [kindEq]
          exact ⟨call, callEq, equalities.1, equalities.2⟩
  | glue =>
      cases rowEq : arm.glueRows[owner.index]? with
      | none => simp [ownerMatchesCheck, kindEq, rowEq] at checked
      | some indexed =>
          have equalities :
              owner.rowStart = indexed.index ∧
                owner.rowEnd = indexed.index + 1 := by
            simpa [ownerMatchesCheck, kindEq, rowEq, Bool.and_eq_true,
              decide_eq_true_eq] using checked
          unfold Owner.Matches
          rw [kindEq]
          exact ⟨indexed, rowEq, equalities.1, equalities.2⟩
  | phaseEnvelope =>
      simp only [ownerMatchesCheck, kindEq, Bool.and_eq_true] at checked
      unfold Owner.Matches
      rw [kindEq]
      exact ⟨of_decide_eq_true checked.1,
        of_decide_eq_true checked.2.1,
        of_decide_eq_true checked.2.2⟩

structure OwnerStepValid
    (arm : RawArm) (cursor : OwnerCursor) (owner : Owner) : Prop where
  rowStart : owner.rowStart = cursor.row
  rowPositive : owner.rowStart < owner.rowEnd
  sourceMatches : owner.Matches arm
  index : owner.index = cursor.index owner.kind

def ownerStepCheck
    (arm : RawArm) (cursor : OwnerCursor) (owner : Owner) : Bool :=
  decide (owner.rowStart = cursor.row) &&
    (decide (owner.rowStart < owner.rowEnd) &&
      (ownerMatchesCheck arm owner &&
        decide (owner.index = cursor.index owner.kind)))

theorem ownerStepCheck_sound
    {arm : RawArm} {cursor : OwnerCursor} {owner : Owner}
    (checked : ownerStepCheck arm cursor owner = true) :
    OwnerStepValid arm cursor owner := by
  simp only [ownerStepCheck, Bool.and_eq_true] at checked
  exact
    { rowStart := of_decide_eq_true checked.1
      rowPositive := of_decide_eq_true checked.2.1
      sourceMatches := ownerMatchesCheck_sound checked.2.2.1
      index := of_decide_eq_true checked.2.2.2 }

def runOwnerPrefix (arm : RawArm) :
    OwnerCursor → List Owner → Option OwnerCursor
  | cursor, [] => some cursor
  | cursor, owner :: rest =>
      match ownerStepCheck arm cursor owner with
      | true =>
          runOwnerPrefix arm
            (cursor.advance owner.kind owner.rowEnd) rest
      | false => none

inductive OwnerPrefixValid (arm : RawArm) :
    OwnerCursor → List Owner → OwnerCursor → Prop
  | nil (cursor : OwnerCursor) : OwnerPrefixValid arm cursor [] cursor
  | cons {cursor final : OwnerCursor} {owner : Owner} {rest : List Owner}
      (step : OwnerStepValid arm cursor owner)
      (tail : OwnerPrefixValid arm
        (cursor.advance owner.kind owner.rowEnd) rest final) :
      OwnerPrefixValid arm cursor (owner :: rest) final

theorem runOwnerPrefix_sound
    {arm : RawArm} {cursor final : OwnerCursor} {owners : List Owner}
    (checked : runOwnerPrefix arm cursor owners = some final) :
    OwnerPrefixValid arm cursor owners final := by
  induction owners generalizing cursor with
  | nil =>
      simp only [runOwnerPrefix, Option.some.injEq] at checked
      subst final
      exact .nil cursor
  | cons owner rest inductionHypothesis =>
      cases stepEq : ownerStepCheck arm cursor owner with
      | false => simp [runOwnerPrefix, stepEq] at checked
      | true =>
          simp only [runOwnerPrefix, stepEq] at checked
          exact .cons (ownerStepCheck_sound stepEq)
            (inductionHypothesis checked)

theorem runOwnerPrefix_append
    {arm : RawArm} {start middle final : OwnerCursor}
    {left right : List Owner}
    (leftChecked : runOwnerPrefix arm start left = some middle)
    (rightChecked : runOwnerPrefix arm middle right = some final) :
    runOwnerPrefix arm start (left ++ right) = some final := by
  induction left generalizing start middle with
  | nil =>
      simp only [runOwnerPrefix, Option.some.injEq] at leftChecked
      subst middle
      simpa using rightChecked
  | cons owner rest inductionHypothesis =>
      cases stepEq : ownerStepCheck arm start owner with
      | false => simp [runOwnerPrefix, stepEq] at leftChecked
      | true =>
          simp only [runOwnerPrefix, stepEq] at leftChecked
          simp only [List.cons_append, runOwnerPrefix, stepEq]
          exact inductionHypothesis leftChecked rightChecked

theorem runOwnerPrefix_of_take_drop
    {arm : RawArm} {start middle final : OwnerCursor}
    {owners : List Owner} {count : Nat}
    (headChecked :
      runOwnerPrefix arm start (owners.take count) = some middle)
    (tailChecked :
      runOwnerPrefix arm middle (owners.drop count) = some final) :
    runOwnerPrefix arm start owners = some final := by
  rw [← List.take_append_drop count owners]
  exact runOwnerPrefix_append headChecked tailChecked

theorem ownerIndices_cons_same
    {kind : OwnerKind} {owner : Owner} {rest : List Owner}
    (same : owner.kind = kind) :
    ownerIndices kind (owner :: rest) =
      owner.index :: ownerIndices kind rest := by
  simp [ownerIndices, same]

theorem ownerIndices_cons_of_ne
    {kind : OwnerKind} {owner : Owner} {rest : List Owner}
    (different : owner.kind ≠ kind) :
    ownerIndices kind (owner :: rest) = ownerIndices kind rest := by
  simp [ownerIndices, different]

inductive ConsecutiveFrom : Nat → List Nat → Prop
  | nil (start : Nat) : ConsecutiveFrom start []
  | cons (start : Nat) {rest : List Nat}
      (tail : ConsecutiveFrom (start + 1) rest) :
      ConsecutiveFrom start (start :: rest)

theorem ConsecutiveFrom.eq_range'
    {start : Nat} {indices : List Nat}
    (consecutive : ConsecutiveFrom start indices) :
    indices = List.range' start indices.length := by
  induction consecutive with
  | nil => rfl
  | cons start tail inductionHypothesis =>
      rw [List.length_cons, List.range'_succ]
      exact congrArg (List.cons start) inductionHypothesis

theorem OwnerPrefixValid.indicesConsecutive
    {arm : RawArm} {cursor final : OwnerCursor} {owners : List Owner}
    (valid : OwnerPrefixValid arm cursor owners final)
    (kind : OwnerKind) :
    ConsecutiveFrom (cursor.index kind) (ownerIndices kind owners) := by
  induction valid with
  | nil =>
      simp only [ownerIndices, List.filter_nil, List.map_nil]
      exact .nil _
  | @cons cursor final owner rest step tail inductionHypothesis =>
      by_cases same : owner.kind = kind
      · have ownerIndex : owner.index = cursor.index kind := by
          rw [← same]
          exact step.index
        rw [ownerIndices_cons_same same, ownerIndex]
        apply ConsecutiveFrom.cons
        have advanceIndex :
            (cursor.advance owner.kind owner.rowEnd).index kind =
              cursor.index kind + 1 := by
          rw [← same]
          exact OwnerCursor.advance_index_same cursor owner.kind owner.rowEnd
        simpa only [advanceIndex] using inductionHypothesis
      · rw [ownerIndices_cons_of_ne same]
        have advanceIndex := OwnerCursor.advance_index_of_ne cursor
          owner.kind kind owner.rowEnd same
        simpa only [advanceIndex] using inductionHypothesis

theorem OwnerPrefixValid.finalIndex_eq_start_add_length
    {arm : RawArm} {cursor final : OwnerCursor} {owners : List Owner}
    (valid : OwnerPrefixValid arm cursor owners final)
    (kind : OwnerKind) :
    final.index kind =
      cursor.index kind + (ownerIndices kind owners).length := by
  induction valid with
  | nil => simp [ownerIndices]
  | @cons cursor final owner rest step tail inductionHypothesis =>
      by_cases same : owner.kind = kind
      · rw [ownerIndices_cons_same same, List.length_cons]
        have tailCount := inductionHypothesis
        have advanceIndex :
            (cursor.advance owner.kind owner.rowEnd).index kind =
              cursor.index kind + 1 := by
          rw [← same]
          exact OwnerCursor.advance_index_same cursor owner.kind owner.rowEnd
        omega
      · rw [ownerIndices_cons_of_ne same]
        have tailCount := inductionHypothesis
        have advanceIndex := OwnerCursor.advance_index_of_ne cursor
          owner.kind kind owner.rowEnd same
        omega

theorem OwnerPrefixValid.ownerIndices_eq_range
    {arm : RawArm} {cursor final : OwnerCursor} {owners : List Owner}
    (valid : OwnerPrefixValid arm cursor owners final)
    (kind : OwnerKind) (target : Nat)
    (initial : cursor.index kind = 0)
    (ending : final.index kind = target) :
    ownerIndices kind owners = List.range target := by
  have consecutive := valid.indicesConsecutive kind
  have count := valid.finalIndex_eq_start_add_length kind
  have lengthEq : (ownerIndices kind owners).length = target := by omega
  calc
    ownerIndices kind owners =
        List.range' (cursor.index kind) (ownerIndices kind owners).length :=
      consecutive.eq_range'
    _ = List.range' 0 target := by rw [initial, lengthEq]
    _ = List.range target := List.range_eq_range'.symm

theorem OwnerPrefixValid.exactOwnerChain
    {arm : RawArm} {cursor final : OwnerCursor} {owners : List Owner}
    (valid : OwnerPrefixValid arm cursor owners final)
    (finalRow : final.row = arm.rowCount) :
    exactOwnerChainFrom arm cursor.row owners = true := by
  induction valid with
  | nil => simpa [exactOwnerChainFrom] using finalRow
  | @cons cursor final owner rest step tail inductionHypothesis =>
      have rowPositive : cursor.row < owner.rowEnd := by
        simpa only [← step.rowStart] using step.rowPositive
      have tailChain : exactOwnerChainFrom arm owner.rowEnd rest = true := by
        simpa only [OwnerCursor.advance_row] using
          inductionHypothesis finalRow
      simp [exactOwnerChainFrom, step.rowStart, rowPositive,
        step.sourceMatches, tailChain]

theorem ownershipValid_of_run
    {arm : RawArm}
    (checked : runOwnerPrefix arm (OwnerCursor.startFor arm) arm.owners =
      some (OwnerCursor.finalFor arm)) :
    arm.OwnershipValid := by
  have valid := runOwnerPrefix_sound checked
  unfold RawArm.OwnershipValid
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact valid.ownerIndices_eq_range .canonical arm.canonicalCalls.length
      (by rfl) (by rfl)
  · exact valid.ownerIndices_eq_range .poseidon2 arm.poseidon2Calls.length
      (by rfl) (by rfl)
  · exact valid.ownerIndices_eq_range .glue arm.glueRows.length
      (by rfl) (by rfl)
  · simpa using valid.ownerIndices_eq_range .phaseEnvelope 1
      (by rfl) (by rfl)
  · exact valid.exactOwnerChain (by rfl)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOwnershipCertificateSupport
