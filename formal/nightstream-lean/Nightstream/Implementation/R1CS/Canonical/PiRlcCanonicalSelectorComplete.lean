import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorHonest

/-!
Contract: active honest completeness of one Lean-owned 54-of-64 selector
scalar.

The witness is constructed from an authoritative 64-candidate list.  This file
proves that the six acceptance-bound rows and all 54 position blocks hold.
It does not assume row satisfaction, a chosen route, or a desired output.

Candidate construction and multi-scalar witness threading are separate proof
responsibilities.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorComplete

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorHonest
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

private theorem range4 :
    List.range slackBitCount = [0, 1, 2, 3] := by
  decide

private theorem finRange11 :
    List.finRange selectionWindow =
      [⟨0, by decide⟩, ⟨1, by decide⟩, ⟨2, by decide⟩,
        ⟨3, by decide⟩, ⟨4, by decide⟩, ⟨5, by decide⟩,
        ⟨6, by decide⟩, ⟨7, by decide⟩, ⟨8, by decide⟩,
        ⟨9, by decide⟩, ⟨10, by decide⟩] := by
  decide

private theorem one_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1) :
    lcEval assignment [(0, 1)] = 1 := by
  simp [lcEval, constantWire, goldilocksP]

private theorem singleton_eval
    (assignment : Nat → Nat) (column : Nat)
    (canonical : assignment column < goldilocksP) :
    lcEval assignment [(column, 1)] = assignment column := by
  simp [lcEval, Nat.mod_eq_of_lt canonical]

private theorem bitRow_complete
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (column : Nat) (bounded : assignment column ≤ 1) :
    RowHolds assignment (bitRow column) := by
  have cases : assignment column = 0 ∨ assignment column = 1 := by
    omega
  rcases cases with zero | one
  · simp [RowHolds, bitRow, lcEval, constantWire, zero]
  · simp [RowHolds, bitRow, lcEval, constantWire, one, goldilocksP]

theorem slack_recomposes
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar) :
    (List.range slackBitCount).foldl
        (fun value offset =>
          value + 2 ^ offset * slackBit candidates offset)
        0 =
      slackValue candidates := by
  have bound := slackValue_le_ten candidates lengthExact
  unfold slackBit
  rw [range4]
  simp only [List.foldl, Nat.zero_add, Nat.pow_zero, Nat.one_mul]
  omega

private theorem slackTerms_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough)
        (slackTerms selectorBase coordinate) =
      slackValue candidates := by
  have raw :
      (slackTerms selectorBase coordinate).foldl
          (fun value term =>
            value + term.2 *
              scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates lengthExact
                  enough term.1)
          0 =
        slackValue candidates := by
    unfold slackTerms
    rw [range4]
    simp only [List.map, List.foldl]
    rw [scalarWitness_slackBit duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough
      (offset := 0) (by decide)]
    rw [scalarWitness_slackBit duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough
      (offset := 1) (by decide)]
    rw [scalarWitness_slackBit duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough
      (offset := 2) (by decide)]
    rw [scalarWitness_slackBit duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough
      (offset := 3) (by decide)]
    simpa only [range4, List.foldl] using
      slack_recomposes candidates lengthExact
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt (slackValue_le_ten candidates lengthExact)
    (by decide)

private theorem acceptanceBoundRows_complete
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (candidates : List ProductionAlphabet.Chunk)
    (sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        assignment candidates)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (positive : 0 < scalarBase selectorBase coordinate) :
    Satisfies
      (acceptanceBoundRows duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates sources.lengthExact
          enough) := by
  let z :=
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates sources.lengthExact enough
  have one : z 0 = 1 :=
    scalarWitness_constant duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates sources.lengthExact
        enough positive constantWire
  have preserved := sources.preserved duplexBase u64Base candidateBase
    selectorBase initialBuilder coordinate assignment candidates enough below
  intro row member
  simp only [acceptanceBoundRows, List.mem_append] at member
  rcases member with bitMember | tailMember
  · rcases List.mem_map.mp bitMember with ⟨offset, inRange, rfl⟩
    apply bitRow_complete z one
    change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates sources.lengthExact
            enough
          (slackBitColumn selectorBase coordinate offset) ≤
        1
    rw [scalarWitness_slackBit duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates sources.lengthExact
      enough (List.mem_range.mp inRange)]
    exact slackBit_le_one candidates offset
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at tailMember
    rcases tailMember with rfl | rfl
    · have slackLt :
          slackValue candidates < goldilocksP :=
        Nat.lt_of_le_of_lt
          (slackValue_le_ten candidates sources.lengthExact)
          (by decide)
      have slackEval :
          lcEval z [(slackColumn selectorBase coordinate, 1)] =
            slackValue candidates := by
        rw [singleton_eval z _]
        · exact scalarWitness_slack duplexBase u64Base candidateBase
            selectorBase initialBuilder coordinate assignment candidates
            sources.lengthExact enough
        · change
            scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates
                  sources.lengthExact enough
                (slackColumn selectorBase coordinate) <
              goldilocksP
          rw [scalarWitness_slack duplexBase u64Base candidateBase
            selectorBase initialBuilder coordinate assignment candidates
            sources.lengthExact enough]
          exact slackLt
      simp only [RowHolds]
      rw [slackEval, one_eval z one,
        slackTerms_witness_eval duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates sources.lengthExact
            enough]
      simp [Nat.mod_eq_of_lt slackLt]
    · have acceptedBound :=
        acceptedCount_le_length ProductionAlphabet.verifier candidates
      have exactLength : candidates.length = 64 := by
        simpa [PiRlcCanonicalCandidates.candidatesPerScalar] using
          sources.lengthExact
      rw [exactLength] at acceptedBound
      have finalLt :
          FirstAccepted.acceptedCount ProductionAlphabet.verifier candidates <
            goldilocksP :=
        Nat.lt_of_le_of_lt acceptedBound (by decide)
      have slackBound := slackValue_le_ten candidates sources.lengthExact
      have sumLt :
          slackValue candidates + outputCount < goldilocksP := by
        simp only [outputCount]
        have modulus : 64 < goldilocksP := by decide
        omega
      have finalEval :
          lcEval z
              [(finalCountSource duplexBase u64Base candidateBase
                initialBuilder coordinate, 1)] =
            FirstAccepted.acceptedCount ProductionAlphabet.verifier
              candidates := by
        rw [singleton_eval z _]
        · exact preserved.finalCount
        · change
            scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates
                  sources.lengthExact enough
                (finalCountSource duplexBase u64Base candidateBase
                  initialBuilder coordinate) <
              goldilocksP
          rw [scalarWitness_finalCountSource duplexBase u64Base candidateBase
            selectorBase initialBuilder coordinate assignment candidates
            sources.lengthExact enough below]
          rw [sources.finalCount]
          exact finalLt
      have rightEval :
          lcEval z
              [(slackColumn selectorBase coordinate, 1), (0, outputCount)] =
            slackValue candidates + outputCount := by
        have slackAt :
            z (slackColumn selectorBase coordinate) =
              slackValue candidates := by
          change
            scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates
                  sources.lengthExact enough
                (slackColumn selectorBase coordinate) =
              slackValue candidates
          exact scalarWitness_slack duplexBase u64Base candidateBase
            selectorBase initialBuilder coordinate assignment candidates
            sources.lengthExact enough
        unfold lcEval
        simp only [List.foldl, Nat.one_mul, Nat.mul_one, Nat.zero_add,
          slackAt, one]
        rw [Nat.mod_eq_of_lt sumLt]
      simp only [RowHolds]
      rw [finalEval, one_eval z one, rightEval]
      unfold FirstAccepted.Enough at enough
      unfold slackValue
      have countLe := acceptedBound
      simp only [outputCount] at enough ⊢
      have exactDifference :
          FirstAccepted.acceptedCount ProductionAlphabet.verifier candidates =
            (FirstAccepted.acceptedCount ProductionAlphabet.verifier
                candidates -
              54) +
              54 := by
        omega
      rw [Nat.mul_one, Nat.mod_eq_of_lt finalLt]
      exact exactDifference

private theorem sum_ite_zero_of_not_mem
    {α : Type} [DecidableEq α] (items : List α) (target : α)
    (value : α → Nat) (absent : target ∉ items) :
    (items.map
      (fun item => if item = target then value item else 0)).sum = 0 := by
  induction items with
  | nil => rfl
  | cons head tail hypothesis =>
      have headNe : head ≠ target := by
        intro same
        exact absent (List.mem_cons.2 (Or.inl same.symm))
      have tailAbsent : target ∉ tail := by
        intro member
        exact absent (List.mem_cons_of_mem _ member)
      simp [headNe, hypothesis tailAbsent]

private theorem sum_ite_single
    {α : Type} [DecidableEq α] (items : List α) (target : α)
    (value : α → Nat) (nodup : items.Nodup) (member : target ∈ items) :
    (items.map
      (fun item => if item = target then value item else 0)).sum =
      value target := by
  induction items with
  | nil => simp at member
  | cons head tail hypothesis =>
      rw [List.nodup_cons] at nodup
      rcases List.mem_cons.1 member with same | inTail
      · subst target
        simp [sum_ite_zero_of_not_mem tail head value nodup.1]
      · have headNe : head ≠ target := by
          intro same
          exact nodup.1 (same ▸ inTail)
        simp [headNe, hypothesis nodup.2 inTail]

private theorem foldl_ite_single
    {α : Type} [DecidableEq α] (items : List α) (target : α)
    (value : α → Nat) (nodup : items.Nodup) (member : target ∈ items) :
    items.foldl
        (fun total item =>
          total + if item = target then value item else 0)
        0 =
      value target := by
  rw [← List.foldl_map, ← List.sum_eq_foldl]
  exact sum_ite_single items target value nodup member

theorem selector_weighted_sum
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount)
    (value : Fin selectionWindow → Nat)
    (canonical : ∀ offset, value offset < goldilocksP) :
    (List.finRange selectionWindow).foldl
        (fun total offset =>
          total +
            selectorValue candidates lengthExact enough position offset *
              value offset % goldilocksP)
        0 =
      value (selectedOffset candidates lengthExact enough position) := by
  have pointwise :
      (fun offset : Fin selectionWindow =>
        selectorValue candidates lengthExact enough position offset *
            value offset %
          goldilocksP) =
      (fun offset =>
        if offset =
            selectedOffset candidates lengthExact enough position
        then value offset else 0) := by
    funext offset
    by_cases selected :
        offset = selectedOffset candidates lengthExact enough position
    · subst offset
      simp only [selectorValue_selected, Nat.one_mul, ↓reduceIte]
      exact Nat.mod_eq_of_lt (canonical _)
    · simp [selectorValue, selected]
  have stepEq :
      (fun total offset =>
        total +
          selectorValue candidates lengthExact enough position offset *
              value offset %
            goldilocksP) =
      (fun total offset =>
        total +
          if offset =
              selectedOffset candidates lengthExact enough position
          then value offset else 0) := by
    funext total offset
    exact congrArg (fun term => total + term) (congrFun pointwise offset)
  rw [stepEq]
  exact foldl_ite_single (List.finRange selectionWindow)
    (selectedOffset candidates lengthExact enough position) value
    (by decide) (List.mem_finRange _)

private theorem selectorTerms_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough)
        (selectorTerms selectorBase coordinate position) =
      1 := by
  have raw :
      (selectorTerms selectorBase coordinate position).foldl
          (fun total term =>
            total + term.2 *
              scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates lengthExact
                  enough term.1)
          0 =
        1 := by
    unfold selectorTerms
    rw [List.foldl_map]
    simp only [Function.comp_apply, Nat.one_mul, scalarWitness_selector]
    have stepEq :
        (fun total offset =>
          total +
            selectorValue candidates lengthExact enough position offset) =
        (fun total offset =>
          total +
            if offset =
                selectedOffset candidates lengthExact enough position
            then 1 else 0) := by
      funext total offset
      by_cases selected :
          offset = selectedOffset candidates lengthExact enough position
      · subst offset
        simp only [selectorValue_selected, ↓reduceIte]
      · simp [selectorValue, selected]
    rw [stepEq]
    exact foldl_ite_single (List.finRange selectionWindow)
      (selectedOffset candidates lengthExact enough position) (fun _ => 1)
      (by decide) (List.mem_finRange _)
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt (by decide : 1 < goldilocksP)]

private theorem oneHotRows_complete
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (constantWire : assignment 0 = 1)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (positive : 0 < scalarBase selectorBase coordinate)
    (position : Fin outputCount) :
    Satisfies (oneHotRows selectorBase coordinate position)
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates lengthExact enough) := by
  let z :=
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates lengthExact enough
  have one : z 0 = 1 :=
    scalarWitness_constant duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough
        positive constantWire
  intro row member
  simp only [oneHotRows, List.mem_append] at member
  rcases member with bitMember | sumMember
  · rcases List.mem_map.mp bitMember with ⟨offset, _, rfl⟩
    apply bitRow_complete z one
    change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough
          (selectorColumn selectorBase coordinate position offset) ≤
        1
    rw [scalarWitness_selector duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough]
    exact selectorValue_le_one candidates lengthExact enough position offset
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at sumMember
    subst row
    simp only [RowHolds]
    rw [selectorTerms_witness_eval duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough position, one_eval z one]
    decide

private theorem routedProduct_complete
    (z : Nat → Nat) (selector source : LinComb) (target : Nat)
    (selectorValue sourceValue : Nat)
    (selectorEval : lcEval z selector = selectorValue)
    (sourceEval : lcEval z source = sourceValue)
    (targetAt :
      z target = selectorValue * sourceValue % goldilocksP) :
    RowHolds z ⟨selector, source, [(target, 1)]⟩ := by
  have targetLt :
      selectorValue * sourceValue % goldilocksP < goldilocksP :=
    Nat.mod_lt _ (by decide)
  have targetEval :
      lcEval z [(target, 1)] =
        selectorValue * sourceValue % goldilocksP := by
    rw [singleton_eval z target]
    · exact targetAt
    · rw [targetAt]
      exact targetLt
  unfold RowHolds
  rw [selectorEval, sourceEval, targetEval]

private theorem selector_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough)
        [(selectorColumn selectorBase coordinate position offset, 1)] =
      selectorValue candidates lengthExact enough position offset := by
  let z :=
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates lengthExact enough
  rw [singleton_eval z _]
  · exact scalarWitness_selector duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough
      position offset
  · change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough
          (selectorColumn selectorBase coordinate position offset) <
        goldilocksP
    rw [scalarWitness_selector duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough]
    exact Nat.lt_of_le_of_lt
      (selectorValue_le_one candidates lengthExact enough position offset)
      (by decide)

private theorem symbolSource_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough)
        [(symbolSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate, 1)] =
      assignment
        (symbolSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) := by
  let z :=
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates lengthExact enough
  rw [singleton_eval z _]
  · exact scalarWitness_symbolSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough below candidate
  · change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough
          (symbolSource duplexBase u64Base candidateBase initialBuilder
            coordinate candidate) <
        goldilocksP
    rw [scalarWitness_symbolSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough below candidate]
    exact canonical _

private theorem acceptSource_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (candidate : Fin PiRlcCanonicalCandidates.candidatesPerScalar) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough)
        [(acceptSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate, 1)] =
      assignment
        (acceptSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate) := by
  let z :=
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates lengthExact enough
  rw [singleton_eval z _]
  · exact scalarWitness_acceptSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough below candidate
  · change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough
          (acceptSource duplexBase u64Base candidateBase initialBuilder
            coordinate candidate) <
        goldilocksP
    rw [scalarWitness_acceptSource duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough below candidate]
    exact canonical _

private theorem productRowsAt_complete
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (position : Fin outputCount) (offset : Fin selectionWindow) :
    Satisfies
      (productRowsAt duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate position offset)
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates lengthExact enough) := by
  let z :=
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates lengthExact enough
  let candidate := candidateAt position offset
  have selectorEval :
      lcEval z [(selectorColumn selectorBase coordinate position offset, 1)] =
        selectorValue candidates lengthExact enough position offset :=
    selector_witness_eval duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates lengthExact enough
      position offset
  intro row member
  simp only [productRowsAt, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · apply routedProduct_complete z _ _ _
      (selectorValue candidates lengthExact enough position offset)
      (assignment
        (symbolSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate))
      selectorEval
      (symbolSource_witness_eval duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment canonical candidates lengthExact
        enough below candidate)
    change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough
          (symbolProductColumn selectorBase coordinate position offset) =
        _
    exact scalarWitness_symbolProduct duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough position offset
  · apply routedProduct_complete z _ _ _
      (selectorValue candidates lengthExact enough position offset)
      (assignment
        (acceptSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate))
      selectorEval
      (acceptSource_witness_eval duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment canonical candidates lengthExact
        enough below candidate)
    change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough
          (acceptProductColumn selectorBase coordinate position offset) =
        _
    exact scalarWitness_acceptProduct duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough position offset
  · apply routedProduct_complete z _ _ _
      (selectorValue candidates lengthExact enough position offset)
      (lcEval assignment
        (prefixSource duplexBase u64Base candidateBase initialBuilder
          coordinate candidate))
      selectorEval
      (scalarWitness_prefixSource duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates lengthExact enough
        below candidate)
    change
      scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough
          (prefixProductColumn selectorBase coordinate position offset) =
        _
    exact scalarWitness_prefixProduct duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates lengthExact
      enough position offset

private theorem productRows_complete
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (position : Fin outputCount) :
    Satisfies
      (productRows duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate position)
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates lengthExact enough) := by
  intro row member
  rcases List.mem_flatMap.mp member with
    ⟨offset, _, rowMember⟩
  exact productRowsAt_complete duplexBase u64Base candidateBase selectorBase
    initialBuilder coordinate assignment canonical candidates lengthExact
    enough below position offset row rowMember

private theorem symbolProductTerms_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (lengthExact :
      candidates.length = PiRlcCanonicalCandidates.candidatesPerScalar)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates lengthExact enough)
        (symbolProductTerms selectorBase coordinate position) =
      assignment
        (symbolSource duplexBase u64Base candidateBase initialBuilder
          coordinate
          (selectedCandidate candidates lengthExact enough position)) := by
  have weighted :=
    selector_weighted_sum candidates lengthExact enough position
      (fun offset =>
        assignment
          (symbolSource duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateAt position offset)))
      (fun offset => canonical _)
  have raw :
      (symbolProductTerms selectorBase coordinate position).foldl
          (fun total term =>
            total + term.2 *
              scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates lengthExact
                  enough term.1)
          0 =
        assignment
          (symbolSource duplexBase u64Base candidateBase initialBuilder
            coordinate
            (selectedCandidate candidates lengthExact enough position)) := by
    unfold symbolProductTerms
    rw [List.foldl_map]
    simp only [Function.comp_apply, Nat.one_mul, scalarWitness_symbolProduct]
    simpa [selectedCandidate] using weighted
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt (canonical _)]

private theorem acceptProductTerms_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (candidates : List ProductionAlphabet.Chunk)
    (sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        assignment candidates)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates sources.lengthExact
            enough)
        (acceptProductTerms selectorBase coordinate position) =
      1 := by
  have selectedAccepted :
      assignment
          (acceptSource duplexBase u64Base candidateBase initialBuilder
            coordinate
            (selectedCandidate candidates sources.lengthExact enough
              position)) =
        1 := by
    rw [sources.accept]
    simp only [(selectedCandidate_spec candidates sources.lengthExact enough
      position).1, ↓reduceIte]
  have weighted :=
    selector_weighted_sum candidates sources.lengthExact enough position
      (fun offset =>
        assignment
          (acceptSource duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateAt position offset)))
      (fun offset => canonical _)
  have raw :
      (acceptProductTerms selectorBase coordinate position).foldl
          (fun total term =>
            total + term.2 *
              scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates
                  sources.lengthExact enough term.1)
          0 =
        1 := by
    unfold acceptProductTerms
    rw [List.foldl_map]
    simp only [Function.comp_apply, Nat.one_mul, scalarWitness_acceptProduct]
    rw [weighted]
    simpa [selectedCandidate] using selectedAccepted
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt (by decide : 1 < goldilocksP)]

private theorem prefixProductTerms_witness_eval
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (candidates : List ProductionAlphabet.Chunk)
    (sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        assignment candidates)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (position : Fin outputCount) :
    lcEval
        (scalarWitness duplexBase u64Base candidateBase selectorBase
          initialBuilder coordinate assignment candidates sources.lengthExact
            enough)
        (prefixProductTerms selectorBase coordinate position) =
      position.val := by
  have selectedPrefix :
      lcEval assignment
          (prefixSource duplexBase u64Base candidateBase initialBuilder
            coordinate
            (selectedCandidate candidates sources.lengthExact enough
              position)) =
        position.val := by
    exact (sources.prefixExact
      (selectedCandidate candidates sources.lengthExact enough position)).trans
      (selectedCandidate_spec candidates sources.lengthExact enough position).2
  have weighted :=
    selector_weighted_sum candidates sources.lengthExact enough position
      (fun offset =>
        lcEval assignment
          (prefixSource duplexBase u64Base candidateBase initialBuilder
            coordinate (candidateAt position offset)))
      (fun offset => Nat.mod_lt _ (by decide))
  have raw :
      (prefixProductTerms selectorBase coordinate position).foldl
          (fun total term =>
            total + term.2 *
              scalarWitness duplexBase u64Base candidateBase selectorBase
                initialBuilder coordinate assignment candidates
                  sources.lengthExact enough term.1)
          0 =
        position.val := by
    unfold prefixProductTerms
    rw [List.foldl_map]
    simp only [Function.comp_apply, Nat.one_mul, scalarWitness_prefixProduct]
    rw [weighted]
    simpa [selectedCandidate] using selectedPrefix
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_trans position.isLt (by decide)

private theorem positionTerms_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (position : Fin outputCount) :
    lcEval assignment (positionTerms position) = position.val := by
  unfold positionTerms
  split
  · rename_i zero
    simp [lcEval, zero]
  · simp only [lcEval, List.foldl, Nat.zero_add, constantWire, Nat.mul_one]
    rw [Nat.mod_eq_of_lt]
    exact Nat.lt_trans position.isLt (by decide)

private theorem bindingRows_complete
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (candidates : List ProductionAlphabet.Chunk)
    (sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        assignment candidates)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (positive : 0 < scalarBase selectorBase coordinate)
    (position : Fin outputCount) :
    Satisfies (bindingRows selectorBase coordinate position)
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates sources.lengthExact
          enough) := by
  let z :=
    scalarWitness duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate assignment candidates sources.lengthExact enough
  have one : z 0 = 1 :=
    scalarWitness_constant duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment candidates sources.lengthExact
        enough positive constantWire
  have outputEval :
      lcEval z [(outputColumn selectorBase coordinate position, 1)] =
        (assignment
            (symbolSource duplexBase u64Base candidateBase initialBuilder
              coordinate
              (selectedCandidate candidates sources.lengthExact enough
                position)) +
          (goldilocksP - 2)) % goldilocksP := by
    rw [singleton_eval z _]
    · exact scalarWitness_output duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates sources.lengthExact
        enough position
    · change
        scalarWitness duplexBase u64Base candidateBase selectorBase
            initialBuilder coordinate assignment candidates sources.lengthExact
              enough
            (outputColumn selectorBase coordinate position) <
          goldilocksP
      rw [scalarWitness_output duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates sources.lengthExact
        enough position]
      exact Nat.mod_lt _ (by decide)
  have centeredEval :
      lcEval z (centeredSymbolTerms selectorBase coordinate position) =
        (assignment
            (symbolSource duplexBase u64Base candidateBase initialBuilder
              coordinate
              (selectedCandidate candidates sources.lengthExact enough
                position)) +
          (goldilocksP - 2)) % goldilocksP := by
    rw [centeredSymbolTerms, KHorner.lcEval_append,
      symbolProductTerms_witness_eval duplexBase u64Base candidateBase
        selectorBase initialBuilder coordinate assignment canonical candidates
        sources.lengthExact enough position]
    simp [lcEval, one, goldilocksP]
  intro row member
  simp only [bindingRows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl
  · simp only [RowHolds]
    rw [acceptProductTerms_witness_eval duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment canonical candidates
      sources enough position, one_eval z one]
    decide
  · simp only [RowHolds]
    rw [prefixProductTerms_witness_eval duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment candidates sources
      enough position, one_eval z one, positionTerms_eval z one position]
    rw [Nat.mul_one, Nat.mod_eq_of_lt]
    exact Nat.lt_trans position.isLt (by decide)
  · simp only [RowHolds]
    rw [outputEval, one_eval z one, centeredEval]
    simp

private theorem positionRows_complete
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (candidates : List ProductionAlphabet.Chunk)
    (sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        assignment candidates)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (positive : 0 < scalarBase selectorBase coordinate)
    (position : Fin outputCount) :
    Satisfies
      (positionRows duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate position)
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates sources.lengthExact
          enough) := by
  intro row member
  simp only [positionRows, List.mem_append] at member
  rcases member with (oneHotMember | productMember) | bindingMember
  · exact oneHotRows_complete duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment constantWire candidates
      sources.lengthExact enough positive position row oneHotMember
  · exact productRows_complete duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment canonical candidates
      sources.lengthExact enough below position row productMember
  · exact bindingRows_complete duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment canonical constantWire candidates
      sources enough positive position row bindingMember

/-- Active honest completeness of one complete selector scalar.  The selected
route and every allocated value are computed from the authoritative candidate
list; row satisfaction is a conclusion, never a premise. -/
theorem scalarRows_complete
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (candidates : List ProductionAlphabet.Chunk)
    (sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        assignment candidates)
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount candidates)
    (below :
      SourcesBelow duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate)
    (positive : 0 < scalarBase selectorBase coordinate) :
    Satisfies
      (scalarRows duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate)
      (scalarWitness duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate assignment candidates sources.lengthExact
          enough) := by
  intro row member
  simp only [scalarRows, List.mem_append] at member
  rcases member with acceptanceMember | positionMember
  · exact acceptanceBoundRows_complete duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate assignment constantWire candidates
      sources enough below positive row acceptanceMember
  · rcases List.mem_flatMap.mp positionMember with
      ⟨position, _, rowMember⟩
    exact positionRows_complete duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate assignment canonical constantWire candidates
      sources enough below positive position row rowMember

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorComplete
