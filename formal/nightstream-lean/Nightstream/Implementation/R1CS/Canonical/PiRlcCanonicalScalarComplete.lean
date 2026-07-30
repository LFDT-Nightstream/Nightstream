import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesHonest

/-!
Contract: honest completeness of one complete Lean-owned `Pi_RLC` sampler
scalar after its canonical-u64 source bits have been materialized.

The candidate vector is reconstructed from those authoritative bits.  The
candidate witnesses are threaded in physical order, the selector's source
equalities are derived from that threaded assignment, and the first-accepted
selector witness is then constructed from the derived vector.

This module owns neither the upstream Poseidon2/canonical-u64 witness nor the
multi-coordinate threading.  Shortfall remains an explicit semantic
precondition: no satisfying selector witness is claimed when fewer than 54 of
the 64 candidates are accepted.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalScalarComplete

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateHonest
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateSound
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidates
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesHonest
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidatesSound
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelector
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSelectorHonest
open Nightstream.SuperNeo.Sampling
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

/-- Verifier-owned candidate reconstructed directly from one honest source-bit
slice. -/
def honestCandidate
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (candidateIndex : Fin candidatesPerScalar) :
    ProductionAlphabet.Chunk :=
  PiRlcCanonicalCandidateSound.candidate initial
    (candidateLayout duplexBase u64Base candidateBase initialBuilder
      coordinate candidateIndex)
    (sourceBits candidateIndex)

/-- Exact 64-candidate vector in physical source order. -/
def honestCandidates
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)) :
    List ProductionAlphabet.Chunk :=
  List.ofFn fun candidate =>
    honestCandidate duplexBase u64Base candidateBase initialBuilder coordinate
      initial sourceBits candidate

@[simp] theorem honestCandidates_length
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)) :
    (honestCandidates duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits).length =
        candidatesPerScalar := by
  simp [honestCandidates]

theorem honestCandidates_get
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (candidateIndex : Fin candidatesPerScalar) :
    (honestCandidates duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits).get
        ⟨candidateIndex.val, by simpa using candidateIndex.isLt⟩ =
      honestCandidate duplexBase u64Base candidateBase initialBuilder
        coordinate initial sourceBits candidateIndex := by
  simp [honestCandidates]

private theorem acceptValue_eq_verifier
    (initial : Nat → Nat) (layout : Layout)
    (sourceBits : SourceBitsBoolean initial layout) :
    acceptValue initial layout =
      if ProductionAlphabet.verifier.accepts
          (PiRlcCanonicalCandidateSound.candidate initial layout sourceBits)
        then 1 else 0 := by
  by_cases rejected :
      sourceValue initial layout = ProductionAlphabet.rejectionBucket
  · have notAccepted :
        ProductionAlphabet.verifier.accepts
            (PiRlcCanonicalCandidateSound.candidate initial layout sourceBits) =
          false := by
      apply Bool.eq_false_iff.mpr
      intro accepted
      have notRejected :=
        (ProductionAlphabet.accepts_eq_true_iff_ne_rejectionBucket _).mp
          accepted
      exact notRejected rejected
    simp [acceptValue, rejected, notAccepted]
  · have accepted :
        ProductionAlphabet.verifier.accepts
            (PiRlcCanonicalCandidateSound.candidate initial layout sourceBits) =
          true := by
      apply
        (ProductionAlphabet.accepts_eq_true_iff_ne_rejectionBucket _).mpr
      exact rejected
    simp [acceptValue, rejected, accepted]

private theorem residue_eq_verifier
    (initial : Nat → Nat) (layout : Layout)
    (sourceBits : SourceBitsBoolean initial layout) :
    residue initial layout =
      (ProductionAlphabet.verifier.symbol
        (PiRlcCanonicalCandidateSound.candidate initial layout sourceBits)).val :=
  rfl

/-- Independent accepted count of the reconstructed source prefix. -/
def honestPrefixCount
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (index : Nat) : Nat :=
  FirstAccepted.acceptedCount ProductionAlphabet.verifier
    ((honestCandidates duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits).take index)

theorem honestPrefixCount_succ
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (candidateIndex : Fin candidatesPerScalar) :
    honestPrefixCount duplexBase u64Base candidateBase initialBuilder
        coordinate initial sourceBits (candidateIndex.val + 1) =
      honestPrefixCount duplexBase u64Base candidateBase initialBuilder
          coordinate initial sourceBits candidateIndex.val +
        (if ProductionAlphabet.verifier.accepts
            (honestCandidate duplexBase u64Base candidateBase initialBuilder
              coordinate initial sourceBits candidateIndex)
          then 1 else 0) := by
  rw [honestPrefixCount, List.take_succ_eq_append_getElem (by
    simpa using candidateIndex.isLt)]
  cases decision :
      ProductionAlphabet.verifier.accepts
        (honestCandidate duplexBase u64Base candidateBase initialBuilder
          coordinate initial sourceBits candidateIndex) <;>
    simp [honestPrefixCount, FirstAccepted.acceptedCount,
      FirstAccepted.acceptedCandidates, honestCandidates, decision]

/-- The witness-side recurrence and the independent filtered-list count are
the same function on every bounded prefix. -/
theorem honestAcceptedPrefix_eq_count
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)) :
    ∀ index, index ≤ candidatesPerScalar →
      honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
          coordinate initial index =
        honestPrefixCount duplexBase u64Base candidateBase initialBuilder
          coordinate initial sourceBits index := by
  intro index bounded
  induction index with
  | zero =>
      simp [honestAcceptedPrefix, honestPrefixCount,
        FirstAccepted.acceptedCount, FirstAccepted.acceptedCandidates]
  | succ index hypothesis =>
      have indexLt : index < candidatesPerScalar := by omega
      let candidateIndex : Fin candidatesPerScalar := ⟨index, indexLt⟩
      have candidateEq : candidateOfNat index = candidateIndex := by
        apply Fin.ext
        exact candidateOfNat_val indexLt
      rw [honestAcceptedPrefix, hypothesis (by omega)]
      rw [candidateEq]
      have countSucc :=
        honestPrefixCount_succ duplexBase u64Base candidateBase initialBuilder
          coordinate initial sourceBits candidateIndex
      have accepted :=
        acceptValue_eq_verifier initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidateIndex)
          (sourceBits candidateIndex)
      simpa [candidateIndex, honestCandidate] using
        congrArg
          (fun value =>
            honestPrefixCount duplexBase u64Base candidateBase initialBuilder
                coordinate initial sourceBits index +
              value)
          accepted |>.trans countSucc.symm

/-- Final threaded candidate assignment at one scalar. -/
def candidateWitness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat) : Nat → Nat :=
  prefixWitness field duplexBase u64Base candidateBase initialBuilder
    coordinate initial candidatesPerScalar

theorem candidateWitness_accept
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourcesBelow :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidateIndex : Fin candidatesPerScalar) :
    candidateWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial
        (acceptColumn
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidateIndex)) =
      acceptValue initial
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidateIndex) := by
  unfold candidateWitness
  rw [prefixWitness_stable field duplexBase u64Base candidateBase
    initialBuilder coordinate initial
    (start := candidateIndex.val + 1) (finish := candidatesPerScalar)
    (by omega) (Nat.le_refl _) (by
      simp only [acceptColumn, candidateLayout, occurrenceBase,
        occurrenceIndex, prefixBoundary, candidatesPerScalar, auxiliaryCount]
      omega)]
  rw [prefixWitness_succ field duplexBase u64Base candidateBase
    initialBuilder coordinate initial candidateIndex.isLt]
  rw [candidateOfNat_eq candidateIndex, witness_accept]
  exact prefixWitness_acceptValue field duplexBase u64Base candidateBase
    initialBuilder coordinate initial candidateIndex.val sourcesBelow
    candidateIndex

theorem candidateWitness_residue
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourcesBelow :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidateIndex : Fin candidatesPerScalar) :
    candidateWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial
        (residueColumn
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidateIndex)) =
      residue initial
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidateIndex) := by
  unfold candidateWitness
  rw [prefixWitness_stable field duplexBase u64Base candidateBase
    initialBuilder coordinate initial
    (start := candidateIndex.val + 1) (finish := candidatesPerScalar)
    (by omega) (Nat.le_refl _) (by
      simp only [residueColumn, candidateLayout, occurrenceBase,
        occurrenceIndex, prefixBoundary, candidatesPerScalar, auxiliaryCount]
      omega)]
  rw [prefixWitness_succ field duplexBase u64Base candidateBase
    initialBuilder coordinate initial candidateIndex.isLt]
  rw [candidateOfNat_eq candidateIndex, witness_residue]
  unfold residue
  rw [prefixWitness_sourceValue field duplexBase u64Base candidateBase
    initialBuilder coordinate initial candidateIndex.val sourcesBelow
    candidateIndex]

theorem candidateWitness_prior
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourcesBelow :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidateIndex : Fin candidatesPerScalar) :
    lcEval
        (candidateWitness field duplexBase u64Base candidateBase
          initialBuilder coordinate initial)
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidateIndex).prior =
      honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
        coordinate initial candidateIndex.val := by
  unfold candidateWitness
  calc
    lcEval
        (prefixWitness field duplexBase u64Base candidateBase initialBuilder
          coordinate initial candidatesPerScalar)
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidateIndex).prior =
      lcEval
        (prefixWitness field duplexBase u64Base candidateBase initialBuilder
          coordinate initial candidateIndex.val)
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidateIndex).prior := by
        apply
          Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
        intro column mentioned
        apply prefixWitness_stable field duplexBase u64Base candidateBase
          initialBuilder coordinate initial
          (start := candidateIndex.val) (finish := candidatesPerScalar)
          (Nat.le_of_lt candidateIndex.isLt) (Nat.le_refl _)
        have below :=
          inputsBelowBase duplexBase u64Base candidateBase initialBuilder
            coordinate sourcesBelow candidateIndex
        unfold Mentions at mentioned
        rcases List.mem_map.mp mentioned with
          ⟨⟨termColumn, coefficient⟩, termMember, rfl⟩
        have termLt := below.prior termColumn coefficient termMember
        simp only [candidateLayout, occurrenceBase, occurrenceIndex,
          prefixBoundary, candidatesPerScalar, auxiliaryCount] at termLt ⊢
        omega
    _ =
      honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
        coordinate initial candidateIndex.val := by
        have exactPrior :=
          prefixWitness_prior_eval field duplexBase u64Base candidateBase
            initialBuilder coordinate initial sourcesBelow candidateIndex.val
            candidateIndex.isLt
        simpa [candidateOfNat_eq candidateIndex] using exactPrior

theorem candidateWitness_cumulative
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourcesBelow :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (candidateIndex : Fin candidatesPerScalar) :
    candidateWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial
        (cumulativeColumn
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidateIndex)) =
      honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
        coordinate initial (candidateIndex.val + 1) := by
  unfold candidateWitness
  rw [prefixWitness_stable field duplexBase u64Base candidateBase
    initialBuilder coordinate initial
    (start := candidateIndex.val + 1) (finish := candidatesPerScalar)
    (by omega) (Nat.le_refl _) (by
      simp only [cumulativeColumn, candidateLayout, occurrenceBase,
        occurrenceIndex, prefixBoundary, candidatesPerScalar, auxiliaryCount]
      omega)]
  rw [prefixWitness_succ field duplexBase u64Base candidateBase
    initialBuilder coordinate initial candidateIndex.isLt]
  rw [candidateOfNat_eq candidateIndex, witness_cumulative]
  unfold cumulative
  have priorExact :=
    prefixWitness_prior_eval field duplexBase u64Base candidateBase
      initialBuilder coordinate initial sourcesBelow candidateIndex.val
      candidateIndex.isLt
  have priorExact' :
      lcEval
          (prefixWitness field duplexBase u64Base candidateBase initialBuilder
            coordinate initial candidateIndex.val)
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidateIndex).prior =
        honestAcceptedPrefix duplexBase u64Base candidateBase initialBuilder
          coordinate initial candidateIndex.val := by
    simpa [candidateOfNat_eq candidateIndex] using priorExact
  rw [priorExact']
  rw [prefixWitness_acceptValue field duplexBase u64Base candidateBase
    initialBuilder coordinate initial candidateIndex.val sourcesBelow
    candidateIndex]
  rw [honestAcceptedPrefix, candidateOfNat_eq candidateIndex]

/-- The selector source contract is constructed from the candidate witness;
it is not a caller-supplied semantic conclusion. -/
theorem candidateSourcesMatch
    (field : FieldInverse)
    (duplexBase u64Base candidateBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourcesBelow :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate)) :
    SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
      (candidateWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial)
      (honestCandidates duplexBase u64Base candidateBase initialBuilder
        coordinate initial sourceBits) where
  lengthExact :=
    honestCandidates_length duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits
  accept candidateIndex := by
    rw [acceptSource, candidateSourceLayout,
      candidateWitness_accept field duplexBase u64Base candidateBase
        initialBuilder coordinate initial sourcesBelow candidateIndex]
    rw [honestCandidates_get duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits candidateIndex]
    exact acceptValue_eq_verifier initial
      (candidateLayout duplexBase u64Base candidateBase initialBuilder
        coordinate candidateIndex)
      (sourceBits candidateIndex)
  symbol candidateIndex := by
    rw [symbolSource, candidateSourceLayout,
      candidateWitness_residue field duplexBase u64Base candidateBase
        initialBuilder coordinate initial sourcesBelow candidateIndex]
    rw [honestCandidates_get duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits candidateIndex]
    exact residue_eq_verifier initial
      (candidateLayout duplexBase u64Base candidateBase initialBuilder
        coordinate candidateIndex)
      (sourceBits candidateIndex)
  prefixExact candidateIndex := by
    rw [prefixSource, candidateSourceLayout,
      candidateWitness_prior field duplexBase u64Base candidateBase
        initialBuilder coordinate initial sourcesBelow candidateIndex]
    exact honestAcceptedPrefix_eq_count duplexBase u64Base candidateBase
      initialBuilder coordinate initial sourceBits candidateIndex.val
      (Nat.le_of_lt candidateIndex.isLt)
  finalCount := by
    let last : Fin candidatesPerScalar := ⟨63, by decide⟩
    have lastExact :=
      candidateWitness_cumulative field duplexBase u64Base candidateBase
        initialBuilder coordinate initial sourcesBelow last
    have prefixExact :=
      honestAcceptedPrefix_eq_count duplexBase u64Base candidateBase
        initialBuilder coordinate initial sourceBits candidatesPerScalar
        (Nat.le_refl _)
    have lastEnd : last.val + 1 = candidatesPerScalar := by decide
    rw [lastEnd, prefixExact] at lastExact
    simpa [finalCountSource, candidateSourceLayout, honestPrefixCount, last,
      candidatesPerScalar] using lastExact

/-- Placing this scalar's selector after its complete candidate prefix
constructs the selector freshness contract. -/
theorem selectorSourcesBelow
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (candidateSources :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (ordered :
      prefixBoundary candidateBase coordinate candidatesPerScalar ≤
        scalarBase selectorBase coordinate) :
    PiRlcCanonicalSelectorHonest.SourcesBelow
      duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate where
  accept candidateIndex := by
    have candidateLt := candidateIndex.isLt
    change candidateIndex.val < 64 at candidateLt
    change
      candidateBase + (coordinate.val * 64 + 64) * 22 ≤
        selectorBase + coordinate.val * 2435 at ordered
    change
      candidateBase + (coordinate.val * 64 + candidateIndex.val) * 22 <
        selectorBase + coordinate.val * 2435
    omega
  symbol candidateIndex := by
    have candidateLt := candidateIndex.isLt
    change candidateIndex.val < 64 at candidateLt
    change
      candidateBase + (coordinate.val * 64 + 64) * 22 ≤
        selectorBase + coordinate.val * 2435 at ordered
    change
      candidateBase + (coordinate.val * 64 + candidateIndex.val) * 22 + 2 <
        selectorBase + coordinate.val * 2435
    omega
  prefixRead candidateIndex column mentioned := by
    have below :=
      inputsBelowBase duplexBase u64Base candidateBase initialBuilder
        coordinate candidateSources candidateIndex
    change
      Mentions
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidateIndex).prior column at mentioned
    unfold Mentions at mentioned
    rcases List.mem_map.mp mentioned with
      ⟨⟨termColumn, coefficient⟩, termMember, rfl⟩
    have termLt := below.prior termColumn coefficient termMember
    have candidateLt := candidateIndex.isLt
    change candidateIndex.val < 64 at candidateLt
    change
      termColumn <
        candidateBase +
          (coordinate.val * 64 + candidateIndex.val) * 22 at termLt
    change
      candidateBase + (coordinate.val * 64 + 64) * 22 ≤
        selectorBase + coordinate.val * 2435 at ordered
    change termColumn < selectorBase + coordinate.val * 2435
    omega
  finalCount := by
    change
      candidateBase + (coordinate.val * 64 + 64) * 22 ≤
        selectorBase + coordinate.val * 2435 at ordered
    change
      candidateBase + (coordinate.val * 64 + 63) * 22 + 21 <
        selectorBase + coordinate.val * 2435
    omega

/-- Candidate classification followed by first-accepted selection for one
scalar coordinate. -/
def scalarRows
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) : List Row :=
  PiRlcCanonicalCandidates.scalarRows duplexBase u64Base candidateBase
      initialBuilder coordinate ++
    PiRlcCanonicalSelector.scalarRows duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate

theorem scalarRows_length
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count) :
    (scalarRows duplexBase u64Base candidateBase selectorBase initialBuilder
      coordinate).length = 4198 := by
  unfold scalarRows
  rw [List.length_append,
    PiRlcCanonicalCandidates.scalarRows_length,
    PiRlcCanonicalSelector.scalarRows_length]
  decide

/-- Complete honest assignment for one candidate-plus-selector scalar. -/
def scalarWitness
    (field : FieldInverse)
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount
        (honestCandidates duplexBase u64Base candidateBase initialBuilder
          coordinate initial sourceBits)) :
    Nat → Nat :=
  PiRlcCanonicalSelectorHonest.scalarWitness
    duplexBase u64Base candidateBase selectorBase initialBuilder coordinate
    (candidateWitness field duplexBase u64Base candidateBase initialBuilder
      coordinate initial)
    (honestCandidates duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits)
    (honestCandidates_length duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits)
    enough

private theorem rowHolds_congr
    (left right : Nat → Nat) (row : Row)
    (agree :
      ∀ column,
        Mentions row.a column ∨ Mentions row.b column ∨
          Mentions row.c column →
        left column = right column) :
    RowHolds left row ↔ RowHolds right row := by
  unfold RowHolds
  rw [Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
      left right row.a (fun column member => agree column (Or.inl member)),
    Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
      left right row.b
      (fun column member => agree column (Or.inr (Or.inl member))),
    Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
      left right row.c
      (fun column member => agree column (Or.inr (Or.inr member)))]

private theorem candidateRows_preserved
    (field : FieldInverse)
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (positive : 0 < candidateBase)
    (constantWire : initial 0 = 1)
    (candidateSources :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount
        (honestCandidates duplexBase u64Base candidateBase initialBuilder
          coordinate initial sourceBits))
    (ordered :
      prefixBoundary candidateBase coordinate candidatesPerScalar ≤
        scalarBase selectorBase coordinate) :
    Satisfies
      (PiRlcCanonicalCandidates.scalarRows duplexBase u64Base candidateBase
        initialBuilder coordinate)
      (scalarWitness field duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate initial sourceBits enough) := by
  have candidateSatisfied :=
    PiRlcCanonicalCandidatesHonest.scalarRows_complete field duplexBase
      u64Base candidateBase initialBuilder coordinate initial positive
      constantWire
      candidateSources sourceBits
  intro row member
  apply
    (rowHolds_congr
      (candidateWitness field duplexBase u64Base candidateBase initialBuilder
        coordinate initial)
      (scalarWitness field duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate initial sourceBits enough)
      row ?_).mp
  · exact candidateSatisfied row member
  · intro column mentioned
    symm
    apply PiRlcCanonicalSelectorHonest.scalarWitness_before
    rcases List.mem_flatMap.mp member with
      ⟨candidateIndex, _, localMember⟩
    have localBound :=
      PiRlcCanonicalCandidatesHonest.candidateRows_mentions_lt
        (candidateLayout duplexBase u64Base candidateBase initialBuilder
          coordinate candidateIndex)
        (by
          simp only [candidateLayout, occurrenceBase, occurrenceIndex]
          omega)
        (inputsBelowBase duplexBase u64Base candidateBase initialBuilder
          coordinate candidateSources candidateIndex)
        row localMember column mentioned
    simp only [candidateLayout, occurrenceBase, occurrenceIndex,
      prefixBoundary, scalarBase, candidatesPerScalar, auxiliaryCount,
      scalarAuxiliaryCount] at localBound ordered ⊢
    have candidateLt := candidateIndex.isLt
    omega

/-- Active honest completeness of one complete scalar sampler.  `Enough` is
the exact named no-shortfall condition; all routes and row values are derived
from the authoritative source bits. -/
theorem complete
    (field : FieldInverse)
    (duplexBase u64Base candidateBase selectorBase : Nat)
    (initialBuilder : SymbolicDuplex.Builder)
    {count : Nat} (coordinate : Fin count)
    (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (positive : 0 < candidateBase)
    (candidateSources :
      PiRlcCanonicalCandidatesHonest.SourcesBelow
        duplexBase u64Base candidateBase initialBuilder coordinate)
    (sourceBits :
      ∀ candidate : Fin candidatesPerScalar,
        SourceBitsBoolean initial
          (candidateLayout duplexBase u64Base candidateBase initialBuilder
            coordinate candidate))
    (enough :
      FirstAccepted.Enough ProductionAlphabet.verifier outputCount
        (honestCandidates duplexBase u64Base candidateBase initialBuilder
          coordinate initial sourceBits))
    (ordered :
      prefixBoundary candidateBase coordinate candidatesPerScalar ≤
        scalarBase selectorBase coordinate) :
    Satisfies
      (scalarRows duplexBase u64Base candidateBase selectorBase initialBuilder
        coordinate)
      (scalarWitness field duplexBase u64Base candidateBase selectorBase
        initialBuilder coordinate initial sourceBits enough) := by
  let candidateAssignment :=
    candidateWitness field duplexBase u64Base candidateBase initialBuilder
      coordinate initial
  let candidates :=
    honestCandidates duplexBase u64Base candidateBase initialBuilder
      coordinate initial sourceBits
  have candidateCanonical :
      ∀ column, candidateAssignment column < goldilocksP := by
    exact prefixWitness_canonical field duplexBase u64Base candidateBase
      initialBuilder coordinate initial initialCanonical candidateSources
      sourceBits candidatesPerScalar (Nat.le_refl _)
  have candidateConstant : candidateAssignment 0 = 1 := by
    exact prefixWitness_constant field duplexBase u64Base candidateBase
      initialBuilder coordinate initial candidatesPerScalar positive
      constantWire
  have sources :
      SourcesMatch duplexBase u64Base candidateBase initialBuilder coordinate
        candidateAssignment candidates := by
    exact candidateSourcesMatch field duplexBase u64Base candidateBase
      initialBuilder coordinate initial candidateSources sourceBits
  have selectorBelow :=
    selectorSourcesBelow duplexBase u64Base candidateBase selectorBase
      initialBuilder coordinate candidateSources ordered
  have selectorPositive : 0 < scalarBase selectorBase coordinate := by
    unfold prefixBoundary at ordered
    omega
  have selectorSatisfied :=
    PiRlcCanonicalSelectorComplete.scalarRows_complete duplexBase u64Base
      candidateBase selectorBase initialBuilder coordinate candidateAssignment
      candidateCanonical candidateConstant candidates sources enough
      selectorBelow selectorPositive
  intro row member
  unfold scalarRows at member
  rcases List.mem_append.mp member with candidateMember | selectorMember
  · exact candidateRows_preserved field duplexBase u64Base candidateBase
      selectorBase initialBuilder coordinate initial positive constantWire
      candidateSources sourceBits enough ordered row candidateMember
  · exact selectorSatisfied row selectorMember

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalScalarComplete
