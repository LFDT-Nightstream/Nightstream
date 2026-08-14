import Nightstream.Implementation.Nebula.Production.Memory.CheckedStepRows
import Nightstream.Protocol.Nebula.ProductionBatchedFPrime

/-!
Contract: exact field-native row program for one production checked-step
batch.

One candidate-specific batch has `E + 1` decoded carry boundaries and `E`
checked memory steps. Each checked step uses the exact adjacent boundary
layouts. Satisfying rows derive every typed boundary, every typed suffix, and
one ordered `ConsumesList` proof across all internal boundaries.

No batch transition, suffix list, carry list, source list, product-update
list, or decoder result is a premise of `derive`.

Does not own absolute generated columns, application-port routing, the
enclosing full-claim carrier, NIFS verification, state hashing, or Rust
refinement.

Emits constraints: yes.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows

open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.Protocol.Nebula.ProductionBatchedFPrime
open Nightstream.Protocol.Nebula.ProductionProfileCandidates
open Nightstream.SuperNeo.Concrete

abbrev StepCount (candidate : Id) : Nat :=
  checkedStepsPerFreshClaim candidate

/-- Candidate-specific physical layouts. Boundary `i` is shared by steps
`i - 1` and `i`; it is decoded only once. -/
structure Layout (candidate : Id) where
  boundaries : Fin (StepCount candidate + 1) ->
    ProductionMemoryCarryRows.Layout
  steps : Fin (StepCount candidate) ->
    ProductionMemoryCheckedStepRows.Layout

/-- Exact physical adjacency and local checked-step ownership. -/
structure Layout.Valid {candidate : Id} (layout : Layout candidate) : Prop where
  stepValid : forall index, (layout.steps index).Valid
  beforeBoundary : forall index,
    (layout.steps index).before = layout.boundaries index.castSucc
  afterBoundary : forall index,
    (layout.steps index).after = layout.boundaries index.succ

def boundaryRows {candidate : Id} (layout : Layout candidate) : List Row :=
  (List.ofFn fun index =>
    ProductionMemoryCarryRows.rows (layout.boundaries index)).flatten

def stepRows {candidate : Id} (layout : Layout candidate) : List Row :=
  (List.ofFn fun index =>
    ProductionMemoryCheckedStepRows.rows (layout.steps index)).flatten

def rows {candidate : Id} (layout : Layout candidate) : List Row :=
  boundaryRows layout ++ stepRows layout

def rowCount (candidate : Id) : Nat :=
  (StepCount candidate + 1) * 178 + StepCount candidate * 27113

private theorem flatten_ofFn_length
    {alpha : Type} {count width : Nat} (blocks : Fin count -> List alpha)
    (each : forall index, (blocks index).length = width) :
    (List.ofFn blocks).flatten.length = count * width := by
  rw [List.length_flatten]
  have constant : forall value, value ∈ (List.ofFn blocks).map List.length ->
      value = width := by
    intro value member
    rcases List.mem_map.mp member with ⟨block, blockMember, rfl⟩
    rcases List.mem_ofFn.mp blockMember with ⟨index, rfl⟩
    exact each index
  rw [List.sum_eq_card_nsmul _ width constant]
  simp

theorem boundaryRows_length_exact
    {candidate : Id} (layout : Layout candidate) :
    (boundaryRows layout).length = (StepCount candidate + 1) * 178 := by
  exact flatten_ofFn_length _ fun index =>
    ProductionMemoryCarryRows.rows_length_exact (layout.boundaries index)

theorem stepRows_length_exact
    {candidate : Id} (layout : Layout candidate) :
    (stepRows layout).length = StepCount candidate * 27113 := by
  exact flatten_ofFn_length _ fun index =>
    ProductionMemoryCheckedStepRows.rows_length_exact (layout.steps index)

theorem rows_length_exact
    {candidate : Id} (layout : Layout candidate) :
    (rows layout).length = rowCount candidate := by
  rw [rows, List.length_append, boundaryRows_length_exact,
    stepRows_length_exact]
  rfl

theorem candidate_row_count_table :
    rowCount .e1 = 27469 /\
      rowCount .e4 = 109342 /\
      rowCount .e8 = 218506 /\
      rowCount .e16 = 436834 := by
  decide

/-- Verifier-owned chain headers are placed at every decoded boundary. -/
def HeadersPlaced
    {candidate : Id} (layout : Layout candidate)
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value) : Prop :=
  forall index, MemoryCarryRows.HeadersPlaced
    (layout.boundaries index).carry assignment headers

private theorem boundaryRows_hold
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment)
    (index : Fin (StepCount candidate + 1)) :
    Satisfies (ProductionMemoryCarryRows.rows
      (layout.boundaries index)) assignment := by
  have allBoundaries : Satisfies (boundaryRows layout) assignment := by
    intro row member
    exact satisfied row (List.mem_append_left _ member)
  exact (satisfies_flatten_iff _ _).mp allBoundaries _
    (List.mem_ofFn.mpr ⟨index, rfl⟩)

private theorem stepRows_hold
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat}
    (satisfied : Satisfies (rows layout) assignment)
    (index : Fin (StepCount candidate)) :
    Satisfies (ProductionMemoryCheckedStepRows.rows
      (layout.steps index)) assignment := by
  have allSteps : Satisfies (stepRows layout) assignment := by
    intro row member
    exact satisfied row (List.mem_append_right _ member)
  exact (satisfies_flatten_iff _ _).mp allSteps _
    (List.mem_ofFn.mpr ⟨index, rfl⟩)

/-- Index-to-list bridge. It proves that exact adjacent indexed transitions
give exact ordered list consumption. -/
theorem consumesList_of_indexed
    {count : Nat} {Digest Challenge Products : Type}
    {balanced : Products -> Prop}
    (states : Fin (count + 1) -> Carry Digest Challenge Products)
    (claims : Fin count -> ClaimSuffix Digest Challenge Products)
    (each : forall index,
      Consumes balanced (states index.castSucc) (claims index)
        (states index.succ)) :
    ConsumesList balanced (states 0) (List.ofFn claims)
      (states (Fin.last count)) := by
  induction count with
  | zero =>
      simpa using (ConsumesList.nil (balanced := balanced) (states 0))
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      apply ConsumesList.cons (each 0)
      let tailStates : Fin (count + 1) -> Carry Digest Challenge Products :=
        fun index => states index.succ
      let tailClaims : Fin count -> ClaimSuffix Digest Challenge Products :=
        fun index => claims index.succ
      have tailEach : forall index,
          Consumes balanced (tailStates index.castSucc) (tailClaims index)
            (tailStates index.succ) := by
        intro index
        exact each index.succ
      simpa [tailStates, tailClaims] using
        inductionHypothesis tailStates tailClaims tailEach

/-- Complete proof-independent output of one generated checked batch. -/
structure Result
    {candidate : Id} (layout : Layout candidate)
    (assignment : Nat -> Nat) (headers : ChainHeaders Digest.Value) where
  boundary : Fin (StepCount candidate + 1) -> MemoryCarryCodec.Value
  boundaryParsed : forall index,
    MemoryCarryPublicRows.ParsedColumnsMatch
      (layout.boundaries index).reference assignment headers (boundary index)
  semantic : Fin (StepCount candidate + 1) ->
    Carry Digest.Value (Challenges K) (ProductState.State K)
  semanticExact : forall index,
    semantic index = MemoryCarryParser.semanticCarry (boundary index)
      (boundaryParsed index).parserCanonical.stepIndex
  claim : Fin (StepCount candidate) -> MemoryClaimCodec.Claim
  claimParsed : forall index,
    MemoryClaimRows.ParsedColumnsMatch
      (layout.steps index).claim.reference assignment (claim index)
  counterWord : forall index counter,
    ((layout.steps index).claim.counters.word counter).digits assignment =
      WasmStateCodec.encodeWord counter.width
        (counter.claimValue (claim index))
  source : forall index,
    MemorySourceRows.Sound (layout.steps index).source assignment (claim index)
  productUpdate : forall index,
    MemoryClaimProductUpdate.mapState (claim index).productsAfter =
      ProductState.update
        Nightstream.Implementation.Nebula.ConcreteField.encode
        (MemoryClaimProductUpdate.mapChallenges (claim index).challenge)
        (MemoryClaimProductUpdate.mapState (claim index).productsBefore)
        (source index).records.chunk
  consumesAt : forall index,
    Consumes MemoryProductBalanceRows.ConcreteBalanced
      (semantic index.castSucc) (claim index) (semantic index.succ)
  consumes : ConsumesList MemoryProductBalanceRows.ConcreteBalanced
    (semantic 0) (List.ofFn claim) (semantic (Fin.last (StepCount candidate)))

/-- Exact authority payload contributed by the row-derived memory batch. -/
def Result.suffixBatch
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (result : Result layout assignment headers) :
    SuffixBatch candidate Digest.Value (Challenges K) (ProductState.State K) :=
  { suffixes := List.ofFn result.claim
    length_exact := by simp [StepCount] }

theorem Result.consumes_suffixBatch
    {candidate : Id} {layout : Layout candidate}
    {assignment : Nat -> Nat} {headers : ChainHeaders Digest.Value}
    (result : Result layout assignment headers) :
    ConsumesList MemoryProductBalanceRows.ConcreteBalanced
      (result.semantic 0) result.suffixBatch.suffixes
      (result.semantic (Fin.last (StepCount candidate))) :=
  result.consumes

/-- Satisfying candidate rows derive every boundary, every claim, and the
complete ordered batch transition. -/
def derive
    {candidate : Id} {layout : Layout candidate}
    (valid : layout.Valid)
    {assignment : Nat -> Nat}
    (headers : ChainHeaders Digest.Value)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headersPlaced : HeadersPlaced layout assignment headers)
    (satisfied : Satisfies (rows layout) assignment) :
    Result layout assignment headers := by
  let boundaryResult := fun index =>
    ProductionMemoryCarryRows.derive headers canonical one
      (headersPlaced index) (boundaryRows_hold satisfied index)
  let boundary := fun index => (boundaryResult index).value
  have boundaryParsed : forall index,
      MemoryCarryPublicRows.ParsedColumnsMatch
        (layout.boundaries index).reference assignment headers
        (boundary index) := by
    intro index
    exact (boundaryResult index).parsed
  let semantic := fun index =>
    MemoryCarryParser.semanticCarry (boundary index)
      (boundaryParsed index).parserCanonical.stepIndex
  have beforeParsed : forall index,
      MemoryCarryPublicRows.ParsedColumnsMatch
        (layout.steps index).before.reference assignment headers
        (boundary index.castSucc) := by
    intro index
    rw [valid.beforeBoundary index]
    exact boundaryParsed index.castSucc
  have afterParsed : forall index,
      MemoryCarryPublicRows.ParsedColumnsMatch
        (layout.steps index).after.reference assignment headers
        (boundary index.succ) := by
    intro index
    rw [valid.afterBoundary index]
    exact boundaryParsed index.succ
  let checked := fun index =>
    ProductionMemoryCheckedStepRows.derive (valid.stepValid index)
      headers canonical one (stepRows_hold satisfied index)
      (boundary index.castSucc) (boundary index.succ)
      (beforeParsed index) (afterParsed index)
  let claim := fun index => (checked index).claim
  have consumesAt : forall index,
      Consumes MemoryProductBalanceRows.ConcreteBalanced
        (semantic index.castSucc) (claim index) (semantic index.succ) := by
    intro index
    simpa [semantic, claim] using (checked index).consumes
  exact
    { boundary := boundary
      boundaryParsed := boundaryParsed
      semantic := semantic
      semanticExact := by intro index; rfl
      claim := claim
      claimParsed := by intro index; exact (checked index).claimParsed
      counterWord := by
        intro index counter
        exact (checked index).counterWord counter
      source := by intro index; exact (checked index).source
      productUpdate := by intro index; exact (checked index).productUpdate
      consumesAt := consumesAt
      consumes := consumesList_of_indexed semantic claim consumesAt }

/-- The result cannot omit or reorder an internal boundary: its final
semantic statement is the exact `List.ofFn` order of the row-derived claims. -/
theorem rows_imply_exact_ordered_batch
    {candidate : Id} {layout : Layout candidate}
    (valid : layout.Valid)
    {assignment : Nat -> Nat}
    (headers : ChainHeaders Digest.Value)
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (headersPlaced : HeadersPlaced layout assignment headers)
    (satisfied : Satisfies (rows layout) assignment) :
    exists result : Result layout assignment headers,
      ConsumesList MemoryProductBalanceRows.ConcreteBalanced
        (result.semantic 0) (List.ofFn result.claim)
        (result.semantic (Fin.last (StepCount candidate))) := by
  let result := derive valid headers canonical one headersPlaced satisfied
  exact ⟨result, result.consumes⟩

end Nightstream.Implementation.Nebula.ProductionMemoryCheckedBatchRows
