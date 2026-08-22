import Std

/-! Provenance: copied from `formal/nightstream-lean/Nightstream/SuperNeo/Sampling/FirstAccepted.lean`
at commit `fb7a8a99aefbb8ebb5474681ecf80f1b95a1b7a2`; namespaces renamed, otherwise unchanged. -/

/-!
Generic first-accepted rejection-selection semantics.

Owns: verifier-owned acceptance/symbol functions, finite-prefix sampling,
shortfall, and a relational reference execution over an unbounded candidate
stream.

Does not own: any concrete chunk width, rejection bucket, output length,
alphabet, strong sampling set, probability distribution, transcript, hash,
Poseidon2 permutation, Rust program, R1CS relation, or row count.

Emits backend obligations: no.

Authority boundary: candidates come from an abstract stream. This file proves
only what follows from the verifier-owned `accepts` and `symbol` functions.
`ReferenceExecution` records a finite stopping witness without asserting that
every stream terminates.

| Protocol | Phase | Constraint family | Mathematical obligation |
|---|---|---|---|
| generic sampler | candidate classification | acceptance | filter only candidates accepted by the verifier-owned predicate |
| generic sampler | output materialization | first accepted | return the first requested accepted symbols in source order |
| generic sampler | output materialization | indexed provenance | an accepted source with exactly `position` prior accepts is output `position` |
| generic sampler | bounded execution | shortfall | return `none` exactly when the bounded prefix has too few accepts |
| generic sampler | reference execution | stopping prefix | stop at the unique least prefix containing enough accepted candidates |
| generic sampler | prefix extension | stability | later candidates cannot change an already complete output |
-/

namespace NightstreamFPrime.Spec.Sampling.FirstAccepted

universe uCandidate uSymbol

/-- The verifier, rather than the candidate source, owns both decisions used
by rejection selection. -/
structure Verifier (Candidate : Type uCandidate) (Symbol : Type uSymbol) where
  accepts : Candidate -> Bool
  symbol : Candidate -> Symbol

/-- Accepted candidates in their original source order. -/
def acceptedCandidates
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (candidates : List Candidate) :
    List Candidate :=
  candidates.filter verifier.accepts

/-- Number of accepted candidates in a finite prefix. -/
def acceptedCount
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (candidates : List Candidate) : Nat :=
  (acceptedCandidates verifier candidates).length

/-- Accepted symbols in source order, before imposing an output bound. -/
def acceptedSymbols
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (candidates : List Candidate) :
    List Symbol :=
  (acceptedCandidates verifier candidates).map verifier.symbol

/-- The canonical first-accepted output from a finite candidate prefix. -/
def firstAccepted
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (candidates : List Candidate) : List Symbol :=
  (acceptedSymbols verifier candidates).take need

/-- Every returned symbol comes from an actually accepted source candidate.
This is the generic provenance theorem needed by protocol-specific strong-set
wrappers; it does not assert anything about what acceptance means. -/
theorem mem_firstAccepted
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} {value : Symbol}
    (member : value ∈ firstAccepted verifier need candidates) :
    exists candidate,
      candidate ∈ candidates /\
        verifier.accepts candidate = true /\
        verifier.symbol candidate = value := by
  have acceptedMember : value ∈ acceptedSymbols verifier candidates :=
    List.mem_of_mem_take member
  rcases List.mem_map.mp acceptedMember with
    ⟨candidate, candidateMember, symbolEq⟩
  have filtered := List.mem_filter.mp candidateMember
  exact ⟨candidate, filtered.1, filtered.2, symbolEq⟩

/-- A finite prefix contains enough accepted candidates for `need` outputs. -/
def Enough
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (candidates : List Candidate) : Prop :=
  need <= acceptedCount verifier candidates

/-- The named bounded-sampler failure event. -/
def Shortfall
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (candidates : List Candidate) : Prop :=
  acceptedCount verifier candidates < need

/-- Executable bounded sampler. Failure is explicit rather than filled with a
prover-chosen default. -/
def boundedSample
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (candidates : List Candidate) : Option (List Symbol) :=
  if need <= acceptedCount verifier candidates then
    some (firstAccepted verifier need candidates)
  else
    none

theorem acceptedCandidates_append
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol)
    (initial suffix : List Candidate) :
    acceptedCandidates verifier (initial ++ suffix) =
      acceptedCandidates verifier initial ++
        acceptedCandidates verifier suffix := by
  simp [acceptedCandidates]

theorem acceptedSymbols_append
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol)
    (initial suffix : List Candidate) :
    acceptedSymbols verifier (initial ++ suffix) =
      acceptedSymbols verifier initial ++ acceptedSymbols verifier suffix := by
  simp [acceptedSymbols, acceptedCandidates]

/-- Indexed source-order characterization. If a source candidate is accepted
and exactly `position` earlier candidates are accepted, then every requested
prefix extending past `position` contains its symbol exactly at that position. -/
theorem getElem?_firstAccepted_eq_symbol_of_prefix
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} {index position : Nat}
    (indexLt : index < candidates.length)
    (accepted : verifier.accepts candidates[index] = true)
    (before : acceptedCount verifier (candidates.take index) = position)
    (positionLt : position < need) :
    (firstAccepted verifier need candidates)[position]? =
      some (verifier.symbol candidates[index]) := by
  let candidateValue : Candidate := candidates[index]
  have acceptedValue : verifier.accepts candidateValue = true := by
    simpa [candidateValue] using accepted
  have prefixLength :
      (acceptedSymbols verifier (candidates.take index)).length = position := by
    simpa [acceptedSymbols, acceptedCount] using before
  have sourceDecomposition :
      candidates = candidates.take index ++
        candidateValue :: candidates.drop (index + 1) := by
    calc
      candidates = candidates.take (index + 1) ++
          candidates.drop (index + 1) :=
        (List.take_append_drop (index + 1) candidates).symm
      _ = candidates.take index ++
          candidateValue :: candidates.drop (index + 1) := by
        rw [List.take_succ_eq_append_getElem indexLt]
        simp [candidateValue]
  have symbolDecomposition :
      acceptedSymbols verifier candidates =
        acceptedSymbols verifier (candidates.take index) ++
          verifier.symbol candidateValue ::
            acceptedSymbols verifier (candidates.drop (index + 1)) := by
    calc
      acceptedSymbols verifier candidates =
          acceptedSymbols verifier
            (candidates.take index ++
              candidateValue :: candidates.drop (index + 1)) :=
        congrArg (acceptedSymbols verifier) sourceDecomposition
      _ = acceptedSymbols verifier (candidates.take index) ++
          acceptedSymbols verifier
            (candidateValue :: candidates.drop (index + 1)) :=
        acceptedSymbols_append verifier _ _
      _ = acceptedSymbols verifier (candidates.take index) ++
          verifier.symbol candidateValue ::
            acceptedSymbols verifier (candidates.drop (index + 1)) := by
        simp [acceptedSymbols, acceptedCandidates, acceptedValue]
  unfold firstAccepted
  rw [List.getElem?_take_of_lt positionLt, symbolDecomposition]
  rw [List.getElem?_append_right (by omega)]
  rw [prefixLength]
  simp [candidateValue]

theorem enough_append
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {initial suffix : List Candidate}
    (enough : Enough verifier need initial) :
    Enough verifier need (initial ++ suffix) := by
  unfold Enough acceptedCount at *
  rw [acceptedCandidates_append, List.length_append]
  exact Nat.le_add_right_of_le enough

/-- Once a prefix contains enough accepts, every extension has the same
canonical first-accepted output. -/
theorem firstAccepted_append_of_enough
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {initial suffix : List Candidate}
    (enough : Enough verifier need initial) :
    firstAccepted verifier need (initial ++ suffix) =
      firstAccepted verifier need initial := by
  unfold firstAccepted
  rw [acceptedSymbols_append]
  apply List.take_append_of_le_length
  simpa [acceptedSymbols, acceptedCount] using enough

theorem boundedSample_eq_some_iff
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} {output : List Symbol} :
    boundedSample verifier need candidates = some output <->
      Enough verifier need candidates /\
        output = firstAccepted verifier need candidates := by
  by_cases enough : need <= acceptedCount verifier candidates
  · simp [boundedSample, enough, Enough, eq_comm]
  · simp [boundedSample, enough, Enough]

/-- Every successful bounded execution returns exactly the first requested
accepted symbols in source order. -/
theorem bounded_success_exact
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} {output : List Symbol}
    (success : boundedSample verifier need candidates = some output) :
    output = firstAccepted verifier need candidates :=
  (boundedSample_eq_some_iff.mp success).2

theorem firstAccepted_length_of_enough
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate}
    (enough : Enough verifier need candidates) :
    (firstAccepted verifier need candidates).length = need := by
  unfold Enough acceptedCount at enough
  simp [firstAccepted, acceptedSymbols, Nat.min_eq_left enough]

/-- A successful bounded sampler returns exactly `need` symbols. -/
theorem bounded_success_length
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} {output : List Symbol}
    (success : boundedSample verifier need candidates = some output) :
    output.length = need := by
  rw [bounded_success_exact success]
  exact firstAccepted_length_of_enough
    (boundedSample_eq_some_iff.mp success).1

theorem bounded_success_unique
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} {left right : List Symbol}
    (leftSuccess : boundedSample verifier need candidates = some left)
    (rightSuccess : boundedSample verifier need candidates = some right) :
    left = right := by
  rw [leftSuccess] at rightSuccess
  exact Option.some.inj rightSuccess

theorem shortfall_iff_acceptedCount_lt
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} :
    Shortfall verifier need candidates <->
      acceptedCount verifier candidates < need := by
  rfl

theorem shortfall_iff_not_enough
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} :
    Shortfall verifier need candidates <->
      ¬ Enough verifier need candidates := by
  simp [Shortfall, Enough]

theorem boundedSample_eq_none_iff_shortfall
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {candidates : List Candidate} :
    boundedSample verifier need candidates = none <->
      Shortfall verifier need candidates := by
  simp [boundedSample, Shortfall, Nat.not_le]

/-- Successful bounded sampling is stable under every list extension. -/
theorem boundedSample_append_of_success
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {initial suffix : List Candidate} {output : List Symbol}
    (success : boundedSample verifier need initial = some output) :
    boundedSample verifier need (initial ++ suffix) = some output := by
  have enough := (boundedSample_eq_some_iff.mp success).1
  apply boundedSample_eq_some_iff.mpr
  refine ⟨enough_append enough, ?_⟩
  rw [firstAccepted_append_of_enough enough]
  exact bounded_success_exact success

/-! ## Unbounded reference semantics without a termination assumption -/

/-- An abstract infinite source of candidates. -/
abbrev CandidateStream (Candidate : Type uCandidate) := Nat -> Candidate

/-- The first `count` candidates of an abstract stream. -/
def streamPrefix
    {Candidate : Type uCandidate}
    (stream : CandidateStream Candidate) (count : Nat) : List Candidate :=
  (List.range count).map stream

/-- Executable least-cursor search over the finite interval `[0, bound]`. -/
def boundedCursor
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (stream : CandidateStream Candidate) (bound : Nat) : Option Nat :=
  (Array.range (bound + 1)).find? fun consumed =>
    decide (need <= acceptedCount verifier (streamPrefix stream consumed))

theorem boundedCursor_eq_some_iff
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound consumed : Nat}
    {stream : CandidateStream Candidate} :
    boundedCursor verifier need stream bound = some consumed <->
      consumed <= bound /\
        Enough verifier need (streamPrefix stream consumed) /\
        forall earlier,
          earlier < consumed ->
            Shortfall verifier need (streamPrefix stream earlier) := by
  simp [boundedCursor, Enough, Shortfall, Nat.lt_succ_iff,
    and_left_comm]

@[simp] theorem streamPrefix_length
    {Candidate : Type uCandidate}
    (stream : CandidateStream Candidate) (count : Nat) :
    (streamPrefix stream count).length = count := by
  simp [streamPrefix]

theorem streamPrefix_add
    {Candidate : Type uCandidate}
    (stream : CandidateStream Candidate) (left right : Nat) :
    streamPrefix stream (left + right) =
      streamPrefix stream left ++
        (List.range right).map (fun index => stream (left + index)) := by
  simp [streamPrefix, List.range_add, List.map_append, List.map_map,
    Function.comp_def]

theorem streamPrefix_extension
    {Candidate : Type uCandidate}
    (stream : CandidateStream Candidate) {short long : Nat}
    (within : short <= long) :
    exists suffix,
      streamPrefix stream long = streamPrefix stream short ++ suffix := by
  obtain ⟨extra, rfl⟩ := Nat.exists_eq_add_of_le within
  exact ⟨(List.range extra).map (fun index => stream (short + index)),
    streamPrefix_add stream short extra⟩

/-- A terminating run is represented by its finite stopping witness. No theorem
asserts that every abstract stream has such a witness. -/
structure ReferenceExecution
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (stream : CandidateStream Candidate) (output : List Symbol)
    (consumed : Nat) : Prop where
  enough : Enough verifier need (streamPrefix stream consumed)
  minimal : forall earlier,
    earlier < consumed ->
      Shortfall verifier need (streamPrefix stream earlier)
  output_eq : output =
    firstAccepted verifier need (streamPrefix stream consumed)

/-- A successful finite execution carries both its exact output and the least
cursor found by `boundedCursor`. -/
structure BoundedExecution
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (stream : CandidateStream Candidate) (bound : Nat) where
  output : List Symbol
  consumed : Nat
  cursor_eq : boundedCursor verifier need stream bound = some consumed
  within : consumed <= bound
  reference : ReferenceExecution verifier need stream output consumed

/-- Reference semantics is existential only over an actual finite stopping
witness, so nontermination is not assumed away. -/
def ReferenceSemantics
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    (verifier : Verifier Candidate Symbol) (need : Nat)
    (stream : CandidateStream Candidate) (output : List Symbol) : Prop :=
  exists consumed, ReferenceExecution verifier need stream output consumed

theorem ReferenceExecution.consumed_eq
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {stream : CandidateStream Candidate} {leftOutput rightOutput : List Symbol}
    {leftConsumed rightConsumed : Nat}
    (left : ReferenceExecution verifier need stream leftOutput leftConsumed)
    (right : ReferenceExecution verifier need stream rightOutput rightConsumed) :
    leftConsumed = rightConsumed := by
  rcases Nat.lt_trichotomy leftConsumed rightConsumed with
    leftBefore | equal | rightBefore
  · have shortfall := right.minimal leftConsumed leftBefore
    exact False.elim <|
      (shortfall_iff_not_enough.mp shortfall) left.enough
  · exact equal
  · have shortfall := left.minimal rightConsumed rightBefore
    exact False.elim <|
      (shortfall_iff_not_enough.mp shortfall) right.enough

theorem ReferenceExecution.output_unique
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {stream : CandidateStream Candidate} {leftOutput rightOutput : List Symbol}
    {leftConsumed rightConsumed : Nat}
    (left : ReferenceExecution verifier need stream leftOutput leftConsumed)
    (right : ReferenceExecution verifier need stream rightOutput rightConsumed) :
    leftOutput = rightOutput := by
  have consumedEqual := left.consumed_eq right
  rw [left.output_eq, right.output_eq, consumedEqual]

theorem ReferenceSemantics.output_unique
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {stream : CandidateStream Candidate} {left right : List Symbol}
    (leftRun : ReferenceSemantics verifier need stream left)
    (rightRun : ReferenceSemantics verifier need stream right) :
    left = right := by
  rcases leftRun with ⟨leftConsumed, leftExecution⟩
  rcases rightRun with ⟨rightConsumed, rightExecution⟩
  exact leftExecution.output_unique rightExecution

/-- A successful bounded prefix constructs the unique terminating reference
run; termination is not assumed for streams whose bounded prefixes all
shortfall. -/
theorem ReferenceExecution.exists_of_bounded_success
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound : Nat}
    {stream : CandidateStream Candidate} {output : List Symbol}
    (success : boundedSample verifier need (streamPrefix stream bound) =
      some output) :
    exists consumed,
      consumed <= bound /\
        ReferenceExecution verifier need stream output consumed := by
  have boundedEnough :
      Enough verifier need (streamPrefix stream bound) :=
    (boundedSample_eq_some_iff.mp success).1
  cases cursorResult : boundedCursor verifier need stream bound with
  | none =>
      have noCursor := cursorResult
      unfold boundedCursor at noCursor
      have boundRejected :=
        (Array.find?_range_eq_none.mp noCursor) bound (Nat.lt_succ_self bound)
      have notEnough :
          ¬ Enough verifier need (streamPrefix stream bound) := by
        simpa [Enough] using boundRejected
      exact False.elim (notEnough boundedEnough)
  | some consumed =>
      have cursorSpec := boundedCursor_eq_some_iff.mp cursorResult
      obtain ⟨suffix, prefixSplit⟩ :=
        streamPrefix_extension stream cursorSpec.1
      have outputAtBound :
          output = firstAccepted verifier need (streamPrefix stream bound) :=
        bounded_success_exact success
      have outputAtConsumed :
          output = firstAccepted verifier need (streamPrefix stream consumed) := by
        rw [outputAtBound, prefixSplit,
          firstAccepted_append_of_enough cursorSpec.2.1]
      exact ⟨consumed, cursorSpec.1, {
        enough := cursorSpec.2.1
        minimal := cursorSpec.2.2
        output_eq := outputAtConsumed
      }⟩

/-- Successful bounded sampling constructs a finite execution carrying the
executable least cursor. -/
theorem BoundedExecution.exists_of_bounded_success
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound : Nat}
    {stream : CandidateStream Candidate} {output : List Symbol}
    (success : boundedSample verifier need (streamPrefix stream bound) =
      some output) :
    exists execution : BoundedExecution verifier need stream bound,
      execution.output = output := by
  rcases ReferenceExecution.exists_of_bounded_success success with
    ⟨consumed, within, reference⟩
  have cursorEq : boundedCursor verifier need stream bound = some consumed :=
    boundedCursor_eq_some_iff.mpr
      ⟨within, reference.enough, reference.minimal⟩
  exact ⟨{
    output := output
    consumed := consumed
    cursor_eq := cursorEq
    within := within
    reference := reference
  }, rfl⟩

theorem BoundedExecution.output_length
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound : Nat}
    {stream : CandidateStream Candidate}
    (execution : BoundedExecution verifier need stream bound) :
    execution.output.length = need := by
  rw [execution.reference.output_eq]
  exact firstAccepted_length_of_enough execution.reference.enough

theorem ReferenceExecution.consumed_le_of_prefix_enough
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound consumed : Nat}
    {stream : CandidateStream Candidate} {output : List Symbol}
    (execution : ReferenceExecution verifier need stream output consumed)
    (boundedEnough : Enough verifier need (streamPrefix stream bound)) :
    consumed <= bound := by
  rcases Nat.le_total consumed bound with within | boundWithin
  · exact within
  · rcases Nat.eq_or_lt_of_le boundWithin with equal | boundBefore
    · exact Nat.le_of_eq equal.symm
    · have shortfall := execution.minimal bound boundBefore
      exact False.elim ((shortfall_iff_not_enough.mp shortfall) boundedEnough)

/-- Every terminating reference execution agrees with any already-complete
bounded prefix of the same stream. -/
theorem ReferenceExecution.agrees_with_bounded_prefix
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound consumed : Nat}
    {stream : CandidateStream Candidate} {output : List Symbol}
    (execution : ReferenceExecution verifier need stream output consumed)
    (boundedEnough : Enough verifier need (streamPrefix stream bound)) :
    boundedSample verifier need (streamPrefix stream bound) = some output := by
  have consumedWithin :=
    execution.consumed_le_of_prefix_enough boundedEnough
  obtain ⟨suffix, prefixSplit⟩ :=
    streamPrefix_extension stream consumedWithin
  apply boundedSample_eq_some_iff.mpr
  refine ⟨boundedEnough, ?_⟩
  rw [execution.output_eq, prefixSplit,
    firstAccepted_append_of_enough execution.enough]

/-- Exact conditional equivalence between bounded success and a terminating
reference execution whose least cursor lies within that bounded prefix. -/
theorem boundedSample_eq_some_iff_referenceExecution_within
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound : Nat}
    {stream : CandidateStream Candidate} {output : List Symbol} :
    boundedSample verifier need (streamPrefix stream bound) = some output <->
      exists consumed,
        consumed <= bound /\
          ReferenceExecution verifier need stream output consumed := by
  constructor
  · exact ReferenceExecution.exists_of_bounded_success
  · intro referenceWithin
    rcases referenceWithin with ⟨consumed, consumedWithin, execution⟩
    obtain ⟨suffix, prefixSplit⟩ :=
      streamPrefix_extension stream consumedWithin
    have boundedEnough :
        Enough verifier need (streamPrefix stream bound) := by
      rw [prefixSplit]
      exact enough_append execution.enough
    exact execution.agrees_with_bounded_prefix boundedEnough

theorem boundedSample_eq_some_iff_boundedExecution
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need bound : Nat}
    {stream : CandidateStream Candidate} {output : List Symbol} :
    boundedSample verifier need (streamPrefix stream bound) = some output <->
      exists execution : BoundedExecution verifier need stream bound,
        execution.output = output := by
  constructor
  · exact BoundedExecution.exists_of_bounded_success
  · intro boundedExecution
    rcases boundedExecution with ⟨execution, outputEq⟩
    rw [← outputEq]
    exact boundedSample_eq_some_iff_referenceExecution_within.mpr
      ⟨execution.consumed, execution.within, execution.reference⟩

/-- The abstract cursor is exactly the consumed stream position. Two reference
executions therefore consume the same candidate prefix. -/
theorem ReferenceExecution.consumedPrefix_eq
    {Candidate : Type uCandidate} {Symbol : Type uSymbol}
    {verifier : Verifier Candidate Symbol} {need : Nat}
    {stream : CandidateStream Candidate} {leftOutput rightOutput : List Symbol}
    {leftConsumed rightConsumed : Nat}
    (left : ReferenceExecution verifier need stream leftOutput leftConsumed)
    (right : ReferenceExecution verifier need stream rightOutput rightConsumed) :
    streamPrefix stream leftConsumed = streamPrefix stream rightConsumed := by
  rw [left.consumed_eq right]

end NightstreamFPrime.Spec.Sampling.FirstAccepted
