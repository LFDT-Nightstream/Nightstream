import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Sampler.Acceptance

/-!
Owns: list-level reference semantics for first-accepted selection.

Does not own: chunk arithmetic, transcript origin, or concrete R1CS row
replacement.

Emits constraints: no.

Authority boundary: selection preserves the order of an already validated
chunk list and cannot authenticate that list.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `firstAcceptedSymbols` | `challenge.sampler.selection` | Preserves order and returns the first 54 accepted symbols | Validated chunk list | No — Rust refinement open |
| `SelectionRowHolds`, `WindowedSelectionRowHolds` | `challenge.sampler.selection.one_hot` | Position `j` selects only from original indices `j..j+10` | At most ten rejected chunks | No — Rust refinement open |
| `selectionRow_exact`, `selectionValue_iff_windowed` | `challenge.sampler.selection.bind` | Witness and windowed relations exactly characterize the selected value | Enough accepts | No — Rust refinement open |
| `selectionAccepts_iff_eq_first` | `challenge.sampler.selection` | Candidate list is accepted exactly when it equals the reference output | Enough accepts | No — Rust refinement open |

The concrete one-hot/product replacement theorem belongs in
`Refinement/SelectionRows.lean`.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- All accepted symbols in original transcript order. -/
def acceptedSymbols (chunks : List Chunk) : List Int :=
  (chunks.filter acceptBit).map symbol

/-- Exact semantic output of one fixed sampler invocation. -/
def firstAcceptedSymbols (chunks : List Chunk) : List Int :=
  (acceptedSymbols chunks).take outputLength

/-- Semantic value selected at one accepted-output position. -/
def SelectionValue
    (chunks : List Chunk) (position : Nat) (output : Int) : Prop :=
  (acceptedSymbols chunks)[position]? = some output

/-- Original-list decomposition represented by one current one-hot selector. -/
structure SelectionWitness where
  before : List Chunk
  selected : Chunk
  after : List Chunk
deriving Repr, DecidableEq

/--
Protocol-level equations behind one one-hot row: the selected chunk is in the
original list, accepts, and has exactly `position` accepted predecessors.
-/
def SelectionRowHolds
    (chunks : List Chunk) (position : Nat) (output : Int)
    (witness : SelectionWitness) : Prop :=
  chunks = witness.before ++ witness.selected :: witness.after ∧
    acceptedCount witness.before = position ∧
    acceptBit witness.selected = true ∧
    output = symbol witness.selected

/-- Selection row restricted to the only indices reachable with at most ten rejections. -/
def WindowedSelectionRowHolds
    (chunks : List Chunk) (position : Nat) (output : Int)
    (witness : SelectionWitness) : Prop :=
  SelectionRowHolds chunks position output witness ∧
    position ≤ witness.before.length ∧
    witness.before.length ≤ position + maxRejections

/-- Extensional contract for the complete first-54 selection branch. -/
def SelectionAccepts (chunks : List Chunk) (candidate : List Int) : Prop :=
  candidate.length = outputLength ∧ candidate <+: acceptedSymbols chunks

theorem acceptedSymbols_length (chunks : List Chunk) :
    (acceptedSymbols chunks).length = acceptedCount chunks := by
  simp [acceptedSymbols, acceptedCount]

theorem firstAcceptedSymbols_length
    (chunks : List Chunk)
    (hEnough : EnoughAccepts chunks) :
    (firstAcceptedSymbols chunks).length = outputLength := by
  rw [firstAcceptedSymbols, List.length_take, acceptedSymbols_length]
  simp only [EnoughAccepts] at hEnough
  exact min_eq_left hEnough

theorem firstAcceptedSymbols_prefix (chunks : List Chunk) :
    firstAcceptedSymbols chunks <+: acceptedSymbols chunks := by
  exact List.take_prefix outputLength (acceptedSymbols chunks)

theorem selectionRow_sound
    {chunks : List Chunk} {position : Nat} {output : Int}
    {witness : SelectionWitness}
    (hRow : SelectionRowHolds chunks position output witness) :
    SelectionValue chunks position output := by
  rcases hRow with ⟨hChunks, hBefore, hAccepted, hOutput⟩
  subst chunks
  subst output
  simp only [SelectionValue, acceptedSymbols, List.filter_append,
    List.map_append]
  simp only [acceptedCount] at hBefore
  simp [hAccepted, hBefore]

theorem selectionRow_index_bounds
    {chunks : List Chunk} {position : Nat} {output : Int}
    {witness : SelectionWitness}
    (hLength : chunks.length = chunksPerSample)
    (hEnough : EnoughAccepts chunks)
    (hRow : SelectionRowHolds chunks position output witness) :
    position ≤ witness.before.length ∧
      witness.before.length ≤ position + maxRejections := by
  rcases hRow with ⟨hChunks, hBefore, hAccepted, _⟩
  have hBeforePartition := accepted_add_rejected witness.before
  have hRejectedBound : rejectedCount chunks ≤ maxRejections := by
    have hBound := (enoughAccepts_iff_rejected_le chunks hLength).mp hEnough
    simpa [maxRejections, chunksPerSample, outputLength] using hBound
  have hRejectedDecomposition :
      rejectedCount chunks =
        rejectedCount witness.before + rejectedCount witness.after := by
    subst chunks
    simp [rejectedCount, hAccepted]
  have hBeforeRejected : rejectedCount witness.before ≤ rejectedCount chunks := by
    rw [hRejectedDecomposition]
    exact Nat.le_add_right _ _
  constructor <;> omega

theorem selectionRow_iff_windowed
    {chunks : List Chunk} {position : Nat} {output : Int}
    {witness : SelectionWitness}
    (hLength : chunks.length = chunksPerSample)
    (hEnough : EnoughAccepts chunks) :
    SelectionRowHolds chunks position output witness ↔
      WindowedSelectionRowHolds chunks position output witness := by
  constructor
  · intro hRow
    exact ⟨hRow, selectionRow_index_bounds hLength hEnough hRow⟩
  · exact fun hWindowed => hWindowed.1

theorem selectionWitness_exists :
    ∀ (chunks : List Chunk) (position : Nat),
      position < acceptedCount chunks →
      ∃ output witness, SelectionRowHolds chunks position output witness
  | [], position, hPosition => by
      simp [acceptedCount] at hPosition
  | chunk :: rest, position, hPosition => by
      by_cases hAccepted : acceptBit chunk = true
      · cases position with
        | zero =>
            refine ⟨symbol chunk,
              { before := [], selected := chunk, after := rest }, ?_⟩
            simp [SelectionRowHolds, acceptedCount, hAccepted]
        | succ priorPosition =>
            have hRest : priorPosition < acceptedCount rest := by
              simpa [acceptedCount, hAccepted, Nat.succ_lt_succ_iff] using
                hPosition
            rcases selectionWitness_exists rest priorPosition hRest with
              ⟨output, witness, hWitness⟩
            rcases hWitness with
              ⟨hRestDecomposition, hBefore, hSelected, hOutput⟩
            refine ⟨output,
              { before := chunk :: witness.before
                selected := witness.selected
                after := witness.after }, ?_⟩
            simp only [SelectionRowHolds]
            refine ⟨?_, ?_, hSelected, hOutput⟩
            · simp [hRestDecomposition]
            · simp only [acceptedCount, List.filter_cons, hAccepted,
                ↓reduceIte, List.length_cons]
              simpa only [acceptedCount] using congrArg Nat.succ hBefore
      · have hRejected : acceptBit chunk = false :=
          Bool.eq_false_of_not_eq_true hAccepted
        have hRest : position < acceptedCount rest := by
          simpa [acceptedCount, hRejected] using hPosition
        rcases selectionWitness_exists rest position hRest with
          ⟨output, witness, hWitness⟩
        rcases hWitness with
          ⟨hRestDecomposition, hBefore, hSelected, hOutput⟩
        refine ⟨output,
          { before := chunk :: witness.before
            selected := witness.selected
            after := witness.after }, ?_⟩
        simp only [SelectionRowHolds]
        refine ⟨?_, ?_, hSelected, hOutput⟩
        · simp [hRestDecomposition]
        · simp only [acceptedCount, List.filter_cons, hRejected,
            Bool.false_eq]
          simpa only [acceptedCount] using hBefore

theorem selectionRow_complete
    {chunks : List Chunk} {position : Nat} {output : Int}
    (hValue : SelectionValue chunks position output) :
    ∃ witness, SelectionRowHolds chunks position output witness := by
  have hValue' := hValue
  rcases List.getElem?_eq_some_iff.mp hValue' with ⟨hPosition, _⟩
  have hAcceptedPosition : position < acceptedCount chunks := by
    simpa only [acceptedSymbols_length] using hPosition
  rcases selectionWitness_exists chunks position hAcceptedPosition with
    ⟨candidate, witness, hWitness⟩
  have hCandidate := selectionRow_sound hWitness
  unfold SelectionValue at hValue hCandidate
  rw [hValue] at hCandidate
  have hEqual : candidate = output := (Option.some.inj hCandidate).symm
  subst candidate
  exact ⟨witness, hWitness⟩

theorem selectionRow_exact
    (chunks : List Chunk) (position : Nat) (output : Int) :
    (∃ witness, SelectionRowHolds chunks position output witness) ↔
      SelectionValue chunks position output := by
  exact ⟨fun ⟨_, hRow⟩ => selectionRow_sound hRow,
    selectionRow_complete⟩

theorem selectionValue_iff_windowed
    {chunks : List Chunk} {position : Nat} {output : Int}
    (hLength : chunks.length = chunksPerSample)
    (hEnough : EnoughAccepts chunks) :
    SelectionValue chunks position output ↔
      ∃ witness, WindowedSelectionRowHolds chunks position output witness := by
  rw [← selectionRow_exact]
  constructor
  · rintro ⟨witness, hRow⟩
    exact ⟨witness, (selectionRow_iff_windowed hLength hEnough).mp hRow⟩
  · rintro ⟨witness, hWindowed⟩
    exact ⟨witness, (selectionRow_iff_windowed hLength hEnough).mpr hWindowed⟩

theorem selectionValue_unique
    {chunks : List Chunk} {position : Nat} {left right : Int}
    (hLeft : SelectionValue chunks position left)
    (hRight : SelectionValue chunks position right) :
    left = right := by
  unfold SelectionValue at hLeft hRight
  rw [hLeft] at hRight
  exact Option.some.inj hRight

/--
The length-plus-prefix contract accepts exactly the first 54 accepted symbols.
Assumes `EnoughAccepts`; does not assume how the chunks were generated. Maps to
the complete `selection::select_first_n_accepts` branch.
-/
theorem selectionAccepts_iff_eq_first
    {chunks : List Chunk} {candidate : List Int}
    (hEnough : EnoughAccepts chunks) :
    SelectionAccepts chunks candidate ↔
      candidate = firstAcceptedSymbols chunks := by
  constructor
  · rintro ⟨hLength, hPrefix⟩
    have hTake := List.prefix_iff_eq_take.mp hPrefix
    simpa only [firstAcceptedSymbols, hLength] using hTake
  · rintro rfl
    exact ⟨firstAcceptedSymbols_length chunks hEnough,
      firstAcceptedSymbols_prefix chunks⟩

theorem selectionAccepts_unique
    {chunks : List Chunk} {left right : List Int}
    (hEnough : EnoughAccepts chunks)
    (hLeft : SelectionAccepts chunks left)
    (hRight : SelectionAccepts chunks right) :
    left = right := by
  rw [(selectionAccepts_iff_eq_first hEnough).mp hLeft,
    (selectionAccepts_iff_eq_first hEnough).mp hRight]

theorem firstAcceptedSymbols_mem_alphabet
    (chunks : List Chunk) (value : Int)
    (hMember : value ∈ firstAcceptedSymbols chunks) :
    (-2 : Int) ≤ value ∧ value ≤ 2 := by
  have hAccepted : value ∈ acceptedSymbols chunks :=
    List.mem_of_mem_take hMember
  rcases List.mem_map.mp hAccepted with ⟨chunk, _, rfl⟩
  exact symbol_mem_alphabet chunk

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
