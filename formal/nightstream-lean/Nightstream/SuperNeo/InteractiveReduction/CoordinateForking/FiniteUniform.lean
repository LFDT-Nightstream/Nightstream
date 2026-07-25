import Init.Data.List.Find
import Init.Data.List.FinRange
import Init.Data.List.Nat.TakeDrop
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking
import Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
import Nightstream.SuperNeo.InteractiveReduction.FiniteUniform.Rejection
set_option autoImplicit false
namespace Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteWords
universe uScalar uAnswer
variable {Scalar : Type uScalar} {Answer : Type uAnswer} {coordinates : Nat}
structure CyclicPoint (Scalar : Type uScalar) where
  value : Scalar
  alternatives : List Scalar
def cyclicPointsAux
    (processedPrefix : List Scalar) : List Scalar -> List (CyclicPoint Scalar)
  | [] => []
  | value :: remaining =>
      { value := value, alternatives := remaining ++ processedPrefix } ::
        cyclicPointsAux (processedPrefix ++ [value]) remaining
def cyclicPoints (values : List Scalar) : List (CyclicPoint Scalar) :=
  cyclicPointsAux [] values
@[simp] theorem cyclicPointsAux_values
    (processedPrefix remaining : List Scalar) :
    (cyclicPointsAux processedPrefix remaining).map CyclicPoint.value = remaining := by
  induction remaining generalizing processedPrefix with
  | nil => rfl
  | cons value remaining inductionHypothesis =>
      simp [cyclicPointsAux, inductionHypothesis]
@[simp] theorem cyclicPoints_values (values : List Scalar) :
    (cyclicPoints values).map CyclicPoint.value = values := by
  exact cyclicPointsAux_values [] values
theorem cyclicPoints_length (values : List Scalar) :
    (cyclicPoints values).length = values.length := by
  have mappedLength := congrArg List.length (cyclicPoints_values values)
  simpa only [List.length_map] using mappedLength
theorem mem_cyclicPointsAux_iff
    (processedPrefix remaining : List Scalar)
    (point : CyclicPoint Scalar) :
    point ∈ cyclicPointsAux processedPrefix remaining <->
      exists before value after,
        remaining = before ++ value :: after ∧
        point = {
          value := value
          alternatives := after ++ processedPrefix ++ before
        } := by
  induction remaining generalizing processedPrefix with
  | nil => simp [cyclicPointsAux]
  | cons head tail inductionHypothesis =>
      constructor
      · intro member
        rcases List.mem_cons.mp member with pointIsHead | pointInTail
        · refine ⟨[], head, tail, rfl, ?_⟩
          simpa [cyclicPointsAux] using pointIsHead
        · rcases (inductionHypothesis (processedPrefix ++ [head])).mp
              pointInTail with
            ⟨before, value, after, tailEquation, pointEquation⟩
          refine ⟨head :: before, value, after, ?_, ?_⟩
          · simp [tailEquation]
          · rw [pointEquation]
            congr 1
            simp [List.append_assoc]
      · rintro ⟨before, value, after, remainingEquation, pointEquation⟩
        cases before with
        | nil =>
            simp only [List.nil_append, List.cons.injEq] at remainingEquation
            rcases remainingEquation with ⟨rfl, rfl⟩
            subst point
            simp [cyclicPointsAux]
        | cons first rest =>
            simp only [List.cons_append, List.cons.injEq] at remainingEquation
            rcases remainingEquation with ⟨rfl, tailEquation⟩
            apply List.mem_cons.mpr
            right
            apply (inductionHypothesis (processedPrefix ++ [head])).mpr
            refine ⟨rest, value, after, tailEquation, ?_⟩
            rw [pointEquation]
            congr 1
            simp [List.append_assoc]
theorem mem_cyclicPoints_iff
    (values : List Scalar)
    (point : CyclicPoint Scalar) :
    point ∈ cyclicPoints values <->
      exists before value after,
        values = before ++ value :: after ∧
        point = {
          value := value
          alternatives := after ++ before
        } := by
  simpa [cyclicPoints, List.append_assoc] using
    mem_cyclicPointsAux_iff ([] : List Scalar) values point
theorem cyclicPoints_nodup
    (values : List Scalar)
    (valuesNodup : values.Nodup) :
    (cyclicPoints values).Nodup := by
  have mappedNodup :
      ((cyclicPoints values).map CyclicPoint.value).Nodup := by
    simpa only [cyclicPoints_values] using valuesNodup
  generalize cyclicPoints values = points at mappedNodup ⊢
  induction points with
  | nil => exact List.nodup_nil
  | cons point points inductionHypothesis =>
      rw [List.map_cons, List.nodup_cons] at mappedNodup
      apply List.nodup_cons.mpr
      refine ⟨?_, inductionHypothesis mappedNodup.2⟩
      intro pointMember
      exact mappedNodup.1
        (List.mem_map.mpr ⟨point, pointMember, rfl⟩)
private theorem value_not_mem_before
    {before after : List Scalar}
    {value : Scalar}
    (nodup : (before ++ value :: after).Nodup) :
    value ∉ before := by
  intro member
  have split := List.nodup_append.mp nodup
  exact split.2.2 value member value (by simp) rfl
private theorem value_not_mem_after
    {before after : List Scalar}
    {value : Scalar}
    (nodup : (before ++ value :: after).Nodup) :
    value ∉ after := by
  have split := List.nodup_append.mp nodup
  exact (List.nodup_cons.mp split.2.1).1
theorem mem_alternatives_iff
    (values : List Scalar)
    (valuesNodup : values.Nodup)
    (point : CyclicPoint Scalar)
    (pointMember : point ∈ cyclicPoints values)
    (candidate : Scalar) :
    candidate ∈ point.alternatives <->
      candidate ∈ values ∧ candidate ≠ point.value := by
  rcases (mem_cyclicPoints_iff values point).mp pointMember with
    ⟨before, value, after, valuesEquation, rfl⟩
  subst values
  have notBefore := value_not_mem_before valuesNodup
  have notAfter := value_not_mem_after valuesNodup
  change candidate ∈ after ++ before <->
    candidate ∈ before ++ value :: after ∧ candidate ≠ value
  constructor
  · intro member
    rcases List.mem_append.mp member with inAfter | inBefore
    · refine ⟨by simp [inAfter], ?_⟩
      intro equal
      subst candidate
      exact notAfter inAfter
    · refine ⟨by simp [inBefore], ?_⟩
      intro equal
      subst candidate
      exact notBefore inBefore
  · rintro ⟨member, distinct⟩
    rcases List.mem_append.mp member with inBefore | inTail
    · exact List.mem_append.mpr (Or.inr inBefore)
    · rcases List.mem_cons.mp inTail with equal | inAfter
      · exact False.elim (distinct equal)
      · exact List.mem_append.mpr (Or.inl inAfter)
theorem alternatives_nodup
    (values : List Scalar)
    (valuesNodup : values.Nodup)
    (point : CyclicPoint Scalar)
    (pointMember : point ∈ cyclicPoints values) :
    point.alternatives.Nodup := by
  rcases (mem_cyclicPoints_iff values point).mp pointMember with
    ⟨before, value, after, valuesEquation, rfl⟩
  subst values
  have split := List.nodup_append.mp valuesNodup
  have beforeNodup := split.1
  have tailNodup := split.2.1
  have afterNodup := (List.nodup_cons.mp tailNodup).2
  apply List.nodup_append.mpr
  refine ⟨afterNodup, beforeNodup, ?_⟩
  intro afterValue afterMember beforeValue beforeMember equal
  exact split.2.2 beforeValue beforeMember afterValue
    (by simp [afterMember]) equal.symm
theorem alternatives_length
    (values : List Scalar)
    (point : CyclicPoint Scalar)
    (pointMember : point ∈ cyclicPoints values) :
    point.alternatives.length + 1 = values.length := by
  rcases (mem_cyclicPoints_iff values point).mp pointMember with
    ⟨before, value, after, valuesEquation, rfl⟩
  subst values
  simp only [List.length_append, List.length_cons]
  omega
def cyclicPointSupport (alphabet : Support Scalar) :
    Support (CyclicPoint Scalar) where
  values := cyclicPoints alphabet.values
  nodup := cyclicPoints_nodup alphabet.values alphabet.nodup
  nonempty := by
    intro pointsEmpty
    have lengthZero := congrArg List.length pointsEmpty
    rw [cyclicPoints_length, List.length_nil] at lengthZero
    exact alphabet.nonempty (List.length_eq_zero_iff.mp lengthZero)
@[simp] theorem cyclicPointSupport_cardinality
    (alphabet : Support Scalar) :
    (cyclicPointSupport alphabet).cardinality = alphabet.cardinality := by
  exact cyclicPoints_length alphabet.values
def scanLength : List Bool -> Nat
  | [] => 0
  | accepted :: remaining =>
      if accepted then 1 else 1 + scanLength remaining
theorem scanLength_le_length (bits : List Bool) :
    scanLength bits ≤ bits.length := by
  induction bits with
  | nil => exact Nat.le_refl 0
  | cons accepted remaining inductionHypothesis =>
      cases accepted with
      | false =>
          simpa [scanLength, Nat.add_comm] using
            Nat.add_le_add_left inductionHypothesis 1
      | true => simp [scanLength]
theorem scanLength_append_le
    (left right : List Bool) :
    scanLength left ≤ scanLength (left ++ right) := by
  induction left with
  | nil => simp [scanLength]
  | cons accepted remaining inductionHypothesis =>
      cases accepted with
      | false =>
          simpa [scanLength] using
            Nat.add_le_add_left inductionHypothesis 1
      | true => simp [scanLength]
theorem scanLength_append_eq_of_true_mem
    (left right : List Bool)
    (containsTrue : true ∈ left) :
    scanLength (left ++ right) = scanLength left := by
  induction left with
  | nil => simp at containsTrue
  | cons accepted remaining inductionHypothesis =>
      cases accepted with
      | false =>
          have tailContains : true ∈ remaining := by
            rcases List.mem_cons.mp containsTrue with impossible | member
            · exact Bool.noConfusion impossible
            · exact member
          simp [scanLength, inductionHypothesis tailContains]
      | true => simp [scanLength]
def forwardCost (terminal : List Bool) : List Bool -> Nat
  | [] => 0
  | accepted :: remaining =>
      (if accepted then scanLength (remaining ++ terminal) else 0) +
        forwardCost terminal remaining
theorem scanLength_add_forwardCost
    (terminal bits : List Bool) :
    scanLength (bits ++ terminal) + forwardCost terminal bits =
      bits.length + scanLength terminal := by
  induction bits with
  | nil => simp [scanLength, forwardCost]
  | cons accepted remaining inductionHypothesis =>
      have lifted := congrArg (fun cost => 1 + cost) inductionHypothesis
      cases accepted <;>
        simpa [scanLength, forwardCost, Nat.add_assoc, Nat.add_comm,
          Nat.add_left_comm] using lifted
def cyclicCostAux (processedPrefix : List Bool) : List Bool -> Nat
  | [] => 0
  | accepted :: remaining =>
      (if accepted then scanLength (remaining ++ processedPrefix) else 0) +
        cyclicCostAux (processedPrefix ++ [accepted]) remaining
def cyclicCost (bits : List Bool) : Nat :=
  cyclicCostAux [] bits
private theorem cyclicCostAux_false_prefix
    (initial falsePrefix remaining : List Bool)
    (allFalse : forall bit, bit ∈ falsePrefix -> bit = false) :
    cyclicCostAux initial (falsePrefix ++ remaining) =
      cyclicCostAux (initial ++ falsePrefix) remaining := by
  induction falsePrefix generalizing initial with
  | nil => simp [cyclicCostAux]
  | cons bit falsePrefix inductionHypothesis =>
      have bitFalse : bit = false := allFalse bit (by simp)
      subst bit
      simp only [List.cons_append, cyclicCostAux, Bool.false_eq_true,
        ↓reduceIte, Nat.zero_add]
      rw [inductionHypothesis]
      · simp [List.append_assoc]
      · intro tailBit member
        exact allFalse tailBit (by simp [member])
private theorem cyclicCostAux_eq_forwardCost
    (terminal processed bits : List Bool)
    (terminalHasTrue : true ∈ terminal) :
    cyclicCostAux (terminal ++ processed) bits =
      forwardCost terminal bits := by
  induction bits generalizing processed with
  | nil => rfl
  | cons accepted remaining inductionHypothesis =>
      simp only [cyclicCostAux, forwardCost]
      have scanAbsorbs :
          scanLength (remaining ++ (terminal ++ processed)) =
            scanLength (remaining ++ terminal) := by
        rw [← List.append_assoc]
        exact scanLength_append_eq_of_true_mem
          (remaining ++ terminal) processed
          (List.mem_append.mpr (Or.inr terminalHasTrue))
      rw [scanAbsorbs]
      rw [show (terminal ++ processed) ++ [accepted] =
          terminal ++ (processed ++ [accepted]) by simp [List.append_assoc]]
      rw [inductionHypothesis (processed ++ [accepted])]
private theorem cyclicCostAux_zero_of_all_false
    (processedPrefix bits : List Bool)
    (allFalse : forall bit, bit ∈ bits -> bit = false) :
    cyclicCostAux processedPrefix bits = 0 := by
  induction bits generalizing processedPrefix with
  | nil => rfl
  | cons bit remaining inductionHypothesis =>
      have bitFalse : bit = false := allFalse bit (by simp)
      subst bit
      simp only [cyclicCostAux, Bool.false_eq_true, ↓reduceIte, Nat.zero_add]
      exact inductionHypothesis (processedPrefix ++ [false]) (by
        intro tailBit member
        exact allFalse tailBit (by simp [member]))
theorem cyclicCost_le_length (bits : List Bool) :
    cyclicCost bits ≤ bits.length := by
  by_cases hasTrue : true ∈ bits
  · have found : bits.find? id = some true := by
      have someTrue : exists value, value ∈ bits ∧ id value = true :=
        ⟨true, hasTrue, rfl⟩
      rcases List.find?_isSome.mpr someTrue with foundSome
      cases equation : bits.find? id with
      | none => simp [equation] at foundSome
      | some value =>
          have valueTrue : value = true := by
            have := List.find?_some equation
            simpa using this
          simpa [valueTrue] using equation
    rcases List.find?_eq_some_iff_append.mp found with
      ⟨_, rejectedPrefix, remaining, bitsEquation, prefixRejected⟩
    have prefixFalse : forall bit, bit ∈ rejectedPrefix -> bit = false := by
      intro bit member
      have rejected := prefixRejected bit member
      cases bit <;> simp_all
    subst bits
    let terminal : List Bool := rejectedPrefix ++ [true]
    have terminalHasTrue : true ∈ terminal := by simp [terminal]
    have decomposition :
        cyclicCost (rejectedPrefix ++ true :: remaining) =
          scanLength (remaining ++ rejectedPrefix) +
            cyclicCostAux terminal remaining := by
      unfold cyclicCost
      rw [cyclicCostAux_false_prefix [] rejectedPrefix (true :: remaining)
        prefixFalse]
      simp [cyclicCostAux, terminal, List.append_assoc]
    have continuationIsForward :
        cyclicCostAux terminal remaining = forwardCost terminal remaining := by
      simpa using
        cyclicCostAux_eq_forwardCost terminal [] remaining terminalHasTrue
    have outstandingBound :
        scanLength (remaining ++ rejectedPrefix) ≤
          scanLength (remaining ++ terminal) := by
      unfold terminal
      simpa only [List.append_assoc] using
        scanLength_append_le (remaining ++ rejectedPrefix) [true]
    have conserved := scanLength_add_forwardCost terminal remaining
    have firstStep :
        scanLength (remaining ++ rejectedPrefix) + forwardCost terminal remaining ≤
          scanLength (remaining ++ terminal) +
            forwardCost terminal remaining :=
      Nat.add_le_add_right outstandingBound _
    have terminalLengthBound := scanLength_le_length terminal
    calc
      cyclicCost (rejectedPrefix ++ true :: remaining) =
          scanLength (remaining ++ rejectedPrefix) +
            cyclicCostAux terminal remaining := decomposition
      _ = scanLength (remaining ++ rejectedPrefix) +
          forwardCost terminal remaining := by rw [continuationIsForward]
      _ ≤
          scanLength (remaining ++ terminal) +
            forwardCost terminal remaining := firstStep
      _ = remaining.length + scanLength terminal := conserved
      _ ≤ remaining.length + terminal.length :=
        Nat.add_le_add_left terminalLengthBound _
      _ = (rejectedPrefix ++ true :: remaining).length := by
        simp [terminal, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
  · have allFalse : forall bit, bit ∈ bits -> bit = false := by
      intro bit member
      cases bit with
      | false => rfl
      | true => exact False.elim (hasTrue member)
    unfold cyclicCost
    rw [cyclicCostAux_zero_of_all_false [] bits allFalse]
    exact Nat.zero_le _
structure FirstAccepted (Query : Type uScalar) where
  selected : Query
  attempts : List Query
  found : Bool
def firstAccepted
    {Query : Type uScalar}
    (accepts : Query -> Bool)
    (fallback : Query) : List Query -> FirstAccepted Query
  | [] => { selected := fallback, attempts := [], found := false }
  | candidate :: remaining =>
      if accepts candidate then
        { selected := candidate, attempts := [candidate], found := true }
      else
        let result := firstAccepted accepts fallback remaining
        { result with attempts := candidate :: result.attempts }
@[simp] theorem firstAccepted_found
    {Query : Type uScalar}
    (accepts : Query -> Bool)
    (fallback : Query)
    (candidates : List Query) :
    (firstAccepted accepts fallback candidates).found =
      candidates.any accepts := by
  induction candidates with
  | nil => rfl
  | cons candidate remaining inductionHypothesis =>
      cases accepted : accepts candidate <;>
        simp [firstAccepted, accepted, inductionHypothesis]
theorem firstAccepted_found_spec
    {Query : Type uScalar}
    (accepts : Query -> Bool)
    (fallback : Query)
    (candidates : List Query)
    (found : (firstAccepted accepts fallback candidates).found = true) :
    (firstAccepted accepts fallback candidates).selected ∈ candidates ∧
      accepts (firstAccepted accepts fallback candidates).selected = true := by
  induction candidates with
  | nil => simp [firstAccepted] at found
  | cons candidate remaining inductionHypothesis =>
      cases accepted : accepts candidate with
      | false =>
          simp only [firstAccepted, accepted, Bool.false_eq_true, ↓reduceIte]
          have tailFound :
              (firstAccepted accepts fallback remaining).found = true := by
            simpa [firstAccepted, accepted] using found
          rcases inductionHypothesis tailFound with ⟨member, selectedAccepted⟩
          exact ⟨by simp [member], selectedAccepted⟩
      | true =>
          simp [firstAccepted, accepted]
theorem firstAccepted_attempts_length
    {Query : Type uScalar}
    (accepts : Query -> Bool)
    (fallback : Query)
    (candidates : List Query) :
    (firstAccepted accepts fallback candidates).attempts.length =
      scanLength (candidates.map accepts) := by
  induction candidates with
  | nil => rfl
  | cons candidate remaining inductionHypothesis =>
      cases accepted : accepts candidate <;>
        simp [firstAccepted, scanLength, accepted, inductionHypothesis,
          Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]
theorem firstAccepted_not_found_selected
    {Query : Type uScalar}
    (accepts : Query -> Bool)
    (fallback : Query)
    (candidates : List Query)
    (notFound : (firstAccepted accepts fallback candidates).found = false) :
    (firstAccepted accepts fallback candidates).selected = fallback := by
  induction candidates with
  | nil => rfl
  | cons candidate remaining inductionHypothesis =>
      cases accepted : accepts candidate with
      | false =>
          have tailNotFound :
              (firstAccepted accepts fallback remaining).found = false := by
            simpa [firstAccepted, accepted] using notFound
          simpa [firstAccepted, accepted] using inductionHypothesis tailNotFound
      | true => simp [firstAccepted, accepted] at notFound
abbrev PointWord
    (Scalar : Type uScalar)
    (coordinates : Nat) :=
  Fin coordinates -> CyclicPoint Scalar
def decodeWord
    {coordinates : Nat}
    (word : PointWord Scalar coordinates) :
    ChallengeVector Scalar coordinates :=
  fun coordinate => (word coordinate).value
def replaceCoordinate
    {coordinates : Nat}
    (base : ChallengeVector Scalar coordinates)
    (coordinate : Fin coordinates)
    (replacement : Scalar) : ChallengeVector Scalar coordinates :=
  fun index => if index = coordinate then replacement else base index
@[simp] theorem replaceCoordinate_same
    {coordinates : Nat}
    (base : ChallengeVector Scalar coordinates)
    (coordinate : Fin coordinates)
    (replacement : Scalar) :
    replaceCoordinate base coordinate replacement coordinate = replacement := by
  simp [replaceCoordinate]
theorem replaceCoordinate_other
    {coordinates : Nat}
    (base : ChallengeVector Scalar coordinates)
    (coordinate index : Fin coordinates)
    (replacement : Scalar)
    (different : index ≠ coordinate) :
    replaceCoordinate base coordinate replacement index = base index := by
  simp [replaceCoordinate, different]
def coordinateCandidates
    {coordinates : Nat}
    (word : PointWord Scalar coordinates)
    (coordinate : Fin coordinates) :
    List (ChallengeVector Scalar coordinates) :=
  (word coordinate).alternatives.map fun replacement =>
    replaceCoordinate (decodeWord word) coordinate replacement
structure RunResult (Scalar : Type uScalar) (coordinates : Nat) where
  base : ChallengeVector Scalar coordinates
  searches : Fin coordinates ->
    FirstAccepted (ChallengeVector Scalar coordinates)
def run
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates) :
    RunResult Scalar coordinates :=
  let base := decodeWord word
  if accepts base then
    {
      base := base
      searches := fun coordinate =>
        firstAccepted accepts base (coordinateCandidates word coordinate)
    }
  else
    {
      base := base
      searches := fun _ =>
        { selected := base, attempts := [], found := false }
    }
def RunResult.sample
    (result : RunResult Scalar coordinates) :
    ForkSample Scalar coordinates where
  base := result.base
  forks := fun coordinate => (result.searches coordinate).selected
def RunResult.trace
    (result : RunResult Scalar coordinates) :
    List (ChallengeVector Scalar coordinates) :=
  result.base :: (List.finRange coordinates).flatMap fun coordinate =>
    (result.searches coordinate).attempts
def RunResult.queryCount
    (result : RunResult Scalar coordinates) : Nat :=
  result.trace.length
def RunResult.successBool
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (result : RunResult Scalar coordinates) : Bool :=
  accepts result.base &&
    (List.finRange coordinates).all fun coordinate =>
      (result.searches coordinate).found
def RunResult.badBool
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (result : RunResult Scalar coordinates) : Bool :=
  accepts result.base &&
    !(List.finRange coordinates).all fun coordinate =>
      (result.searches coordinate).found
@[simp] theorem run_base
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates) :
    (run accepts word).base = decodeWord word := by
  by_cases accepted : accepts (decodeWord word) = true <;>
    simp [run, accepted]
theorem run_rejected_search
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates)
    (rejected : accepts (decodeWord word) = false)
    (coordinate : Fin coordinates) :
    (run accepts word).searches coordinate =
      { selected := decodeWord word, attempts := [], found := false } := by
  simp [run, rejected]
theorem run_rejected_trace
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates)
    (rejected : accepts (decodeWord word) = false) :
    (run accepts word).trace = [decodeWord word] := by
  unfold RunResult.trace
  rw [run_base]
  apply congrArg (List.cons (decodeWord word))
  apply List.flatMap_eq_nil_iff.mpr
  intro coordinate _
  rw [run_rejected_search accepts word rejected coordinate]
theorem run_accepted_search
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates)
    (accepted : accepts (decodeWord word) = true)
    (coordinate : Fin coordinates) :
    (run accepts word).searches coordinate =
      firstAccepted accepts (decodeWord word)
        (coordinateCandidates word coordinate) := by
  simp [run, accepted]
theorem coordinateCandidates_member
    {word : PointWord Scalar coordinates}
    {coordinate : Fin coordinates}
    {candidate : ChallengeVector Scalar coordinates}
    (member : candidate ∈ coordinateCandidates word coordinate) :
    exists replacement,
      replacement ∈ (word coordinate).alternatives ∧
      candidate = replaceCoordinate (decodeWord word) coordinate replacement := by
  rcases List.mem_map.mp member with ⟨replacement, replacementMember, rfl⟩
  exact ⟨replacement, replacementMember, rfl⟩
theorem coordinateCandidates_agreeExcept
    {word : PointWord Scalar coordinates}
    {coordinate index : Fin coordinates}
    {candidate : ChallengeVector Scalar coordinates}
    (member : candidate ∈ coordinateCandidates word coordinate)
    (different : index ≠ coordinate) :
    candidate index = decodeWord word index := by
  rcases coordinateCandidates_member member with
    ⟨replacement, _, rfl⟩
  exact replaceCoordinate_other _ _ _ _ different
theorem coordinateCandidates_changed
    (alphabet : Support Scalar)
    {word : PointWord Scalar coordinates}
    (wordMember : word ∈
      vectors (cyclicPointSupport alphabet).values coordinates)
    (coordinate : Fin coordinates)
    {candidate : ChallengeVector Scalar coordinates}
    (member : candidate ∈ coordinateCandidates word coordinate) :
    decodeWord word coordinate ≠ candidate coordinate := by
  have pointMember : word coordinate ∈ cyclicPoints alphabet.values :=
    (mem_vectors_iff (cyclicPointSupport alphabet).values coordinates word).mp
      wordMember coordinate
  rcases coordinateCandidates_member member with
    ⟨replacement, replacementMember, rfl⟩
  have replacementDistinct : replacement ≠ (word coordinate).value :=
    (mem_alternatives_iff alphabet.values alphabet.nodup
      (word coordinate) pointMember replacement).mp replacementMember |>.2
  intro equal
  apply replacementDistinct
  simpa only [replaceCoordinate_same] using equal.symm
/-- Every decoded seed coordinate belongs to the original scalar alphabet. -/
theorem decodedWord_mem
    (alphabet : Support Scalar)
    {word : PointWord Scalar coordinates}
    (wordMember : word ∈
      vectors (cyclicPointSupport alphabet).values coordinates) :
    forall coordinate, decodeWord word coordinate ∈ alphabet.values := by
  intro coordinate
  have pointMember : word coordinate ∈ cyclicPoints alphabet.values :=
    (mem_vectors_iff (cyclicPointSupport alphabet).values coordinates word).mp
      wordMember coordinate
  have mappedMember : (word coordinate).value ∈
      (cyclicPoints alphabet.values).map CyclicPoint.value :=
    List.mem_map.mpr ⟨word coordinate, pointMember, rfl⟩
  simpa only [cyclicPoints_values] using mappedMember

theorem decodedWord_valid
    (alphabet : Support Scalar)
    (valid : Scalar -> Prop)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values -> valid scalar)
    {word : PointWord Scalar coordinates}
    (wordMember : word ∈
      vectors (cyclicPointSupport alphabet).values coordinates) :
    forall coordinate, valid (decodeWord word coordinate) := by
  intro coordinate
  exact alphabetValid _ (decodedWord_mem alphabet wordMember coordinate)
theorem coordinateCandidate_valid
    (alphabet : Support Scalar)
    (valid : Scalar -> Prop)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values -> valid scalar)
    {word : PointWord Scalar coordinates}
    (wordMember : word ∈
      vectors (cyclicPointSupport alphabet).values coordinates)
    {coordinate : Fin coordinates}
    {candidate : ChallengeVector Scalar coordinates}
    (member : candidate ∈ coordinateCandidates word coordinate) :
    forall index, valid (candidate index) := by
  rcases coordinateCandidates_member member with
    ⟨replacement, replacementMember, rfl⟩
  have pointMember : word coordinate ∈ cyclicPoints alphabet.values :=
    (mem_vectors_iff (cyclicPointSupport alphabet).values coordinates word).mp
      wordMember coordinate
  have replacementValid : valid replacement := by
    exact alphabetValid replacement
      ((mem_alternatives_iff alphabet.values alphabet.nodup
        (word coordinate) pointMember replacement).mp replacementMember).1
  intro index
  by_cases equal : index = coordinate
  · subst index
    simpa using replacementValid
  · rw [replaceCoordinate_other _ _ _ _ equal]
    exact decodedWord_valid alphabet valid alphabetValid wordMember index
theorem successBool_implies_acceptedCoordinateFork
    (alphabet : Support Scalar)
    (valid : Scalar -> Prop)
    (verify : ChallengeVector Scalar coordinates -> Answer -> Prop)
    (oracle : Oracle Scalar Answer coordinates)
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (acceptsIff : forall challenge,
      accepts challenge = true <-> verify challenge (oracle challenge))
    (alphabetValid : forall scalar, scalar ∈ alphabet.values -> valid scalar)
    {word : PointWord Scalar coordinates}
    (wordMember : word ∈
      vectors (cyclicPointSupport alphabet).values coordinates)
    (success : (run accepts word).successBool accepts = true) :
    AcceptedCoordinateFork valid verify oracle (run accepts word).sample := by
  have successParts := Bool.and_eq_true_iff.mp success
  have baseAcceptedBool : accepts (decodeWord word) = true := by
    simpa only [run_base] using successParts.1
  have allFound : forall coordinate : Fin coordinates,
      ((run accepts word).searches coordinate).found = true := by
    intro coordinate
    have coordinateMember : coordinate ∈ List.finRange coordinates :=
      List.mem_finRange coordinate
    exact (List.all_eq_true.mp successParts.2) coordinate coordinateMember
  have selectedFacts : forall coordinate : Fin coordinates,
      ((run accepts word).searches coordinate).selected ∈
          coordinateCandidates word coordinate ∧
        accepts ((run accepts word).searches coordinate).selected = true := by
    intro coordinate
    rw [run_accepted_search accepts word baseAcceptedBool coordinate]
    exact firstAccepted_found_spec accepts (decodeWord word)
      (coordinateCandidates word coordinate) (by
        rw [← run_accepted_search accepts word baseAcceptedBool coordinate]
        exact allFound coordinate)
  constructor
  · simpa [RunResult.sample, run_base] using
      (acceptsIff _).mp baseAcceptedBool
  · intro coordinate
    simpa [RunResult.sample] using
      (acceptsIff _).mp (selectedFacts coordinate).2
  · simpa [RunResult.sample, run_base] using
      decodedWord_valid alphabet valid alphabetValid wordMember
  · intro coordinate
    simpa [RunResult.sample] using
      coordinateCandidate_valid alphabet valid alphabetValid wordMember
        (selectedFacts coordinate).1
  · intro coordinate index different
    have agreement := coordinateCandidates_agreeExcept
      (selectedFacts coordinate).1 different
    simpa [RunResult.sample, run_base] using agreement.symm
  · intro coordinate
    simpa [RunResult.sample, run_base] using
      coordinateCandidates_changed alphabet wordMember coordinate
        (selectedFacts coordinate).1
def oneDimensionalCost
    (acceptsValue : Scalar -> Bool)
    (point : CyclicPoint Scalar) : Nat :=
  if acceptsValue point.value then
    scanLength (point.alternatives.map acceptsValue)
  else
    0
def oneDimensionalBad
    (acceptsValue : Scalar -> Bool)
    (point : CyclicPoint Scalar) : Bool :=
  acceptsValue point.value && !point.alternatives.any acceptsValue
private theorem cyclicPointsAux_cost_eq
    (processedPrefix remaining : List Scalar)
    (acceptsValue : Scalar -> Bool) :
    ((cyclicPointsAux processedPrefix remaining).map
      (oneDimensionalCost acceptsValue)).sum =
        cyclicCostAux (processedPrefix.map acceptsValue)
          (remaining.map acceptsValue) := by
  induction remaining generalizing processedPrefix with
  | nil => rfl
  | cons value remaining inductionHypothesis =>
      simp [cyclicPointsAux, oneDimensionalCost, cyclicCostAux,
        inductionHypothesis, List.map_append, List.append_assoc]
theorem oneDimensionalCost_sum_le
    (alphabet : Support Scalar)
    (acceptsValue : Scalar -> Bool) :
    ((cyclicPoints alphabet.values).map
      (oneDimensionalCost acceptsValue)).sum ≤ alphabet.cardinality := by
  rw [show ((cyclicPoints alphabet.values).map
        (oneDimensionalCost acceptsValue)).sum =
      cyclicCost (alphabet.values.map acceptsValue) by
        exact cyclicPointsAux_cost_eq [] alphabet.values acceptsValue]
  have bound := cyclicCost_le_length (alphabet.values.map acceptsValue)
  simpa [Support.cardinality] using bound
private theorem cyclicPoint_eq_of_value_eq
    (alphabet : Support Scalar)
    {left right : CyclicPoint Scalar}
    (leftMember : left ∈ cyclicPoints alphabet.values)
    (rightMember : right ∈ cyclicPoints alphabet.values)
    (valuesEqual : left.value = right.value) :
    left = right := by
  let points := cyclicPoints alphabet.values
  let relation : CyclicPoint Scalar -> CyclicPoint Scalar -> Prop :=
    fun first second => first.value = second.value -> first = second
  have mappedNodup : (points.map CyclicPoint.value).Nodup := by
    simpa [points] using alphabet.nodup
  have valuesPairwise :
      points.Pairwise (fun first second => first.value ≠ second.value) := by
    exact List.pairwise_map.mp mappedNodup
  have forward : points.Pairwise relation :=
    valuesPairwise.imp (by
      intro first second distinct equal
      exact False.elim (distinct equal))
  have backward : points.Pairwise (flip relation) :=
    valuesPairwise.imp (by
      intro first second distinct equal
      exact False.elim (distinct equal.symm))
  have reflexive : forall point, point ∈ points -> relation point point := by
    intro point _ _
    rfl
  exact List.Pairwise.forall_of_forall_of_flip reflexive forward backward
    leftMember rightMember valuesEqual
private theorem countP_le_one_of_unique
    {Element : Type uScalar}
    (values : List Element)
    (event : Element -> Bool)
    (nodup : values.Nodup)
    (unique : forall left, left ∈ values ->
      forall right, right ∈ values ->
        event left = true -> event right = true -> left = right) :
    values.countP event ≤ 1 := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      have nodupParts := List.nodup_cons.mp nodup
      cases headEvent : event head with
      | false =>
          simpa [List.countP_cons, headEvent] using
            inductionHypothesis nodupParts.2 (by
              intro left leftMember right rightMember
              exact unique left (by simp [leftMember]) right
                (by simp [rightMember]))
      | true =>
          have tailZero : tail.countP event = 0 := by
            apply List.countP_eq_zero.mpr
            intro point pointMember pointEvent
            have equal := unique head (by simp) point
              (by simp [pointMember]) headEvent pointEvent
            exact nodupParts.1 (equal ▸ pointMember)
          simp [List.countP_cons, headEvent, tailZero]
theorem oneDimensionalBad_count_le_one
    (alphabet : Support Scalar)
    (acceptsValue : Scalar -> Bool) :
    (cyclicPoints alphabet.values).countP
      (oneDimensionalBad acceptsValue) ≤ 1 := by
  apply countP_le_one_of_unique
    (cyclicPoints alphabet.values) (oneDimensionalBad acceptsValue)
    (cyclicPoints_nodup alphabet.values alphabet.nodup)
  intro left leftMember right rightMember leftBad rightBad
  have leftParts := Bool.and_eq_true_iff.mp leftBad
  have rightParts := Bool.and_eq_true_iff.mp rightBad
  by_cases valuesEqual : left.value = right.value
  · exact cyclicPoint_eq_of_value_eq alphabet leftMember rightMember
      valuesEqual
  · have rightAlternative : right.value ∈ left.alternatives :=
      (mem_alternatives_iff alphabet.values alphabet.nodup left leftMember
        right.value).mpr ⟨by
          have mapped : right.value ∈
              (cyclicPoints alphabet.values).map CyclicPoint.value :=
            List.mem_map.mpr ⟨right, rightMember, rfl⟩
          simpa only [cyclicPoints_values] using mapped,
        fun equal => valuesEqual equal.symm⟩
    have anyAccepted : left.alternatives.any acceptsValue = true :=
      List.any_eq_true.mpr ⟨right.value, rightAlternative, rightParts.1⟩
    simp [anyAccepted] at leftParts
private theorem nat_sum_map_le
    {Element : Type uScalar}
    (values : List Element)
    (left right : Element -> Nat)
    (ordered : forall value, value ∈ values -> left value ≤ right value) :
    (values.map left).sum ≤ (values.map right).sum := by
  induction values with
  | nil => exact Nat.le_refl 0
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons]
      exact Nat.add_le_add (ordered head (by simp))
        (inductionHypothesis (by
          intro value member
          exact ordered value (by simp [member])))
private theorem nat_sum_map_add
    {Element : Type uScalar}
    (values : List Element)
    (left right : Element -> Nat) :
    (values.map fun value => left value + right value).sum =
      (values.map left).sum + (values.map right).sum := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, inductionHypothesis]
      ac_rfl
private theorem nat_sum_swap
    {Left : Type uScalar}
    {Right : Type uAnswer}
    (left : List Left)
    (right : List Right)
    (value : Left -> Right -> Nat) :
    (left.map fun leftValue =>
      (right.map fun rightValue => value leftValue rightValue).sum).sum =
    (right.map fun rightValue =>
      (left.map fun leftValue => value leftValue rightValue).sum).sum := by
  induction left with
  | nil => simp [List.map_const', List.sum_replicate_nat]
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, inductionHypothesis]
      exact (nat_sum_map_add right
        (fun rightValue => value head rightValue)
        (fun rightValue =>
          (tail.map fun leftValue => value leftValue rightValue).sum)).symm
private theorem countP_eq_indicator_sum
    {Element : Type uScalar}
    (values : List Element)
    (event : Element -> Bool) :
    values.countP event =
      (values.map fun value => if event value then 1 else 0).sum := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      rw [List.countP_cons, List.map_cons, List.sum_cons, inductionHypothesis]
      exact Nat.add_comm _ _
private theorem countP_sum_swap
    {Left : Type uScalar}
    {Right : Type uAnswer}
    (left : List Left)
    (right : List Right)
    (event : Left -> Right -> Bool) :
    (left.map fun leftValue => right.countP (event leftValue)).sum =
      (right.map fun rightValue =>
        left.countP fun leftValue => event leftValue rightValue).sum := by
  simpa only [countP_eq_indicator_sum] using
    nat_sum_swap left right fun leftValue rightValue =>
      if event leftValue rightValue then 1 else 0
private theorem nat_sum_map_one_add
    {Element : Type uScalar}
    (values : List Element)
    (cost : Element -> Nat) :
    (values.map fun value => 1 + cost value).sum =
      values.length + (values.map cost).sum := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.sum_cons, List.length_cons,
        inductionHypothesis]
      omega
def coordinateCost
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates)
    (coordinate : Fin coordinates) : Nat :=
  if accepts (decodeWord word) then
    scanLength ((coordinateCandidates word coordinate).map accepts)
  else
    0
def coordinateBad
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates)
    (coordinate : Fin coordinates) : Bool :=
  accepts (decodeWord word) &&
    !(coordinateCandidates word coordinate).any accepts
@[simp] theorem decodeWord_prepend
    (head : CyclicPoint Scalar)
    (tail : PointWord Scalar coordinates) :
    decodeWord (prepend head tail) =
      prepend head.value (decodeWord tail) := by
  funext index
  refine Fin.cases ?_ ?_ index <;> simp [decodeWord, prepend]
@[simp] theorem coordinateCandidates_prepend_zero
    (head : CyclicPoint Scalar)
    (tail : PointWord Scalar coordinates) :
    coordinateCandidates (prepend head tail) 0 =
      head.alternatives.map fun replacement =>
        prepend replacement (decodeWord tail) := by
  apply List.map_congr_left
  intro replacement _
  funext index
  refine Fin.cases ?_ (fun tailIndex => ?_) index
  · simp [coordinateCandidates, replaceCoordinate, decodeWord, prepend]
  · have notZero := Fin.succ_ne_zero tailIndex
    simp [coordinateCandidates, replaceCoordinate, decodeWord, prepend,
      notZero]
@[simp] theorem coordinateCandidates_prepend_succ
    (head : CyclicPoint Scalar)
    (tail : PointWord Scalar coordinates)
    (coordinate : Fin coordinates) :
    coordinateCandidates (prepend head tail) coordinate.succ =
      (coordinateCandidates tail coordinate).map
        (prepend head.value) := by
  unfold coordinateCandidates
  rw [List.map_map]
  apply List.map_congr_left
  intro replacement _
  funext index
  refine Fin.cases ?_ ?_ index
  · have zeroNe : (0 : Fin (coordinates + 1)) ≠ coordinate.succ :=
      fun equality => Fin.succ_ne_zero coordinate equality.symm
    simp [replaceCoordinate, decodeWord, prepend, zeroNe]
  · intro tailIndex
    by_cases equal : tailIndex = coordinate
    · subst tailIndex
      simp [replaceCoordinate, decodeWord, prepend]
    · simp [replaceCoordinate, decodeWord, prepend, equal,
        Fin.succ_inj]
private theorem sum_vectors_succ
    {Value : Type uScalar}
    (alphabet : List Value)
    (coordinates : Nat)
    (cost : (Fin (coordinates + 1) -> Value) -> Nat) :
    ((vectors alphabet (coordinates + 1)).map cost).sum =
      (alphabet.map fun head =>
        ((vectors alphabet coordinates).map fun tail =>
          cost (prepend head tail)).sum).sum := by
  have nat_sum_flatMap : forall
      (values : List Value) (mapping : Value -> List Nat),
      (values.flatMap mapping).sum =
        (values.map fun value => (mapping value).sum).sum := by
    intro values mapping
    induction values with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.flatMap_cons, List.sum_append_nat, List.map_cons,
          List.sum_cons, inductionHypothesis]
  rw [vectors, List.map_flatMap]
  simp only [List.map_map, Function.comp_def]
  change (alphabet.flatMap fun head =>
      (vectors alphabet coordinates).map fun tail =>
        cost (prepend head tail)).sum =
    (alphabet.map fun head =>
      ((vectors alphabet coordinates).map fun tail =>
        cost (prepend head tail)).sum).sum
  exact nat_sum_flatMap alphabet fun head =>
    (vectors alphabet coordinates).map fun tail => cost (prepend head tail)
private theorem countP_vectors_succ
    {Value : Type uScalar}
    (alphabet : List Value)
    (coordinates : Nat)
    (event : (Fin (coordinates + 1) -> Value) -> Bool) :
    (vectors alphabet (coordinates + 1)).countP event =
      (alphabet.map fun head =>
        (vectors alphabet coordinates).countP fun tail =>
          event (prepend head tail)).sum := by
  simp [vectors, List.countP_flatMap, Function.comp_def]
theorem coordinateCost_total_le
    (alphabet : Support Scalar)
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (coordinate : Fin coordinates) :
    ((vectors (cyclicPointSupport alphabet).values coordinates).map
      (fun word => coordinateCost accepts word coordinate)).sum ≤
        alphabet.cardinality ^ coordinates := by
  induction coordinates with
  | zero => exact Fin.elim0 coordinate
  | succ coordinates inductionHypothesis =>
      refine Fin.cases ?_ ?_ coordinate
      · rw [sum_vectors_succ]
        rw [nat_sum_swap]
        calc
          ((vectors (cyclicPointSupport alphabet).values coordinates).map
              fun tail =>
                (((cyclicPointSupport alphabet).values.map fun head =>
                  coordinateCost accepts (prepend head tail) 0).sum)).sum ≤
              ((vectors (cyclicPointSupport alphabet).values coordinates).map
                fun _ => alphabet.cardinality).sum := by
                  apply nat_sum_map_le
                  intro tail _
                  simpa [coordinateCost, oneDimensionalCost,
                    Function.comp_def] using
                    oneDimensionalCost_sum_le alphabet
                      (fun replacement =>
                        accepts (prepend replacement (decodeWord tail)))
          _ = alphabet.cardinality ^ (coordinates + 1) := by
            rw [List.map_const', List.sum_replicate_nat, vectors_length]
            have receiptLength :
                (cyclicPointSupport alphabet).values.length =
                  alphabet.cardinality := by
              simpa [Support.cardinality] using
                cyclicPointSupport_cardinality alphabet
            rw [receiptLength, Nat.pow_succ]
      · intro tailCoordinate
        rw [sum_vectors_succ]
        calc
          (((cyclicPointSupport alphabet).values.map fun head =>
              ((vectors (cyclicPointSupport alphabet).values coordinates).map
                fun tail => coordinateCost accepts
                  (prepend head tail) tailCoordinate.succ).sum).sum) ≤
              (((cyclicPointSupport alphabet).values.map fun _ =>
                alphabet.cardinality ^ coordinates).sum) := by
                  apply nat_sum_map_le
                  intro head _
                  simpa [coordinateCost, Function.comp_def] using
                    inductionHypothesis
                      (fun challenge => accepts (prepend head.value challenge))
                      tailCoordinate
          _ = alphabet.cardinality ^ (coordinates + 1) := by
            rw [List.map_const', List.sum_replicate_nat]
            have receiptLength :
                (cyclicPointSupport alphabet).values.length =
                  alphabet.cardinality := by
              simpa [Support.cardinality] using
                cyclicPoints_length alphabet.values
            rw [receiptLength, Nat.pow_succ]
            exact Nat.mul_comm _ _
theorem coordinateBad_count_le
    (alphabet : Support Scalar)
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (coordinate : Fin coordinates) :
    (vectors (cyclicPointSupport alphabet).values coordinates).countP
      (fun word => coordinateBad accepts word coordinate) ≤
        alphabet.cardinality ^ (coordinates - 1) := by
  induction coordinates with
  | zero => exact Fin.elim0 coordinate
  | succ coordinates inductionHypothesis =>
      refine Fin.cases ?_ ?_ coordinate
      · rw [countP_vectors_succ]
        rw [show (cyclicPointSupport alphabet).values =
          cyclicPoints alphabet.values by rfl]
        have transposed := countP_sum_swap
          (cyclicPoints alphabet.values)
          (vectors (cyclicPoints alphabet.values) coordinates)
          (fun head tail => coordinateBad accepts (prepend head tail) 0)
        rw [transposed]
        calc
          ((vectors (cyclicPoints alphabet.values) coordinates).map
              fun tail =>
                (cyclicPoints alphabet.values).countP fun head =>
                  coordinateBad accepts (prepend head tail) 0).sum ≤
              ((vectors (cyclicPoints alphabet.values) coordinates).map
                fun _ => 1).sum := by
                  apply nat_sum_map_le
                  intro tail _
                  simpa [coordinateBad, oneDimensionalBad,
                    Function.comp_def] using
                    oneDimensionalBad_count_le_one alphabet
                      (fun replacement =>
                        accepts (prepend replacement (decodeWord tail)))
          _ = alphabet.cardinality ^ ((coordinates + 1) - 1) := by
            rw [List.map_const', List.sum_replicate_nat, vectors_length,
              cyclicPoints_length]
            change alphabet.cardinality ^ coordinates * 1 =
              alphabet.cardinality ^ ((coordinates + 1) - 1)
            simp
      · intro tailCoordinate
        have coordinatesPos : 0 < coordinates := by
          exact Nat.zero_lt_of_lt tailCoordinate.isLt
        rw [countP_vectors_succ]
        calc
          ((cyclicPoints alphabet.values).map fun head =>
              (vectors (cyclicPointSupport alphabet).values coordinates).countP
                fun tail => coordinateBad accepts
                  (prepend head tail) tailCoordinate.succ).sum ≤
              ((cyclicPoints alphabet.values).map fun _ =>
                alphabet.cardinality ^ (coordinates - 1)).sum := by
                  apply nat_sum_map_le
                  intro head _
                  simpa [coordinateBad, Function.comp_def] using
                    inductionHypothesis
                      (fun challenge => accepts (prepend head.value challenge))
                      tailCoordinate
          _ = alphabet.cardinality ^ ((coordinates + 1) - 1) := by
            rw [List.map_const', List.sum_replicate_nat, cyclicPoints_length]
            change alphabet.cardinality *
                alphabet.cardinality ^ (coordinates - 1) =
              alphabet.cardinality ^ coordinates
            calc
              _ = alphabet.cardinality ^ (coordinates - 1) *
                    alphabet.cardinality := Nat.mul_comm _ _
              _ = alphabet.cardinality ^ ((coordinates - 1) + 1) :=
                (Nat.pow_succ _ _).symm
              _ = _ := by
                rw [Nat.sub_add_cancel coordinatesPos]
def wordBad
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates) : Bool :=
  (List.finRange coordinates).any fun coordinate =>
    coordinateBad accepts word coordinate
private theorem countP_or_le
    {Element : Type uScalar}
    (values : List Element)
    (left right : Element -> Bool) :
    values.countP (fun value => left value || right value) ≤
      values.countP left + values.countP right := by
  rw [countP_eq_indicator_sum, countP_eq_indicator_sum,
    countP_eq_indicator_sum, ← nat_sum_map_add]
  apply nat_sum_map_le
  intro value _
  cases left value <;> cases right value <;> simp
private theorem countP_any_le_sum
    {Element : Type uScalar}
    {Index : Type uAnswer}
    (values : List Element)
    (indices : List Index)
    (event : Index -> Element -> Bool) :
    values.countP (fun value => indices.any fun index => event index value) ≤
      (indices.map fun index => values.countP (event index)).sum := by
  induction indices with
  | nil => simp
  | cons index indices inductionHypothesis =>
      have union :
          values.countP (fun value =>
            event index value || indices.any fun next => event next value) ≤
            values.countP (event index) +
              values.countP (fun value =>
                indices.any fun next => event next value) :=
        countP_or_le values (event index)
          (fun value => indices.any fun next => event next value)
      simpa only [List.any_cons, List.map_cons, List.sum_cons] using
        Nat.le_trans union (Nat.add_le_add_left inductionHypothesis _)
theorem wordBad_count_le
    (alphabet : Support Scalar)
    (accepts : ChallengeVector Scalar coordinates -> Bool) :
    (vectors (cyclicPointSupport alphabet).values coordinates).countP
      (wordBad accepts) ≤
        coordinates * alphabet.cardinality ^ (coordinates - 1) := by
  unfold wordBad
  refine Nat.le_trans (countP_any_le_sum
    (values := vectors (cyclicPointSupport alphabet).values coordinates)
    (indices := List.finRange coordinates)
    (event := fun (coordinate : Fin coordinates)
        (word : PointWord Scalar coordinates) =>
      coordinateBad accepts word coordinate)) ?_
  calc
    ((List.finRange coordinates).map fun coordinate =>
        (vectors (cyclicPointSupport alphabet).values coordinates).countP
          fun word => coordinateBad accepts word coordinate).sum ≤
      ((List.finRange coordinates).map fun _ =>
        alphabet.cardinality ^ (coordinates - 1)).sum := by
          apply nat_sum_map_le
          intro coordinate _
          exact coordinateBad_count_le alphabet accepts coordinate
    _ = coordinates * alphabet.cardinality ^ (coordinates - 1) := by
      simp [List.map_const', List.sum_replicate_nat]
theorem base_success_or_wordBad
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates)
    (baseAccepted : accepts (decodeWord word) = true) :
    (run accepts word).successBool accepts = true \/
      wordBad accepts word = true := by
  by_cases allFound :
      (List.finRange coordinates).all (fun coordinate =>
        ((run accepts word).searches coordinate).found) = true
  · left
    simp [RunResult.successBool, run_base, baseAccepted, allFound]
  · right
    have allFalse :
        (List.finRange coordinates).all (fun coordinate =>
          ((run accepts word).searches coordinate).found) = false := by
      cases value : (List.finRange coordinates).all (fun coordinate =>
        ((run accepts word).searches coordinate).found) <;> simp_all
    rcases List.all_eq_false.mp allFalse with
      ⟨coordinate, coordinateMember, notFound⟩
    have foundFalse :
        ((run accepts word).searches coordinate).found = false := by
      cases found : ((run accepts word).searches coordinate).found <;>
        simp_all
    have noCandidate :
        (coordinateCandidates word coordinate).any accepts = false := by
      rw [run_accepted_search accepts word baseAccepted coordinate,
        firstAccepted_found] at foundFalse
      exact foundFalse
    unfold wordBad coordinateBad
    apply List.any_eq_true.mpr
    exact ⟨coordinate, coordinateMember, by simp [baseAccepted, noCandidate]⟩
theorem run_trace_length
    (accepts : ChallengeVector Scalar coordinates -> Bool)
    (word : PointWord Scalar coordinates) :
    (run accepts word).trace.length =
      1 + ((List.finRange coordinates).map fun coordinate =>
        coordinateCost accepts word coordinate).sum := by
  cases accepted : accepts (decodeWord word) with
  | false =>
      rw [run_rejected_trace accepts word accepted]
      simp [coordinateCost, accepted, List.map_const',
        List.sum_replicate_nat]
  | true =>
      unfold RunResult.trace
      simp only [List.length_cons, List.length_flatMap]
      rw [Nat.add_comm 1]
      apply congrArg (fun total => total + 1)
      apply congrArg List.sum
      apply List.map_congr_left
      intro coordinate _
      rw [run_accepted_search accepts word accepted coordinate,
        firstAccepted_attempts_length]
      simp [coordinateCost, accepted]
theorem total_trace_length_le
    (alphabet : Support Scalar)
    (accepts : ChallengeVector Scalar coordinates -> Bool) :
    ((vectors (cyclicPointSupport alphabet).values coordinates).map fun word =>
      (run accepts word).trace.length).sum ≤
        (coordinates + 1) * alphabet.cardinality ^ coordinates := by
  rw [show ((vectors (cyclicPointSupport alphabet).values coordinates).map
      fun word => (run accepts word).trace.length).sum =
        (vectors (cyclicPointSupport alphabet).values coordinates).length +
          ((vectors (cyclicPointSupport alphabet).values coordinates).map
            fun word => ((List.finRange coordinates).map fun coordinate =>
              coordinateCost accepts word coordinate).sum).sum by
      simpa only [run_trace_length] using nat_sum_map_one_add
        (vectors (cyclicPointSupport alphabet).values coordinates)
        (fun word => ((List.finRange coordinates).map fun coordinate =>
          coordinateCost accepts word coordinate).sum)]
  have transposed := nat_sum_swap
    (vectors (cyclicPointSupport alphabet).values coordinates)
    (List.finRange coordinates)
    (fun word (coordinate : Fin coordinates) =>
      coordinateCost accepts word coordinate)
  rw [transposed]
  have coordinateBound :
      ((List.finRange coordinates).map fun coordinate =>
        ((vectors (cyclicPointSupport alphabet).values coordinates).map
          fun word => coordinateCost accepts word coordinate).sum).sum ≤
        coordinates * alphabet.cardinality ^ coordinates := by
    calc
      _ ≤ ((List.finRange coordinates).map fun _ =>
          alphabet.cardinality ^ coordinates).sum := by
            apply nat_sum_map_le
            intro coordinate _
            exact coordinateCost_total_le alphabet accepts coordinate
      _ = coordinates * alphabet.cardinality ^ coordinates := by
        simp [List.map_const', List.sum_replicate_nat]
  rw [vectors_length]
  have receiptLength :
      (cyclicPointSupport alphabet).values.length = alphabet.cardinality := by
    simpa [Support.cardinality] using
      cyclicPointSupport_cardinality alphabet
  rw [receiptLength]
  let power := alphabet.cardinality ^ coordinates
  change power +
      ((List.finRange coordinates).map fun coordinate =>
        ((vectors (cyclicPointSupport alphabet).values coordinates).map
          fun word => coordinateCost accepts word coordinate).sum).sum ≤
    (coordinates + 1) * power
  calc
    _ ≤ power + coordinates * power :=
      Nat.add_le_add_left coordinateBound power
    _ = (coordinates + 1) * power := by
      rw [Nat.add_mul, Nat.one_mul, Nat.add_comm]

end Nightstream.SuperNeo.InteractiveReduction.CoordinateForking.FiniteUniform
