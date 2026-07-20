import Nightstream.Implementation.R1CS.Core.Program

/-!
Generic bounded certificate partitioning for the selective `y_zcol` slice.

Owns: lossless `take`/`drop` splitting by an explicit length vector, exact
chunk lengths and flattening, and transport from compact Boolean chunk checks
to every member of the original list.

Does not own: any concrete artifact, checker predicate, row semantics,
assignment construction, source authority, or security event.

Emits constraints: no.

| Utility leaf | Exact obligation | Authority class |
|---|---|---|
| chunk construction | an exact length vector splits and flattens to the original list in order | derived |
| size bound | every produced chunk has its declared length | derived |
| checker lift | true Boolean summaries imply the element predicate | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness.Chunking

def splitByLengths {α : Type} : List Nat → List α → List (List α)
  | [], _ => []
  | length :: lengths, values =>
      values.take length ::
        splitByLengths lengths (values.drop length)

theorem splitByLengths_lengths
    {α : Type} (lengths : List Nat) (values : List α)
    (within : lengths.sum ≤ values.length) :
    (splitByLengths lengths values).map List.length = lengths := by
  induction lengths generalizing values with
  | nil => rfl
  | cons length lengths inductionHypothesis =>
      simp only [List.sum_cons] at within
      simp only [splitByLengths, List.map_cons, List.cons.injEq]
      constructor
      · rw [List.length_take]
        omega
      · apply inductionHypothesis
        rw [List.length_drop]
        omega

theorem splitByLengths_flatten
    {α : Type} (lengths : List Nat) (values : List α)
    (exact : lengths.sum = values.length) :
    (splitByLengths lengths values).flatten = values := by
  induction lengths generalizing values with
  | nil =>
      simp only [List.sum_nil] at exact
      have empty : values = [] := List.eq_nil_of_length_eq_zero exact.symm
      simp [splitByLengths, empty]
  | cons length lengths inductionHypothesis =>
      simp only [List.sum_cons] at exact
      simp only [splitByLengths, List.flatten_cons]
      rw [inductionHypothesis]
      · exact List.take_append_drop length values
      · rw [List.length_drop]
        omega

def chunkChecks {α : Type} (parts : List (List α))
    (check : α → Bool) : List Bool :=
  parts.map fun part => part.all check

theorem check_eq_true_of_chunkChecks
    {α : Type} (lengths : List Nat) (items : List α)
    (check : α → Bool)
    (exact : lengths.sum = items.length)
    (checked :
      (chunkChecks (splitByLengths lengths items) check).all
          (fun value => value) = true) :
    ∀ item ∈ items, check item = true := by
  intro item member
  have flattenedMember :
      item ∈ (splitByLengths lengths items).flatten := by
    rw [splitByLengths_flatten lengths items exact]
    exact member
  rcases List.mem_flatten.mp flattenedMember with
    ⟨part, partMember, itemMember⟩
  have partCheckMember :
      part.all check ∈
        chunkChecks (splitByLengths lengths items) check :=
    List.mem_map.mpr ⟨part, partMember, rfl⟩
  have partChecked : part.all check = true :=
    (List.all_eq_true.mp checked) _ partCheckMember
  exact (List.all_eq_true.mp partChecked) item itemMember

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Completeness.Chunking
