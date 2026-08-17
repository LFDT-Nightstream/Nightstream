import Nightstream.Assurance.CompactSourceArtifact

/-!
Chunk-decomposed family-certificate validity. Owns the bridge from
bounded per-chunk leaf certificates (candidate-filter equalities and
per-scalar support checks) to `FamilyCertificate.Valid` over the
artifact a `Wire` denotes. Owns no wire expansion and no certificate
semantics; those live in CompactSourceArtifact and
ConstraintMinimization.
-/

namespace Nightstream.Assurance.CompactSourceArtifact

open Nightstream.Assurance.ConstraintMinimization

/-- Filtering the chunked rows is the flat map of per-chunk filters. -/
theorem filter_artifactRows (wire : Wire) (pred : IndexedRow → Bool) :
    (artifactRows wire).filter pred =
      (List.range wire.chunkCount).flatMap
        (fun k => (rowsChunk wire k).filter pred) := by
  unfold artifactRows
  induction List.range wire.chunkCount with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp [List.flatMap_cons, List.filter_append, inductionHypothesis]

/-- Chunk membership lifts to artifact membership. -/
theorem mem_artifactRows_of_mem_chunk (wire : Wire) {row : IndexedRow}
    {chunk : Nat} (bound : chunk < wire.chunkCount)
    (member : row ∈ rowsChunk wire chunk) :
    row ∈ artifactRows wire := by
  unfold artifactRows
  exact List.mem_flatMap.mpr ⟨chunk, List.mem_range.mpr bound, member⟩

/-- One support's obligations, checked inside its own row chunk. -/
def supportOk (wire : Wire) (plan : List String) (family : String)
    (support : ScalarSupport) : Bool :=
  decide (support.source.sourceIndex / wire.chunkRows < wire.chunkCount) &&
    decide (support.source ∈
      rowsChunk wire (support.source.sourceIndex / wire.chunkRows)) &&
      decide (support.source.family ∈ plan) &&
        decide (support.source.family ≠ family)

theorem support_facts_of_supportOk (wire : Wire) (plan : List String)
    (family : String) (support : ScalarSupport)
    (ok : supportOk wire plan family support = true) :
    support.source ∈ artifactRows wire ∧
      support.source.family ∈ plan ∧
        support.source.family ≠ family := by
  unfold supportOk at ok
  simp only [Bool.and_eq_true, decide_eq_true_eq] at ok
  obtain ⟨⟨⟨bound, member⟩, planMember⟩, distinct⟩ := ok
  exact ⟨mem_artifactRows_of_mem_chunk wire bound member, planMember, distinct⟩

/-- One scalar certificate of the duplicate class: a single support
with coefficient one whose source row equals the candidate row. The
polynomial identity `residual = scalarCombination` is noncomputable in
general, but for this class it is a decidable structural fact. -/
def duplicateOk (scalar : ScalarCertificate) : Bool :=
  match scalar.support with
  | [support] =>
      decide (scalar.candidate.row = support.source.row) &&
        decide (support.coefficient = 1)
  | _ => false

theorem valid_of_duplicateOk (scalar : ScalarCertificate)
    (ok : duplicateOk scalar = true) : scalar.Valid := by
  unfold duplicateOk at ok
  match h : scalar.support with
  | [] => rw [h] at ok; cases ok
  | _ :: _ :: _ => rw [h] at ok; cases ok
  | [support] =>
      rw [h] at ok
      simp only [Bool.and_eq_true, decide_eq_true_eq] at ok
      obtain ⟨rowEq, coefficientOne⟩ := ok
      show Algebraic.residual scalar.candidate.row =
        scalarCombination scalar.support
      rw [h, rowEq]
      simp [scalarCombination, coefficientOne]

/-- Every support row cited by one part list. -/
def leafSupports (parts : List ScalarCertificate) : List IndexedRow :=
  parts.flatMap (fun scalar => scalar.support.map (fun support => support.source))

/-- All of one chunk's cited supports are rows of that chunk, with the
chunk expanded exactly once (the `let` keeps the expansion out of the
per-row loop). Rows whose home chunk differs are skipped here and
caught by the home census. -/
def supportsCovered (wire : Wire) (chunk : Nat)
    (parts : List ScalarCertificate) : Bool :=
  decide (chunk < wire.chunkCount) &&
    (let rows := rowsChunk wire chunk
     (leafSupports parts).all (fun source =>
       decide (source.sourceIndex / wire.chunkRows ≠ chunk) ||
         decide (source ∈ rows)))

/-- Duplicate shape plus plan facts for every part, with no chunk
expansion at all. -/
def scalarShapeOk (plan : List String) (family : String)
    (parts : List ScalarCertificate) : Bool :=
  parts.all (fun scalar =>
    duplicateOk scalar &&
      scalar.support.all (fun support =>
        decide (support.source.family ∈ plan) &&
          decide (support.source.family ≠ family)))

/-- Per-scalar validity facts from the three bounded leaf checks: the
home census routes every support to a listed chunk, `supportsCovered`
pins it inside that chunk, and the shape check carries the duplicate
and plan facts. -/
theorem scalar_facts_of_leaf_checks (wire : Wire) (plan : List String)
    (family : String) (parts : List ScalarCertificate)
    (chunks : List Nat)
    (covered : ∀ chunk ∈ chunks, supportsCovered wire chunk parts = true)
    (homes : (leafSupports parts).all (fun source =>
      decide (source.sourceIndex / wire.chunkRows ∈ chunks)) = true)
    (shape : scalarShapeOk plan family parts = true) :
    ∀ scalar ∈ parts,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ plan ∧
              support.source.family ≠ family := by
  intro scalar scalarMember
  unfold scalarShapeOk at shape
  rw [List.all_eq_true] at shape homes
  have shapeFacts := shape scalar scalarMember
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at shapeFacts
  obtain ⟨scalarDuplicate, planFacts⟩ := shapeFacts
  refine ⟨valid_of_duplicateOk scalar scalarDuplicate, ?_⟩
  intro support supportMember
  have sourceMember : support.source ∈ leafSupports parts :=
    List.mem_flatMap.mpr
      ⟨scalar, scalarMember,
        List.mem_map.mpr ⟨support, supportMember, rfl⟩⟩
  have home := homes support.source sourceMember
  rw [decide_eq_true_eq] at home
  have coveredHome := covered _ home
  unfold supportsCovered at coveredHome
  simp only [Bool.and_eq_true, decide_eq_true_eq] at coveredHome
  obtain ⟨bound, inChunk⟩ := coveredHome
  rw [List.all_eq_true] at inChunk
  have hit := inChunk support.source sourceMember
  simp only [Bool.or_eq_true, decide_eq_true_eq] at hit
  have member : support.source ∈
      rowsChunk wire (support.source.sourceIndex / wire.chunkRows) := by
    rcases hit with offChunk | member
    · exact absurd rfl offChunk
    · exact member
  have planFact := planFacts support supportMember
  exact ⟨mem_artifactRows_of_mem_chunk wire bound member,
    planFact.1, planFact.2⟩

/-- Assemble `FamilyCertificate.Valid` from per-chunk candidate leaves
and any per-scalar evidence (for example `scalar_facts_of_leaf_checks`). -/
theorem familyCertificate_valid_of_scalar_facts
    (wire : Wire) (plan : List String) (family : String)
    (parts : Nat → List ScalarCertificate)
    (memberFam : family ∈ wire.completeFamilies)
    (candLeaves : ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).filter (fun row => decide (row.family = family)) =
        (parts k).map (fun scalar => scalar.candidate))
    (scalars : ∀ k, k < wire.chunkCount → ∀ scalar ∈ parts k,
      scalar.Valid ∧
        ∀ support ∈ scalar.support,
          support.source ∈ artifactRows wire ∧
            support.source.family ∈ plan ∧
              support.source.family ≠ family) :
    FamilyCertificate.Valid
      ⟨family, (List.range wire.chunkCount).flatMap parts⟩
      (sourceArtifactOf wire) plan := by
  refine ⟨memberFam, ?_, ?_⟩
  · show ((List.range wire.chunkCount).flatMap parts).map
        (fun scalar => scalar.candidate) =
      candidateRows (sourceArtifactOf wire) family
    unfold candidateRows
    show _ = (artifactRows wire).filter
      (fun row => decide (row.family = family))
    rw [filter_artifactRows, List.map_flatMap]
    apply List.flatMap_congr
    intro chunk member
    rw [List.mem_range] at member
    exact (candLeaves chunk member).symm
  · intro scalar member
    rw [List.mem_flatMap] at member
    obtain ⟨chunk, chunkMember, scalarMember⟩ := member
    rw [List.mem_range] at chunkMember
    exact scalars chunk chunkMember scalar scalarMember

/-- Assemble `FamilyCertificate.Valid` from bounded per-chunk leaves.
The certificate list is the chunk-ordered flat map of `parts`, so the
candidate census composes chunk by chunk and every support fact stays
inside one chunk. -/
theorem familyCertificate_valid_of_chunk_parts
    (wire : Wire) (plan : List String) (family : String)
    (parts : Nat → List ScalarCertificate)
    (memberFam : family ∈ wire.completeFamilies)
    (candLeaves : ∀ k, k < wire.chunkCount →
      (rowsChunk wire k).filter (fun row => decide (row.family = family)) =
        (parts k).map (fun scalar => scalar.candidate))
    (scalarLeaves : ∀ k, k < wire.chunkCount →
      (parts k).all (fun scalar =>
        duplicateOk scalar &&
          scalar.support.all (supportOk wire plan family)) = true) :
    FamilyCertificate.Valid
      ⟨family, (List.range wire.chunkCount).flatMap parts⟩
      (sourceArtifactOf wire) plan := by
  refine ⟨memberFam, ?_, ?_⟩
  · show ((List.range wire.chunkCount).flatMap parts).map
        (fun scalar => scalar.candidate) =
      candidateRows (sourceArtifactOf wire) family
    unfold candidateRows
    show _ = (artifactRows wire).filter
      (fun row => decide (row.family = family))
    rw [filter_artifactRows, List.map_flatMap]
    apply List.flatMap_congr
    intro chunk member
    rw [List.mem_range] at member
    exact (candLeaves chunk member).symm
  · intro scalar member
    rw [List.mem_flatMap] at member
    obtain ⟨chunk, chunkMember, scalarMember⟩ := member
    rw [List.mem_range] at chunkMember
    have leaf := scalarLeaves chunk chunkMember
    rw [List.all_eq_true] at leaf
    have facts := leaf scalar scalarMember
    simp only [Bool.and_eq_true, List.all_eq_true] at facts
    obtain ⟨scalarDuplicate, supports⟩ := facts
    refine ⟨valid_of_duplicateOk scalar scalarDuplicate, ?_⟩
    intro support supportMember
    have ok := supports support supportMember
    have supportFacts :=
      support_facts_of_supportOk wire plan family support ok
    exact ⟨supportFacts.1, supportFacts.2.1, supportFacts.2.2⟩

end Nightstream.Assurance.CompactSourceArtifact
