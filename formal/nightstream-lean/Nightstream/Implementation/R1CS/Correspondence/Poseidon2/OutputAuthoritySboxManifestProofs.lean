import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.OutputAuthoritySboxManifestSchema

/-!
Contract: model-level consequences of a kernel-checked output-authority S-box
manifest certificate.

Owns: transport across every generated call, compact candidate-count and
family derivations, mapped-interval facts, uniqueness and boundary
disjointness, and consequences of an explicitly supplied complete-matrix
no-escape witness.

Does not own: the generated production data, source-row matching, a complete
matrix-use extractor, Rust conformance, centered substitution, or permission
to remove rows or committed slots.

Emits constraints: no.

Authority boundary: `Manifest.Certificate` checks generated geometry and
census data. `WholeMatrixNoEscape` is a separate premise; Rust acceptance
metadata never discharges it inside Lean.

Assurance tier: model-level, with artifact-checked instantiation only after
the generated `.expected` file is reviewed and promoted.

| Theorem | Mathematical obligation | Guarantee | Remaining premise | Permits row removal? |
|---|---|---|---|---|
| `allCalls_transport` | 17 renamed isolated layouts | `poseidon2Call_transport` holds for every listed call | source rows still need `SourceCallRowsMatch` | no |
| `candidate_count` | compact Cartesian census | exactly 430 derived candidate entries | manifest certificate | no |
| `candidateColumns_nodup` | mapped interval injectivity | all derived candidate columns are unique | manifest certificate | no |
| `candidate_in_allocated_interval` | local mapped range | each offset stays inside its call's 600 fresh columns | manifest certificate | no |
| `family_census` | three Poseidon2 S-box phases | `32 + 22 + 32 = 86` per call | manifest certificate | no |
| `noEscape_use_roles` | complete matrix occurrence census | one C definition plus eight A uses per candidate | explicit `WholeMatrixNoEscape` | no |
-/

namespace Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest

open Nightstream.Implementation.R1CS.Poseidon2Sbox7OutputLayout

theorem candidateColumnsFor_length
    (calls : List CallGeometry) (offsets : List Nat) :
    (candidateColumnsFor calls offsets).length =
      calls.length * offsets.length := by
  induction calls with
  | nil => simp [candidateColumnsFor]
  | cons call rest ih =>
      simp [candidateColumnsFor, CallGeometry.candidateColumns, ih,
        Nat.succ_mul, Nat.add_comm]

theorem Manifest.candidateColumns_length (manifest : Manifest) :
    manifest.candidateColumns.length =
      manifest.calls.length * manifest.isolatedOutputOffsets.length := by
  exact candidateColumnsFor_length manifest.calls manifest.isolatedOutputOffsets

private theorem columnsStrictlyIncreasing_tail
    (head : Nat) (tail : List Nat)
    (ordered : columnsStrictlyIncreasing (head :: tail) = true) :
    columnsStrictlyIncreasing tail = true := by
  cases tail with
  | nil => rfl
  | cons next rest =>
      have pair : decide (head < next) = true ∧
          columnsStrictlyIncreasing (next :: rest) = true := by
        simpa only [columnsStrictlyIncreasing, Bool.and_eq_true] using ordered
      exact pair.2

private theorem columnsStrictlyIncreasing_head_lt
    (head : Nat) (tail : List Nat)
    (ordered : columnsStrictlyIncreasing (head :: tail) = true) :
    ∀ column ∈ tail, head < column := by
  induction tail generalizing head with
  | nil => simp
  | cons next rest ih =>
      have pair : decide (head < next) = true ∧
          columnsStrictlyIncreasing (next :: rest) = true := by
        simpa only [columnsStrictlyIncreasing, Bool.and_eq_true] using ordered
      have headNext : head < next := of_decide_eq_true pair.1
      have tailOrdered : columnsStrictlyIncreasing (next :: rest) = true := pair.2
      intro column member
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · exact headNext
      · exact Nat.lt_trans headNext (ih next tailOrdered column member)

theorem columnsStrictlyIncreasing_nodup
    (columns : List Nat)
    (ordered : columnsStrictlyIncreasing columns = true) :
    columns.Nodup := by
  induction columns with
  | nil => exact List.nodup_nil
  | cons head tail ih =>
      rw [List.nodup_cons]
      constructor
      · intro member
        exact (Nat.lt_irrefl head)
          (columnsStrictlyIncreasing_head_lt head tail ordered head member)
      · exact ih (columnsStrictlyIncreasing_tail head tail ordered)

/-- The compact manifest yields `5 * 86 = 430` candidate entries without
listing those entries in generated Lean. -/
theorem candidate_count
    (manifest : Manifest) (certificate : manifest.Certificate) :
    manifest.candidateColumns.length = 430 := by
  rw [manifest.candidateColumns_length, certificate.callCount,
    certificate.offsetCount]
  rcases certificate.censusValid with
    ⟨_, _, _, _, _, _, permutations, sboxes, _, _, _, _, _, _, _⟩
  simp [permutations, sboxes]

/-- Linear generated order gives global injectivity across all calls and all
86 local offsets. -/
theorem candidateColumns_nodup
    (manifest : Manifest) (certificate : manifest.Certificate) :
    manifest.candidateColumns.Nodup :=
  columnsStrictlyIncreasing_nodup manifest.candidateColumns
    certificate.candidateColumnsIncreasing

theorem candidate_in_allocated_interval
    (manifest : Manifest) (certificate : manifest.Certificate)
    (call : CallGeometry) (callMember : call ∈ manifest.calls)
    (offset : Nat) (offsetMember : offset ∈ manifest.isolatedOutputOffsets) :
    call.firstAllocatedColumn ≤ call.firstAllocatedColumn + offset ∧
      call.firstAllocatedColumn + offset <
        call.firstAllocatedColumn + call.allocatedColumnCount := by
  have callValid := certificate.everyCallValid call callMember
  have offsetBound := certificate.offsetsInAllocatedRange offset offsetMember
  rcases callValid with ⟨_, _, allocatedColumns, _⟩
  rw [allocatedColumns]
  omega

theorem candidate_offset_injective
    (call : CallGeometry) {left right : Nat}
    (equal : call.firstAllocatedColumn + left =
      call.firstAllocatedColumn + right) :
    left = right := by
  omega

theorem candidates_disjoint_from_boundaries
    (manifest : Manifest) (certificate : manifest.Certificate) :
    manifest.CandidatesDisjointFromBoundaries :=
  certificate.boundaryDisjoint

theorem family_census
    (manifest : Manifest) (certificate : manifest.Certificate) :
    manifest.families.initialExternal.width = 32 ∧
      manifest.families.partialRounds.width = 22 ∧
      manifest.families.terminalExternal.width = 32 ∧
      manifest.isolatedOutputOffsets.length = 86 := by
  rcases certificate.familiesValid with ⟨initial, middle, terminal⟩
  rcases certificate.censusValid with
    ⟨_, _, _, _, _, _, _, sboxes, _, _, _, _, _, _, _⟩
  rw [initial, middle, terminal]
  simp [NatRange.width, certificate.offsetCount, sboxes]

/-- The exact isolated 86-output schedule exported by
`Sbox7OutputLayout` is the manifest's compact offset schedule. -/
theorem offsets_match_isolated_layout
    (manifest : Manifest) (certificate : manifest.Certificate) :
    manifest.isolatedOutputOffsets =
      Poseidon2Sbox7OutputLayout.outputColumns.map (· - 9) :=
  certificate.offsetsExact

/-- Lift the existing isolated-layout theorem uniformly over every generated
call. This does not prove that those rows occur in a global program; that is
the separate `SourceCallRowsMatch` obligation. -/
theorem allCalls_transport (manifest : Manifest) :
    ∀ call ∈ manifest.calls,
      TransportedLayout call.toCall.columnMap call.toCall.rows := by
  intro call _
  exact poseidon2Call_transport call.toCall

theorem use_census
    (manifest : Manifest) (certificate : manifest.Certificate) :
    manifest.census.definitionCUses = 430 ∧
      manifest.census.linearAUses = 3440 ∧
      manifest.census.totalMatrixUses = 3870 := by
  rcases certificate.censusValid with
    ⟨_, _, _, _, _, _, permutations, sboxes, _, _, _, candidates,
      definitions, linear, total⟩
  simp [permutations, sboxes] at candidates
  simp [candidates] at definitions
  simp [candidates] at linear
  simp [definitions, linear] at total
  exact ⟨definitions, linear, total⟩

theorem expectedCandidateUses_length : expectedCandidateUses.length = 9 := by
  decide

/-- Only an explicit complete-matrix witness yields the per-candidate role
census in Lean. The Rust evidence boolean is intentionally unused. -/
theorem noEscape_use_roles
    (manifest : Manifest)
    (completeUses : Nat → List MatrixUseRole)
    (noEscape : manifest.WholeMatrixNoEscape completeUses) :
    ∀ column ∈ manifest.candidateColumns,
      completeUses column = expectedCandidateUses ∧
        (completeUses column).length = 9 := by
  intro column member
  have exactRoles := noEscape column member
  exact ⟨exactRoles, by rw [exactRoles]; exact expectedCandidateUses_length⟩

/-- The generated booleans are exposed only as Rust conformance evidence. -/
theorem rustEvidence_declared
    (manifest : Manifest) (certificate : manifest.Certificate) :
    manifest.rustEvidence.exactCallRowsAccepted = true ∧
      manifest.rustEvidence.wholeMatrixNoEscapeAccepted = true :=
  ⟨certificate.exactRowsEvidence, certificate.noEscapeEvidence⟩

end Nightstream.Implementation.R1CS.OutputAuthoritySboxManifest
