import NightstreamFPrime.Spec.Compression.Akita.Key30
import NightstreamFPrime.Spec.Compression.Akita.Key20Complete

/-! The numbers of the honest prover of the opening of Akita for the schedule file
`crates/ns-compress/artifacts/key30.aks` (Akita, ePrint 2026/1983, revision of 2026-11-23,
Theorem 10.9 with Lemma 10.7, Section 9 with (136), Figure 4 and Section 8.2). The conditions on
the numbers that `Key30.lean` does not state, in the form of `Key20Complete.lean` and
`Key20Root.lean`.

- `root_folds`, `root_cells` and `root_path`: the response base 8 of each group of the root, its
  tables of `2^30`, `2^vars` and `2^28` cells, and the compression maps of its matrix `D`.
- `level₁_folds`, `level₁_path` and `level₁_bounded`: the response base 16 of each group of
  level 1, the compression maps of its matrix `D`, and the bounded source of its witness group.
  The group of the setup prefix has the full-field source (`honest`).
- `level₁_openings`: the base `b_1 = 16` of each group of level 1.
- `lifts`: the levels 2 and 3 have the bounded source, `b_z ≤ b⋆` and the compression maps of
  their matrix `D`.
- `widths`: a message of the range tree has 6 field elements for the bases 8 and 16, at most 8
  for the base 32, and at most 8 for the base 16 with the direct norm check.
- `source₄` to `sourceTerm`, `fold₄` to `fold₆`: the conditions `SourceSized` and `FoldSized` of
  the levels 4 to 6 and of the terminal, with the segments `(ρ_pack, ρ_pos, ρ_blk)` of the lengths
  (6, 9, 3), (5, 9, 3), (5, 8, 3) and (5, 8, 3). The levels 5 and 6 and the terminal have the
  numbers of `key20.aks`.
- `links`: the link from level 3 to level 4 and the links of the levels 4 to 6, in the form of
  `Key20.links` and `Key20.lifts`. `capped`: the bounds `S_max` of the levels 5 and 6 are at most
  `3029804899`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Key30

open NightstreamFPrime.Spec.Compression.Akita.Extract

/-- Each group of the root has the response base `b_z = 8` with the conditions `Radix`, and
`b_z ≤ b⋆`. -/
theorem root_folds : ∀ r ∈ rows, ∀ g, Radix ((root r).fold g).responseBase ∧
    ((root r).fold g).responseBase ≤ ((root r).shape g).certifiedBase := by
  have facts : ∀ r ∈ rows, ∀ g, ((root r).fold g).responseBase % 2 = 0 ∧
      2 ≤ ((root r).fold g).responseBase ∧ ((root r).fold g).responseBase < fieldModulus ∧
      ((root r).fold g).responseBase ≤ ((root r).shape g).certifiedBase := by decide
  exact fun r member g => ⟨⟨(facts r member g).1, (facts r member g).2.1,
    (facts r member g).2.2.1⟩, (facts r member g).2.2.2⟩

/-- The blocks of the groups of the root have `2^30`, `2^vars` and `2^28` cells. -/
theorem root_cells : ∀ r ∈ rows, ∀ g, ((root r).shape g).blockLength *
    ((root r).shape g).innerDegree * ((root r).shape g).blocks = 2 ^ ![30, r.vars, 28] g := by
  decide

/-- The inputs of the compression maps of the shared matrix `D` of the root have a position for
each digit. -/
theorem root_path : ∀ r ∈ rows,
    PathFits (root r).openDegree (1 * (root r).openRank) (root r).chain := by
  intro r member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> exact pathFits_of (by decide) (by decide)

/-- Each group of level 1 has the response base `b_z = 16` with the conditions `Radix`, and
`b_z ≤ b⋆`. -/
theorem level₁_folds : ∀ r ∈ rows, ∀ g, Radix ((level₁ r).fold g).responseBase ∧
    ((level₁ r).fold g).responseBase ≤ ((level₁ r).shape g).certifiedBase := by
  have facts : ∀ r ∈ rows, ∀ g, ((level₁ r).fold g).responseBase % 2 = 0 ∧
      2 ≤ ((level₁ r).fold g).responseBase ∧ ((level₁ r).fold g).responseBase < fieldModulus ∧
      ((level₁ r).fold g).responseBase ≤ ((level₁ r).shape g).certifiedBase := by decide
  exact fun r member g => ⟨⟨(facts r member g).1, (facts r member g).2.1,
    (facts r member g).2.2.1⟩, (facts r member g).2.2.2⟩

/-- The inputs of the compression maps of the shared matrix `D` of level 1 have a position for
each digit. -/
theorem level₁_path : ∀ r ∈ rows,
    PathFits (level₁ r).openDegree (1 * (level₁ r).openRank) (level₁ r).chain := by
  intro r member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;> exact pathFits_of (by decide) (by decide)

/-- The shape of the witness group of level 1 satisfies the conditions `Shape.Bounded` of the
commitment algorithm with a bounded source. -/
theorem level₁_bounded : ∀ r ∈ rows, (shapeW₁ r).Bounded := by
  intro r member
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;>
    exact ⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide,
      pathFits_of (by decide) (by decide)⟩

/-- Each group of level 1 has the base `b_1 = 16` with 16 digits for each residue, and
`b_1 ≤ b⋆`. -/
theorem level₁_openings : ∀ r ∈ rows, ∀ g,
    Base ((level₁ r).shape g).openingBase ((level₁ r).shape g).openingDigits ∧
      ((level₁ r).shape g).openingBase ≤ ((level₁ r).shape g).certifiedBase := by
  intro r member g
  fin_cases g
  · exact ⟨honest.2.opening, honest.2.certified⟩
  · exact ⟨(level₁_bounded r member).opening, (level₁_bounded r member).certified⟩

/-- The levels 2 and 3 satisfy the conditions `Shape.Bounded`, have `b_z = 16 ≤ b⋆` with the
conditions `Radix`, and the compression maps of their matrix `D` have a position for each binary
digit. -/
theorem lifts : (∀ r ∈ rows, (level₂ r).shape.Bounded ∧ Radix (level₂ r).fold.responseBase ∧
      (level₂ r).fold.responseBase ≤ (level₂ r).shape.certifiedBase ∧
      PathFits (level₂ r).opening.openDegree (1 * (level₂ r).opening.openRank)
        (level₂ r).opening.chain) ∧
    level₃.shape.Bounded ∧ Radix level₃.fold.responseBase ∧
      level₃.fold.responseBase ≤ level₃.shape.certifiedBase ∧
      PathFits level₃.opening.openDegree (1 * level₃.opening.openRank) level₃.opening.chain := by
  refine ⟨fun r member => ?_, ⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide,
    by decide, pathFits_of (by decide) (by decide)⟩, ⟨by decide, by decide, by decide⟩, by decide,
    pathFits_of (by decide) (by decide)⟩
  simp only [rows, List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl <;>
    exact ⟨⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide,
      pathFits_of (by decide) (by decide)⟩, ⟨by decide, by decide, by decide⟩, by decide,
      pathFits_of (by decide) (by decide)⟩

/-- The largest number of field elements in one message of the range tree: 6 for the bases 8
and 16, at most 8 for the base 32, and at most 8 for the base 16 with the direct norm check. -/
theorem widths : rangeWidth 8 = 6 ∧ rangeWidth 16 = 6 ∧ rangeWidth 32 ≤ 8 ∧
    rangeWidthSq 16 ≤ 8 := by decide

theorem source₄ : SourceSized level₄.shape 32 6 9 3 :=
  ⟨⟨⟨by decide, by decide, by decide, by decide⟩, by decide, by decide, by decide,
    pathFits_of (by decide) (by decide)⟩,
    (ZMod.isUnit_iff_coprime _ _).mpr (by decide), by decide, by decide, by decide⟩

theorem source₅ : SourceSized level₅.shape 16 5 9 3 := Key20.source₄

theorem source₆ : SourceSized level₆.shape 16 5 8 3 := Key20.source₅

theorem sourceTerm : SourceSized terminal 16 5 8 3 := Key20.source₆

theorem fold₄ : FoldSized level₄.shape level₄.opening level₄.fold 8 :=
  ⟨⟨by decide, by decide, by decide⟩, by decide, pathFits_of (by decide) (by decide), by decide,
    by decide, by decide, by decide⟩

theorem fold₅ : FoldSized level₅.shape level₅.opening level₅.fold 8 := Key20.fold₄

theorem fold₆ : FoldSized level₆.shape level₆.opening level₆.fold 8 := Key20.fold₅

/-- The conditions of the link from level 3 to level 4 and of the links of the levels 4 to 6:
`μ = ν + 1` at the levels 3 to 6, the count `ν` of the coordinates of the point of each receiver,
and the cells of the quotient digits of level 3 and of each flat witness of the levels 4 to 6 in
the live blocks of its receiver, as `2·B·M·(d_A/k)` coordinates of `Z_q` (Akita Section 4.5, (40)
to (42)). -/
theorem links :
    (level₃.vars = level₃.tail + 1 ∧ level₄.vars = level₄.tail + 1 ∧
      level₅.vars = level₅.tail + 1 ∧ level₆.vars = level₆.tail + 1) ∧
    (level₃.tail = 6 + (9 + 3) ∧ level₄.tail = 5 + (9 + 3) ∧ level₅.tail = 5 + (8 + 3) ∧
      level₆.tail = 5 + (8 + 3)) ∧
    level₃.opening.quotientRoom level₃.fold quotientDigits ≤
        2 * (level₄.shape.blocks * (level₄.shape.blockLength * (2 * 32))) ∧
      level₄.opening.cells level₄.fold ≤
        2 * (level₅.shape.blocks * (level₅.shape.blockLength * (2 * 16))) ∧
      level₅.opening.cells level₅.fold ≤
        2 * (level₆.shape.blocks * (level₆.shape.blockLength * (2 * 16))) ∧
      level₆.opening.cells level₆.fold ≤
        2 * (terminal.blocks * (terminal.blockLength * (2 * 16))) := by decide

/-- The bounds `S_max` of the levels 5 and 6 are at most `3029804899`, the energy that a wire
message of the checks can have. -/
theorem capped : norm₅.limit ≤ 3029804899 ∧ norm₆.limit ≤ 3029804899 := by decide

end NightstreamFPrime.Spec.Compression.Akita.Key30
