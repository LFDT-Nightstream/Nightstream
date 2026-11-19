import NightstreamFPrime.Spec.Compression.Akita
import NightstreamFPrime.Spec.Compression.Akita.Commit

/-! Binding of the Akita commitment of the compression key (ePrint 2026/1983, revision of
2026-10-15), by a reduction to solutions of Module-SIS.

- `chainCollision` and `chainCollision_isSome` are Akita Lemma 4.3: two openings of one payload
  that differ give a solution for the original map or for a compression map.
- `collision` and `collision_isSome` are Akita Corollary 10.25 with a reference occurrence on one
  side (Akita (162) to (164)): a certified decommitment and the reference decommitment of one
  payload with different tables give a solution for `A`, `B`, `F_1` or `F_2`.
- `pcs` is the scheme whose decommitment relation is `Certified.Opens`, and `use` is an opening of
  a final proof on it. `binding` gives `Use.Binding` when the reduction `solve` returns a solution
  on the output of an extractor with probability at most `bindingError`. `commit_binding` is
  `binding` for the output of the commitment algorithm. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Game (event_ne_top)

/-- A solution of `MSIS(n_{F,j}, m_{F,j}, 1)` for a compression map `F_1` or `F_2` of a path
(Akita Lemma 4.3, item 2). A path without compression maps has none. -/
def PathKernel : (chain : Option ChainShape) → PathKey chain → Type
  | none, _ => Empty
  | some _, key => ShortKernel (ChainKey.first key) 1 ⊕ ShortKernel (ChainKey.second key) 1

/-- The difference of the digit vectors of `F_2`, or else of `F_1`, when it is a solution. -/
noncomputable def pathCollision : (chain : Option ChainShape) → (key : PathKey chain) →
    PathDigits chain → PathDigits chain → Option (PathKernel chain key)
  | none, _, _, _ => none
  | some _, key, digits, digits' =>
      ((ShortKernel.ofCollision? (ChainKey.second key) 1 (ChainDigits.second digits)
        (ChainDigits.second digits')).map Sum.inr).or
      ((ShortKernel.ofCollision? (ChainKey.first key) 1 (ChainDigits.first digits)
        (ChainDigits.first digits')).map Sum.inl)

theorem isSome_or_map {α β γ : Type} {a : Option α} {b : Option β} {f : α → γ} {g : β → γ} :
    ((a.map f).or (b.map g)).isSome ↔ a.isSome ∨ b.isSome := by
  cases a <;> cases b <;> simp

section Chain

variable {d n slices : ℕ} [NeZero d] {cols : Type} [Fintype cols]

/-- The reduction of Akita Lemma 4.3 for the map `M` on each of `slices` slices and the path
`chain`: from two openings `(x, digits)` and `(x', digits')`, a solution for a compression map, or
else a solution of `MSIS(n, m, η_in)` for `M` from the first slice whose inputs collide. -/
noncomputable def chainCollision (chain : Option ChainShape) (key : PathKey chain)
    (M : Matrix (Fin n) cols (Rq d)) (bound : ℕ) (x x' : Fin slices → cols → Rq d)
    (digits digits' : PathDigits chain) : Option (ShortKernel M bound ⊕ PathKernel chain key) :=
  ((pathCollision chain key digits digits').map Sum.inr).or
    (((List.finRange slices).findSome? fun s =>
      ShortKernel.ofCollision? M bound (x s) (x' s)).map Sum.inl)

/-- Akita Lemma 4.3: two openings that satisfy the rows (47) or (48) with one payload, with
`‖x_s − x'_s‖_∞ ≤ η_in` on each slice and compression digits in `{−1, 0}`, and that differ, give a
solution of Module-SIS for `M` with the bound `η_in` or for a compression map with the bound 1. -/
theorem chainCollision_isSome {chain : Option ChainShape} {key : PathKey chain}
    {M : Matrix (Fin n) cols (Rq d)} {bound : ℕ} {x x' : Fin slices → cols → Rq d}
    {digits digits' : PathDigits chain} {payload : Payload d (slices * n) chain}
    (rows : Rows chain key (stack M x) digits payload)
    (rows' : Rows chain key (stack M x') digits' payload)
    (short : ∀ s, vecNormInf (x s - x' s) ≤ bound)
    (binary : PathDigits.Binary chain digits) (binary' : PathDigits.Binary chain digits')
    (different : x ≠ x' ∨ digits ≠ digits') :
    (chainCollision chain key M bound x x' digits digits').isSome := by
  have outer (same : stack M x = stack M x') (differs : x ≠ x') :
      ((List.finRange slices).findSome? fun s =>
        ShortKernel.ofCollision? M bound (x s) (x' s)).isSome := by
    obtain ⟨s, differs⟩ := Function.ne_iff.mp differs
    exact List.findSome?_isSome_iff.mpr ⟨s, List.mem_finRange s,
      ShortKernel.ofCollision?_isSome ⟨differs, mulVec_eq_of_stack_eq same s, short s⟩⟩
  refine isSome_or_map.mpr ?_
  cases chain with
  | none =>
      have image : stack M x = payload := rows
      have image' : stack M x' = payload := rows'
      exact Or.inr (outer (image.trans image'.symm) (different.resolve_right fun differs =>
        differs rfl))
  | some c =>
      obtain ⟨first, second, last⟩ : _ ∧ _ ∧ _ := rows
      obtain ⟨first', second', last'⟩ : _ ∧ _ ∧ _ := rows'
      obtain ⟨firstBinary, secondBinary⟩ : _ ∧ _ := binary
      obtain ⟨firstBinary', secondBinary'⟩ : _ ∧ _ := binary'
      by_cases secondSame : ChainDigits.second digits = ChainDigits.second digits'
      · by_cases firstSame : ChainDigits.first digits = ChainDigits.first digits'
        · refine Or.inr (outer ?_ (different.resolve_right fun differs =>
            differs (ChainDigits.ext firstSame secondSame)))
          rw [first, first', firstSame]
        · have image : (ChainKey.first key).mulVec (ChainDigits.first digits) =
              (ChainKey.first key).mulVec (ChainDigits.first digits') := by
            rw [second, second', secondSame]
          exact Or.inl (isSome_or_map.mpr (Or.inr (ShortKernel.ofCollision?_isSome
            ⟨firstSame, image, vecNormInf_sub_le_one firstBinary firstBinary'⟩)))
      · exact Or.inl (isSome_or_map.mpr (Or.inl (ShortKernel.ofCollision?_isSome
          ⟨secondSame, last.trans last'.symm, vecNormInf_sub_le_one secondBinary secondBinary'⟩)))

end Chain

section Reference

variable {S : Shape}

/-- A solution of Module-SIS for one matrix of the commitment (Akita Table 3): for `A` with the
bound `innerBound`, for `B` with the bound `b⋆ − 1` (Akita (49)), or for `F_1` or `F_2` with the
bound 1. -/
abbrev Break (key : Key S) (innerBound : ℕ) : Type :=
  ShortKernel key.inner innerBound ⊕
    (ShortKernel key.outer (S.certifiedBase - 1) ⊕ PathKernel S.chain key.chain)

/-- The reduction of Akita Corollary 10.25 for a certified decommitment `st` and a reference
decommitment: the solution of Lemma 4.3 for `B`, `F_1` or `F_2`, or else the solution
`ι(c̄_i)·s_i − ι(c̄_i)·s⁰_i` for `A` from the first block where it is one. -/
noncomputable def collision (key : Key S) (innerBound : ℕ) (st : Certified S)
    (reference : Decommitment S) : Option (Break key innerBound) :=
  ((chainCollision S.chain key.chain key.outer (S.certifiedBase - 1) (S.sliceInput st.inner)
    (S.sliceInput reference.inner) st.chain reference.chain).map Sum.inr).or
  (((List.finRange S.blocks).findSome? fun i =>
    ShortKernel.ofCollision? key.inner innerBound (st.factor i • st.source i)
      (st.factor i • reference.source i)).map Sum.inl)

/-- Akita Corollary 10.25 with a reference occurrence on one side: a certified decommitment with
the bounds `κ̄_1` and `β̄_∞` and a reference decommitment with `‖s⁰_i‖_∞ ≤ ρ` (certificate
`c̄ = 1`, Akita (162) and (163)) of one payload to different tables give a solution of Module-SIS:
for `A` with the bound `β̄_∞ + κ̄_1·ρ` of Akita (164), for `B` with the bound `b⋆ − 1`, or for a
compression map with the bound 1. -/
theorem collision_isSome {key : Key S} {factorBound sourceBound referenceBound : ℕ}
    {st : Certified S} {reference : Decommitment S} {payload : S.Payload} {table table' : Table}
    (opens : st.Opens factorBound sourceBound key payload table)
    (referenceOpens : reference.Opens key payload table')
    (short : ∀ i, vecNormInf (reference.source i) ≤ referenceBound)
    (different : table ≠ table') :
    (collision key (sourceBound + factorBound * referenceBound) st reference).isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits :
      S.sliceInput st.inner ≠ S.sliceInput reference.inner ∨ st.chain ≠ reference.chain
  · exact Or.inl (chainCollision_isSome opens.source.rows referenceOpens.rows
      (S.sliceInput_sub_le fun i => S.outerView_sub_le (opens.source.innerDigits i)
        (referenceOpens.innerDigits i))
      opens.source.chainDigits referenceOpens.chainDigits digits)
  · have inner : st.inner = reference.inner :=
      S.sliceInput_injective (not_not.mp fun differs => digits (Or.inl differs))
    have sources : st.source ≠ reference.source := fun same =>
      different (by rw [opens.source.decodes, referenceOpens.decodes, same])
    obtain ⟨i, differs⟩ := Function.ne_iff.mp sources
    refine Or.inr (List.findSome?_isSome_iff.mpr ⟨i, List.mem_finRange i,
      ShortKernel.ofCollision?_isSome ⟨?_, ?_, ?_⟩⟩)
    · exact fun same => differs ((opens.unit i).smul_left_cancel.mp same)
    · rw [Matrix.mulVec_smul, Matrix.mulVec_smul, opens.source.innerRows,
        referenceOpens.innerRows, inner]
    · exact (vecNormInf_sub_le _ _).trans (Nat.add_le_add (opens.sourceNorm i)
        ((vecNormInf_smul_le _ _).trans (Nat.mul_le_mul (opens.factorNorm i) (short i))))

end Reference

section Scheme

variable (S : Shape) (factorBound sourceBound : ℕ) {Point Value Proof : Type}

/-- Akita as a polynomial commitment scheme for one polynomial under the shape `S`: the parameters
are the matrices, a commitment is a payload, and the decommitment relation is `Certified.Opens`
with the bounds `κ̄_1` and `β̄_∞`. The evaluation of a table and the verifier are arguments. -/
def pcs (eval : Table → Point → Value)
    (verify : Key S → (n : ℕ) → (Fin n → S.Payload × Point × Value) → Proof → Prop) : Pcs where
  Params := Key S
  Commitment := S.Payload
  Table := Table
  State := Certified S
  Point := Point
  Value := Value
  Proof := Proof
  eval := eval
  relCom key payload table st := st.Opens factorBound sourceBound key payload table
  verify := verify

variable (eval : Table → Point → Value)
  (verify : Key S → (n : ℕ) → (Fin n → S.Payload × Point × Value) → Proof → Prop)
  (claims : ℕ) (schedule : Schedule) (key : Key S) (index : Fin claims)
  (keyCommitment : S.Payload) (keyTable : Table)
  (Admitted : Prover (pcs S factorBound sourceBound eval verify) claims → Prop)
  (Efficient : (prover : Prover (pcs S factorBound sourceBound eval verify) claims) →
    Extractor prover → Prop)
  (advantage : Prover (pcs S factorBound sourceBound eval verify) claims →
    Fin schedule.instances → ℝ)
  (bindingError : Prover (pcs S factorBound sourceBound eval verify) claims → ℝ)

/-- The Akita opening of a final proof on the scheme `pcs`, with the matrices `key` and the
commitment `keyCommitment` of the compression key at the claim `index`. -/
def use : Use where
  scheme := pcs S factorBound sourceBound eval verify
  claims := claims
  schedule := schedule
  params := key
  key := index
  keyCommitment := keyCommitment
  keyTable := keyTable
  Admitted := Admitted
  Efficient := Efficient
  advantage := advantage
  bindingError := bindingError

/-- The reduction from a break of the binding of the key commitment to Module-SIS: the collision
of the decommitment of the claim `index` in the output of an extractor with the reference
decommitment. -/
noncomputable def solve {key : Key S} (innerBound : ℕ) (reference : Decommitment S)
    (index : Fin claims) (output : Option (Fin claims → Table × Certified S)) :
    Option (Break key innerBound) :=
  output.bind fun witness => collision key innerBound (witness index).2 reference

/-- Binding of the commitment of the compression key (`Use.Binding`) from Module-SIS. The
reference decommitment opens `keyCommitment` to `keyTable` with `‖s⁰_i‖_∞ ≤ ρ`. For each admitted
prover and each efficient extractor, `solve` returns a solution of Module-SIS for `A` with the
bound `β̄_∞ + κ̄_1·ρ`, for `B` with the bound `b⋆ − 1`, or for `F_1` or `F_2` with the bound 1,
with probability at most `bindingError` (`msis`). -/
theorem binding (reference : Decommitment S) (referenceBound : ℕ)
    (opens : reference.Opens key keyCommitment keyTable)
    (short : ∀ i, vecNormInf (reference.source i) ≤ referenceBound)
    (msis : ∀ (prover : Prover (pcs S factorBound sourceBound eval verify) claims)
      (extractor : Extractor prover), Admitted prover → Efficient prover extractor →
      extractor.chance (fun draw => (solve S claims (key := key)
        (sourceBound + factorBound * referenceBound) reference index
        (extractor.run draw.1 draw.2)).isSome) ≤ bindingError prover) :
    (use S factorBound sourceBound eval verify claims schedule key index keyCommitment keyTable
      Admitted Efficient advantage bindingError).Binding := by
  intro prover extractor admitted efficient
  refine le_trans (ENNReal.toReal_mono (event_ne_top _ _) (MeasureTheory.measure_mono ?_))
    (msis prover extractor admitted efficient)
  rintro draw ⟨witness, ran, certified, different⟩
  change (solve S claims _ reference index (extractor.run draw.1 draw.2)).isSome = true
  rw [ran]
  exact collision_isSome certified opens short different

/-- `binding` for the commitment algorithm of Akita Section 4.5: the payload `(commit key table).1`
binds the padded table. The reference decommitment is the commitment state, with `ρ = b/2` for
its balanced digits. -/
theorem commit_binding (table : Table) (honest : S.Honest)
    (msis : ∀ (prover : Prover (pcs S factorBound sourceBound eval verify) claims)
      (extractor : Extractor prover), Admitted prover → Efficient prover extractor →
      extractor.chance (fun draw => (solve S claims (key := key)
        (sourceBound + factorBound * (S.sourceBase / 2)) (commit key table).2 index
        (extractor.run draw.1 draw.2)).isSome) ≤ bindingError prover) :
    (use S factorBound sourceBound eval verify claims schedule key index (commit key table).1
      (S.pad table) Admitted Efficient advantage bindingError).Binding :=
  binding S factorBound sourceBound eval verify claims schedule key index _ _ Admitted Efficient
    advantage bindingError (commit key table).2 (S.sourceBase / 2) (commit_opens honest key table)
    (fun _ => (decompose_balanced honest.source _).vecNormInf_le) msis

end Scheme

end NightstreamFPrime.Spec.Compression.Akita
