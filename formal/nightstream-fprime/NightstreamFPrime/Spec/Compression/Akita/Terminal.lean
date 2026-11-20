import NightstreamFPrime.Spec.Compression.Akita.TraceFold

/-! The evaluation-trace terminal of Akita, for its one claim (ePrint 2026/1983, revision of
2026-10-15, Section 8.2 and Proposition 10.29).

The numbers of the terminal are those of a commitment shape `S`: `B_term = S.blocks`,
`M_term = S.blockLength`, the base `b_term = S.sourceBase` with `δ_com = S.sourceDigits`, the
degree `d_{A,term} = S.innerDegree` and the rank `n_{A,term} = S.innerRank`.

- `TerminalStatement` has the public data: the matrix `A_term`, the state `t_i = A_term·s_i` of the
  predecessor, the embedding `E ⊆ R_term`, the weights `χ_blk,term` and `a_term`, the functional
  `T_{ρ_pack,term}` of (97), and the pair `(θ_term, v_tensor)`.
- `TerminalStatement.Accepts` is the checks (128), (129) and (130) for the clear partials `e_i`,
  the challenges `c_i` and the clear response `z`.
- `TerminalFork.source` is the source block `s⋆_i = (Δc_i)^{-1}·(z^{(i,1)} − z^{(i,0)})` of the
  proof of Akita Proposition 10.29, and `terminal_inner` and `terminal_partial` are its two
  equations `A_term·s⋆_i = t_i` and `e_i = a_term^⊤·G_{b_term,M_term}·s⋆_i`.
- `terminalCollision_isSome` gives the solution for `A_term` with the radius
  `η_{A,term,∞} = 8·κ_{1,term}·Z_{∞,term}` of Akita (181) from two extracted candidates for one
  block that differ, and `terminalCollisionSq_isSome` has the radius
  `η²_{A,term,2} = 64·Γ_term²·S_max,term` of Akita (182). -/

namespace NightstreamFPrime.Spec.Compression.Akita

variable {S : Shape}

/-- The public data of the evaluation-trace terminal (Akita Section 8.2): the matrix `A_term`,
the state `(t_i)_i` that the predecessor binds, the embedding of the fixed subfield
`E ⊆ R_term`, the weights `χ_blk,term(i)` and `a_term,x`, the functional `T_{ρ_pack,term}`, and the
pair `(θ_term, v_tensor)`. -/
structure TerminalStatement (S : Shape) (E : Type) [CommRing E] [Module Zq E] where
  matrix : Matrix (Fin S.innerRank) (Fin S.blockLength × Fin S.sourceDigits) (Rq S.innerDegree)
  state : Fin S.blocks → Fin S.innerRank → Rq S.innerDegree
  subfield : E →+* Rq S.innerDegree
  block : Fin S.blocks → E
  position : Fin S.blockLength → E
  functional : Rq S.innerDegree →ₗ[Zq] E
  scale : E
  value : E

variable {E : Type} [CommRing E] [Module Zq E]

/-- The checks of the terminal for the clear partials `e_i`, the challenges `c_i` and the clear
response `z` (Akita Section 8.2): `inner` is (128), `A_term·z = Σ_i c_i·t_i`; `evaluation` is
(129), `Σ_i c_i·e_i = a_term^⊤·G_{b_term,M_term}·z`; `value` is (130),
`θ_term·Σ_i χ_blk,term(i)·T_{ρ_pack,term}(e_i) = v_tensor`. -/
structure TerminalStatement.Accepts (st : TerminalStatement S E)
    (partials challenge : Fin S.blocks → Rq S.innerDegree) (response : S.Source) : Prop where
  inner : st.matrix.mulVec response = ∑ i, challenge i • st.state i
  evaluation : ∑ i, challenge i * partials i =
    traceContract st.subfield st.position (gadget S.sourceBase response)
  value : st.scale * ∑ i, st.block i * st.functional (partials i) = st.value

/-- One accepting extension of a terminal transcript through the fold challenges: the challenges
`c_i` and the clear response `z`. -/
structure TerminalChild (S : Shape) where
  challenge : Fin S.blocks → Rq S.innerDegree
  response : S.Source

/-- Two children for each block `i` (the coordinate-wise fold siblings of Akita
Proposition 10.29). -/
structure TerminalFork (S : Shape) where
  first : Fin S.blocks → TerminalChild S
  second : Fin S.blocks → TerminalChild S

namespace TerminalFork

variable {st : TerminalStatement S E} {fork fork' : TerminalFork S}
  {family : Set (Rq S.innerDegree)} {partials partials' : Fin S.blocks → Rq S.innerDegree}

/-- The two children of block `i` have challenges of the family at `i` that differ, their
challenges agree at each other block, and each child passes the checks of the terminal for the
partials `(e_i)_i`, which are fixed before the challenges (Akita Proposition 10.29, proof). -/
structure Accepts (family : Set (Rq S.innerDegree)) (fork : TerminalFork S)
    (st : TerminalStatement S E) (partials : Fin S.blocks → Rq S.innerDegree) : Prop where
  firstMember : ∀ i, (fork.first i).challenge i ∈ family
  secondMember : ∀ i, (fork.second i).challenge i ∈ family
  different : ∀ i, (fork.first i).challenge i ≠ (fork.second i).challenge i
  agree : ∀ i j, j ≠ i → (fork.first i).challenge j = (fork.second i).challenge j
  firstAccepts : ∀ i, st.Accepts partials (fork.first i).challenge (fork.first i).response
  secondAccepts : ∀ i, st.Accepts partials (fork.second i).challenge (fork.second i).response

/-- `Δc_i = c_i^{(i,1)} − c_i^{(i,0)}`. -/
noncomputable def difference (fork : TerminalFork S) (i : Fin S.blocks) : Rq S.innerDegree :=
  (fork.second i).challenge i - (fork.first i).challenge i

/-- `z^{(i,1)} − z^{(i,0)}`. -/
noncomputable def responseDifference (fork : TerminalFork S) (i : Fin S.blocks) : S.Source :=
  (fork.second i).response - (fork.first i).response

/-- `s⋆_i = (Δc_i)^{-1}·(z^{(i,1)} − z^{(i,0)})` (Akita Proposition 10.29, proof). -/
noncomputable def source (fork : TerminalFork S) (i : Fin S.blocks) : S.Source :=
  Ring.inverse (fork.difference i) • fork.responseDifference i

theorem Accepts.isUnit_difference (accepts : fork.Accepts family st partials)
    (units : UnitDifferences family) (i : Fin S.blocks) : IsUnit (fork.difference i) :=
  units _ (accepts.secondMember i) _ (accepts.firstMember i) (accepts.different i).symm

theorem difference_smul_source (fork : TerminalFork S) {i : Fin S.blocks}
    (unit : IsUnit (fork.difference i)) :
    fork.difference i • fork.source i = fork.responseDifference i := by
  rw [source, smul_smul, Ring.mul_inverse_cancel _ unit, one_smul]

/-- The difference of the checks (128) of the two children of block `i`:
`A_term·(z^{(i,1)} − z^{(i,0)}) = Δc_i·t_i`. -/
theorem Accepts.inner_difference (accepts : fork.Accepts family st partials) (i : Fin S.blocks) :
    st.matrix.mulVec (fork.responseDifference i) = fork.difference i • st.state i := by
  rw [responseDifference, Matrix.mulVec_sub, (accepts.secondAccepts i).inner,
    (accepts.firstAccepts i).inner, difference]
  exact sum_smul_sub_sum_smul (fun j => (fork.second i).challenge j)
    (fun j => (fork.first i).challenge j) st.state i
    (fun j other => (accepts.agree i j other).symm)

/-- The difference of the checks (129) of the two children of block `i`:
`Δc_i·e_i = a_term^⊤·G_{b_term,M_term}·(z^{(i,1)} − z^{(i,0)})`. -/
theorem Accepts.partial_difference (accepts : fork.Accepts family st partials)
    (i : Fin S.blocks) :
    fork.difference i * partials i = traceContract st.subfield st.position
      (gadget S.sourceBase (fork.responseDifference i)) := by
  rw [responseDifference, gadget_sub, traceContract_sub, ← (accepts.secondAccepts i).evaluation,
    ← (accepts.firstAccepts i).evaluation]
  exact (sum_smul_sub_sum_smul (fun j => (fork.second i).challenge j)
    (fun j => (fork.first i).challenge j) partials i
    (fun j other => (accepts.agree i j other).symm)).symm

end TerminalFork

section Extraction

variable {st st' : TerminalStatement S E} {fork fork' : TerminalFork S}
  {family : Set (Rq S.innerDegree)} {partials partials' : Fin S.blocks → Rq S.innerDegree}

/-- Akita Proposition 10.29, extraction step: `A_term·s⋆_i = t_i`. -/
theorem terminal_inner (units : UnitDifferences family)
    (accepts : fork.Accepts family st partials) (i : Fin S.blocks) :
    st.matrix.mulVec (fork.source i) = st.state i := by
  rw [TerminalFork.source, Matrix.mulVec_smul, accepts.inner_difference i, smul_smul,
    Ring.inverse_mul_cancel _ (accepts.isUnit_difference units i), one_smul]

/-- Akita Proposition 10.29, extraction step: `e_i = a_term^⊤·G_{b_term,M_term}·s⋆_i` in
`R_term`. -/
theorem terminal_partial (units : UnitDifferences family)
    (accepts : fork.Accepts family st partials) (i : Fin S.blocks) :
    partials i =
      traceContract st.subfield st.position (gadget S.sourceBase (fork.source i)) := by
  rw [TerminalFork.source, gadget_smul, traceContract_smul, ← accepts.partial_difference i,
    ← mul_assoc, Ring.inverse_mul_cancel _ (accepts.isUnit_difference units i), one_mul]

/-- The vector `z_A = Δc'_i·(Δc_i·s⋆_i) − Δc_i·(Δc'_i·s⋆'_i)` for two extracted candidates of
block `i`, when it is a solution for `A_term` (Akita Proposition 10.29, proof). -/
noncomputable def terminalCollision (st : TerminalStatement S E) (bound : ℕ)
    (fork fork' : TerminalFork S) (i : Fin S.blocks) : Option (ShortKernel st.matrix bound) :=
  ShortKernel.ofCollision? st.matrix bound (fork'.difference i • fork.responseDifference i)
    (fork.difference i • fork'.responseDifference i)

/-- The solution `z_A` for `A_term` in the Euclidean norm. -/
noncomputable def terminalCollisionSq (st : TerminalStatement S E) (bound : ℕ)
    (fork fork' : TerminalFork S) (i : Fin S.blocks) : Option (ShortKernelSq st.matrix bound) :=
  ShortKernelSq.ofCollision? st.matrix bound (fork'.difference i • fork.responseDifference i)
    (fork.difference i • fork'.responseDifference i)

theorem terminal_ne (units : UnitDifferences family) (accepts : fork.Accepts family st partials)
    (accepts' : fork'.Accepts family st' partials') {i : Fin S.blocks}
    (different : fork.source i ≠ fork'.source i) :
    fork'.difference i • fork.responseDifference i ≠
      fork.difference i • fork'.responseDifference i := by
  have unit := accepts.isUnit_difference units i
  have unit' := accepts'.isUnit_difference units i
  intro same
  rw [← fork.difference_smul_source unit, ← fork'.difference_smul_source unit', smul_smul,
    smul_smul, mul_comm] at same
  exact different ((unit.mul unit').smul_left_cancel.mp same)

theorem terminal_image (accepts : fork.Accepts family st partials)
    (accepts' : fork'.Accepts family st' partials') (matrix : st'.matrix = st.matrix)
    (state : st'.state = st.state) (i : Fin S.blocks) :
    st.matrix.mulVec (fork'.difference i • fork.responseDifference i) =
      st.matrix.mulVec (fork.difference i • fork'.responseDifference i) := by
  have image := accepts'.inner_difference i
  rw [matrix, state] at image
  rw [Matrix.mulVec_smul, Matrix.mulVec_smul, accepts.inner_difference i, image, smul_comm]

/-- Akita Proposition 10.29 with the radius (181). Two forks pass the checks of two terminal
statements with one matrix `A_term` and one state `(t_i)_i`, each challenge has
`‖c‖_1 = κ_{1,term}`, and each clear response has `‖z‖_∞ ≤ Z_{∞,term}`. If the extracted candidates
for block `i` differ, `z_A` is a solution of Module-SIS for `A_term` with the bound
`8·κ_{1,term}·Z_{∞,term}`. -/
theorem terminalCollision_isSome {mass peak response : ℕ} (units : UnitDifferences family)
    (sparse : ∀ c ∈ family, Sparse mass peak c) (accepts : fork.Accepts family st partials)
    (accepts' : fork'.Accepts family st' partials') (matrix : st'.matrix = st.matrix)
    (state : st'.state = st.state)
    (short : ∀ i, vecNormInf (fork.first i).response ≤ response ∧
      vecNormInf (fork.second i).response ≤ response)
    (short' : ∀ i, vecNormInf (fork'.first i).response ≤ response ∧
      vecNormInf (fork'.second i).response ≤ response)
    {i : Fin S.blocks} (different : fork.source i ≠ fork'.source i) :
    (terminalCollision st (8 * mass * response) fork fork' i).isSome := by
  have factor : normOne (fork.difference i) ≤ 2 * mass :=
    (sparse _ (accepts.secondMember i)).normOne_sub_le (sparse _ (accepts.firstMember i))
  have factor' : normOne (fork'.difference i) ≤ 2 * mass :=
    (sparse _ (accepts'.secondMember i)).normOne_sub_le (sparse _ (accepts'.firstMember i))
  have delta : vecNormInf (fork.responseDifference i) ≤ 2 * response :=
    (vecNormInf_sub_le _ _).trans
      ((Nat.add_le_add (short i).2 (short i).1).trans_eq (Nat.two_mul response).symm)
  have delta' : vecNormInf (fork'.responseDifference i) ≤ 2 * response :=
    (vecNormInf_sub_le _ _).trans
      ((Nat.add_le_add (short' i).2 (short' i).1).trans_eq (Nat.two_mul response).symm)
  refine ShortKernel.ofCollision?_isSome ⟨terminal_ne units accepts accepts' different,
    terminal_image accepts accepts' matrix state i, ?_⟩
  exact (vecNormInf_sub_le _ _).trans ((Nat.add_le_add
    ((vecNormInf_smul_le _ _).trans (Nat.mul_le_mul factor' delta))
    ((vecNormInf_smul_le _ _).trans (Nat.mul_le_mul factor delta'))).trans_eq (by ring))

/-- Akita Proposition 10.29 with the radius (182). Two forks pass the checks of two terminal
statements with one matrix `A_term` and one state `(t_i)_i`, each challenge has
`‖c‖_mul,2 ≤ Γ_term`, and each clear response has `‖z‖_2² ≤ S_max,term`. If the extracted
candidates for block `i` differ, `z_A` is a solution of Module-SIS for `A_term` with
`‖z_A‖_2² ≤ 64·Γ_term²·S_max,term`. -/
theorem terminalCollisionSq_isSome {operator energy : ℕ} (units : UnitDifferences family)
    (bounded : ∀ c ∈ family, MulBound operator c) (accepts : fork.Accepts family st partials)
    (accepts' : fork'.Accepts family st' partials') (matrix : st'.matrix = st.matrix)
    (state : st'.state = st.state)
    (short : ∀ i, vecNormSq (fork.first i).response ≤ energy ∧
      vecNormSq (fork.second i).response ≤ energy)
    (short' : ∀ i, vecNormSq (fork'.first i).response ≤ energy ∧
      vecNormSq (fork'.second i).response ≤ energy)
    {i : Fin S.blocks} (different : fork.source i ≠ fork'.source i) :
    (terminalCollisionSq st (64 * operator ^ 2 * energy) fork fork' i).isSome := by
  have factor : MulBound (2 * operator) (fork.difference i) :=
    (bounded _ (accepts.secondMember i)).sub (bounded _ (accepts.firstMember i))
  have factor' : MulBound (2 * operator) (fork'.difference i) :=
    (bounded _ (accepts'.secondMember i)).sub (bounded _ (accepts'.firstMember i))
  have delta : vecNormSq (fork.responseDifference i) ≤ 4 * energy :=
    (vecNormSq_sub_le (short i).2 (short i).1 (sq _).ge).trans_eq (by ring)
  have delta' : vecNormSq (fork'.responseDifference i) ≤ 4 * energy :=
    (vecNormSq_sub_le (short' i).2 (short' i).1 (sq _).ge).trans_eq (by ring)
  refine ShortKernelSq.ofCollision?_isSome ⟨terminal_ne units accepts accepts' different,
    terminal_image accepts accepts' matrix state i, ?_⟩
  exact (vecNormSq_sub_le
    ((factor'.vecNormSq_smul_le _).trans (Nat.mul_le_mul_left _ delta))
    ((factor.vecNormSq_smul_le _).trans (Nat.mul_le_mul_left _ delta')) (sq _).ge).trans_eq
    (by ring)

end Extraction

end NightstreamFPrime.Spec.Compression.Akita
