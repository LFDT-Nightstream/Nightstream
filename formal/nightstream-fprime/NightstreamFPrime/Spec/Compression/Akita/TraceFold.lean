import NightstreamFPrime.Spec.Compression.Akita.Euclid

/-! One fold of Akita with the evaluation-trace opening method, for one claim in one group
(ePrint 2026/1983, revision of 2026-10-15, Sections 5.3, 7.1 and 7.2, Definition 10.22,
Lemma 10.23 and Corollary 10.32).

At a trace level the fold challenges are elements of the A ring `R = R_{q,d_A}`, and a partial
opening evaluation is an element of `R` with `d_A` base-field coordinates. The opening shape `O`
of such a level has `h_car = 1` and `k·d_car = d_A`, so the view of `D` has the width `d_A`.

- `TraceStatement` has the public data of one fold: the matrices and payloads, the embedding
  `E ⊆ R` of the fixed subfield, the weights `χ_blk` of (95) and `a_x = ẽq(ρ_pos, x)` of (66), the
  functional `T_{ρ_pack}` of (97), the factor `θ` of (101) and the target.
- `traceContract` and `TraceStatement.contract` are `F ↦ ⟨a, F⟩` of Akita (66),
  `OpeningShape.tracePartial` is the partial `e_i` that the digits `ê` give, and
  `TraceStatement.scalar` is the left side of the scalar row (101) for one claim, which is `θ`
  times the left side of (98).
- `Witness.TraceSatisfies` is the exact branch relation for one challenge vector: the rows (103),
  (104), (105), (106) and (101) of Akita Figure 5 with the alphabets.
- `Weak.TraceOpens` is the weak opening of Akita Definition 10.22 for a trace level.
- `traceCollision_isSome` is Akita Lemma 10.23 with the bound (174), and `traceCollisionSq_isSome`
  has the second line of the bound (177).
- `traceExtract_opens_or_collision` is Akita Corollary 10.32 for one claim with challenges of the
  A ring, and `traceExtract_euclidean` is Akita Corollary 10.26 for the extracted opening. -/

namespace NightstreamFPrime.Spec.Compression.Akita

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

/-- The partial `e_i ∈ R_{q,d_A}` of a trace level: coefficient `j` is coordinate `j` of the
`d_A` coordinates that the digits of block `i` give in base `b_1` (Akita (66) and
Definition 10.22 (iv)). -/
noncomputable def OpeningShape.tracePartial (O : OpeningShape S) (digits : O.Partials)
    (i : Fin S.blocks) : Rq S.innerDegree :=
  ofCoeffs fun j => sequence
    (gadget S.openingBase fun entry : Fin O.subcolumns × Fin S.openingDigits => digits (i, entry))
    j

/-- The public data of one fold for one claim at a trace level (Akita Definition 10.22, "The
fixed folding statement"): the matrices of the commitment and of the opening path, the payloads
`u_pub` and `v_pub`, the embedding of the fixed subfield `E ⊆ R_{q,d_A}` (Akita Section 3.1), the
weights `χ_blk(i) = ẽq(ρ_blk, i)` of (95) and `a_x = ẽq(ρ_pos, x)` of (66), the functional
`T_{ρ_pack}` of (97), the factor `θ` of (101) and the reduced target `v̄`. -/
structure TraceStatement (O : OpeningShape S) (E : Type) [CommRing E] [Module Zq E] where
  key : Key S
  openKey : OpeningKey O
  payload : S.Payload
  openPayload : O.Payload
  subfield : E →+* Rq S.innerDegree
  block : Fin S.blocks → E
  position : Fin S.blockLength → E
  functional : Rq S.innerDegree →ₗ[Zq] E
  scale : E
  value : E

variable {E : Type} [CommRing E] [Module Zq E]

section Contract

variable {d : ℕ} {ι : Type} [Fintype ι]

/-- `F ↦ ⟨a, F⟩ = Σ_x a_x·F_x` in `R_{q,d_A}`, for the weights `a_x ∈ E ⊆ R_{q,d_A}`
(Akita (66), and the right side of (103) and (129)). -/
noncomputable def traceContract (subfield : E →+* Rq d) (position : ι → E) (F : ι → Rq d) :
    Rq d :=
  ∑ x, subfield (position x) * F x

omit [Module Zq E] in
/-- The contraction is linear over the A ring: `⟨a, c·F⟩ = c·⟨a, F⟩`. -/
theorem traceContract_smul (subfield : E →+* Rq d) (position : ι → E) (c : Rq d)
    (F : ι → Rq d) :
    traceContract subfield position (c • F) = c * traceContract subfield position F := by
  rw [traceContract, traceContract, Finset.mul_sum]
  exact Finset.sum_congr rfl fun x _ => mul_left_comm _ _ _

omit [Module Zq E] in
theorem traceContract_sub (subfield : E →+* Rq d) (position : ι → E) (F G : ι → Rq d) :
    traceContract subfield position (F - G) =
      traceContract subfield position F - traceContract subfield position G := by
  rw [traceContract, traceContract, traceContract, ← Finset.sum_sub_distrib]
  exact Finset.sum_congr rfl fun x _ => mul_sub _ _ _

end Contract

namespace TraceStatement

variable (st : TraceStatement O E)

/-- `F ↦ ⟨a, F⟩` for the weights of the statement (Akita (66), and the right side of (103)). -/
noncomputable def contract (F : Fin S.blockLength → Rq S.innerDegree) : Rq S.innerDegree :=
  traceContract st.subfield st.position F

theorem contract_smul (c : Rq S.innerDegree) (F : Fin S.blockLength → Rq S.innerDegree) :
    st.contract (c • F) = c * st.contract F :=
  traceContract_smul st.subfield st.position c F

theorem contract_sub (F G : Fin S.blockLength → Rq S.innerDegree) :
    st.contract (F - G) = st.contract F - st.contract G :=
  traceContract_sub st.subfield st.position F G

/-- The left side of the scalar row (101) of Akita for one claim:
`θ · Σ_i χ_blk(i)·T_{ρ_pack}(e_i)`, which is `θ` times the left side of (98). -/
noncomputable def scalar (partials : Fin S.blocks → Rq S.innerDegree) : E :=
  st.scale * ∑ i, st.block i * st.functional (partials i)

end TraceStatement

/-- The rows of Akita Figure 5 that have no fold challenge, at a trace level, with their
alphabets: the rows (105) of `B` with the rows (47) or (48) to `u_pub`, the row (106) of `D` with
the rows (47) or (48) to `v_pub`, the scalar row (101) for one claim, the alphabet `A_{b⋆}` of `t̂`
and `ê`, and the alphabet `{−1, 0}` of the chain digits. -/
structure Carried.TraceOpens (digits : Carried O) (st : TraceStatement O E) : Prop where
  rows : Rows S.chain st.key.chain (stack st.key.outer (S.sliceInput digits.inner)) digits.chain
    st.payload
  openRows : OpeningRows st.openKey digits.partials digits.openChain st.openPayload
  value : st.scalar (O.tracePartial digits.partials) = st.value
  innerDigits : ∀ i, Balanced S.certifiedBase (digits.inner i)
  chainDigits : PathDigits.Binary S.chain digits.chain
  partialDigits : Balanced S.certifiedBase digits.partials
  openDigits : PathDigits.Binary O.chain digits.openChain

/-- The reduction of Akita Lemma 10.23, "Digit paths", at a trace level: a solution for the path
of `B`, or else for the path of `D`, from two families of carried digits. -/
noncomputable def traceCarriedCollision (st : TraceStatement O E) (digits digits' : Carried O) :
    Option (OuterBreak st.key ⊕ OpenBreak st.openKey) :=
  ((outerCollision st.key digits.inner digits'.inner digits.chain digits'.chain).map Sum.inl).or
    ((openCollision st.openKey digits.partials digits'.partials digits.openChain
      digits'.openChain).map Sum.inr)

/-- Akita Lemma 10.23, "Digit paths", at a trace level: two families of carried digits that
satisfy the rows of one statement and that differ give a solution of Module-SIS for `B` or `D`
with the bound `b⋆ − 1`, or for a compression map with the bound 1. -/
theorem traceCarriedCollision_isSome {st : TraceStatement O E} {digits digits' : Carried O}
    (opens : digits.TraceOpens st) (opens' : digits'.TraceOpens st)
    (different : digits ≠ digits') : (traceCarriedCollision st digits digits').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases outer : digits.inner ≠ digits'.inner ∨ digits.chain ≠ digits'.chain
  · exact Or.inl (outerCollision_isSome opens.rows opens'.rows opens.innerDigits
      opens'.innerDigits opens.chainDigits opens'.chainDigits outer)
  · refine Or.inr (openCollision_isSome opens.openRows opens'.openRows opens.partialDigits
      opens'.partialDigits opens.openDigits opens'.openDigits ?_)
    by_contra same
    obtain ⟨inner, chain⟩ := not_or.mp outer
    obtain ⟨partials, openChain⟩ := not_or.mp same
    exact different (Carried.ext (not_not.mp inner) (not_not.mp chain) (not_not.mp partials)
      (not_not.mp openChain))

/-- The exact branch relation of one fold at a trace level for the challenges `c_i ∈ R_{q,d_A}`
(Akita Section 10, before Corollary 10.32), for one claim. `carried` has the rows without a
challenge. `evaluation` is the row (103): `Σ_i c_i·e_i = a^⊤·G_{b,M}·G_{b_z,m_A}·ẑ` in
`R_{q,d_A}`. `innerRows` is the row (104) with `c^A_i = c_i`:
`A·G_{b_z,m_A}·ẑ = Σ_i c_i·G_{b_1,n_A}·t̂_i`. `responseDigits` is the alphabet `A_{b⋆}` of `ẑ`. -/
structure Witness.TraceSatisfies (w : Witness O fold) (st : TraceStatement O E)
    (challenge : Fin S.blocks → Rq S.innerDegree) : Prop where
  carried : w.toCarried.TraceOpens st
  evaluation : ∑ i, challenge i * O.tracePartial w.partials i =
    st.contract (gadget S.sourceBase (gadget fold.responseBase w.response))
  innerRows : st.key.inner.mulVec (gadget fold.responseBase w.response) =
    ∑ i, challenge i • gadget S.openingBase (w.inner i)
  responseDigits : Balanced S.certifiedBase w.response

/-- Akita Definition 10.22 for one claim at a trace level, on the coefficient-norm route: `ω` is a
weak opening of the statement, and its source blocks decode to the table. `certified` is the
conditions (i) and (ii), with each factor `c̄_i` a unit of `R_{q,d_A}`. `openRows` is (iii).
`partials` and `value` are (iv): `e_i = ⟨a, G_{b,M}·s_i⟩` of (66), and the scalar row (101).
`partialDigits` and `openDigits` are the part of (v) for the opening path. -/
structure Weak.TraceOpens (factorBound sourceBound : ℕ) (ω : Weak O) (st : TraceStatement O E)
    (table : Table) : Prop where
  certified : ω.toCertified.Opens factorBound sourceBound st.key st.payload table
  openRows : OpeningRows st.openKey ω.partials ω.openChain st.openPayload
  partials : ∀ i, O.tracePartial ω.partials i = st.contract (gadget S.sourceBase (ω.source i))
  value : st.scalar (O.tracePartial ω.partials) = st.value
  partialDigits : Balanced S.certifiedBase ω.partials
  openDigits : PathDigits.Binary O.chain ω.openChain

/-- A solution of Module-SIS for one matrix of a fold at a trace level (Akita Table 3): for `A`
with the bound `innerBound`, for `B` or `D` with the bound `b⋆ − 1`, or for `F_1`, `F_2`, `H_1` or
`H_2` with the bound 1. -/
abbrev TraceBreak (st : TraceStatement O E) (innerBound : ℕ) : Type :=
  ShortKernel st.key.inner innerBound ⊕ (OuterBreak st.key ⊕ OpenBreak st.openKey)

/-- The reduction of Akita Lemma 10.23 for two weak openings at a trace level: the solution of
the digit paths, or else the solution `z_A` for `A`. -/
noncomputable def traceCollision (st : TraceStatement O E) (innerBound : ℕ) (ω ω' : Weak O) :
    Option (TraceBreak st innerBound) :=
  ((traceCarriedCollision st ω.carried ω'.carried).map Sum.inr).or
    ((sourceCollision st.key innerBound ω.toCertified ω'.toCertified).map Sum.inl)

/-- A solution of Module-SIS for one matrix of a fold at a trace level on the Euclidean route
(Akita Table 3): for `A` with `‖z‖_2² ≤ bound`, or for another matrix as in `TraceBreak`. -/
abbrev TraceBreakSq (st : TraceStatement O E) (bound : ℕ) : Type :=
  ShortKernelSq st.key.inner bound ⊕ (OuterBreak st.key ⊕ OpenBreak st.openKey)

/-- The reduction of Akita Lemma 10.23 for two weak openings at a trace level on the Euclidean
route. -/
noncomputable def traceCollisionSq (st : TraceStatement O E) (bound : ℕ) (ω ω' : Weak O) :
    Option (TraceBreakSq st bound) :=
  ((traceCarriedCollision st ω.carried ω'.carried).map Sum.inr).or
    ((sourceCollisionSq st.key bound ω.toCertified ω'.toCertified).map Sum.inl)

section Binding

variable {st : TraceStatement O E} {ω ω' : Weak O}
  {factorBound sourceBound factorBound' sourceBound' : ℕ} {table table' : Table}

theorem Weak.TraceOpens.carried (opens : ω.TraceOpens factorBound sourceBound st table) :
    ω.carried.TraceOpens st :=
  ⟨opens.certified.source.rows, opens.openRows, opens.value, opens.certified.source.innerDigits,
    opens.certified.source.chainDigits, opens.partialDigits, opens.openDigits⟩

/-- Akita Lemma 10.23 (weak binding of one fold) at a trace level with the bound (174): two weak
openings of one statement that differ on a source block or on a carried digit vector give a
solution of Module-SIS: for `A` with each bound `η_A ≥ κ̄'_1·β̄_∞ + κ̄_1·β̄'_∞`, for `B` or `D` with
the bound `b⋆ − 1`, or for a compression map with the bound 1. -/
theorem traceCollision_isSome {innerBound : ℕ}
    (opens : ω.TraceOpens factorBound sourceBound st table)
    (opens' : ω'.TraceOpens factorBound' sourceBound' st table')
    (covers : factorBound' * sourceBound + factorBound * sourceBound' ≤ innerBound)
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (traceCollision st innerBound ω ω').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits : ω.carried = ω'.carried
  · exact Or.inr (sourceCollision_isSome opens.certified opens'.certified covers
      (congrArg Carried.inner digits) (different.resolve_right (not_not.mpr digits)))
  · exact Or.inl (traceCarriedCollision_isSome opens.carried opens'.carried digits)

/-- Akita Lemma 10.23 at a trace level with the second line of the bound (177): two weak openings
with the Euclidean certificates `(κ̄_mul,2, β̄_2²)` that differ give a solution for `A` with
`‖z_A‖_2² ≤ 4·κ̄_mul,2²·β̄_2²`, or a solution for another matrix. With `κ̄_mul,2 = 2Γ` and
`β̄_2² = 4·S_max` the bound is the radius `64·Γ²·S_max` of Akita (179) and Corollary 10.35 (ii). -/
theorem traceCollisionSq_isSome {mulBound sqBound : ℕ}
    (opens : ω.TraceOpens factorBound sourceBound st table)
    (opens' : ω'.TraceOpens factorBound' sourceBound' st table')
    (euclid : ω.toCertified.Euclidean mulBound sqBound)
    (euclid' : ω'.toCertified.Euclidean mulBound sqBound)
    (different : ω.source ≠ ω'.source ∨ ω.carried ≠ ω'.carried) :
    (traceCollisionSq st (4 * mulBound ^ 2 * sqBound) ω ω').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits : ω.carried = ω'.carried
  · exact Or.inr (sourceCollisionSq_isSome_of_bounds opens.certified opens'.certified euclid
      euclid' (congrArg Carried.inner digits) (different.resolve_right (not_not.mpr digits)))
  · exact Or.inl (traceCarriedCollision_isSome opens.carried opens'.carried digits)

end Binding

section Extraction

/-- One accepting extension of a transcript prefix through the fold challenge at a trace level (a
child of Akita Definition 10.30): its challenges `c_i ∈ R_{q,d_A}` and the digit vectors of its
branch relation. -/
structure TraceChild (O : OpeningShape S) (fold : Descriptor) where
  challenge : Fin S.blocks → Rq S.innerDegree
  witness : Witness O fold

/-- Two children for each block `i` (Akita Definition 10.30 (ii), with `Q` the set of blocks). -/
structure TraceFork (O : OpeningShape S) (fold : Descriptor) where
  first : Fin S.blocks → TraceChild O fold
  second : Fin S.blocks → TraceChild O fold

variable {st : TraceStatement O E} {fork : TraceFork O fold}
  {family : Set (Rq S.innerDegree)} {digits : Carried O}

/-- The coordinate-wise structure of Akita Definition 10.30 (ii) with the hypothesis of Akita
Corollary 10.32, at a trace level: the two children of block `i` have challenges of the family at
`i` that differ, their challenges agree at each other block, and the digit vectors of each child
satisfy its exact branch relation for the statement. -/
structure TraceFork.Accepts (family : Set (Rq S.innerDegree)) (fork : TraceFork O fold)
    (st : TraceStatement O E) : Prop where
  firstMember : ∀ i, (fork.first i).challenge i ∈ family
  secondMember : ∀ i, (fork.second i).challenge i ∈ family
  different : ∀ i, (fork.first i).challenge i ≠ (fork.second i).challenge i
  agree : ∀ i j, j ≠ i → (fork.first i).challenge j = (fork.second i).challenge j
  firstSatisfies : ∀ i, (fork.first i).witness.TraceSatisfies st (fork.first i).challenge
  secondSatisfies : ∀ i, (fork.second i).witness.TraceSatisfies st (fork.second i).challenge

/-- `c̄_i = c_i^{(i,1)} − c_i^{(i,0)}` (Akita Theorem 10.31, "Coordinate siblings"). -/
noncomputable def TraceFork.difference (fork : TraceFork O fold) (i : Fin S.blocks) :
    Rq S.innerDegree :=
  (fork.second i).challenge i - (fork.first i).challenge i

/-- `z^{(i,1)} − z^{(i,0)}` for the recomposed responses of the two children of block `i`. -/
noncomputable def TraceFork.responseDifference (fork : TraceFork O fold) (i : Fin S.blocks) :
    S.Source :=
  gadget fold.responseBase (fork.second i).witness.response -
    gadget fold.responseBase (fork.first i).witness.response

/-- The weak opening of the proof of Akita Theorem 10.31 at a trace level: the factor `c̄_i`, the
source block `s_i = c̄_i^{-1}·(z^{(i,1)} − z^{(i,0)})`, and the carried digits `digits` of the
children. -/
noncomputable def traceExtract (fork : TraceFork O fold) (digits : Carried O) : Weak O where
  source i := Ring.inverse (fork.difference i) • fork.responseDifference i
  inner := digits.inner
  chain := digits.chain
  factor i := fork.difference i
  partials := digits.partials
  openChain := digits.openChain

/-- The reduction of Akita Corollary 10.32, "Digit paths", at a trace level: the solution of
`traceCarriedCollision` for the first child whose carried digits are not `digits`. -/
noncomputable def traceForkCollision (st : TraceStatement O E) (fork : TraceFork O fold)
    (digits : Carried O) : Option (OuterBreak st.key ⊕ OpenBreak st.openKey) :=
  (List.finRange S.blocks).findSome? fun i =>
    (traceCarriedCollision st (fork.first i).witness.toCarried digits).or
      (traceCarriedCollision st (fork.second i).witness.toCarried digits)

theorem TraceFork.Accepts.isUnit_difference (accepts : fork.Accepts family st)
    (units : UnitDifferences family) (i : Fin S.blocks) : IsUnit (fork.difference i) :=
  units _ (accepts.secondMember i) _ (accepts.firstMember i) (accepts.different i).symm

/-- `c̄_i·s_i = z^{(i,1)} − z^{(i,0)}` for the extracted source block. -/
theorem traceExtract_factor_smul_source (fork : TraceFork O fold) (digits : Carried O)
    (i : Fin S.blocks) (unit : IsUnit (fork.difference i)) :
    (traceExtract fork digits).factor i • (traceExtract fork digits).source i =
      fork.responseDifference i := by
  show fork.difference i • (Ring.inverse (fork.difference i) • fork.responseDifference i) =
    fork.responseDifference i
  rw [smul_smul, Ring.mul_inverse_cancel _ unit, one_smul]

/-- The difference of the rows (104) of the two children of block `i`:
`A·(z^{(i,1)} − z^{(i,0)}) = c̄_i·G_{b_1,n_A}·t̂_i`. -/
theorem TraceFork.Accepts.inner_difference (accepts : fork.Accepts family st)
    (firstDigits : ∀ i, (fork.first i).witness.toCarried = digits)
    (secondDigits : ∀ i, (fork.second i).witness.toCarried = digits) (i : Fin S.blocks) :
    st.key.inner.mulVec (fork.responseDifference i) =
      fork.difference i • gadget S.openingBase (digits.inner i) := by
  have first := (accepts.firstSatisfies i).innerRows
  have second := (accepts.secondSatisfies i).innerRows
  rw [firstDigits i] at first
  rw [secondDigits i] at second
  rw [TraceFork.responseDifference, Matrix.mulVec_sub, first, second, TraceFork.difference]
  exact sum_smul_sub_sum_smul (fun j => (fork.second i).challenge j)
    (fun j => (fork.first i).challenge j) (fun j => gadget S.openingBase (digits.inner j)) i
    (fun j other => (accepts.agree i j other).symm)

/-- The difference of the rows (103) of the two children of block `i`:
`c̄_i·e_i = ⟨a, G_{b,M}·(z^{(i,1)} − z^{(i,0)})⟩`. -/
theorem TraceFork.Accepts.partial_difference (accepts : fork.Accepts family st)
    (firstDigits : ∀ i, (fork.first i).witness.toCarried = digits)
    (secondDigits : ∀ i, (fork.second i).witness.toCarried = digits) (i : Fin S.blocks) :
    fork.difference i * O.tracePartial digits.partials i =
      st.contract (gadget S.sourceBase (fork.responseDifference i)) := by
  have first := (accepts.firstSatisfies i).evaluation
  have second := (accepts.secondSatisfies i).evaluation
  rw [firstDigits i] at first
  rw [secondDigits i] at second
  rw [TraceFork.responseDifference, gadget_sub, st.contract_sub, ← first, ← second]
  exact (sum_smul_sub_sum_smul (fun j => (fork.second i).challenge j)
    (fun j => (fork.first i).challenge j) (fun j => O.tracePartial digits.partials j) i
    (fun j other => (accepts.agree i j other).symm)).symm

/-- Akita Theorem 10.31, "Coordinate siblings" and "Assembly", for one claim at a trace level:
when each child has the carried digits `digits`, the extracted `traceExtract fork digits` is a
weak opening of the statement to the table that its source blocks decode to, with `κ̄_1 = 2ω` and
`β̄_∞ = Δ^cert_f`. The family has units as differences (`units`) and members of the norm
`‖c‖_1 = ω` (`sparse`), and `b_z ≥ 2`. -/
theorem traceExtract_opens {mass peak : ℕ} (units : UnitDifferences family)
    (sparse : ∀ c ∈ family, Sparse mass peak c) (base : 2 ≤ fold.responseBase)
    (accepts : fork.Accepts family st) (opens : digits.TraceOpens st)
    (firstDigits : ∀ i, (fork.first i).witness.toCarried = digits)
    (secondDigits : ∀ i, (fork.second i).witness.toCarried = digits) :
    (traceExtract fork digits).TraceOpens (2 * mass) (fold.sourceBound S) st
      (S.decode (traceExtract fork digits).source) := by
  have unit := fun i => accepts.isUnit_difference units i
  have cancel := fun i => traceExtract_factor_smul_source fork digits i (unit i)
  refine {
    certified := {
      source := {
        decodes := rfl
        innerRows := fun i => ?_
        rows := opens.rows
        innerDigits := opens.innerDigits
        chainDigits := opens.chainDigits }
      unit := unit
      factorNorm := fun i =>
        (sparse _ (accepts.secondMember i)).normOne_sub_le (sparse _ (accepts.firstMember i))
      sourceNorm := fun i => ?_ }
    openRows := opens.openRows
    partials := fun i => ?_
    value := opens.value
    partialDigits := opens.partialDigits
    openDigits := opens.openDigits }
  · show st.key.inner.mulVec (Ring.inverse (fork.difference i) • fork.responseDifference i) =
      gadget S.openingBase (digits.inner i)
    rw [Matrix.mulVec_smul, accepts.inner_difference firstDigits secondDigits i, smul_smul,
      Ring.inverse_mul_cancel _ (unit i), one_smul]
  · rw [cancel i, Descriptor.sourceBound_eq fold S base]
    exact vecNormInf_gadget_sub_le _ (accepts.secondSatisfies i).responseDigits
      (accepts.firstSatisfies i).responseDigits
  · show O.tracePartial digits.partials i = st.contract (gadget S.sourceBase
      (Ring.inverse (fork.difference i) • fork.responseDifference i))
    rw [gadget_smul, st.contract_smul, ← accepts.partial_difference firstDigits secondDigits i,
      ← mul_assoc, Ring.inverse_mul_cancel _ (unit i), one_mul]

/-- Akita Corollary 10.32 for one claim at a trace level, on the coefficient-norm route. `fork`
has two children for each block that satisfy their exact branch relations for one statement, with
challenges of a family of `R_{q,d_A}` with unit differences and `‖c‖_1 = ω`. `digits` is a family
of carried digits that satisfies the rows of the statement, for example that of a child. Then a
child with other carried digits gives a solution of Module-SIS for `B` or `D` with the bound
`b⋆ − 1` or for a compression map with the bound 1, or `traceExtract fork digits` is a weak
opening with `κ̄_1 = 2ω` and `β̄_∞ = Δ^cert_f`. -/
theorem traceExtract_opens_or_collision {mass peak : ℕ} (units : UnitDifferences family)
    (sparse : ∀ c ∈ family, Sparse mass peak c) (base : 2 ≤ fold.responseBase)
    (accepts : fork.Accepts family st) (opens : digits.TraceOpens st) :
    (traceForkCollision st fork digits).isSome ∨
      (traceExtract fork digits).TraceOpens (2 * mass) (fold.sourceBound S) st
        (S.decode (traceExtract fork digits).source) := by
  by_cases same : (∀ i, (fork.first i).witness.toCarried = digits) ∧
      ∀ i, (fork.second i).witness.toCarried = digits
  · exact Or.inr (traceExtract_opens units sparse base accepts opens same.1 same.2)
  · have other : ∃ i, (fork.first i).witness.toCarried ≠ digits ∨
        (fork.second i).witness.toCarried ≠ digits := by
      by_contra none
      exact same ⟨fun i => not_not.mp fun differs => none ⟨i, Or.inl differs⟩,
        fun i => not_not.mp fun differs => none ⟨i, Or.inr differs⟩⟩
    obtain ⟨i, differs⟩ := other
    refine Or.inl (List.findSome?_isSome_iff.mpr ⟨i, List.mem_finRange i, ?_⟩)
    rw [Option.isSome_or, Bool.or_eq_true]
    exact differs.imp
      (traceCarriedCollision_isSome (accepts.firstSatisfies i).carried opens)
      (traceCarriedCollision_isSome (accepts.secondSatisfies i).carried opens)

/-- Akita Corollary 10.26 for the extracted weak opening at a trace level: when each challenge of
the family has `‖c‖_mul,2 ≤ Γ` and each response of a child has `‖G_{b_z,m_A}·ẑ‖_2² ≤ S_max`, the
extracted opening has `κ̄_mul,2 = 2Γ` and `β̄_2² = 4·S_max`. -/
theorem traceExtract_euclidean {operator energy : ℕ} (units : UnitDifferences family)
    (accepts : fork.Accepts family st) (bounded : ∀ c ∈ family, MulBound operator c)
    (first : ∀ i, vecNormSq (gadget fold.responseBase (fork.first i).witness.response) ≤ energy)
    (second : ∀ i,
      vecNormSq (gadget fold.responseBase (fork.second i).witness.response) ≤ energy) :
    (traceExtract fork digits).toCertified.Euclidean (2 * operator) (4 * energy) where
  factorMul i :=
    (bounded _ (accepts.secondMember i)).sub (bounded _ (accepts.firstMember i))
  sourceSq i := by
    rw [traceExtract_factor_smul_source fork digits i (accepts.isUnit_difference units i)]
    exact (vecNormSq_sub_le (second i) (first i) (sq _).ge).trans_eq (by ring)

end Extraction

end NightstreamFPrime.Spec.Compression.Akita
