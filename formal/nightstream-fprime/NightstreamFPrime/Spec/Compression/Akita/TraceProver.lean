import NightstreamFPrime.Spec.Compression.Akita.Terminal
import NightstreamFPrime.Spec.Compression.Akita.TracePack

/-! The honest prover of one fold of Akita with the evaluation-trace opening method and of the
terminal, for one claim (ePrint 2026/1983, revision of 2026-10-15, Figure 4, steps 1 and 2,
Section 8.2, and the deterministic part of Lemma 10.7).

- `Shape.Bounded` has the conditions on a shape for the commitment algorithm with a bounded
  source (Akita Section 4.5: "Bounded source: `s_i := f_i`, `δ_com := 1`"). `Shape.embed` is this
  source, with `G_{b,M}·s_i = f_i` for each `f_i` (`gadget_embed`), and `commitSource` is the
  commitment algorithm for given source blocks. The schedule `key20.aks` has a bounded source at
  each level after the root: `Shape.Honest` is false for these shapes, because `b^{δ_com} < q`.
- `OpeningShape.traceEncode` gives the digits `ê` of the partials `e_i ∈ R_{q,d_A}`, and
  `OpeningShape.tracePartial_encode` states that `OpeningShape.tracePartial` decodes them.
- `traceResponse` is the folded response `z = Σ_i c_i·s_i` with challenges of the A ring.
- `honestTrace` is the witness of the honest prover, and `honestTrace_satisfies` states that it
  satisfies `Witness.TraceSatisfies` when the response is encodable. Akita Theorem 10.9 has the
  probability that no nonce gives an encodable response in the term `ξ_h` of (155). This file has
  no bound for it.
- `packTable` is the table of the ring elements `ψ(f̄)` of a packed polynomial `f̄`, and
  `unpackTable_packTable` states that `unpackTable` returns `f̄`.
- `traceSource` has the bounded source blocks of `packTable`, and `traceScalar_source`
  states that the left side of the row (98) for these blocks is the evaluation of `f̄`: the
  converse of `traceOpens_evaluation` for honest data.
- `terminal_accepts` states that the state `t_i = A_term·s_i`, the clear partials and the response
  `z = Σ_i c_i·s_i` of the honest prover pass the checks (128), (129) and (130) of the
  terminal. -/

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib (eqAt mle mle_congr)

variable {S : Shape} {O : OpeningShape S} {fold : Descriptor}

section Bounded

/-- The conditions on a shape for the commitment algorithm with a bounded source (Akita
Section 4.5): the base `b_1` decomposes each residue, `b_1 ≤ b⋆`, a block is not empty, the
source has a digit, and the inputs of the compression maps have a position for each digit. The
base `b` of the source has no condition. -/
structure Shape.Bounded (S : Shape) : Prop where
  opening : Base S.openingBase S.openingDigits
  certified : S.openingBase ≤ S.certifiedBase
  block : 0 < S.blockLength
  digit : 0 < S.sourceDigits
  path : PathFits S.outerDegree (S.slices * S.outerRank) S.chain

/-- The bounded source of a block `f_i` (Akita Section 4.5, "Bounded source: `s_i := f_i`"): the
first digit is `f_i`, and each other digit is 0. -/
noncomputable def Shape.embed (S : Shape) (F : Fin S.blockLength → Rq S.innerDegree) : S.Source :=
  fun entry => if (entry.2 : ℕ) = 0 then F entry.1 else 0

/-- `G_{b,M}·s_i = f_i` for the bounded source (Akita Section 4.5: "In either encoding,
`G_{b,M} s_i = f_i`"). -/
theorem gadget_embed (positive : 0 < S.sourceDigits) (F : Fin S.blockLength → Rq S.innerDegree) :
    gadget S.sourceBase (S.embed F) = F := by
  funext x
  rw [gadget, Finset.sum_eq_single (⟨0, positive⟩ : Fin S.sourceDigits)]
  · simp [Shape.embed]
  · intro u _ other
    have later : ¬ (u : ℕ) = 0 := fun zero => other (Fin.ext zero)
    simp [Shape.embed, later]
  · intro absent
    exact absurd (Finset.mem_univ _) absent

/-- The commitment algorithm of Akita Section 4.5 for given source blocks:
`t̂_i = G^{-1}_{b_1,n_A}(A s_i)`, the stacked image of `B` on the slice inputs, and the compression
chain. `commit` is this algorithm for the source blocks of a full-field source. -/
noncomputable def commitSource (key : Key S) (source : Fin S.blocks → S.Source) :
    S.Payload × Decommitment S :=
  let inner : Fin S.blocks → S.InnerDigits := fun i =>
    decompose S.openingBase (key.inner.mulVec (source i))
  let path := pathCommit S.chain key.chain (stack key.outer (S.sliceInput inner))
  (path.2, ⟨source, inner, path.1⟩)

/-- The output of `commitSource` is an algebraic decommitment of its payload to the table that the
source blocks decode to (Akita Definition 3.5 for the relation of Definition 10.21). -/
theorem commitSource_opens (shape : S.Bounded) (key : Key S) (source : Fin S.blocks → S.Source) :
    (commitSource key source).2.Opens key (commitSource key source).1 (S.decode source) where
  decodes := rfl
  innerRows _ := (gadget_decompose shape.opening _).symm
  rows := (pathCommit_rows shape.path key.chain _).1
  innerDigits _ := (decompose_balanced shape.opening _).mono shape.certified
  chainDigits := (pathCommit_rows shape.path key.chain _).2

/-- The bounded source blocks of a table decode to the padded table. -/
theorem decode_embed_pack (shape : S.Bounded) (table : Table) :
    (S.decode fun i => S.embed (S.pack table i)) = S.pad table := by
  funext cell
  unfold Shape.decode Shape.pad
  split
  next inside =>
    have positive : 0 < S.blockLength * S.innerDegree :=
      Nat.mul_pos shape.block (NeZero.pos S.innerDegree)
    have below := Nat.mod_lt cell positive
    have entry : cell % (S.blockLength * S.innerDegree) / S.innerDegree < S.blockLength :=
      Nat.div_lt_of_lt_mul (lt_of_lt_of_eq below (Nat.mul_comm _ _))
    have unpacked := sequence_apply
      (S.pack table ⟨cell / (S.blockLength * S.innerDegree), inside⟩)
      ⟨cell % (S.blockLength * S.innerDegree) / S.innerDegree, entry⟩
      (Nat.mod_lt (cell % (S.blockLength * S.innerDegree)) (NeZero.pos S.innerDegree))
    simp only [Nat.div_add_mod'] at unpacked
    rw [gadget_embed shape.digit, unpacked, Shape.pack,
      coeff_ofCoeffs _ (Nat.mod_lt _ (NeZero.pos S.innerDegree))]
    congr 1
    have whole := Nat.div_add_mod cell (S.blockLength * S.innerDegree)
    have part := Nat.div_add_mod' (cell % (S.blockLength * S.innerDegree)) S.innerDegree
    rw [Nat.add_mul, Nat.add_assoc, part, Nat.mul_assoc,
      Nat.mul_comm (cell / (S.blockLength * S.innerDegree))]
    exact whole
  next => rfl

end Bounded

namespace OpeningShape

/-- The native blocks of `R_{q,d_D}` of the `d_A` coordinates of one partial of a trace level:
coefficient `k` of block `y` is coordinate `y·d_D + k`. -/
noncomputable def traceNative (O : OpeningShape S) (value : Rq S.innerDegree) :
    Fin O.subcolumns → Rq O.openDegree :=
  fun y => ofCoeffs fun k => coeff value (y * O.openDegree + k)

/-- The digits `ê` of the partials `e_i ∈ R_{q,d_A}` of a trace level: the balanced digits in base
`b_1` of the native blocks of each partial (Akita Section 5.4). -/
noncomputable def traceEncode (O : OpeningShape S) (partials : Fin S.blocks → Rq S.innerDegree) :
    O.Partials :=
  fun column => decompose S.openingBase (O.traceNative (partials column.1)) column.2

/-- `tracePartial` decodes the digits of `traceEncode`, when the view of `D` has the `d_A`
coordinates of a partial: `d_A ≤ (k·d_car/d_D)·d_D`. -/
theorem tracePartial_encode (O : OpeningShape S) (base : Base S.openingBase S.openingDigits)
    (wide : S.innerDegree ≤ O.subcolumns * O.openDegree)
    (partials : Fin S.blocks → Rq S.innerDegree) (i : Fin S.blocks) :
    O.tracePartial (O.traceEncode partials) i = partials i := by
  refine ext_coeff fun j small => ?_
  have column : j / O.openDegree < O.subcolumns :=
    Nat.div_lt_of_lt_mul ((small.trans_le wide).trans_eq (Nat.mul_comm _ _))
  have digits : (fun entry : Fin O.subcolumns × Fin S.openingDigits =>
      O.traceEncode partials (i, entry)) =
      decompose S.openingBase (O.traceNative (partials i)) := rfl
  have unpacked := sequence_apply (O.traceNative (partials i)) ⟨j / O.openDegree, column⟩
    (Nat.mod_lt j (NeZero.pos O.openDegree))
  simp only [Nat.div_add_mod'] at unpacked
  rw [tracePartial, coeff_ofCoeffs _ small, digits, gadget_decompose base, unpacked]
  simp only [traceNative]
  rw [coeff_ofCoeffs _ (Nat.mod_lt _ (NeZero.pos O.openDegree)), Nat.div_add_mod']

end OpeningShape

section Contract

variable {E : Type} [CommRing E] {d : ℕ} {ι κ : Type} [Fintype ι]

theorem traceContract_sum (subfield : E →+* Rq d) (position : ι → E) (s : Finset κ)
    (F : κ → ι → Rq d) :
    traceContract subfield position (∑ i ∈ s, F i) =
      ∑ i ∈ s, traceContract subfield position (F i) := by
  simp only [traceContract, Finset.sum_apply, Finset.mul_sum]
  exact Finset.sum_comm

end Contract

/-- The folded response `z = Σ_i c_i·s_i` of one claim at a trace level, for the challenges
`c_i ∈ R_{q,d_A}` (Akita (73)). -/
noncomputable def traceResponse (challenge : Fin S.blocks → Rq S.innerDegree)
    (source : Fin S.blocks → S.Source) : S.Source :=
  ∑ i, challenge i • source i

section Honest

variable {E : Type} [CommRing E] [Module Zq E]

/-- The digits `ê` of the honest prover of a trace level: the digits of the partials
`e_i = ⟨a, G_{b,M}·s_i⟩` of Akita (66). -/
noncomputable def tracePartials (O : OpeningShape S) (subfield : E →+* Rq S.innerDegree)
    (position : Fin S.blockLength → E) (source : Fin S.blocks → S.Source) : O.Partials :=
  O.traceEncode fun i => traceContract subfield position (gadget S.sourceBase (source i))

/-- The witness of the honest prover of one trace fold for the commitment state `state` and the
challenges `c_i ∈ R_{q,d_A}` (Akita Figure 4, steps 1 and 2): `t̂` and the digits of `F_1`, `F_2`
of the state, the digits `ê` and the digits of `H_1`, `H_2`, and `ẑ = G^{-1}_{b_z,m_A}(z)` for
the response `z` of Akita (73) and (74). -/
noncomputable def honestTrace (key : OpeningKey O) (subfield : E →+* Rq S.innerDegree)
    (position : Fin S.blockLength → E) (fold : Descriptor)
    (challenge : Fin S.blocks → Rq S.innerDegree) (state : Decommitment S) : Witness O fold where
  inner := state.inner
  chain := state.chain
  partials := tracePartials O subfield position state.source
  openChain := (honestOpening key (tracePartials O subfield position state.source)).1
  response := decompose fold.responseBase (traceResponse challenge state.source)

/-- The deterministic part of Akita Lemma 10.7 for one claim at a trace level: when the response
is admissible, the honest witness satisfies the exact branch relation `Witness.TraceSatisfies` for
the challenges. The state is an algebraic decommitment of `u_pub` (`opens`), the statement has the
payload `v_pub` and the target of the honest prover (`payload`, `value`), the bases satisfy
`b_1 ≤ b⋆` (`shape`, which has no condition on the base of the source) and `b_z ≤ b⋆`
(`certified`), the view of `D` has the `d_A` coordinates of a
partial (`wide`), and the response `z` satisfies `Enc` of Akita Definition 10.5 (`encodable`). The
hypothesis `encodable` is not a consequence of the others: Akita completeness is not perfect
(Theorem 10.9, (155)). -/
theorem honestTrace_satisfies {st : TraceStatement O E} {state : Decommitment S} {table : Table}
    (challenge : Fin S.blocks → Rq S.innerDegree) (shape : S.Bounded)
    (radix : Radix fold.responseBase) (certified : fold.responseBase ≤ S.certifiedBase)
    (fits : PathFits O.openDegree (1 * O.openRank) O.chain)
    (wide : S.innerDegree ≤ O.subcolumns * O.openDegree)
    (opens : state.Opens st.key st.payload table)
    (encodable : Encodable fold.responseBase fold.responseDigits
      (traceResponse challenge state.source))
    (payload : st.openPayload =
      (honestOpening st.openKey (tracePartials O st.subfield st.position state.source)).2)
    (value : st.value =
      st.scalar fun i => st.contract (gadget S.sourceBase (state.source i))) :
    (honestTrace st.openKey st.subfield st.position fold challenge state).TraceSatisfies st
      challenge := by
  have partials : O.tracePartial (tracePartials O st.subfield st.position state.source) =
      fun i => st.contract (gadget S.sourceBase (state.source i)) :=
    funext fun i => O.tracePartial_encode shape.opening wide _ i
  have recomposed := gadget_decompose_of_encodable radix encodable
  refine {
    carried := {
      rows := opens.rows
      openRows := ?_
      value := ?_
      innerDigits := opens.innerDigits
      chainDigits := opens.chainDigits
      partialDigits := fun column =>
        ((decompose_balanced shape.opening _).mono shape.certified) column.2
      openDigits := (pathCommit_rows fits st.openKey.chain _).2 }
    evaluation := ?_
    innerRows := ?_
    responseDigits := (decompose_balanced_of_radix radix _ _).mono certified }
  · show Rows O.chain st.openKey.chain _ _ st.openPayload
    rw [payload]
    exact (pathCommit_rows fits st.openKey.chain _).1
  · show st.scalar (O.tracePartial (tracePartials O st.subfield st.position state.source)) =
      st.value
    rw [partials, value]
  · show ∑ i, challenge i *
        O.tracePartial (tracePartials O st.subfield st.position state.source) i =
      st.contract (gadget S.sourceBase (gadget fold.responseBase
        (decompose fold.responseBase (traceResponse challenge state.source))))
    rw [recomposed, traceResponse, gadget_sum, TraceStatement.contract, traceContract_sum,
      partials]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [gadget_smul]
    exact (st.contract_smul _ _).symm
  · show st.key.inner.mulVec (gadget fold.responseBase
      (decompose fold.responseBase (traceResponse challenge state.source))) =
      ∑ i, challenge i • gadget S.openingBase (state.inner i)
    rw [recomposed, traceResponse, Matrix.mulVec_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Matrix.mulVec_smul, opens.innerRows]

end Honest

section Pack

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E] {d : ℕ} [NeZero d] {m : ℕ}

/-- The table of the ring elements `ψ(f̄)` of a packed polynomial `f̄`: the cells
`r·d, …, r·d + d − 1` have the coefficients of `ψ` of the `d/k = 2·m` values of `f̄` from the value
`r·(d/k)` (Akita Section 3.7 and (21)). -/
noncomputable def packTable (quarter : 4 * m = d) (g : ℕ → E) : Table :=
  fun cell => coeff (psi quarter fun u => g (cell / d * (2 * m) + u)) (cell % d)

theorem ofCoeffs_packTable (quarter : 4 * m = d) (g : ℕ → E) (r : ℕ) :
    (ofCoeffs fun k => packTable quarter g (r * d + k) : Rq d) =
      psi quarter fun u => g (r * (2 * m) + u) := by
  refine ext_coeff fun k small => ?_
  have quotient : (r * d + k) / d = r := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ (NeZero.pos d), Nat.div_eq_of_lt small,
      Nat.zero_add]
  have rest : (r * d + k) % d = k := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt small]
  rw [coeff_ofCoeffs _ small, packTable, quotient, rest]

/-- `unpackTable` returns the packed polynomial of `packTable`, when `d/k` is a unit of `Z_q`. -/
theorem unpackTable_packTable (quarter : 4 * m = d) (unit : IsUnit ((2 * m : ℕ) : Zq))
    (g : ℕ → E) : unpackTable d m (packTable quarter g) = g := by
  have positive : 0 < 2 * m := by
    have := NeZero.pos d
    omega
  funext cell
  rw [unpackTable, ofCoeffs_packTable, unpack_psi quarter unit _ (Nat.mod_lt cell positive),
    Nat.div_add_mod']

end Pack

section Source

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E] {m : ℕ}

/-- The bounded source blocks `s_i = F_i` of the commitment of a trace level to a packed
polynomial `f̄`, for the ring elements `F_{i,x} = ψ(f̄_{i,x})` (Akita Section 4.5 and (21)). -/
noncomputable def traceSource (quarter : 4 * m = S.innerDegree) (g : ℕ → E) (i : Fin S.blocks) :
    S.Source :=
  S.embed (S.pack (packTable quarter g) i)

/-- The table that the source blocks `traceSource` decode to has the packed polynomial `f̄` on the
cube of a point, when `f̄` is 0 after the `B·M·(d_A/k)` values of the live blocks. -/
theorem unpackTable_traceSource (quarter : 4 * m = S.innerDegree) (shape : S.Bounded)
    (unit : IsUnit ((2 * m : ℕ) : Zq)) {g : ℕ → E}
    (live : ∀ cell, S.blocks * (S.blockLength * (2 * m)) ≤ cell → g cell = 0) (cell : ℕ) :
    unpackTable S.innerDegree m (S.decode (traceSource quarter g)) cell = g cell := by
  have positive : 0 < 2 * m := by
    have := NeZero.pos S.innerDegree
    omega
  have decoded : S.decode (traceSource quarter g) = S.pad (packTable quarter g) :=
    decode_embed_pack shape _
  have block : ∀ k < S.innerDegree, (cell / (2 * m) * S.innerDegree + k) /
      (S.blockLength * S.innerDegree) = cell / (2 * m) / S.blockLength := fun k small => by
    rw [Nat.mul_comm S.blockLength, ← Nat.div_div_eq_div_mul, Nat.add_comm,
      Nat.add_mul_div_right _ _ (NeZero.pos S.innerDegree), Nat.div_eq_of_lt small, Nat.zero_add]
  by_cases inside : cell / (2 * m) / S.blockLength < S.blocks
  · have same : (ofCoeffs fun k => S.decode (traceSource quarter g)
        (cell / (2 * m) * S.innerDegree + k) : Rq S.innerDegree) =
        ofCoeffs fun k => packTable quarter g (cell / (2 * m) * S.innerDegree + k) :=
      ext_coeff fun k small => by
        rw [coeff_ofCoeffs _ small, coeff_ofCoeffs _ small, decoded, Shape.pad, block k small,
          ite_eq_left inside]
    rw [unpackTable, same, ← unpackTable, unpackTable_packTable quarter unit]
  · have empty : (ofCoeffs fun k => S.decode (traceSource quarter g)
        (cell / (2 * m) * S.innerDegree + k) : Rq S.innerDegree) = 0 :=
      ext_coeff fun k small => by
        rw [coeff_ofCoeffs _ small, coeff_zero, decoded, Shape.pad, block k small,
          ite_eq_right inside]
    have absent : S.blocks * (S.blockLength * (2 * m)) ≤ cell := by
      have rows : S.blocks * S.blockLength ≤ cell / (2 * m) :=
        (Nat.mul_le_of_le_div _ _ _ (Nat.le_of_not_lt inside))
      calc S.blocks * (S.blockLength * (2 * m)) = S.blocks * S.blockLength * (2 * m) :=
            (Nat.mul_assoc _ _ _).symm
        _ ≤ cell / (2 * m) * (2 * m) := Nat.mul_le_mul_right _ rows
        _ ≤ cell := Nat.div_mul_le_self _ _
    rw [unpackTable, empty, map_zero, live cell absent]

/-- The honest evaluation of a trace level (Akita (98), the converse of `traceOpens_evaluation`):
for the source blocks `traceSource` of a packed polynomial `f̄`, the sum
`Σ_i ẽq(ρ_blk, i)·T_{ρ_pack}(e_i)` of the partials `e_i = ⟨a, G_{b,M}·s_i⟩` is `f̄(ρ)` at
`ρ = (ρ_pack, ρ_pos, ρ_blk)`. `M` and `d_A/k` are the powers of two of the lengths of their
segments, `B ≤ 2^{r_blk}`, and `f̄` is 0 after the values of the live blocks. -/
theorem traceScalar_source (quarter : 4 * m = S.innerDegree) (shape : S.Bounded)
    (unit : IsUnit ((2 * m : ℕ) : Zq)) {g : ℕ → E} (block position pack : List E)
    (blocks : S.blocks ≤ 2 ^ block.length) (length : S.blockLength = 2 ^ position.length)
    (width : 2 * m = 2 ^ pack.length)
    (live : ∀ cell, S.blocks * (S.blockLength * (2 * m)) ≤ cell → g cell = 0) :
    ∑ i : Fin S.blocks, eqAt block i * packFunctional quarter (eqAt pack)
        (traceContract (subfield quarter).toRingHom (fun x : Fin S.blockLength => eqAt position x)
          (gadget S.sourceBase (traceSource quarter g i))) =
      mle (pack ++ (position ++ block)) g := by
  rw [traceScalar_eq_mle quarter block position pack blocks length width]
  exact mle_congr _ fun cell _ => unpackTable_traceSource quarter shape unit live cell

end Source

section Terminal

variable {E : Type} [CommRing E] [Module Zq E]

/-- The honest prover of the terminal (Akita Section 8.2): for the source blocks `s_i`, the state
`t_i = A_term·s_i`, the clear partials `e_i = ⟨a_term, G_{b_term,M_term}·s_i⟩` and the clear
response `z = Σ_i c_i·s_i` pass the checks (128) and (129) for each challenge vector, and the
check (130) when the pair `(θ_term, v_tensor)` has the value of the partials. -/
theorem terminal_accepts {st : TerminalStatement S E} (source : Fin S.blocks → S.Source)
    (challenge : Fin S.blocks → Rq S.innerDegree)
    (state : ∀ i, st.state i = st.matrix.mulVec (source i))
    (value : st.scale * ∑ i, st.block i * st.functional
      (traceContract st.subfield st.position (gadget S.sourceBase (source i))) = st.value) :
    st.Accepts (fun i => traceContract st.subfield st.position (gadget S.sourceBase (source i)))
      challenge (traceResponse challenge source) where
  inner := by
    rw [traceResponse, Matrix.mulVec_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [Matrix.mulVec_smul, state]
  evaluation := by
    rw [traceResponse, gadget_sum, traceContract_sum]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [gadget_smul, traceContract_smul]
  value := value

end Terminal

end NightstreamFPrime.Spec.Compression.Akita
