import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingLevel
import NightstreamFPrime.Spec.Compression.Akita.Replay.WeightPackingBatchOffload
import NightstreamFPrime.Spec.Compression.Akita.Replay.LevelIn
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalJoin

/-! The weight theorem of a packing level with one group in the form of the hypothesis `weighs` of
`packFoldRun_runs` (`Replay/LevelPack.lean`), for the statement `packStatementFor` of the words
of the replay (`Replay/LevelStatement.lean`) and the group of `packGroupsIn` with the claim
coefficient 1 (`packWeight_one`), and the hypothesis `supported` (`packSupport_one`), and the
hypothesis `supported` of `batchFoldRun_runs` and `batchOffloadRun_runs` for the groups of
`batchGroups` (`batchSupport_groups`). `packGroupsIn_batch`: the groups of `packGroupsIn` at a fold
with opening groups (`Schedules.ofBatch`) are `batchGroups` with the drawn fold challenges
`drawnCarrier` and `drawnAmbient`; `batchWeight_groups` is the hypothesis `weighs` of
`batchFoldRun_runs` in the form that `foldRunIn_batch_runs` applies it, `batchSupport_in` the
hypothesis `supported` in this form, and `batchOffload_groups`
the hypothesis `weighs` of `batchOffloadRun_runs` in the form of `offloadRunIn_runs`, for each split
whose `part` and `scale` are those of `Batch.codeScan` (`codeScan_cast`: the split `codeScan` moved
by an equation of statements, as `Statement.over_eq`, keeps its three fields). `drawn_ambiented`,
`drawn_getD`, `batchAmbiented` and `packAmbiented`: the hypothesis `ambiented` for the challenges
that `packGroupsIn` builds from the drawn challenges of `foldChallenges` (`ring_ambient`). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita NightstreamFPrime.Spec.Compression.Lib

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E] {S : Shape}
  {O : OpeningShape S} {fold : Descriptor}

/-- `weighs` of `packFoldRun_runs`: at each point `r_2` of `μ` coordinates, the weight of
`packFinish` is `m_{τ_1}(r_2)` of `liftedReduction` for `packStatementFor`, for the fold
challenges `c_i` of the group in `R_{q,d_car}` and `ι(c_i)` in `R_{q,d_A}`, the basis `(1, u)`
and the inputs of `packInputsIn`. -/
theorem packWeight_one (β : Module.Basis (Fin O.planes) Zq E)
    (based : ∀ t, β t = (extBasis.getD t Ext.zero).into) (F : FoldInputs) (V : PackingValues)
    (data : RhsRun → Array Fq) (source : RhsSource) (coefficients claims : Array Ext)
    (challenges ambient : List (Array Fq)) (point : List Ext)
    (challenge : Fin S.blocks → Rq O.carrierDegree)
    (challenged : ∀ i, challenge i =
      Poly.toRq _ (challenges.getD i (Poly.zero O.carrierDegree)))
    (ambiented : ∀ i : Fin S.blocks, Poly.toRq S.innerDegree
      (ambient.getD i.val (Poly.zero S.innerDegree)) =
        embed O.stride O.stride_tower (challenge i))
    {μ : ℕ} (fits : O.cells fold ≤ 2 ^ μ) {digits : ℕ} (room : O.quotientRoom fold digits ≤ 2 ^ μ)
    {batch : ℕ} (rowsFit : O.rowCount + 1 ≤ 2 ^ batch)
    (numbers : PackGroup.Fits ⟨S, O, fold, challenges, ambient, point, Ext.one⟩ μ)
    (tau1 : F.tau1 = V.tau1) (digited : F.digits = digits) (base : F.base = S.certifiedBase)
    (path : F.path = ⟨O.openDegree, O.openRank, O.chain⟩) (code : F.order = .code)
    (cells : F.cells = {}) (tau : V.tau1.length = batch) :
    ∀ p : List Ext, p.length = μ →
      ((packFinish F [⟨S, O, fold, challenges, ambient, point, Ext.one⟩] V none p).weight.into :
        E) = mle (p.map Ext.into) (((liftedReduction β (packStatementFor O F.setup point data
          source coefficients claims) challenge μ fits digits room batch rowsFit).rows
            V.alpha.into).batchWeight (V.tau1.map Ext.into)) := by
  subst digited
  exact fun p length => foldWeight_one ⟨S, O, fold, challenges, ambient, point, Ext.one⟩ β based F
    V _ (Key.ofSetup_shared _ _) (OpeningKey.ofSetup_shared _ _) (fun _ => rfl) (fun _ => rfl)
    (fun _ => rfl) (fun _ => rfl) challenge challenged ambiented rfl fits room rowsFit numbers tau1
    base path code cells tau p length

omit [Fintype E] in
/-- `supported` of `packFoldRun_runs`: at each point `r_2` of `μ` coordinates, the binary-support
weight of the replay is the multilinear extension of `1_{I_bin}·ẽq(r_virt, ·)` for the support of
the witness of Akita (76). -/
theorem packSupport_one (F : FoldInputs) (V : PackingValues) (challenges ambient : List (Array Fq))
    (point : List Ext) {μ : ℕ} {digits : ℕ} (room : O.quotientRoom fold digits ≤ 2 ^ μ)
    (digited : F.digits = digits) (path : F.path = ⟨O.openDegree, O.openRank, O.chain⟩)
    (virtualLength : V.virtual.length = μ) :
    ∀ p : List Ext, p.length = μ → O.compressed fold = true →
      ((foldSupport (Placement.of [⟨S, O, fold, challenges, ambient, point, Ext.one⟩] F.path
        F.digits) (CellEq.half V.virtual).get (CellEq.half p).get).into : E) =
          mle (p.map Ext.into) fun x =>
            if (O.gaps fold digits).support x then eqAt (V.virtual.map Ext.into) x else 0 := by
  intro p length _
  subst digited
  rw [path]
  exact foldSupport_one ⟨S, O, fold, challenges, ambient, point, Ext.one⟩ F.digits _ _ room
    (by rw [List.length_map, length])
    (fun u small => CellEq.get_half V.virtual (by rw [virtualLength]; exact small))
    (fun x small => CellEq.get_half p (by rw [length]; exact small))

omit [Fintype E] in
/-- `supported` of `batchFoldRun_runs` and `batchOffloadRun_runs`: at each point `r_2` of `μ`
coordinates, the binary-support weight of the replay for the groups in the order of the code is
the multilinear extension of `1_{I_bin}·ẽq(r_virt, ·)` for the support of `Batch.gaps`. -/
theorem batchSupport_groups {n : ℕ} (B : Batch (Fin n)) (cs amb : Fin n → List (Array Fq))
    (pts : Fin n → List Ext) (w : Fin n → Ext) (F : FoldInputs) (V : PackingValues) {μ : ℕ}
    {digits : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ) (digited : F.digits = digits)
    (path : F.path = ⟨B.openDegree, B.openRank, B.chain⟩) (virtualLength : V.virtual.length = μ) :
    ∀ p : List Ext, p.length = μ → B.compressed = true →
      ((foldSupport (Placement.of (batchGroups B cs amb pts w) F.path F.digits)
        (CellEq.half V.virtual).get (CellEq.half p).get).into : E) =
          mle (p.map Ext.into) fun x =>
            if (B.gaps digits).support x then eqAt (V.virtual.map Ext.into) x else 0 := by
  intro p length _
  subst digited
  rw [path]
  exact foldSupport_batch B cs amb pts w F.digits _ _ room (by rw [List.length_map, length])
    (fun u small => CellEq.get_half V.virtual (by rw [virtualLength]; exact small))
    (fun x small => CellEq.get_half p (by rw [length]; exact small))

/-- `ι` depends only on the stride. -/
theorem embed_stride {d dCar s₁ s₂ : ℕ} [NeZero d] [NeZero dCar] (same : s₁ = s₂)
    (t₁ : s₁ * dCar = d) (t₂ : s₂ * dCar = d) : embed s₁ t₁ = embed s₂ t₂ := by
  subst same
  rfl

/-- A drawn challenge of a packing group (`foldChallenges`: the positions of the sampler times
`ring/degree`) in `R_{q,d_A}` is `ι` of the challenge in `R_{q,d_car}` that `packGroupsIn` builds
from it with its positions divided by `d_A/d_car` (`ring_ambient`, Akita (38)). -/
theorem drawn_ambiented {G : FoldGroup} (c : Sparse) {d dCar stride : ℕ} [NeZero d] [NeZero dCar]
    (tower : stride * dCar = d) (ringed : G.ring = d) (degreed : G.degree = dCar) :
    Poly.toRq d (Poly.ofSparse d (ambient G c)) = embed stride tower (Poly.toRq dCar
      (Poly.ofSparse dCar ((ambient G c).map fun x => (x.1 / (d / dCar), x.2)))) := by
  have positive : 0 < dCar := Nat.pos_of_neZero dCar
  have quotient : d / dCar = stride := by rw [← tower, Nat.mul_div_cancel _ positive]
  have back : (ambient G c).map (fun x => (x.1 / (d / dCar), x.2)) = c := by
    unfold ambient
    rw [List.map_map]
    conv_rhs => rw [← List.map_id c]
    refine List.map_congr_left fun x _ => ?_
    rw [Function.comp_apply, ringed, degreed, quotient,
      Nat.mul_div_cancel _ (stride_pos tower)]
    rfl
  subst ringed degreed
  have tower' : G.ring / G.degree * G.degree = G.ring := by rw [quotient, tower]
  rw [back, Poly.toRq_ofSparse_ring, Poly.toRq_ofSparse_ring, ring_ambient tower' c,
    embed_stride quotient tower' tower]

/-- The drawn challenges of a packing group as the lists of `packGroupsIn`: in `R_{q,d_A}` each is
`ι` of the challenge in `R_{q,d_car}`, and both lists have the zero challenge after their end. -/
theorem drawn_getD (G : FoldGroup) (cs : List Sparse) {d dCar stride : ℕ} [NeZero d] [NeZero dCar]
    (tower : stride * dCar = d) (ringed : G.ring = d) (degreed : G.degree = dCar) (i : ℕ) :
    Poly.toRq d (((cs.map (ambient G)).map (Poly.ofSparse d)).getD i (Poly.zero d)) =
      embed stride tower (Poly.toRq dCar (((cs.map (ambient G)).map fun c =>
        Poly.ofSparse dCar (c.map fun x => (x.1 / (d / dCar), x.2))).getD i (Poly.zero dCar))) := by
  rw [List.map_map, List.map_map]
  by_cases inside : i < cs.length
  · rw [List.getD_eq_getElem _ _ (by rw [List.length_map]; exact inside),
      List.getD_eq_getElem _ _ (by rw [List.length_map]; exact inside), List.getElem_map,
      List.getElem_map]
    exact drawn_ambiented cs[i] tower ringed degreed
  · rw [List.getD_eq_default _ _ (by rw [List.length_map]; omega),
      List.getD_eq_default _ _ (by rw [List.length_map]; omega), Poly.toRq_zero, Poly.toRq_zero,
      map_zero]

section Drawn

open Replayer

variable {n : ℕ} (B : Batch (Fin n)) (s : Replayer) (level : ℕ)

/-- The fold challenges of the group at the index `k` of a fold with opening groups in its subring
`R_{q,d_car}` (`packGroupsIn`): each drawn sparse challenge with its positions divided by
`d_A / d_car`. -/
def drawnCarrier (k : Fin n) : List (Array Fq) :=
  (drawnIn s level (groupOrder n k).val).map fun c =>
    Poly.ofSparse (B.carrierDegree (groupOrder n k)) (c.map fun x =>
      (x.1 / ((B.shape (groupOrder n k)).innerDegree / B.carrierDegree (groupOrder n k)), x.2))

/-- The fold challenges of the group at the index `k` of a fold with opening groups in
`R_{q,d_A}` (`packGroupsIn`). -/
def drawnAmbient (k : Fin n) : List (Array Fq) :=
  (drawnIn s level (groupOrder n k).val).map (Poly.ofSparse (B.shape (groupOrder n k)).innerDegree)

/-- The numbers `PackGroup.Fits` of a group of a fold with opening groups do not depend on its fold
challenges, its point and its claim coefficient. -/
theorem batchGroup_fits {cs amb : Fin n → List (Array Fq)} {pts : Fin n → List Ext}
    {w : Fin n → Ext} (cs' amb' : Fin n → List (Array Fq)) (pts' : Fin n → List Ext)
    (w' : Fin n → Ext) {k : Fin n} {μ : ℕ} (numbers : (batchGroup B cs amb pts w k).Fits μ) :
    (batchGroup B cs' amb' pts' w' k).Fits μ := by
  obtain ⟨a, b, c, d, e, f, g, h, i, j, l, m, o, p, q, r, t, u, v⟩ := numbers
  exact ⟨a, b, c, d, e, f, g, h, i, j, l, m, o, p, q, r, t, u, v⟩

/-- The groups of `packGroupsIn` for the slots of a fold with opening groups in the order
`groupOrder` (`Schedules.ofBatch`). -/
theorem packGroupsIn_batch (P : PackShape) (F : Fold)
    (slots : P.slots = (List.finRange n).map fun k => ⟨(groupOrder n k).val,
      B.shape (groupOrder n k), B.opening (groupOrder n k), B.fold (groupOrder n k)⟩) :
    packGroupsIn P s level F = batchGroups B (drawnCarrier B s level) (drawnAmbient B s level)
      (fun k => claimPointIn s level (groupOrder n k).val)
      (fun k => (rowsIn s level F).getD (groupOrder n k).val Ext.zero) := by
  rw [packGroupsIn, slots, List.map_map]
  rfl

variable {B} in
/-- `weighs` of `batchFoldRun_runs` in the form of `foldRunIn_batch_runs`: at each point `r_2` of
`μ` coordinates, the weight of `packFinish` for the groups of `packGroupsIn` is `m_{τ_1}(r_2)` of
`Batch.liftedReduction` for `batchStatementFor`, for the fold challenges `c_{g,i}` of the groups in
`R_{q,d_car,g}` and `ι(c_{g,i})` in `R_{q,d_A,g}`, the basis `(1, u)` and the inputs of
`packInputsIn`. -/
theorem batchWeight_groups {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
    (pub : Public) (F : Fold) {P : PackShape}
    (slots : P.slots = (List.finRange n).map fun k => ⟨(groupOrder n k).val,
      B.shape (groupOrder n k), B.opening (groupOrder n k), B.fold (groupOrder n k)⟩)
    (opening : P.path = ⟨B.openDegree, B.openRank, B.chain⟩) {digits : ℕ}
    (digited : P.digits = digits) (setup : Array Fq) (source : Fin n → RhsSource) {base : ℕ}
    (based : 2 ^ F.logBasis = base) (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : B.Challenge)
    (challenged : ∀ k i, challenge (groupOrder n k) i = Poly.toRq _
      ((drawnCarrier B s level k).getD i (Poly.zero (B.carrierDegree (groupOrder n k)))))
    (ambiented : ∀ k (i : Fin (B.shape (groupOrder n k)).blocks),
      Poly.toRq (B.shape (groupOrder n k)).innerDegree
        ((drawnAmbient B s level k).getD i.val (Poly.zero (B.shape (groupOrder n k)).innerDegree)) =
        embed (B.opening (groupOrder n k)).stride (B.opening (groupOrder n k)).stride_tower
          (challenge (groupOrder n k) i))
    {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (numbers : ∀ g ∈ packGroupsIn P s level F, g.Fits μ) (k₀ : Fin n)
    (tau : (recordedIn s ⟨level, "tau1", none⟩).size = batch) :
    ∀ p : List Ext, p.length = μ →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) none p).weight.into : E) =
      mle (p.map Ext.into) (((Batch.liftedReduction β (batchStatementFor B setup
        (claimPointIn s level) (dataIn pub s level) source (rowsIn s level F)
          (claimsIn s level F)) challenge μ fits base digits room batch rowsFit).rows
            (firstIn s ⟨level, "alpha", none⟩).into).batchWeight
              ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into)) := by
  subst digited based
  intro p length
  rw [packGroupsIn_batch B s level P F slots] at numbers ⊢
  exact foldWeight_batch B _ _ _ _ β basis (packInputsIn P setup s level F)
    (packingValuesIn s level F) _ (fun g => Key.ofSetup_shared _ _) rfl
    (PathKey.ofSetup_shared _ _) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) challenge challenged ambiented fits room rowsFit
    (fun k => numbers _ (List.mem_map_of_mem (List.mem_finRange k))) k₀ rfl opening rfl rfl tau p
    length

variable {B} in
/-- `supported` of `batchFoldRun_runs` and `batchOffloadRun_runs` in the form of
`foldRunIn_batch_runs` and `offloadRunIn_runs`, for the groups of `packGroupsIn`. -/
theorem batchSupport_in {E : Type} [Field E] [Algebra Zq E] [Degree2 E] (F : Fold) {P : PackShape}
    (slots : P.slots = (List.finRange n).map fun k => ⟨(groupOrder n k).val,
      B.shape (groupOrder n k), B.opening (groupOrder n k), B.fold (groupOrder n k)⟩)
    (opening : P.path = ⟨B.openDegree, B.openRank, B.chain⟩) {digits : ℕ}
    (digited : P.digits = digits) {μ : ℕ} (room : B.quotientRoom digits ≤ 2 ^ μ)
    (virtualLength : (recordedIn s ⟨level, "range.leaf.point", none⟩).size = μ) :
    ∀ p : List Ext, p.length = μ → B.compressed = true →
      ((foldSupport (Placement.of (packGroupsIn P s level F) P.path P.digits)
        (CellEq.half (recordedIn s ⟨level, "range.leaf.point", none⟩).toList).get
          (CellEq.half p).get).into : E) = mle (p.map Ext.into) fun x =>
        if (B.gaps digits).support x then
          eqAt ((recordedIn s ⟨level, "range.leaf.point", none⟩).toList.map Ext.into) x
        else 0 := by
  rw [packGroupsIn_batch B s level P F slots]
  exact batchSupport_groups B _ _ _ _ (packInputsIn P #[] s level F) (packingValuesIn s level F)
    room digited opening virtualLength

variable {B} in
/-- `weighs` of `batchOffloadRun_runs` in the form of `offloadRunIn_runs` (the root of
`key30.aks`): at each point `r_2` of `μ` coordinates, the weight of `packFinish` with the claim `σ`
of `stage3.setup_claim` is `part + scale·σ` for each split `scan` whose `part` is `m_local`, the
weight for `batchStatementFor` with the zero setup, and whose `scale` is
`powerScale(α, r_2[..low])`, the fields of `Batch.codeScan` with the block `2^low`. -/
theorem batchOffload_groups {E : Type} [Field E] [Fintype E] [Algebra Zq E] [Degree2 E]
    (pub : Public) (F : Fold) {P : PackShape}
    (slots : P.slots = (List.finRange n).map fun k => ⟨(groupOrder n k).val,
      B.shape (groupOrder n k), B.opening (groupOrder n k), B.fold (groupOrder n k)⟩)
    (opening : P.path = ⟨B.openDegree, B.openRank, B.chain⟩) {digits : ℕ}
    (digited : P.digits = digits) (setup : Array Fq) (source : Fin n → RhsSource) {base : ℕ}
    (based : 2 ^ F.logBasis = base) (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : B.Challenge)
    (challenged : ∀ k i, challenge (groupOrder n k) i = Poly.toRq _
      ((drawnCarrier B s level k).getD i (Poly.zero (B.carrierDegree (groupOrder n k)))))
    (ambiented : ∀ k (i : Fin (B.shape (groupOrder n k)).blocks),
      Poly.toRq (B.shape (groupOrder n k)).innerDegree
        ((drawnAmbient B s level k).getD i.val (Poly.zero (B.shape (groupOrder n k)).innerDegree)) =
        embed (B.opening (groupOrder n k)).stride (B.opening (groupOrder n k)).stride_tower
          (challenge (groupOrder n k) i))
    {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) (room : B.quotientRoom digits ≤ 2 ^ μ) {batch : ℕ}
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    (numbers : ∀ g ∈ packGroupsIn P s level F, g.Fits μ) (k₀ : Fin n)
    (tau : (recordedIn s ⟨level, "tau1", none⟩).size = batch) (lowFits : 2 ^ F.low ≤ B.openDegree)
    (lowShort : F.low ≤ μ) {R : Akita.Reduction E} {ν : ℕ} {flat : Table}
    (scan : SetupScan R ν flat)
    (parted : ∀ α τ r2, scan.part α τ r2 = mle r2 (((Batch.liftedReduction β
      ((batchStatementFor B setup (claimPointIn s level) (dataIn pub s level) source
        (rowsIn s level F) (claimsIn s level F)).over (0 : Fin (2 ^ ν) → Zq)) challenge μ fits
          base digits room batch rowsFit).rows α).batchWeight τ))
    (scaled : ∀ α τ r2, scan.scale α τ r2 = powerScale α (r2.take F.low)) :
    ∀ p : List Ext, p.length = μ →
      ((packFinish (packInputsIn P setup s level F) (packGroupsIn P s level F)
        (packingValuesIn s level F) (some (firstIn s ⟨level, "stage3.setup_claim", none⟩))
          p).weight.into : E) =
      scan.part (firstIn s ⟨level, "alpha", none⟩).into
          ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (p.map Ext.into) +
        scan.scale (firstIn s ⟨level, "alpha", none⟩).into
          ((recordedIn s ⟨level, "tau1", none⟩).toList.map Ext.into) (p.map Ext.into) *
          (firstIn s ⟨level, "stage3.setup_claim", none⟩).into := by
  intro p length
  rw [parted, scaled]
  subst digited based
  rw [packGroupsIn_batch B s level P F slots] at numbers ⊢
  exact foldWeight_offload B _ _ _ _ β basis (packInputsIn P setup s level F)
    (packingValuesIn s level F) _ (fun g => Key.ofSetup_shared _ _) rfl
    (PathKey.ofSetup_shared _ _) (fun _ _ => rfl) (fun _ _ => rfl) (fun _ _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl) challenge challenged ambiented fits room rowsFit
    (fun k => numbers _ (List.mem_map_of_mem (List.mem_finRange k))) k₀ rfl opening rfl rfl tau
    lowFits lowShort (2 ^ ν) _ p length

end Drawn


section Ambient

open Replayer

variable (s : Replayer) (level : ℕ)

variable {n : ℕ} (B : Batch (Fin n)) in
/-- `ambiented` of `batchWeight_groups` and `batchOffload_groups`: when the recorded challenges of
the group at the index `k` are the sampler list of a fold group of the ring `d_A` and the degree
`d_car` moved by `ambient` (`drawn_sample` of the draw), the challenges of `drawnAmbient` are `ι` of
the challenges of `drawnCarrier`. -/
theorem batchAmbiented (G : Fin n → FoldGroup)
    (ringed : ∀ k, (G k).ring = (B.shape (groupOrder n k)).innerDegree)
    (degreed : ∀ k, (G k).degree = B.carrierDegree (groupOrder n k))
    (drawnAt : ∀ k, ∃ cs : List Sparse,
      drawnIn s level (groupOrder n k).val = cs.map (ambient (G k)))
    (challenge : B.Challenge)
    (challenged : ∀ k i, challenge (groupOrder n k) i = Poly.toRq _
      ((drawnCarrier B s level k).getD i (Poly.zero (B.carrierDegree (groupOrder n k))))) :
    ∀ k (i : Fin (B.shape (groupOrder n k)).blocks),
      Poly.toRq (B.shape (groupOrder n k)).innerDegree
        ((drawnAmbient B s level k).getD i.val (Poly.zero (B.shape (groupOrder n k)).innerDegree)) =
        embed (B.opening (groupOrder n k)).stride (B.opening (groupOrder n k)).stride_tower
          (challenge (groupOrder n k) i) := by
  intro k i
  obtain ⟨cs, drawn⟩ := drawnAt k
  rw [challenged, drawnAmbient, drawnCarrier, drawn]
  exact drawn_getD (G k) cs (B.opening (groupOrder n k)).stride_tower (ringed k) (degreed k) i

variable {S : Shape} (O : OpeningShape S) in
/-- `ambiented` of `packWeight_one` for the challenges of the group of `packGroupsIn` at a packing
level with one group. -/
theorem packAmbiented (G : FoldGroup) (ringed : G.ring = S.innerDegree)
    (degreed : G.degree = O.carrierDegree)
    (drawnAt : ∃ cs : List Sparse, drawnIn s level 0 = cs.map (ambient G))
    (challenge : Fin S.blocks → Rq O.carrierDegree)
    (challenged : ∀ i, challenge i = Poly.toRq _ (((drawnIn s level 0).map fun c =>
      Poly.ofSparse O.carrierDegree (c.map fun x => (x.1 / (S.innerDegree / O.carrierDegree), x.2))).getD
        i (Poly.zero O.carrierDegree))) :
    ∀ i : Fin S.blocks, Poly.toRq S.innerDegree
      (((drawnIn s level 0).map (Poly.ofSparse S.innerDegree)).getD i.val
        (Poly.zero S.innerDegree)) = embed O.stride O.stride_tower (challenge i) := by
  intro i
  obtain ⟨cs, drawn⟩ := drawnAt
  rw [challenged, drawn]
  exact drawn_getD G cs O.stride_tower ringed degreed i

end Ambient

section Scan

variable {E : Type} [Field E] [Fintype E] [Algebra Zq E] {n : ℕ} {B : Batch (Fin n)}

/-- A split moved by an equation of statements (`Statement.over_eq` in `Tower.scan`) keeps its
fields `part`, `scale` and `weight`. -/
theorem codeScan_cast (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) {st₁ st₂ : B.Statement E}
    (same : st₁ = st₂) (challenge : B.Challenge) {μ : ℕ} (fits : B.cells ≤ 2 ^ μ) (base digits : ℕ)
    (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ) (rowsFit : B.rowCount + 1 ≤ 2 ^ batch)
    {ν : ℕ} {flat : Table}
    (scan : SetupScan (Batch.liftedReduction β st₁ challenge μ fits base digits room batch rowsFit)
      ν flat) :
    (same ▸ scan : SetupScan (Batch.liftedReduction β st₂ challenge μ fits base digits room batch
        rowsFit) ν flat).part = scan.part ∧
      (same ▸ scan : SetupScan (Batch.liftedReduction β st₂ challenge μ fits base digits room batch
        rowsFit) ν flat).scale = scan.scale ∧
      (same ▸ scan : SetupScan (Batch.liftedReduction β st₂ challenge μ fits base digits room batch
        rowsFit) ν flat).weight = scan.weight := by
  subst same
  exact ⟨rfl, rfl, rfl⟩

end Scan

end NightstreamFPrime.Spec.Compression.Akita.Replay
