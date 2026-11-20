import NightstreamFPrime.Spec.Compression.Akita.Fold

/-! One fold of Akita with direct coefficient packing for a family of opening groups, with one
claim in each group (ePrint 2026/1983, revision of 2026-10-15, Sections 5.1 and 5.4, Figure 5,
Definition 10.22, Lemma 10.23 and Corollary 10.32).

The groups `g : G` have their own commitment shape, challenge ring, opening point and response.
All partial opening evaluations share one unsliced `D/H` path over one ring `R_{q,d_D}`
(Akita (69), (70)). The rows (102), (104) and (105) are local to a group. The row (106) and the
scalar row (100) with the weights `ϑ_g` read each group. The paper has `n_clm,g` claims in a
group; here `n_clm,g = 1`.

- `Batch` has the shapes of the groups and of the shared path, and `Batch.opening g` is the
  `OpeningShape` of the group `g` with the shared path.
- `Batch.Statement`, `Batch.Carried` and `Batch.Witness` have the public data and the digit
  vectors (75) of the fold. `Batch.Witness.Satisfies` is the exact branch relation: the
  group-local relation `Witness.Local` of each group, the row (106) with the rows (47) or (48) of
  the `H` path, and the scalar row (100).
- `Batch.Weak` and `Batch.Weak.Opens` are the weak opening of Akita Definition 10.22 for the
  groups.
- `Batch.weakCollision_isSome` is Akita Lemma 10.23 for the groups. The solution for `D` has the
  bound `max_g (b⋆_g − 1)`.
- `Batch.extract_opens_or_collision` is Akita Corollary 10.32 for the groups: two children for
  each fold coordinate `(g, i)`. It uses `extract_local` for each group and one comparison of the
  shared path.
- `Batch.single` is one group as a family of groups, and `Batch.single_satisfies_iff` states that
  its exact branch relation with the weight 1 is `Witness.Satisfies` of one claim. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

/-- The shapes of one fold with the opening groups `g : G` (Akita Section 5.1 and (53)). The group
`g` has the commitment shape `shape g`, the degree `d_car,g` of its challenge subring, the degree
`k` of the extension field and the factor `h_car,g` with `d_A,g = k·h_car,g·d_car,g` (Akita (55)),
and the response numbers `fold g`. The opening matrix `D` and its compression maps are shared:
`d_D` divides each `k·d_car,g` (Akita Section 5.4, after (68)). -/
structure Batch (G : Type) where
  shape : G → Shape
  carrierDegree : G → ℕ
  [carrierPositive : ∀ g, NeZero (carrierDegree g)]
  planes : G → ℕ
  packing : G → ℕ
  tower : ∀ g, planes g * packing g * carrierDegree g = (shape g).innerDegree
  fold : G → Descriptor
  openDegree : ℕ
  [openPositive : NeZero openDegree]
  openDivides : ∀ g, openDegree ∣ planes g * carrierDegree g
  openRank : ℕ
  chain : Option ChainShape

attribute [instance] Batch.carrierPositive Batch.openPositive

namespace Batch

variable {G : Type} (B : Batch G)

/-- The opening shape of the group `g`: its own numbers with the shared `D/H` path. -/
@[reducible] def opening (g : G) : OpeningShape (B.shape g) where
  carrierDegree := B.carrierDegree g
  planes := B.planes g
  packing := B.packing g
  tower := B.tower g
  openDegree := B.openDegree
  openDivides := B.openDivides g
  openRank := B.openRank
  chain := B.chain

/-- The fold coordinates: one for each group and source block (Akita Section 5.4, "the verifier
samples one challenge `c_{g,c,i}` for every group, local claim, and source block"). -/
abbrev Block : Type := Σ g, Fin (B.shape g).blocks

/-- The positions of the input of `D`: the positions of the groups in their order, with no
padding (Akita (69), (70)). -/
abbrev OpenColumn : Type := Σ g, (B.opening g).OpenColumn

/-- The digit vector `ê = concat(ê_0, …, ê_{G−1})` in the view of `D`. -/
abbrev Partials : Type := B.OpenColumn → Rq B.openDegree

/-- The payload `v_pub` of the shared opening path. -/
abbrev Payload : Type := Akita.Payload B.openDegree (1 * B.openRank) B.chain

/-- The fold challenges: one element of the challenge ring of the group for each source block. -/
abbrev Challenge : Type := (g : G) → Fin (B.shape g).blocks → Rq (B.carrierDegree g)

/-- The matrices of the shared opening path: `D ∈ R_{q,d_D}^{n_D × m_D}` with `m_D = Σ_g m_{D,g}`,
and the compression maps `H_1`, `H_2` (Akita (70)). -/
structure OpenKey where
  matrix : Matrix (Fin B.openRank) B.OpenColumn (Rq B.openDegree)
  chain : PathKey B.chain

/-- The public data of one fold for the groups (Akita Definition 10.22, "The fixed folding
statement", and (172)): for each group the matrices of its commitment, its payload `u_pub,g` and
the weights of its opening point, and for the fold the matrices and the payload `v_pub` of the
shared path, the weights `ϑ_g` of the scalar claims and the target `v_root` of Akita (72). -/
structure Statement (E : Type) where
  key : (g : G) → Key (B.shape g)
  openKey : B.OpenKey
  payload : (g : G) → (B.shape g).Payload
  openPayload : B.Payload
  point : (g : G) → Point (B.opening g) E
  weight : G → E
  value : E

/-- The digit vectors of one fold that do not depend on the fold challenge: for each group the
digits `t̂_g` and the digits of its `F` maps, and for the fold the digits `ê` and the digits of
the `H` maps (Akita (75), (76)). -/
@[ext] structure Carried where
  inner : (g : G) → Fin (B.shape g).blocks → (B.shape g).InnerDigits
  chain : (g : G) → PathDigits (B.shape g).chain
  partials : B.Partials
  openChain : PathDigits B.chain

/-- The digit vectors of one fold: the carried digits and the response digits `ẑ_g` of each
group (Akita (74), (75)). -/
structure Witness extends B.Carried where
  response : (g : G) → (B.shape g).Response (B.fold g)

/-- The parts of a weak opening of one fold (Akita Definition 10.22): a certified decommitment
for each group, the digits `ê` of the partials and the digits of the `H` maps. -/
structure Weak where
  certified : (g : G) → Certified (B.shape g)
  partials : B.Partials
  openChain : PathDigits B.chain

variable {B} {E : Type}

/-- The group-local part of the statement for the group `g`. -/
@[reducible] def Statement.group (st : B.Statement E) (g : G) :
    GroupStatement (B.opening g) E :=
  ⟨st.key g, st.payload g, st.point g⟩

/-- The carried digits of the group `g`: its own digits, its interval `ê_g` of `ê`, and the
digits of the shared `H` maps. -/
def Carried.group (digits : B.Carried) (g : G) : Akita.Carried (B.opening g) :=
  ⟨digits.inner g, digits.chain g, fun column => digits.partials ⟨g, column⟩, digits.openChain⟩

/-- The digit vectors of the group `g`. -/
def Witness.group (w : B.Witness) (g : G) : Akita.Witness (B.opening g) (B.fold g) :=
  ⟨w.toCarried.group g, w.response g⟩

/-- The weak opening of the group `g`. -/
def Weak.group (ω : B.Weak) (g : G) : Akita.Weak (B.opening g) where
  toCertified := ω.certified g
  partials column := ω.partials ⟨g, column⟩
  openChain := ω.openChain

/-- The carried digits of a weak opening. -/
def Weak.carried (ω : B.Weak) : B.Carried :=
  ⟨fun g => (ω.certified g).inner, fun g => (ω.certified g).chain, ω.partials, ω.openChain⟩

variable [Fintype G]

variable (B) in
/-- The row (106) of Akita, `D·concat(ê_0, …, ê_{G−1}) = v`, with the rows (47) from `v` to the
payload `v_pub`, or `D ê = v_pub` (48). The map is unsliced. -/
def OpenRows (key : B.OpenKey) (digits : B.Partials) (chain : PathDigits B.chain)
    (payload : B.Payload) : Prop :=
  Rows B.chain key.chain (stack key.matrix fun _ : Fin 1 => digits) chain payload

variable (B) in
/-- The bound of a difference of two digit vectors `ê` in their alphabets: the largest
`b⋆_g − 1`. -/
def openBound : ℕ := Finset.univ.sup fun g => (B.shape g).certifiedBase - 1

variable (B) in
/-- A solution of Module-SIS for `D` with the bound `max_g (b⋆_g − 1)`, or for `H_1` or `H_2`
with the bound 1 (Akita Table 3). -/
abbrev OpenBreak (key : B.OpenKey) : Type :=
  ShortKernel key.matrix B.openBound ⊕ PathKernel B.chain key.chain

variable (B) in
/-- The reduction of Akita Lemma 4.3 for the shared path of `D`, `H_1` and `H_2`. -/
noncomputable def openCollision (key : B.OpenKey) (partials partials' : B.Partials)
    (chain chain' : PathDigits B.chain) : Option (B.OpenBreak key) :=
  chainCollision B.chain key.chain key.matrix B.openBound (fun _ : Fin 1 => partials)
    (fun _ => partials') chain chain'

theorem openCollision_isSome {key : B.OpenKey} {partials partials' : B.Partials}
    {chain chain' : PathDigits B.chain} {payload : B.Payload}
    (rows : B.OpenRows key partials chain payload)
    (rows' : B.OpenRows key partials' chain' payload)
    (digits : ∀ g, Balanced (B.shape g).certifiedBase fun column :
      (B.opening g).OpenColumn => partials ⟨g, column⟩)
    (digits' : ∀ g, Balanced (B.shape g).certifiedBase fun column :
      (B.opening g).OpenColumn => partials' ⟨g, column⟩)
    (binary : PathDigits.Binary B.chain chain) (binary' : PathDigits.Binary B.chain chain')
    (different : partials ≠ partials' ∨ chain ≠ chain') :
    (B.openCollision key partials partials' chain chain').isSome := by
  refine chainCollision_isSome rows rows' (fun _ => ?_) binary binary'
    (different.imp_left fun differs same => differs (congrFun same 0))
  refine (vecNormInf_le_iff _ _).mpr fun column => ?_
  obtain ⟨g, column⟩ := column
  have within : vecNormInf ((fun column : (B.opening g).OpenColumn => partials ⟨g, column⟩) -
      fun column => partials' ⟨g, column⟩) ≤ (B.shape g).certifiedBase - 1 :=
    (vecNormInf_sub_le_of_digits (digits g) (digits' g)).trans (by omega)
  exact ((normInf_le_vecNormInf _ column).trans within).trans
    (Finset.le_sup (f := fun g => (B.shape g).certifiedBase - 1) (Finset.mem_univ g))

variable [CommRing E] [Module Zq E]

variable (B) in
/-- The left side of the scalar row (100) of Akita: the sum of the scalar openings (59) of the
groups with the weights `ϑ_g`. Each summand uses its own point and challenge subring. -/
noncomputable def scalar (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (st : B.Statement E)
    (partials : B.Partials) : E :=
  ∑ g, st.weight g * Akita.scalar (β g) (st.point g)
    ((B.opening g).partialPlane fun column => partials ⟨g, column⟩)

/-- The rows of Akita Figure 5 that have no fold challenge: the group-local rows of each group,
the shared row (106) with the rows of the `H` path and their alphabet `{−1, 0}`, and the scalar
row (100). -/
structure Carried.Opens (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (digits : B.Carried)
    (st : B.Statement E) : Prop where
  groups : ∀ g, (digits.group g).Local (st.group g)
  openRows : B.OpenRows st.openKey digits.partials digits.openChain st.openPayload
  openDigits : PathDigits.Binary B.chain digits.openChain
  value : B.scalar β st digits.partials = st.value

/-- The exact branch relation of one fold for the groups and the fold challenges `c_{g,i}`
(Akita Section 10, before Corollary 10.32, and Figure 5): the rows (102), (104), (105) of each
group with its alphabets, the shared row (106) with the rows of the `H` path, and the scalar
row (100). -/
structure Witness.Satisfies (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E) (w : B.Witness)
    (st : B.Statement E) (challenge : B.Challenge) : Prop where
  groups : ∀ g, (w.group g).Local (β g) (st.group g) (challenge g)
  openRows : B.OpenRows st.openKey w.partials w.openChain st.openPayload
  openDigits : PathDigits.Binary B.chain w.openChain
  value : B.scalar β st w.partials = st.value

/-- Akita Definition 10.22 for the groups on the coefficient-norm route: `ω` is a weak opening of
the statement, and the source blocks of the group `g` decode to `table g`. `groups` is the
conditions (i), (ii), the partials of (iv) and the alphabet of `ê_g` for each group, with the
bounds `κ̄_1` and `β̄_∞` of the group. `openRows` is (iii) for the shared path, `value` is the
scalar row (100), and `openDigits` is the part of (v) for the `H` maps. -/
structure Weak.Opens (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (factorBound sourceBound : G → ℕ) (ω : B.Weak) (st : B.Statement E) (table : G → Table) :
    Prop where
  groups : ∀ g, (ω.group g).Local (β g) (factorBound g) (sourceBound g) (st.group g) (table g)
  openRows : B.OpenRows st.openKey ω.partials ω.openChain st.openPayload
  openDigits : PathDigits.Binary B.chain ω.openChain
  value : B.scalar β st ω.partials = st.value

variable {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {st : B.Statement E}

theorem Witness.Satisfies.carried {w : B.Witness} {challenge : B.Challenge}
    (satisfies : w.Satisfies β st challenge) : w.toCarried.Opens β st :=
  ⟨fun g => (satisfies.groups g).carried, satisfies.openRows, satisfies.openDigits,
    satisfies.value⟩

theorem Weak.Opens.carried {ω : B.Weak} {factorBound sourceBound : G → ℕ} {table : G → Table}
    (opens : ω.Opens β factorBound sourceBound st table) : ω.carried.Opens β st :=
  ⟨fun g => ⟨(opens.groups g).certified.source.rows, (opens.groups g).certified.source.innerDigits,
    (opens.groups g).certified.source.chainDigits, (opens.groups g).partialDigits⟩,
    opens.openRows, opens.openDigits, opens.value⟩

section Binding

variable (B) in
/-- A solution of Module-SIS for a digit path of one fold (Akita Table 3): for `B_g` with the
bound `b⋆_g − 1` or a map `F_a` of a group, or for `D` or a map `H_a`. -/
abbrev DigitBreak (st : B.Statement E) : Type :=
  (Σ g, OuterBreak (st.key g)) ⊕ B.OpenBreak st.openKey

variable (B) in
/-- The reduction of Akita Lemma 10.23, "Digit paths", for the groups: a solution for the path of
`B_g` of the first group whose digits differ, or else for the shared path of `D`. -/
noncomputable def carriedCollision (st : B.Statement E) (digits digits' : B.Carried) :
    Option (B.DigitBreak st) :=
  ((Finset.univ.toList.findSome? fun g =>
      ((outerCollision (st.key g) (digits.inner g) (digits'.inner g) (digits.chain g)
        (digits'.chain g)).map (Sigma.mk g) : Option (Σ g, OuterBreak (st.key g)))).map Sum.inl).or
    ((B.openCollision st.openKey digits.partials digits'.partials digits.openChain
      digits'.openChain).map Sum.inr)

/-- Akita Lemma 10.23, "Digit paths", for the groups: two families of carried digits that satisfy
the rows of one statement and that differ give a solution of Module-SIS for a matrix `B_g` or for
`D`, or for a compression map with the bound 1. The shared path is compared once. -/
theorem carriedCollision_isSome {digits digits' : B.Carried} (opens : digits.Opens β st)
    (opens' : digits'.Opens β st) (different : digits ≠ digits') :
    (B.carriedCollision st digits digits').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases outer : ∃ g, digits.inner g ≠ digits'.inner g ∨ digits.chain g ≠ digits'.chain g
  · obtain ⟨g, differs⟩ := outer
    refine Or.inl (List.findSome?_isSome_iff.mpr
      ⟨g, Finset.mem_toList.mpr (Finset.mem_univ g), ?_⟩)
    rw [Option.isSome_map]
    exact outerCollision_isSome (opens.groups g).rows (opens'.groups g).rows
      (opens.groups g).innerDigits (opens'.groups g).innerDigits (opens.groups g).chainDigits
      (opens'.groups g).chainDigits differs
  · refine Or.inr (openCollision_isSome opens.openRows opens'.openRows
      (fun g => (opens.groups g).partialDigits) (fun g => (opens'.groups g).partialDigits)
      opens.openDigits opens'.openDigits ?_)
    by_contra same
    obtain ⟨partials, openChain⟩ := not_or.mp same
    refine different (Carried.ext (funext fun g => ?_) (funext fun g => ?_) (not_not.mp partials)
      (not_not.mp openChain))
    · exact not_not.mp fun differs => outer ⟨g, Or.inl differs⟩
    · exact not_not.mp fun differs => outer ⟨g, Or.inr differs⟩

variable (B) in
/-- A solution of Module-SIS for one matrix of a fold with groups (Akita Table 3): for a matrix
`A_g` with the bound `innerBound g`, or for a digit path. -/
abbrev FoldBreak (st : B.Statement E) (innerBound : G → ℕ) : Type :=
  (Σ g, ShortKernel (st.key g).inner (innerBound g)) ⊕ B.DigitBreak st

variable (B) in
/-- The reduction of Akita Lemma 10.23 for two weak openings of the groups: the solution of the
digit paths, or else the solution `z_A` for the matrix `A_g` of the first group whose source
blocks differ. -/
noncomputable def weakCollision (st : B.Statement E) (innerBound : G → ℕ) (ω ω' : B.Weak) :
    Option (B.FoldBreak st innerBound) :=
  ((B.carriedCollision st ω.carried ω'.carried).map Sum.inr).or
    ((Finset.univ.toList.findSome? fun g =>
      ((sourceCollision (st.key g) (innerBound g) (ω.certified g) (ω'.certified g)).map
        (Sigma.mk g) : Option (Σ g, ShortKernel (st.key g).inner (innerBound g)))).map Sum.inl)

/-- Akita Lemma 10.23 (weak binding of one fold) for the groups, with the bound (174) for each
group: two weak openings of one statement that differ on a source block of a group or on a
carried digit vector give a solution of Module-SIS: for `A_g` with each bound
`η_{A,g} ≥ κ̄'_{1,g}·β̄_{∞,g} + κ̄_{1,g}·β̄'_{∞,g}`, for `B_g` with the bound `b⋆_g − 1`, for `D`
with the bound `max_g (b⋆_g − 1)`, or for a compression map with the bound 1. -/
theorem weakCollision_isSome {ω ω' : B.Weak} {factorBound sourceBound factorBound' sourceBound' :
    G → ℕ} {table table' : G → Table} {innerBound : G → ℕ}
    (opens : ω.Opens β factorBound sourceBound st table)
    (opens' : ω'.Opens β factorBound' sourceBound' st table')
    (covers : ∀ g, factorBound' g * sourceBound g + factorBound g * sourceBound' g ≤ innerBound g)
    (different : (∃ g, (ω.certified g).source ≠ (ω'.certified g).source) ∨
      ω.carried ≠ ω'.carried) :
    (B.weakCollision st innerBound ω ω').isSome := by
  refine isSome_or_map.mpr ?_
  by_cases digits : ω.carried = ω'.carried
  · obtain ⟨g, differs⟩ := different.resolve_right (not_not.mpr digits)
    refine Or.inr (List.findSome?_isSome_iff.mpr
      ⟨g, Finset.mem_toList.mpr (Finset.mem_univ g), ?_⟩)
    rw [Option.isSome_map]
    exact sourceCollision_isSome (opens.groups g).certified (opens'.groups g).certified
      (covers g) (congrFun (congrArg Carried.inner digits) g) differs
  · exact Or.inl (carriedCollision_isSome opens.carried opens'.carried digits)

end Binding

section Extraction

variable (B) in
/-- One accepting extension of a transcript prefix through the fold challenge (a child of Akita
Definition 10.30): its challenges `c_{g,i}` and the digit vectors of its branch relation. -/
structure Child where
  challenge : B.Challenge
  witness : B.Witness

variable (B) in
/-- Two children for each fold coordinate `(g, i)` (Akita Definition 10.30 (ii), with `Q` the set
of the pairs of a group and a source block). -/
structure Fork where
  first : B.Block → B.Child
  second : B.Block → B.Child

/-- The children of the fold coordinates of the group `g`, with their challenges and digit vectors
for this group. -/
def Fork.group (fork : B.Fork) (g : G) : Akita.Fork (B.opening g) (B.fold g) :=
  ⟨fun i => ⟨(fork.first ⟨g, i⟩).challenge g, (fork.first ⟨g, i⟩).witness.group g⟩,
    fun i => ⟨(fork.second ⟨g, i⟩).challenge g, (fork.second ⟨g, i⟩).witness.group g⟩⟩

/-- The coordinate-wise structure of Akita Definition 10.30 (ii) with the hypothesis of Akita
Corollary 10.32, for the groups: the two children of the coordinate `(g, i)` have challenges of
the family of `g` at `(g, i)` that differ, their challenges agree at each other coordinate of each
group, and the digit vectors of each child satisfy its exact branch relation for the statement. -/
structure Fork.Accepts (β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E)
    (family : (g : G) → Set (Rq (B.carrierDegree g))) (fork : B.Fork) (st : B.Statement E) :
    Prop where
  firstMember : ∀ g i, (fork.first ⟨g, i⟩).challenge g i ∈ family g
  secondMember : ∀ g i, (fork.second ⟨g, i⟩).challenge g i ∈ family g
  different : ∀ g i, (fork.first ⟨g, i⟩).challenge g i ≠ (fork.second ⟨g, i⟩).challenge g i
  agree : ∀ (b : B.Block) g j, (⟨g, j⟩ : B.Block) ≠ b →
    (fork.first b).challenge g j = (fork.second b).challenge g j
  firstSatisfies : ∀ b, (fork.first b).witness.Satisfies β st (fork.first b).challenge
  secondSatisfies : ∀ b, (fork.second b).witness.Satisfies β st (fork.second b).challenge

variable {family : (g : G) → Set (Rq (B.carrierDegree g))} {fork : B.Fork} {digits : B.Carried}

/-- The children of the coordinates of one group satisfy the group-local conditions `Fork.Local`
of this group: the rows (102) and (104) of a group read the challenges of this group only. -/
theorem Fork.Accepts.group (accepts : fork.Accepts β family st) (g : G) :
    Akita.Fork.Local (O := B.opening g) (β g) (family g) (fork.group g) (st.group g) where
  firstMember := accepts.firstMember g
  secondMember := accepts.secondMember g
  different := accepts.different g
  agree i j other := accepts.agree ⟨g, i⟩ g j fun same =>
    other (eq_of_heq (Sigma.mk.inj same).2)
  firstLocal i := (accepts.firstSatisfies ⟨g, i⟩).groups g
  secondLocal i := (accepts.secondSatisfies ⟨g, i⟩).groups g

omit [Fintype G] [CommRing E] [Module Zq E] in
/-- The weak opening of the proof of Akita Theorem 10.31 for the groups: for each group the
certified decommitment of `extract` for its children, with the digits `ê` and the digits of the
`H` maps of `digits`. -/
noncomputable def extract (fork : B.Fork) (digits : B.Carried) : B.Weak where
  certified g := (Akita.extract (fork.group g) (digits.group g)).toCertified
  partials := digits.partials
  openChain := digits.openChain

variable (B) in
/-- The reduction of Akita Corollary 10.32, "Digit paths", for the groups: the solution of
`carriedCollision` for the first child whose carried digits are not `digits`. -/
noncomputable def forkCollision (st : B.Statement E) (fork : B.Fork) (digits : B.Carried) :
    Option (B.DigitBreak st) :=
  Finset.univ.toList.findSome? fun b : B.Block =>
    (B.carriedCollision st (fork.first b).witness.toCarried digits).or
      (B.carriedCollision st (fork.second b).witness.toCarried digits)

/-- Akita Corollary 10.32 for a fold with opening groups, coefficient packing and one claim in
each group, on the coefficient-norm route. `fork` has two children for each fold coordinate
`(g, i)` that satisfy their exact branch relations for one statement, with challenges of the
family of the group, which has unit differences and `‖c‖_1 = ω_g`. `digits` is a family of carried
digits that satisfies the rows of the statement, for example that of a child. Then a child with
other carried digits gives a solution of Module-SIS for a matrix `B_g` or `D` or for a compression
map, or `extract fork digits` is a weak opening with `κ̄_{1,g} = 2ω_g` and
`β̄_{∞,g} = Δ^cert_{f,g}` for each group. The scalar row (100) with its weights is that of
`digits`: no weight is inverted (Akita Proposition 10.44, proof). -/
theorem extract_opens_or_collision {mass peak : G → ℕ}
    (units : ∀ g, UnitDifferences (family g))
    (sparse : ∀ g, ∀ c ∈ family g, Sparse (mass g) (peak g) c)
    (base : ∀ g, 2 ≤ (B.fold g).responseBase) (accepts : fork.Accepts β family st)
    (opens : digits.Opens β st) :
    (B.forkCollision st fork digits).isSome ∨
      (extract fork digits).Opens β (fun g => 2 * mass g)
        (fun g => (B.fold g).sourceBound (B.shape g)) st
        (fun g => (B.shape g).decode ((extract fork digits).certified g).source) := by
  by_cases same : (∀ b, (fork.first b).witness.toCarried = digits) ∧
      ∀ b, (fork.second b).witness.toCarried = digits
  · exact Or.inr
      ⟨fun g => extract_local (units g) (sparse g) (base g) (accepts.group g) (opens.groups g)
        (fun i => congrArg (Carried.group · g) (same.1 ⟨g, i⟩))
        (fun i => congrArg (Carried.group · g) (same.2 ⟨g, i⟩)),
      opens.openRows, opens.openDigits, opens.value⟩
  · have other : ∃ b, (fork.first b).witness.toCarried ≠ digits ∨
        (fork.second b).witness.toCarried ≠ digits := by
      by_contra none
      exact same ⟨fun b => not_not.mp fun differs => none ⟨b, Or.inl differs⟩,
        fun b => not_not.mp fun differs => none ⟨b, Or.inr differs⟩⟩
    obtain ⟨b, differs⟩ := other
    refine Or.inl (List.findSome?_isSome_iff.mpr
      ⟨b, Finset.mem_toList.mpr (Finset.mem_univ b), ?_⟩)
    rw [Option.isSome_or, Bool.or_eq_true]
    exact differs.imp
      (carriedCollision_isSome (accepts.firstSatisfies b).carried opens)
      (carriedCollision_isSome (accepts.secondSatisfies b).carried opens)

end Extraction

section Single

variable {S : Shape} (O : OpeningShape S) (fold : Descriptor)

/-- One group as a family of groups: the index type has one element (Akita Section 7.2:
"For `G = 1` and one claim, the rows specialize to the local formulas"). -/
def single : Batch Unit where
  shape _ := S
  carrierDegree _ := O.carrierDegree
  carrierPositive _ := O.carrierPositive
  planes _ := O.planes
  packing _ := O.packing
  tower _ := O.tower
  fold _ := fold
  openDegree := O.openDegree
  openPositive := O.openPositive
  openDivides _ := O.openDivides
  openRank := O.openRank
  chain := O.chain

/-- The folding statement of one claim as a statement of one group with the weight 1 (Akita
Section 5.4: "At a singleton direct edge, the only weight is one"). -/
def Statement.single [One E] (st : Akita.Statement O E) : (single O fold).Statement E where
  key _ := st.key
  openKey := ⟨fun row column => st.openKey.matrix row column.2, st.openKey.chain⟩
  payload _ := st.payload
  openPayload := st.openPayload
  point _ := st.point
  weight _ := 1
  value := st.value

/-- The digit vectors of one fold for one claim as the digit vectors of one group. -/
def Witness.single (w : Akita.Witness O fold) : (single O fold).Witness where
  inner _ := w.inner
  chain _ := w.chain
  partials column := w.partials column.2
  openChain := w.openChain
  response _ := w.response

variable {O fold}

omit [CommRing E] [Module Zq E] in
theorem single_openRows [One E] (st : Akita.Statement O E) (w : Akita.Witness O fold) :
    (single O fold).OpenRows (Statement.single O fold st).openKey (Witness.single O fold w).partials
        w.openChain st.openPayload ↔
      OpeningRows st.openKey w.partials w.openChain st.openPayload := by
  have same : stack (Statement.single O fold st).openKey.matrix
      (fun _ : Fin 1 => (Witness.single O fold w).partials) =
        stack st.openKey.matrix fun _ : Fin 1 => w.partials := by
    funext j
    exact Fintype.sum_equiv
      (⟨fun column => column.2, fun column => ⟨(), column⟩, fun _ => rfl, fun _ => rfl⟩ :
        (single O fold).OpenColumn ≃ O.OpenColumn) _ _ fun _ => rfl
  unfold OpenRows OpeningRows
  rw [same]
  exact Iff.rfl

/-- The exact branch relation of one fold for one claim is the exact branch relation for one
group with the weight 1: the family of groups with one element has the relation
`Witness.Satisfies` of `Fold.lean`. -/
theorem single_satisfies_iff (β : Module.Basis (Fin O.planes) Zq E) (w : Akita.Witness O fold)
    (st : Akita.Statement O E) (challenge : Fin S.blocks → Rq O.carrierDegree) :
    (Witness.single O fold w).Satisfies (B := single O fold) (fun _ => β)
        (Statement.single O fold st) (fun _ => challenge) ↔
      w.Satisfies β st challenge := by
  have value : (single O fold).scalar (fun _ => β) (Statement.single O fold st)
      (Witness.single O fold w).partials = Akita.scalar β st.point (O.partialPlane w.partials) := by
    unfold scalar
    rw [Fintype.sum_unique]
    exact one_mul _
  constructor
  · intro satisfies
    have group := satisfies.groups ()
    exact ⟨⟨group.carried.rows, (single_openRows st w).mp satisfies.openRows,
      value.symm.trans satisfies.value, group.carried.innerDigits, group.carried.chainDigits,
      group.carried.partialDigits, satisfies.openDigits⟩, group.evaluation, group.innerRows,
      group.responseDigits⟩
  · intro satisfies
    exact ⟨fun _ => satisfies.toLocal, (single_openRows st w).mpr satisfies.carried.openRows,
      satisfies.carried.openDigits, value.trans satisfies.carried.value⟩

end Single

end Batch

end NightstreamFPrime.Spec.Compression.Akita
