import NightstreamFPrime.Spec.Compression.Akita.Replay.Readback
import NightstreamFPrime.Spec.Compression.Akita.Key30

/-! The shape of the public statement of each schedule row of `key20.aks` and `key30.aks` for
`Public.shaped` (`Replay/Layout.lean`), from the Lean shapes of the root (`Key20.root`,
`Key30.root`).

- `Profile` has the numbers of the profile header of the canonical bytes of a commitment, and
  `Profile.bytes` is the header in the order of `write_group` (Akita revision `d50f700`,
  `crates/akita-types/src/proof/commitment.rs`): the version, the variables, the polynomials, the
  live ring elements of a claim, the positions of a block, the live blocks, the slices, the base
  and the digits of the source, the matrix `A`, the base and the digits of the opening, and the
  matrix `B`. A matrix is three tags, the table digest, the ring degree, the rank, the width and
  the bound of its Module-SIS instance.
- `Profile.ofGroup` gives each number from the shape and the descriptor of a group. The bound of
  `A` is `η_A` (`Descriptor.innerBound`, Akita (50)) and the bound of `B` is `b⋆ − 1` (Akita (49)).
- These numbers are constants of the code: the version `GroupCommitPhaseParams::VERSION`
  (`crates/akita-params/src/schedule/profiles.rs`), one polynomial in each group
  (`ns-compress`, `akita.rs`, `row`), the tags of `SisModulusProfileId::Q64Offset59`,
  `SisSecurityPolicyId::Quantum169BitADPS16` and `SisMatrixRole`, and the table digest
  `SisTableDigest::CURRENT` (`crates/akita-params/src/sis/ajtai_key/table.rs`,
  `crates/akita-params/src/sis/generated_sis_table/mod.rs`).
- `root20` and `root30 r` are the shapes of the three groups of a row. `profiles20` and
  `profiles30` give each number of their headers. `tests/AkitaAllVectors.lean` compares the
  headers with the commitments of real openings.
- `shaped_groups` and `shaped_group` read `Public.shaped`. `shaped20` and `shaped30` are its
  consequences for a row: the lengths of the points, the count of the values, and the bytes of
  each commitment with the canonical coefficients of its rows.
- `axes20` and `axes30`: the variables of each group are the sum of the bits of the four axes of
  its point in the code (`PreparedSubringCoefficientPackingPoint::new`,
  `crates/akita-types/src/subring_coefficient_packing.rs`), so the code adds no zero coordinate
  to a point of these rows. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- The `count` bytes of `value` with the least byte first. -/
def littleEndian (count value : ℕ) : ByteArray :=
  ByteArray.mk ((Array.range count).map fun i => (value / 2 ^ (8 * i) % 256).toUInt8)

theorem littleEndian_size (count value : ℕ) : (littleEndian count value).size = count := by
  simp [littleEndian, ByteArray.size]

/-- `SisTableDigest::CURRENT`. -/
def tableDigest : ByteArray := ByteArray.mk #[
  0xe5, 0xa6, 0xbf, 0x87, 0x0b, 0x92, 0x09, 0xa9, 0xc5, 0x10, 0xa2, 0x51, 0x0c, 0xcb, 0x75, 0x6e,
  0xe5, 0x47, 0x9d, 0x81, 0x1e, 0x3a, 0xa9, 0xeb, 0x35, 0x5a, 0xcd, 0x42, 0xe9, 0x07, 0x47, 0x79]

theorem tableDigest_size : tableDigest.size = 32 := rfl

/-- The numbers of one matrix in the profile header: the tag of its role, its ring degree, its
rank, its width and the bound of its Module-SIS instance. -/
structure MatrixProfile where
  role : ℕ
  degree : ℕ
  rank : ℕ
  width : ℕ
  bound : ℕ
deriving DecidableEq, Repr

/-- The tags of the modulus and of the policy, the tag of the role, the table digest, and the
degree, the rank, the width and the bound in 4, 8, 8 and 16 bytes. -/
def MatrixProfile.bytes (M : MatrixProfile) : ByteArray :=
  ByteArray.mk #[2, 2, M.role.toUInt8] ++ tableDigest ++ littleEndian 4 M.degree ++
    littleEndian 8 M.rank ++ littleEndian 8 M.width ++ littleEndian 16 M.bound

theorem MatrixProfile.bytes_size (M : MatrixProfile) : M.bytes.size = 71 := by
  simp only [MatrixProfile.bytes, ByteArray.size_append, littleEndian_size, tableDigest_size]
  rfl

/-- The numbers of the profile header of a commitment. -/
structure Profile where
  vars : ℕ
  live : ℕ
  positions : ℕ
  blocks : ℕ
  slices : ℕ
  sourceLog : ℕ
  sourceDigits : ℕ
  inner : MatrixProfile
  openingLog : ℕ
  openingDigits : ℕ
  outer : MatrixProfile
deriving DecidableEq, Repr

namespace Profile

/-- The profile header in the order of `write_group`. -/
def bytes (P : Profile) : ByteArray :=
  ByteArray.mk #[4] ++ littleEndian 8 P.vars ++ littleEndian 8 1 ++ littleEndian 8 P.live ++
    littleEndian 8 P.positions ++ littleEndian 8 P.blocks ++ littleEndian 8 P.slices ++
    littleEndian 4 P.sourceLog ++ littleEndian 8 P.sourceDigits ++ P.inner.bytes ++
    littleEndian 4 P.openingLog ++ littleEndian 8 P.openingDigits ++ P.outer.bytes

theorem bytes_size (P : Profile) : P.bytes.size = 215 := by
  simp only [bytes, ByteArray.size_append, littleEndian_size, MatrixProfile.bytes_size]
  rfl

/-- The numbers of the header of a group with the shape `S` and the opening descriptor `fold`. -/
def ofShape (S : Shape) (fold : Descriptor) : Profile where
  vars := Nat.log 2 (S.blocks * S.blockLength * S.innerDegree)
  live := S.blocks * S.blockLength
  positions := S.blockLength
  blocks := S.blocks
  slices := S.slices
  sourceLog := Nat.log 2 S.sourceBase
  sourceDigits := S.sourceDigits
  inner := ⟨1, S.innerDegree, S.innerRank, S.blockLength * S.sourceDigits, fold.innerBound S⟩
  openingLog := Nat.log 2 S.openingBase
  openingDigits := S.openingDigits
  outer := ⟨2, S.outerDegree, S.outerRank,
    S.innerRank * S.ratio * S.openingDigits * S.sliceLength, S.certifiedBase - 1⟩

end Profile

/-- The count of the coefficients of the payload of a commitment: the image of the last
compression map, or the image of `B` without compression. -/
def payloadCoefficients (S : Shape) : ℕ :=
  match S.chain with
  | some c => c.secondRank * c.secondDegree
  | none => S.slices * S.outerRank * S.outerDegree

/-- The shape of the statement of the group `g` of a root. -/
def rootGroup (B : Batch (Fin 3)) (g : Fin 3) : RootGroup where
  vars := (Profile.ofShape (B.shape g) (B.fold g)).vars
  header := (Profile.ofShape (B.shape g) (B.fold g)).bytes
  rows := payloadCoefficients (B.shape g)

/-- The shapes of the statement of a root: the key polynomial, `W` and `V`. -/
def rootGroups (B : Batch (Fin 3)) : Array RootGroup :=
  #[rootGroup B 0, rootGroup B 1, rootGroup B 2]

/-- The shape of the statement of the row of `key20.aks`. -/
def root20 : Array RootGroup := rootGroups Key20.root

/-- The shape of the statement of a row of `key30.aks`. -/
def root30 (r : Key30.Row) : Array RootGroup := rootGroups (Key30.root r)

/-- The numbers of the three headers of `key20.aks`. -/
theorem profiles20 : ∀ g, Profile.ofShape (Key20.root.shape g) (Key20.root.fold g) =
    ![⟨20, 1024, 32, 32, 4, 10, 7, ⟨1, 1024, 1, 224, 32505732⟩, 3, 22, ⟨2, 128, 1, 1408, 7⟩⟩,
      ⟨16, 128, 8, 16, 8, 10, 7, ⟨1, 512, 2, 56, 53477172⟩, 3, 22, ⟨2, 128, 1, 352, 7⟩⟩,
      ⟨16, 64, 8, 8, 1, 10, 7, ⟨1, 1024, 1, 56, 53477172⟩, 3, 22, ⟨2, 128, 1, 1408, 7⟩⟩] g := by
  decide

/-- The numbers of the three headers of each row of `key30.aks`. -/
theorem profiles30 : ∀ r ∈ Key30.rows, ∀ g,
    Profile.ofShape ((Key30.root r).shape g) ((Key30.root r).fold g) =
    ![⟨30, 524288, 1024, 512, 4, 16, 4, ⟨1, 2048, 1, 4096, 16642998148⟩, 3, 22,
        ⟨2, 256, 1, 22528, 7⟩⟩,
      ⟨r.vars, 256 * r.length, r.length, 256, 8, 16, 4,
        ⟨1, 2048, 1, r.length * 4, 16642998148⟩, 3, 22, ⟨2, 128, 1, 11264, 7⟩⟩,
      ⟨26, 131072, 512, 256, 2, 13, 5, ⟨1, 512, 3, 2560, 2080374660⟩, 3, 22,
        ⟨2, 256, 1, 16896, 7⟩⟩] g := by
  decide

theorem root20_vars (g : Fin 3) : (rootGroup Key20.root g).vars = ![20, 16, 16] g :=
  (congrArg Profile.vars (profiles20 g)).trans (by revert g; decide)

theorem root30_vars {r : Key30.Row} (member : r ∈ Key30.rows) (g : Fin 3) :
    (rootGroup (Key30.root r) g).vars = ![30, r.vars, 26] g :=
  (congrArg Profile.vars (profiles30 r member g)).trans (by fin_cases g <;> rfl)

theorem rootGroup_header_size (B : Batch (Fin 3)) (g : Fin 3) :
    (rootGroup B g).header.size = 215 := Profile.bytes_size _

theorem root20_rows (g : Fin 3) : (rootGroup Key20.root g).rows = 32 := by
  revert g
  decide

theorem root30_rows (r : Key30.Row) (g : Fin 3) : (rootGroup (Key30.root r) g).rows = 32 := by
  fin_cases g <;> rfl

theorem rootGroups_getElem? (B : Batch (Fin 3)) (g : Fin 3) :
    (rootGroups B)[g.val]? = some (rootGroup B g) := by
  fin_cases g <;> rfl

/-- The variables of each group of `key20.aks` are the sum of the bits of the axes of its point
in the code: the stride, the carrier degree, the positions of a block and the blocks. -/
theorem axes20 (g : Fin 3) :
    Nat.log 2 (Key20.root.opening g).stride + Nat.log 2 (Key20.root.carrierDegree g) +
      Nat.log 2 (Key20.root.shape g).blockLength + Nat.log 2 (Key20.root.shape g).blocks =
      (rootGroup Key20.root g).vars := by
  obtain ⟨blocks, length, stride, carrier⟩ := Key20.root_points g
  rw [root20_vars, blocks, length, stride, carrier]
  simp only [Nat.log_pow (by decide : 1 < 2)]
  revert g
  decide

/-- The variables of each group of a row of `key30.aks` are the sum of the bits of the axes of
its point in the code. -/
theorem axes30 {r : Key30.Row} (member : r ∈ Key30.rows) (g : Fin 3) :
    Nat.log 2 ((Key30.root r).opening g).stride + Nat.log 2 ((Key30.root r).carrierDegree g) +
      Nat.log 2 ((Key30.root r).shape g).blockLength +
      Nat.log 2 ((Key30.root r).shape g).blocks = (rootGroup (Key30.root r) g).vars := by
  obtain ⟨blocks, length, stride, carrier, sum⟩ := Key30.root_points r member g
  rw [root30_vars member, blocks, length, stride, carrier, ← sum]
  simp only [Nat.log_pow (by decide : 1 < 2)]
  omega

@[simp] theorem Public.sealed_shaped (pub : Public) : pub.sealed.shaped = pub.shaped := rfl

@[simp] theorem Public.sealed_layout (pub : Public) : pub.sealed.layout = pub.layout := rfl

@[simp] theorem Public.sealed_plan (pub : Public) : pub.sealed.plan = pub.plan := rfl

@[simp] theorem Public.sealed_groups (pub : Public) : pub.sealed.groups = pub.groups := rfl

@[simp] theorem Public.sealed_root (pub : Public) : pub.sealed.root = pub.root := rfl

theorem Public.shaped_word : @Public.shaped.word = @Wire.word := rfl

/-- The checks of `Public.shaped` on one group of the statement: one value, a point of the
length of the variables, and a commitment that is the header and then canonical coefficients. -/
structure Statement.Shaped (g : Statement) (r : RootGroup) : Prop where
  value : g.value.size = 1
  point : g.point.size = r.vars
  size : g.commitment.size = r.header.size + 8 * r.rows
  header : (g.commitment.extract 0 r.header.size).data = r.header.data
  canonical : ∀ j < r.rows, (Wire.word g.commitment (r.header.size + 8 * j)).toNat < fieldModulus

/-- Each shape of a group has a statement: the header and then zero coefficients. -/
theorem Statement.exists_shaped (r : RootGroup) : ∃ s : Statement, s.Shaped r := by
  have size : (zeros (8 * r.rows)).size = 8 * r.rows := by simp [zeros, ByteArray.size]
  refine ⟨⟨r.header ++ zeros (8 * r.rows), Array.replicate r.vars Ext.zero, #[Ext.zero]⟩,
    rfl, Array.size_replicate, by rw [ByteArray.size_append, size], ?_, fun j below => ?_⟩
  · rw [ByteArray.data_extract, ByteArray.data_append, ← ByteArray.size_data]
    simp
  · have eight : (le64 0).data = Array.replicate 8 0 :=
      Array.ext (by simp [le64]) fun i _ _ => by simp [le64]
    have count : 8 * r.rows = 8 * j + (8 + 8 * (r.rows - j - 1)) := by omega
    have split : (r.header ++ zeros (8 * r.rows)).data =
        r.header.data ++ Array.replicate (8 * j) 0 ++ (le64 0).data ++
          Array.replicate (8 * (r.rows - j - 1)) 0 := by
      rw [ByteArray.data_append, eight, Array.append_assoc, Array.append_assoc,
        Array.replicate_append_replicate, Array.replicate_append_replicate, ← count]
      rfl
    have word := word_of_data (by decide : 0 < 2 ^ 64) split
    rw [Array.size_append, Array.size_replicate, ByteArray.size_data] at word
    rw [word]
    decide

/-- `Public.shaped` holds exactly when the statement has the count of the groups of the row and
each group has the checks of `Statement.Shaped`. -/
theorem shaped_groups (pub : Public) :
    pub.shaped = true ↔ pub.groups.size = pub.root.size ∧
      ∀ i (inGroups : i < pub.groups.size) (inRoot : i < pub.root.size),
        pub.groups[i].Shaped pub.root[i] := by
  unfold Public.shaped
  rw [Bool.and_eq_true, decide_eq_true_eq, Array.all_eq_true]
  refine and_congr_right fun _ => ⟨fun all i inGroups inRoot => ?_, fun all i inside => ?_⟩
  · have group := all i (by rw [Array.size_zip, Nat.lt_min]; exact ⟨inGroups, inRoot⟩)
    simp only [Array.getElem_zip, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true,
      List.mem_range] at group
    obtain ⟨⟨⟨⟨value, point⟩, size⟩, header⟩, canonical⟩ := group
    refine ⟨value, point, size, header, fun j below => ?_⟩
    have decoded := canonical j below
    by_contra above
    simp only [Public.shaped_word, Fq.ofWord?, above, ↓reduceIte] at decoded
    exact Bool.false_ne_true decoded
  · rw [Array.size_zip, Nat.lt_min] at inside
    obtain ⟨value, point, size, header, canonical⟩ := all i inside.1 inside.2
    simp only [Array.getElem_zip, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true,
      List.mem_range]
    refine ⟨⟨⟨⟨value, point⟩, size⟩, header⟩, fun j below => ?_⟩
    simp only [Public.shaped_word, Fq.ofWord?, canonical j below, ↓reduceIte]
    rfl

/-- The group `g` of a statement with three groups in the shape of a root. -/
theorem shaped_group {pub : Public} {B : Batch (Fin 3)} (root : pub.root = rootGroups B)
    (shaped : pub.shaped = true) (g : Fin 3) :
    ∃ s, pub.groups[g.val]? = some s ∧ s.Shaped (rootGroup B g) := by
  obtain ⟨count, all⟩ := (shaped_groups pub).mp shaped
  have three : pub.root.size = 3 := by rw [root]; rfl
  have inGroups : g.val < pub.groups.size := by omega
  have inRoot : g.val < pub.root.size := by omega
  have row : pub.root[g.val] = rootGroup B g :=
    Option.some.inj ((Array.getElem?_eq_getElem inRoot).symm.trans
      (by rw [root, rootGroups_getElem?]))
  refine ⟨pub.groups[g.val], Array.getElem?_eq_getElem inGroups, ?_⟩
  rw [← row]
  exact all g.val inGroups inRoot

/-- A statement in the shape of a root has three values. -/
theorem shaped_values {pub : Public} {B : Batch (Fin 3)} (root : pub.root = rootGroups B)
    (shaped : pub.shaped = true) : (pub.groups.flatMap (·.value)).size = 3 := by
  obtain ⟨count, all⟩ := (shaped_groups pub).mp shaped
  have three : pub.groups.size = 3 := by rw [count, root]; rfl
  obtain ⟨a, b, c, groups⟩ : ∃ a b c, pub.groups = #[a, b, c] := by
    generalize pub.groups = groups at three
    obtain ⟨list⟩ := groups
    rcases list with _ | ⟨a, _ | ⟨b, _ | ⟨c, _ | ⟨d, rest⟩⟩⟩⟩
    all_goals first
      | exact ⟨_, _, _, rfl⟩
      | (exfalso
         simp only [List.size_toArray, List.length_cons, List.length_nil] at three
         omega)
  have first : a.value.size = 1 := by simpa [groups] using (all 0 (by omega) (by omega)).value
  have second : b.value.size = 1 := by simpa [groups] using (all 1 (by omega) (by omega)).value
  have third : c.value.size = 1 := by simpa [groups] using (all 2 (by omega) (by omega)).value
  simp [groups, first, second, third]

/-- The coefficients of the commitment rows of a group in the shape of a row are canonical. -/
theorem Statement.Shaped.rows {s : Statement} {r : RootGroup} (shaped : s.Shaped r) :
    ∀ a ∈ commitmentRows s.commitment r.rows, a.word.toNat < fieldModulus := by
  intro a member
  simp only [commitmentRows, Array.mem_map, Array.mem_range] at member
  obtain ⟨j, below, rfl⟩ := member
  have canonical := shaped.canonical j below
  have at_ : s.commitment.size - 8 * r.rows + 8 * j = r.header.size + 8 * j := by
    rw [shaped.size]
    omega
  rw [at_]
  exact canonical

/-- The statement of the row of `key20.aks` after `Public.shaped`: the hypotheses `points` and
`values` of `presents20_accepts`, and each commitment has 471 bytes, the profile header of its
group and 32 canonical coefficients of its rows. -/
theorem shaped20 {pub : Public} (root : pub.root = root20) (shaped : pub.shaped = true) :
    (∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![20, 16, 16] g) ∧
    (pub.groups.flatMap (·.value)).size = 3 ∧
    ∀ g : Fin 3, ∃ s, pub.groups[g.val]? = some s ∧ s.commitment.size = 471 ∧
      (s.commitment.extract 0 215).data = (rootGroup Key20.root g).header.data ∧
      ∀ a ∈ commitmentRows s.commitment 32, a.word.toNat < fieldModulus := by
  refine ⟨fun g => ?_, shaped_values root shaped, fun g => ?_⟩
  · obtain ⟨s, found, group⟩ := shaped_group root shaped g
    rw [found, Option.map_some, Option.getD_some, group.point, root20_vars]
  · obtain ⟨s, found, group⟩ := shaped_group root shaped g
    have rows := group.rows
    have size := group.size
    have header := group.header
    rw [rootGroup_header_size] at size header
    rw [root20_rows] at size rows
    exact ⟨s, found, size, header, rows⟩

/-- The statement of a row of `key30.aks` after `Public.shaped`: the lengths of the points, the
count of the values, and each commitment has 471 bytes, the profile header of its group and 32
canonical coefficients of its rows. -/
theorem shaped30 {pub : Public} {r : Key30.Row} (member : r ∈ Key30.rows)
    (root : pub.root = root30 r) (shaped : pub.shaped = true) :
    (∀ g : Fin 3, ((pub.groups[g.val]?.map (·.point)).getD #[]).size = ![30, r.vars, 26] g) ∧
    (pub.groups.flatMap (·.value)).size = 3 ∧
    ∀ g : Fin 3, ∃ s, pub.groups[g.val]? = some s ∧ s.commitment.size = 471 ∧
      (s.commitment.extract 0 215).data = (rootGroup (Key30.root r) g).header.data ∧
      ∀ a ∈ commitmentRows s.commitment 32, a.word.toNat < fieldModulus := by
  refine ⟨fun g => ?_, shaped_values root shaped, fun g => ?_⟩
  · obtain ⟨s, found, group⟩ := shaped_group root shaped g
    rw [found, Option.map_some, Option.getD_some, group.point, root30_vars member]
  · obtain ⟨s, found, group⟩ := shaped_group root shaped g
    have rows := group.rows
    have size := group.size
    have header := group.header
    rw [rootGroup_header_size] at size header
    rw [root30_rows] at size rows
    exact ⟨s, found, size, header, rows⟩

end NightstreamFPrime.Spec.Compression.Akita.Replay
