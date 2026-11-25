import NightstreamFPrime.Spec.Compression.Akita.Replay.FinalGame
import NightstreamFPrime.Spec.Compression.Akita.Replay.Schedules

/-! The game of the offloaded root of a row of `key30.aks` when `runProof` accepts, with no
hypothesis of the layout (Akita revision `d50f700`; the root of `layout30`, `Key30.root`,
`Key30.Tower.scan`).

- `key30Root_low`, `key30Root_numbers`, `key30Root_pow` and `key30Root_fits`:
  the numbers of the root of each row that the hypotheses of `verifyProof_offload_game` read (the
  block `2^7` of `Key30.root_blocked`, the response base `2^3`, the `2^23` cells of the setup
  prefix, the degrees, and `PackGroup.Fits` of each group).
- `verifyProof_key30_root`: `verifyProof_offload_game` for the fold `F` of the root of
  `layout30 r zMax rice` (`fold`). The hypotheses are the shapes of the script at its `fused.end`,
  `endsFit`, the setup prefixes, and the choices of the basis, the challenge of the specification,
  `w` and `flat`. The split is `Batch.codeScan` with the block `2^low = 2^7`, the split of
  `Key30.Tower.scan` of the claims. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita

open Replayer

/-- `Batch.low` of the root of each row is the `κ = 7` of `Key30.root_blocked`. -/
theorem key30Root_low : ∀ r ∈ Key30.rows, (Key30.root r).low = 7 := by
  have least : ∀ r ∈ Key30.rows, (List.ofFn fun g => min (min ((Key30.root r).shape g).innerDegree
      ((Key30.root r).shape g).outerDegree) ((Key30.root r).carrierDegree g)).foldl min
        (Key30.root r).openDegree = 2 ^ 7 := by
    decide
  intro r member
  unfold Batch.low
  rw [least r member, Nat.log_pow (by norm_num)]

/-- The numbers of the root of each row: the response base `2^3`, the views of `A`, `B` and `D` in
the `2^23` cells of the setup prefix, and the block `2^low` in the degree of `D` and in `μ`. -/
theorem key30Root_numbers : ∀ r ∈ Key30.rows, (∀ g, ((Key30.root r).fold g).responseBase = 2 ^ 3) ∧
    (∀ g, (((Key30.root r).shape g).innerInstance false 0).size ≤ 2 ^ Key30.setupVars) ∧
    (∀ g, ((Key30.root r).shape g).outerInstance.size ≤ 2 ^ Key30.setupVars) ∧
    (Key30.root r).openInstance.size ≤ 2 ^ Key30.setupVars ∧
    2 ^ (Key30.root r).low ≤ (Key30.root r).openDegree ∧ (Key30.root r).low ≤ Key30.rootVars := by
  decide

/-- The degrees of `A`, `B` and `D` of the root are powers of two. -/
theorem key30Root_pow (r : Key30.Row) :
    (∀ g, 2 ^ Nat.log 2 ((Key30.root r).shape g).innerDegree =
      ((Key30.root r).shape g).innerDegree) ∧
    (∀ g, 2 ^ Nat.log 2 ((Key30.root r).shape g).outerDegree =
      ((Key30.root r).shape g).outerDegree) ∧
    2 ^ Nat.log 2 (Key30.root r).openDegree = (Key30.root r).openDegree := by
  refine ⟨fun g => ?_, fun g => ?_, pow_log_of (k := 8) rfl⟩
  · fin_cases g
    · exact pow_log_of (k := 11) rfl
    · exact pow_log_of (k := 11) rfl
    · exact pow_log_of (k := 9) rfl
  · fin_cases g
    · exact pow_log_of (k := 8) rfl
    · exact pow_log_of (k := 7) rfl
    · exact pow_log_of (k := 8) rfl

/-- `PackGroup.Fits` of each group of the root of each row. -/
theorem key30Root_fits : ∀ r ∈ Key30.rows, ∀ k, (batchGroup (Key30.root r) (fun _ => [])
    (fun _ => []) (fun _ => []) (fun _ => Ext.one) k).Fits Key30.rootVars := by
  intro r member k
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;> fin_cases k
  all_goals first
    | exact ⟨⟨7, rfl⟩, ⟨9, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
        by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
        by decide, by decide, by decide⟩
    | exact ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨8, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
        by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
        by decide, by decide, by decide⟩
    | exact ⟨⟨7, rfl⟩, ⟨11, rfl⟩, ⟨7, rfl⟩, ⟨8, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩, ⟨6, rfl⟩, ⟨5, rfl⟩,
        by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
        by decide, by decide, by decide⟩

/-- The fold groups of the root of a row of `layout30` (`ofBatch`). -/
def key30RootGroup (r : Key30.Row) (g : Fin 3) : FoldGroup :=
  ⟨((Key30.root r).shape g).blocks, ((Key30.root r).shape g).innerDegree,
    (Key30.root r).carrierDegree g, (Key30.family 0 g).1, (Key30.family 0 g).2, false⟩

/-- The packing shape of the root of a row of `layout30` (`ofBatch`). -/
def key30RootShape (r : Key30.Row) : PackShape :=
  ⟨(List.finRange 3).map fun k => ⟨(groupOrder 3 k).val, (Key30.root r).shape (groupOrder 3 k),
      (Key30.root r).opening (groupOrder 3 k), (Key30.root r).fold (groupOrder 3 k)⟩,
    ⟨(Key30.root r).openDegree, (Key30.root r).openRank, (Key30.root r).chain⟩,
    Key30.quotientDigits₀⟩

/-- The fields of the fold `F` of the root of a row of `layout30` that the hypotheses of
`verifyProof_offload_game` read. -/
structure Key30RootFold (r : Key30.Row) (F : Fold) : Prop where
  low : F.low = (Key30.root r).low
  logBasis : F.logBasis = 3
  stage3 : F.stage3 = some Key30.setupVars
  euclidean : F.euclidean = false
  compressed : F.compressed = (Key30.root r).compressed
  fused : F.fused = .packing (key30RootShape r)
  groups : F.groups = (List.finRange 3).map (key30RootGroup r)
  rows : (F.rhs, F.openingRow) = relation (fun k => (Key30.root r).shape (groupOrder 3 k))
    (Key30.root r).openRank (Key30.root r).openDegree (Key30.root r).chain
      fun k => .root (groupOrder 3 k).val
  fits : (Key30.root r).cells ≤ 2 ^ F.vars
  room : (Key30.root r).quotientRoom Key30.quotientDigits₀ ≤ 2 ^ F.vars
  rowsFit : (Key30.root r).rowCount + 1 ≤ 2 ^ F.batch
  blocked : (Key30.root r).Blocked F.low
  lowFits : 2 ^ F.low ≤ (Key30.root r).openDegree
  lowShort : F.low ≤ F.vars

/-- The fields of the root fold of each row that are numbers of the layout. -/
theorem key30Root_fields {r : Key30.Row} (member : r ∈ Key30.rows) {zMax rice : ℕ} {F : Fold}
    (fold : (layout30 r zMax rice).folds[0]? = some F) :
    F.vars = Key30.rootVars ∧ F.batch = Key30.rootBatch ∧ F.low = 7 ∧ F.logBasis = 3 ∧
      F.stage3 = some Key30.setupVars ∧ F.euclidean = false ∧
      F.compressed = (Key30.root r).compressed ∧ F.fused = .packing (key30RootShape r) ∧
      F.groups = (List.finRange 3).map (key30RootGroup r) ∧
      (F.rhs, F.openingRow) = relation (fun k => (Key30.root r).shape (groupOrder 3 k))
        (Key30.root r).openRank (Key30.root r).openDegree (Key30.root r).chain
          fun k => .root (groupOrder 3 k).val := by
  rw [show (layout30 r zMax rice).folds = (layout30 r 0 0).folds from rfl] at fold
  have get : ∀ {α : Type} (f : Fold → α) {x : α},
      (layout30 r 0 0).folds[0]?.map f = some x → f F = x := by
    intro α f x computed
    rw [fold] at computed
    exact Option.some.inj computed
  simp only [Key30.rows, List.mem_cons, List.mem_nil_iff, or_false] at member
  rcases member with rfl | rfl <;>
    exact ⟨get Fold.vars rfl, get Fold.batch rfl, get Fold.low (by decide),
      get Fold.logBasis (by decide), get Fold.stage3 rfl, get Fold.euclidean rfl,
      get Fold.compressed rfl, get Fold.fused rfl, get Fold.groups rfl,
      get (fun F => (F.rhs, F.openingRow)) rfl⟩

theorem key30Root_fold {r : Key30.Row} (member : r ∈ Key30.rows) {zMax rice : ℕ} {F : Fold}
    (fold : (layout30 r zMax rice).folds[0]? = some F) : Key30RootFold r F := by
  obtain ⟨vars, batch, low, logBasis, stage3, euclidean, compressed, fused, groups, rows⟩ :=
    key30Root_fields member fold
  have sized := Key30.root_sized r member
  have numbers := key30Root_numbers r member
  have rootLow := key30Root_low r member
  refine ⟨low.trans rootLow.symm, logBasis, stage3, euclidean, compressed, fused, groups, rows,
    ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [vars]; exact sized.1
  · rw [vars]; exact sized.2.1
  · rw [batch]; exact sized.2.2.1
  · rw [low]; exact Key30.root_blocked r member
  · rw [low, ← rootLow]; exact numbers.2.2.2.2.1
  · rw [low, vars]; decide

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

/-- The offloaded root of a row of `key30.aks`: when `runProof` accepts and the check
`fused.end` at `c` has the fold `F` of the root of `layout30 r zMax rice` (`fold`), Stage 3 gives
`some o` and `offloadCheck` of `Batch.liftedReduction` of the root over the setup prefix `flat`,
with the split `Batch.codeScan` of the claims, runs on the path of the code to acceptance with the
recorded `(fused.point, witness_evaluation)` and `o`. -/
theorem verifyProof_key30_root (r : Key30.Row) (member : r ∈ Key30.rows) {zMax rice : ℕ}
    (pub : Public) (proof : ByteArray) {ops : Array Op}
    (scripted : script pub.layout pub.plan = .ok ops) (ends : endsFit ops.zipIdx.toList = true)
    {full : Array Fq} (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full)
    (accepted : runProof pub proof = .verdict .accept) {messages : Array Message} {rest : ℕ}
    (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ} {key : Key} {level : ℕ}
    {F : Fold} (at_ : ops[c]? = some (.check key (.fused level F)))
    (fold : (layout30 r zMax rice).folds[0]? = some F) {i a e b t : ℕ}
    (shape : offloadShape (ops.zipIdx.toList.take c) level F
      (SetupPlan.ofBatch (Key30.root r) (groupOrder 3) F.low 3) i a e b t = true)
    (game : gameShape (ops.zipIdx.toList.take c) level F = true)
    (setupShape : roundsAt (ops.zipIdx.toList.take c) ⟨level, "stage3", none⟩ Key30.setupVars 2 =
      true)
    (drawShape : drawnShape (ops.zipIdx.toList.take c) level F.groups = true) {s : Replayer}
    (state : stateBefore pub proof messages ops c = s) {setup : Array Fq}
    (expanded : pub.setupFor F.setupCount = some setup)
    (β : ∀ g, Module.Basis (Fin ((Key30.root r).planes g)) Zq E)
    (basis : ∀ g t, β g t = (extBasis.getD t Ext.zero).into) (challenge : (Key30.root r).Challenge)
    (challenged : ∀ k i, challenge (groupOrder 3 k) i = Poly.toRq _
      ((drawnCarrier (Key30.root r) s level k).getD i
        (Poly.zero ((Key30.root r).carrierDegree (groupOrder 3 k)))))
    (w flat : Table) :
    ∃ o, stage3In s level (SetupPlan.ofBatch (Key30.root r) (groupOrder 3) F.low 3) = some o ∧
      (offloadCheck (Batch.liftedReduction β ((batchStatementFor (Key30.root r) setup
          (claimPointIn s level) (dataIn pub s level) (fun k => .root (groupOrder 3 k).val)
          (rowsIn s level F) (claimsIn s level F)).over fun p : Fin (2 ^ Key30.setupVars) => flat p)
          challenge F.vars (key30Root_fold member fold).fits 8 Key30.quotientDigits₀
          (key30Root_fold member fold).room F.batch (key30Root_fold member fold).rowsFit) 8
          ((Key30.root r).gaps Key30.quotientDigits₀).support (Key30.root r).compressed w
          Key30.setupVars flat
          (Batch.codeScan β (batchStatementFor (Key30.root r) setup (claimPointIn s level)
            (dataIn pub s level) (fun k => .root (groupOrder 3 k).val) (rowsIn s level F)
            (claimsIn s level F)) challenge F.vars (key30Root_fold member fold).fits 8
            Key30.quotientDigits₀ (key30Root_fold member fold).room F.batch
            (key30Root_fold member fold).rowsFit Key30.setupVars flat F.low
            (key30Root_fold member fold).blocked (key30Root_fold member fold).lowShort)).Runs
        (offloadPathIn E pub s level F setup F.vars F.batch)
        (some (((recordedIn s ⟨level, "fused.point", none⟩).toList.map Ext.into,
            (firstIn s ⟨level, "witness_evaluation", none⟩).into),
          (o.1.map Ext.into, o.2.into))) := by
  have root := key30Root_fold member fold
  have numbers := key30Root_numbers r member
  exact verifyProof_offload_game pub proof scripted ends terminalSetup accepted decoded at_ _ shape
    game setupShape drawShape state root.fused root.stage3 expanded
    (fun k => .root (groupOrder 3 k).val) root.rows 8 (by rw [root.logBasis]; rfl) root.low β basis
    challenge challenged root.fits root.room root.rowsFit w flat root.euclidean root.compressed
    (by rw [root.logBasis]; rfl) (by rw [root.logBasis]; rfl) rfl rfl rfl
    (fun k => by
      rw [show F.vars = Key30.rootVars from (key30Root_fields member fold).1]
      exact key30Root_fits r member k)
    (key30RootGroup r) root.groups (fun _ => rfl) (fun _ => rfl) 0 root.lowFits root.lowShort
    root.blocked (by rw [root.logBasis]) (by rw [root.logBasis]; exact numbers.1)
    (key30Root_pow r).1 (key30Root_pow r).2.1 (key30Root_pow r).2.2 numbers.2.1 numbers.2.2.1
    numbers.2.2.2.1

end NightstreamFPrime.Spec.Compression.Akita.Replay
