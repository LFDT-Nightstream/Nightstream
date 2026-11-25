import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinAccept

/-! Theorem B of the join, direction "→", at the levels 1 to 5 of `key20.aks` (Akita revision
`d50f700`; ePrint 2026/1983, revision of 2026-11-23, Figure 4, steps 0, 3 and 5;
`Tier.TraceLinked` of `Extract/TraceLink.lean`).

- `path20_trace`, `receiver20_lt`: the path of a level with the tensor reduction of the next
  level, and the index where that reduction has ended.
- `first20_numbers` to `fourth20_numbers`: the numbers of the levels 1 to 4 of the model.
- `presents20_level1` to `presents20_level4`: the link `Tier.TraceLinked` from the level `ℓ` to
  the level `ℓ + 1`, for `ℓ` = 1 to 4. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game
open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Akita.Extract (Tier Presents view Prefix)

open Replayer

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E] [Degree2 E]

omit [Fintype E] [DecidableEq E] in
/-- The path of the level `ℓ ≥ 1`: its fold and the tensor reduction of its claim. -/
theorem path20_trace (pub : Public) (proof : ByteArray) (messages : Array Message) (ℓ : Fin 6)
    (positive : ℓ.val ≠ 0) :
    path20 pub proof messages E ℓ =
      foldPathIn E pub (fused20 pub proof messages ℓ) ℓ (fold20 ℓ) (setup20 pub ℓ) (fold20 ℓ).vars
          (fold20 ℓ).batch ++
        tensoredPath E (tail20 ℓ)
          (some ((recordedIn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ))
              ⟨ℓ, "fused.point", none⟩).toList,
            firstIn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ))
              ⟨ℓ, "witness_evaluation", none⟩))
          ((recordedIn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ))
            ⟨ℓ + 1, "tensor.partials", none⟩).getD 0 Ext.zero)
          ((recordedIn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ))
            ⟨ℓ + 1, "tensor.partials", none⟩).getD 1 Ext.zero)
          (firstIn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ))
            ⟨ℓ + 1, "tensor.eta", none⟩)
          (roundsIn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ))
            ⟨ℓ + 1, "tensor.sumcheck", none⟩)
          (firstIn (stateBefore pub proof messages pub.ops (receiver20 pub ℓ))
            ⟨ℓ + 1, "tensor.final_claims", none⟩) := by
  unfold path20
  dsimp only
  split
  · exact absurd ‹_› positive
  · rfl

theorem receiver20_lt (pub : Public) {ℓ : ℕ} (below : ℓ < 5) :
    receiver20 pub ℓ = fusedIndex pub.layout (ℓ + 1) := by
  simp [receiver20, below]

variable {Coord : Type} [Fintype Coord] [DecidableEq Coord] {total : ℕ}
  (o : Key20.Tower E (Key20.Wire E) Coord total)

local notation "code₁" =>
  (CodeSlot.open Key20.open64 Key20.open64_fits :
    CodeSlot E → Option (Rq Key20.level₁.opening.carrierDegree))

local notation "code₂" =>
  (CodeSlot.open Key20.open128 Key20.open128_fits :
    CodeSlot E → Option (Rq Key20.level₂.shape.innerDegree))

local notation "zeroth" =>
  Extract.liftBatch o.β o.key o.openKey o.claim Key20.rootVars (And.left Key20.root_sized) 8
    Key20.quotientDigits Key20.root_room Key20.rootBatch (And.left (And.right Key20.root_sized))
    o.block (Key20.rootDecode E) o.read Key20.rootMass (0 + 1)

local notation "first" =>
  Tier.tensored (Extract.liftFold o.β₁ o.key₁ o.openKey₁ o.claim₁ Key20.level₁.vars
    (And.left Key20.level₁_sized) Key20.quotientDigits (And.left (And.right Key20.level₁_sized))
    Key20.level₁.batch (And.left (And.right (And.right Key20.level₁_sized))) o.block₁ code₁ o.read
    51 (Tier.finish zeroth)) Key20.level₁.tail

local notation "second" =>
  Tier.tensored (Extract.committedTrace (fold := Key20.level₂.fold) o.key₂ o.openKey₂ o.claim₂
    Key20.level₂.vars (And.left Key20.sized₂) Key20.level₂.batch (And.left (And.right Key20.sized₂))
    o.block₂ code₂ o.read 31 (Tier.finish first)) Key20.level₂.tail

omit [Fintype Coord] [DecidableEq Coord] in
/-- The numbers of the level 1 of the model: its fold stage 51, its 68 check stages (with the
tensor reduction of the level 2), its end 120, and its work bits are those of `bits20` at the
stages 51 to 119. -/
theorem first20_numbers : (first).start = 51 ∧ (first).costs.length = 68 ∧ (first).finish = 120 ∧
    (first).foldBits = bits20.flatten.getD 51 0 ∧
    ∀ index < 68, (first).bits.getD index 0 = bits20.flatten.getD (51 + 1 + index) 0 := by
  simp only [Extract.liftBatch, Extract.liftFold, Tier.tensored, Tier.finish,
    Extract.foldCosts_length, List.length_append, Extract.tensorCosts, List.length_cons,
    List.length_replicate]
  decide

local notation "code₃" =>
  (CodeSlot.kept128 : CodeSlot E → Option (Rq Key20.level₃.shape.innerDegree))

local notation "code₄" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₄.shape.innerDegree))

local notation "code₅" =>
  (CodeSlot.kept64 : CodeSlot E → Option (Rq Key20.level₅.shape.innerDegree))

local notation "third" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₃.fold) o.key₃ o.openKey₃ o.claim₃
    Key20.level₃.vars (And.left Key20.sized₃) Key20.level₃.batch
    (And.left (And.right Key20.sized₃)) o.block₃ code₃ o.read 31 Key20.norm₃.operator
    Key20.norm₃.limit (Tier.finish second)) Key20.level₃.tail

local notation "fourth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₄.fold) o.key₄ o.openKey₄ o.claim₄
    Key20.level₄.vars (And.left Key20.sized₄) Key20.level₄.batch
    (And.left (And.right Key20.sized₄)) o.block₄ code₄ o.read 53 Key20.norm₄.operator
    Key20.norm₄.limit (Tier.finish third)) Key20.level₄.tail

local notation "fifth" =>
  Tier.tensored (Extract.committedTraceSq (fold := Key20.level₅.fold) o.key₅ o.openKey₅ o.claim₅
    Key20.level₅.vars (And.left Key20.sized₅) Key20.level₅.batch
    (And.left (And.right Key20.sized₅)) o.block₅ code₅ o.read 53 Key20.norm₅.operator
    Key20.norm₅.limit (Tier.finish fourth)) Key20.level₅.tail

omit [Fintype Coord] [DecidableEq Coord] in
/-- The numbers of the level 2 of the model: its fold stage 120, its 82 check stages (with the
tensor reduction of the level 3), its end 203, and its work bits are those of `bits20` at the
stages 120 to 202. -/
theorem second20_numbers : (second).start = 120 ∧ (second).costs.length = 82 ∧ (second).finish = 203 ∧
    (second).foldBits = bits20.flatten.getD 120 0 ∧
    ∀ index < 82, (second).bits.getD index 0 = bits20.flatten.getD (120 + 1 + index) 0 := by
  simp only [Extract.liftBatch, Extract.liftFold, Extract.committedTrace, Tier.tensored,
    Tier.finish, Extract.foldCosts_length, List.length_append, Extract.tensorCosts,
    List.length_cons, List.length_replicate]
  decide

omit [Fintype Coord] [DecidableEq Coord] in
/-- The numbers of the level 3 of the model: its fold stage 203, its 80 check stages (with the
tensor reduction of the level 4), its end 284, and its work bits are those of `bits20` at the
stages 203 to 283. -/
theorem third20_numbers : (third).start = 203 ∧ (third).costs.length = 80 ∧ (third).finish = 284 ∧
    (third).foldBits = bits20.flatten.getD 203 0 ∧
    ∀ index < 80, (third).bits.getD index 0 = bits20.flatten.getD (203 + 1 + index) 0 := by
  simp only [Extract.liftBatch, Extract.liftFold, Extract.committedTrace, Extract.committedTraceSq,
    Tier.tensored, Tier.finish, Extract.foldCosts_length, Extract.foldCostsSq_length,
    List.length_append, Extract.tensorCosts, List.length_cons, List.length_replicate]
  decide

omit [Fintype Coord] [DecidableEq Coord] in
/-- The numbers of the level 4 of the model: its fold stage 284, its 93 check stages (with the
tensor reduction of the level 5), its end 378, and its work bits are those of `bits20` at the
stages 284 to 377. -/
theorem fourth20_numbers : (fourth).start = 284 ∧ (fourth).costs.length = 93 ∧ (fourth).finish = 378 ∧
    (fourth).foldBits = bits20.flatten.getD 284 0 ∧
    ∀ index < 93, (fourth).bits.getD index 0 = bits20.flatten.getD (284 + 1 + index) 0 := by
  simp only [Extract.liftBatch, Extract.liftFold, Extract.committedTrace, Extract.committedTraceSq,
    Tier.tensored, Tier.finish, Extract.foldCosts_length, Extract.foldCostsSq_length,
    List.length_append, Extract.tensorCosts, List.length_cons, List.length_replicate]
  decide +kernel

variable (pub : Public) (proof : ByteArray) (messages : Array Message)
  (tLast : ℕ → Rq Key20.terminal.innerDegree) (eLast : Rq Key20.terminal.innerDegree)
  {block : Key20.root.Block → Coord} {block₁ : Fin Key20.level₁.shape.blocks → Coord}
  {block₂ : Fin Key20.level₂.shape.blocks → Coord} {block₃ : Fin Key20.level₃.shape.blocks → Coord}
  {block₄ : Fin Key20.level₄.shape.blocks → Coord} {block₅ : Fin Key20.level₅.shape.blocks → Coord}
  {blockL : Fin Key20.terminal.blocks → Coord}

/-- Theorem B "→" at the level 1 (Akita Figure 4, steps 0, 3 and 5): on a transcript that
presents the bytes of an accepted proof of `layout20`, the link `Tier.TraceLinked` from the level
1 with the tensor reduction of its claim to the level 2 holds. `expanded`: the setup prefix of the
level 1 exists. -/
theorem presents20_level1 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 1).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    Tier.TraceLinked (first) (second) (And.left Key20.quarters) o.commit₂ point := by
  obtain ⟨started, length, finished, foldBits, bits⟩ := first20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₁
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  have data := presents.data
  subst tower
  obtain ⟨scripted, ends, levels⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, pack, tensor, -⟩ := verifyProof_frames20 layout accepted
  have at₀ : pub.ops[fusedIndex pub.layout 0]? =
      some (.check ⟨0, "fused.end", none⟩ (.fused 0 (fold20 0))) := (levels 0).at_
  have game₀ : gameShape (pub.ops.zipIdx.toList.take (fusedIndex pub.layout 0)) 0 (fold20 0) =
      true := (levels 0).game
  have found : pub.setupFor (fold20 1).setupCount = some (setup20 pub 1) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have packed := (verifyProof_key20_packLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded at₀ (layout20_fold zMax rice 0) game₀ pack.pointAt pack.openingsAt pack.pointBefore
    pack.openingsBefore pack.pointAfter pack.openingsAfter pack.pointHeld pack.openingsHeld
    pack.pointFresh pack.openingsFresh pack.pointKept pack.openingsKept
    (layout20_fold zMax rice 1) (setup20 pub 1)).2.2.2.2.2
  obtain ⟨fits, room, rowsFit, runs⟩ := verifyProof_key20_level1_of (E := E) pub proof layout
    accepted terminalSetup decoded rfl found (basis20 E) basis20_apply
    (challenge20₁ pub proof messages) (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  have linked := verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded List.mem_cons_self tensor.at_ tensor.rounds tensor.after tensor.kept
    (layout20_fold zMax rice 2) (by decide) (setup20 pub 2)
  dsimp only at tensored linked
  obtain ⟨-, -, runs'⟩ := tensored
  obtain ⟨segments, blocks, positioned, packedLength, -⟩ := linked
  have packed' : packStatementFor (E := E) Key20.level₁.opening (setup20 pub 1)
      (claimPointIn (fused20 pub proof messages 1) 1 0) (dataIn pub (fused20 pub proof messages 1) 1)
      .witness (rowsIn (fused20 pub proof messages 1) 1 (fold20 1))
      (claimsIn (fused20 pub proof messages 1) 1 (fold20 1)) =
      ((point.wire 51).data.claim₁).statement (Key.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 1)) Key20.level₁.opening) := by
    rw [data]
    exact packed
  rw [packed'] at runs'
  rw [path20_trace pub proof messages 1 (by decide), receiver20_lt pub (by decide)] at presented
  refine ⟨by rw [finished]; have := presents.long; omega, ⟨⟨reply, answered, ?_, fun b => ?_⟩,
    fun index inside other found' => ?_, ?_⟩, _, _, _,
    mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
        ⟨1 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
      (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
        ⟨1, "fused.point", none⟩).toList.tail).map Ext.into)
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
          ⟨1 + 1, "tensor.eta", none⟩).into),
    (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
      ⟨1 + 1, "tensor.final_claims", none⟩).into,
    payloadOf Key20.level₂.opening.chain _ _
      (((stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)).fields.get?
        ⟨2, "opening_payload", none⟩).getD #[]), ?_, blocks, positioned, packedLength, ?_⟩
  · rw [foldBits]
    exact (presents.work 51 (by decide) reply answered).mpr (ok _)
  · rw [decodedAt b]
    rfl
  · rw [length] at inside
    rw [bits index inside]
    exact (presents.work (51 + 1 + index) (by omega) other found').mpr (ok _)
  · exact (presents.nonce 1 (by decide)).mpr (ok _)
  · refine (Extract.Tier.game_eq_pure_iff_of_runs _ 0 point presented (result :=
      some (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
          ⟨1 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into,
        mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
            ⟨1 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
          (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
            ⟨1, "fused.point", none⟩).toList.tail).map Ext.into)
            (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
              ⟨1 + 1, "tensor.eta", none⟩).into),
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2))
          ⟨1 + 1, "tensor.final_claims", none⟩).into)) ?_ _).mpr ?_
    · unfold Tier.challenge
      rw [challenged]
      exact runs'
    · rw [segments]
  · show ((point.wire _).data.claim₂) = Extract.TraceClaim.ofPoint _ ((point.wire _).data.commit₂) _ _ _ _ _ _
    rw [data, data]
    rfl

/-- Theorem B "→" at the level 2: on a transcript that presents the bytes of an accepted proof of
`layout20`, the link `Tier.TraceLinked` from the level 2 with the tensor reduction of its claim to
the level 3 holds. `expanded`: the setup prefix of the level 2 exists. -/
theorem presents20_level2 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 2).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    Tier.TraceLinked (second) (third) (And.left (And.right Key20.quarters)) o.commit₃ point := by
  obtain ⟨started, length, finished, foldBits, bits⟩ := second20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₂
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  have data := presents.data
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.1
  have tensor := frames.2.1
  have found : pub.setupFor (fold20 2).setupCount = some (setup20 pub 2) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded List.mem_cons_self tensorBefore.at_ tensorBefore.rounds tensorBefore.after
    tensorBefore.kept (layout20_fold zMax rice 2) (by decide) (setup20 pub 2)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_trace_of (E := E) pub proof layout
    accepted terminalSetup decoded rfl found (challengeTrace20 pub proof messages 2 Key20.level₂)
    (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  have linked := verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded (List.mem_cons_of_mem _ List.mem_cons_self) tensor.at_ tensor.rounds tensor.after tensor.kept
    (layout20_fold zMax rice 3) (by decide) (setup20 pub 3)
  dsimp only at tensored linked before
  obtain ⟨-, -, runs'⟩ := tensored
  obtain ⟨segments, blocks, positioned, packedLength, -⟩ := linked
  have before' : traceStatementFor (E := E) Key20.level₂.opening quarter (setup20 pub 2)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2)) (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2)) (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 2)) 2 (fold20 2)) =
      ((point.wire 120).data.claim₂).statement (Key.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 2)) Key20.level₂.opening) := by
    rw [data]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 2 (by decide), receiver20_lt pub (by decide)] at presented
  refine ⟨by rw [finished]; have := presents.long; omega, ⟨⟨reply, answered, ?_, fun b => ?_⟩,
    fun index inside other found' => ?_, ?_⟩, _, _, _,
    mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
        ⟨2 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
      (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
        ⟨2, "fused.point", none⟩).toList.tail).map Ext.into)
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
          ⟨2 + 1, "tensor.eta", none⟩).into),
    (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
      ⟨2 + 1, "tensor.final_claims", none⟩).into,
    payloadOf Key20.level₃.opening.chain _ _
      (((stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3)).fields.get?
        ⟨3, "opening_payload", none⟩).getD #[]), ?_, blocks, positioned, packedLength, ?_⟩
  · rw [foldBits]
    exact (presents.work 120 (by decide) reply answered).mpr (ok _)
  · rw [decodedAt b]
    rfl
  · rw [length] at inside
    rw [bits index inside]
    exact (presents.work (120 + 1 + index) (by omega) other found').mpr (ok _)
  · exact (presents.nonce 2 (by decide)).mpr (ok _)
  · refine (Extract.Tier.game_eq_pure_iff_of_runs _ 0 point presented (result :=
      some (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
          ⟨2 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into,
        mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
            ⟨2 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
          (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
            ⟨2, "fused.point", none⟩).toList.tail).map Ext.into)
            (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
              ⟨2 + 1, "tensor.eta", none⟩).into),
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 3))
          ⟨2 + 1, "tensor.final_claims", none⟩).into)) ?_ _).mpr ?_
    · unfold Tier.challenge
      rw [challenged]
      exact runs'
    · rw [segments]
  · show ((point.wire _).data.claim₃) =
      Extract.TraceClaim.ofPoint _ ((point.wire _).data.commit₃) _ _ _ _ _ _
    rw [data, data]
    rfl

/-- Theorem B "→" at the level 3: on a transcript that presents the bytes of an accepted proof of
`layout20`, the link `Tier.TraceLinked` from the level 3 with the tensor reduction of its claim to
the level 4 holds. `expanded`: the setup prefix of the level 3 exists. -/
theorem presents20_level3 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 3).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    Tier.TraceLinked (third) (fourth) (And.left (And.right (And.right Key20.quarters))) o.commit₄ point := by
  obtain ⟨started, length, finished, foldBits, bits⟩ := third20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₃
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  have data := presents.data
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.2.1
  have tensor := frames.2.2.1
  have found : pub.setupFor (fold20 3).setupCount = some (setup20 pub 3) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded (List.mem_cons_of_mem _ List.mem_cons_self) tensorBefore.at_ tensorBefore.rounds tensorBefore.after
    tensorBefore.kept (layout20_fold zMax rice 3) (by decide) (setup20 pub 3)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_traceSq_of (E := E) 3 Key20.level₃
    List.mem_cons_self pub proof layout accepted terminalSetup decoded rfl found
    (challengeTrace20 pub proof messages 3 Key20.level₃) (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  have linked := verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)) tensor.at_ tensor.rounds tensor.after tensor.kept
    (layout20_fold zMax rice 4) (by decide) (setup20 pub 4)
  dsimp only at tensored linked before
  obtain ⟨-, -, runs'⟩ := tensored
  obtain ⟨segments, blocks, positioned, packedLength, -⟩ := linked
  have before' : traceStatementFor (E := E) Key20.level₃.opening quarter (setup20 pub 3)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ) (fold20 3)) (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ) (fold20 3)) (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((3 : Fin 6) : ℕ))) ((3 : Fin 6) : ℕ) (fold20 3)) =
      ((point.wire 203).data.claim₃).statement (Key.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 3)) Key20.level₃.opening) := by
    rw [data]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 3 (by decide), receiver20_lt pub (by decide)] at presented
  refine ⟨by rw [finished]; have := presents.long; omega, ⟨⟨reply, answered, ?_, fun b => ?_⟩,
    fun index inside other found' => ?_, ?_⟩, _, _, _,
    mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
        ⟨3 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
      (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
        ⟨3, "fused.point", none⟩).toList.tail).map Ext.into)
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
          ⟨3 + 1, "tensor.eta", none⟩).into),
    (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
      ⟨3 + 1, "tensor.final_claims", none⟩).into,
    payloadOf Key20.level₄.opening.chain _ _
      (((stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4)).fields.get?
        ⟨4, "opening_payload", none⟩).getD #[]), ?_, blocks, positioned, packedLength, ?_⟩
  · rw [foldBits]
    exact (presents.work 203 (by decide) reply answered).mpr (ok _)
  · rw [decodedAt b]
    rfl
  · rw [length] at inside
    rw [bits index inside]
    exact (presents.work (203 + 1 + index) (by omega) other found').mpr (ok _)
  · exact (presents.nonce 3 (by decide)).mpr (ok _)
  · refine (Extract.Tier.game_eq_pure_iff_of_runs _ 0 point presented (result :=
      some (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
          ⟨3 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into,
        mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
            ⟨3 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
          (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
            ⟨3, "fused.point", none⟩).toList.tail).map Ext.into)
            (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
              ⟨3 + 1, "tensor.eta", none⟩).into),
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 4))
          ⟨3 + 1, "tensor.final_claims", none⟩).into)) ?_ _).mpr ?_
    · unfold Tier.challenge
      rw [challenged]
      exact runs'
    · rw [segments]
  · show ((point.wire _).data.claim₄) =
      Extract.TraceClaim.ofPoint _ ((point.wire _).data.commit₄) _ _ _ _ _ _
    rw [data, data]
    rfl

/-- Theorem B "→" at the level 4: on a transcript that presents the bytes of an accepted proof of
`layout20`, the link `Tier.TraceLinked` from the level 4 with the tensor reduction of its claim to
the level 5 holds. `expanded`: the setup prefix of the level 4 exists. -/
theorem presents20_level4 {zMax rice : ℕ} (layout : pub.layout = layout20 zMax rice)
    (accepted : verifyProof pub proof = .verdict .accept) {full : Array Fq}
    (terminalSetup : pub.setupFor pub.layout.terminal.setupCount = some full) {rest : ℕ}
    (decoded : decode pub.ops proof = .ok (messages, rest))
    (expanded : (pub.setupFor (fold20 4).setupCount).isSome)
    (tower : o = tower20 pub block block₁ block₂ block₃ block₄ block₅ blockL total 472)
    {point : Prefix (Key20.Wire E) (Option Coord → CodeSlot E) total}
    (presents : PresentsAt20 o pub proof messages tLast eLast point) :
    Tier.TraceLinked (fourth) (fifth) (And.left (And.right (And.right (And.right Key20.quarters)))) o.commit₅ point := by
  obtain ⟨started, length, finished, foldBits, bits⟩ := fourth20_numbers o
  obtain ⟨⟨reply, answered, decodedAt⟩, presented⟩ := presents.level₄
  have challenged := Extract.challengeAt_of_reply _ _ _ answered decodedAt
  have data := presents.data
  subst tower
  obtain ⟨scripted, ends, -⟩ := verifyProof_script20 layout accepted
  obtain ⟨-, -, frames⟩ := verifyProof_frames20 layout accepted
  have tensorBefore := frames.2.2.1
  have tensor := frames.2.2.2.1
  have found : pub.setupFor (fold20 4).setupCount = some (setup20 pub 4) := by
    obtain ⟨setup, found⟩ := Option.isSome_iff_exists.mp expanded
    simp [setup20, found]
  have ok := stepOk_of_accept scripted ends terminalSetup accepted decoded
  have before := (verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup
    accepted decoded (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self)) tensorBefore.at_ tensorBefore.rounds tensorBefore.after
    tensorBefore.kept (layout20_fold zMax rice 4) (by decide) (setup20 pub 4)).2.2.2.2
  obtain ⟨quarter, fits, rowsFit, runs⟩ := verifyProof_key20_traceSq_of (E := E) 4 Key20.level₄
    (List.mem_cons_of_mem _ List.mem_cons_self) pub proof layout accepted terminalSetup decoded rfl found
    (challengeTrace20 pub proof messages 4 Key20.level₄) (fun _ => rfl) 0
  have tensored := verifyProof_tensored pub proof scripted ends terminalSetup accepted decoded
    tensor.at_ tensor.rounds tensor.before tensor.after tensor.held tensor.kept 0 runs
  have linked := verifyProof_key20_traceLink (E := E) pub proof scripted ends terminalSetup accepted
    decoded (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ (List.mem_cons_of_mem _ List.mem_cons_self))) tensor.at_ tensor.rounds tensor.after tensor.kept
    (layout20_fold zMax rice 5) (by decide) (setup20 pub 5)
  dsimp only at tensored linked before
  obtain ⟨-, -, runs'⟩ := tensored
  obtain ⟨segments, blocks, positioned, packedLength, -⟩ := linked
  have before' : traceStatementFor (E := E) Key20.level₄.opening quarter (setup20 pub 4)
      (traceValuesIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ) (fold20 4)) (dataIn pub (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ)) .witness
      (rowsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ) (fold20 4)) (claimsIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout ((4 : Fin 6) : ℕ))) ((4 : Fin 6) : ℕ) (fold20 4)) =
      ((point.wire 284).data.claim₄).statement (Key.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.shape)
        (OpeningKey.ofSetup (flatOf (setup20 pub 4)) Key20.level₄.opening) := by
    rw [data]
    exact before
  rw [before'] at runs'
  rw [path20_trace pub proof messages 4 (by decide), receiver20_lt pub (by decide)] at presented
  refine ⟨by rw [finished]; have := presents.long; omega, ⟨⟨reply, answered, ?_, fun b => ?_⟩,
    fun index inside other found' => ?_, ?_⟩, _, _, _,
    mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
        ⟨4 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
      (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
        ⟨4, "fused.point", none⟩).toList.tail).map Ext.into)
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
          ⟨4 + 1, "tensor.eta", none⟩).into),
    (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
      ⟨4 + 1, "tensor.final_claims", none⟩).into,
    payloadOf Key20.level₅.opening.chain _ _
      (((stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5)).fields.get?
        ⟨5, "opening_payload", none⟩).getD #[]), ?_, blocks, positioned, packedLength, ?_⟩
  · rw [foldBits]
    exact (presents.work 284 (by omega) reply (started ▸ answered)).mpr (ok _)
  · rw [decodedAt b]
    rfl
  · rw [length] at inside
    rw [bits index inside]
    rw [started] at found'
    exact (presents.work (284 + 1 + index) (by omega) other found').mpr (ok _)
  · rw [started]
    exact (presents.nonce 4 (by decide)).mpr (ok _)
  · refine (Extract.Tier.game_eq_pure_iff_of_runs _ 0 point presented (result :=
      some (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
          ⟨4 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into,
        mle (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
            ⟨4 + 1, "tensor.sumcheck.point", none⟩).toList).map Ext.into)
          (tensorWeight (((recordedIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
            ⟨4, "fused.point", none⟩).toList.tail).map Ext.into)
            (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
              ⟨4 + 1, "tensor.eta", none⟩).into),
        (firstIn (stateBefore pub proof messages pub.ops (fusedIndex pub.layout 5))
          ⟨4 + 1, "tensor.final_claims", none⟩).into)) ?_ _).mpr ?_
    · unfold Tier.challenge
      rw [challenged, started]
      exact runs'
    · rw [segments]
  · show ((point.wire _).data.claim₅) =
      Extract.TraceClaim.ofPoint _ ((point.wire _).data.commit₅) _ _ _ _ _ _
    rw [data, data]
    rfl

end NightstreamFPrime.Spec.Compression.Akita.Replay
