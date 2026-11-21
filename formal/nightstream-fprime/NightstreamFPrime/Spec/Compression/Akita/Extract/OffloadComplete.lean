import NightstreamFPrime.Spec.Compression.Akita.Extract.Offload
import NightstreamFPrime.Spec.Compression.Akita.Extract.CheckComplete

/-! The completeness of the checks of an offloaded fold of Akita (ePrint 2026/1983, revision of
2026-10-15, Section 9 with (132) to (136), and Theorem 10.9: "All honest algebraic checks are
exact identities").

- `offloadDraws` is the number of the draws of `offloadCheck`: the draws of `foldCheck` and the
  `ν_S` rounds of Stage 3 (`offloadDraws_eq`).
- `setupCheck_winsAt` is the completeness of Stage 3: for the true value `σ_S`, the prover sends
  the true round polynomials and `s_{ρ_S} = S̃^♭(ρ_S)`.
- `offloadCheck_winsAt` is the completeness of `offloadCheck`: for a witness with the relation,
  the digit range statement and the binary statement, the prover sends the messages of
  `foldCheck`, then `v_W = w̃(r_2)` and the value `σ` of the weight of the code without a draw
  between them, and then the messages of Stage 3. The checks end in the two true claims
  `(r_2, v_W)` and `(ρ_S, s_{ρ_S})` of Akita (136).
- `Batch.liftedFold_holds` gives the three statements of the quotient-lift route for the witness
  of Akita (76) with the digits of the quotients, and `offBatch_winsAt` is the completeness of the
  checks of the level `offBatch` (Akita Lemma 7.1, item 1). `offBatch_costs_length` counts its
  draws. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita

open NightstreamFPrime.Spec.Compression.Lib NightstreamFPrime.Spec.Compression.Game Polynomial

local instance : Fact (Nat.Prime fieldModulus) := ⟨FieldPrime.fieldModulus_natPrime⟩

namespace Batch

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [Algebra Zq E]
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {st : B.Statement E} {challenge : B.Challenge}
  {μ : ℕ} {fits : B.cells ≤ 2 ^ μ} {base digits : ℕ}

/-- The witness of Akita (76) with the digits of the quotients of digit vectors with the exact
branch relation has the relation of `Batch.liftedReduction`, the digit range statement and the
binary statement (Akita Lemma 7.1, item 1, with Lemma 10.7). These are the hypotheses of the
completeness of the checks of a fold with opening groups. -/
theorem liftedFold_holds (uniform : ∀ g, (B.shape g).certifiedBase = base) (radix : Radix base)
    (covers : fieldModulus ≤ base ^ digits) (room : B.quotientRoom digits ≤ 2 ^ μ) (batch : ℕ)
    (rowsFit : B.rowCount + 1 ≤ 2 ^ batch) {w : B.Witness}
    (satisfies : w.Satisfies β st challenge) :
    (liftedReduction β st challenge μ fits base digits room batch rowsFit).Holds
        ((B.flatRelation β st challenge μ fits).completion (B.gaps digits) base digits w.flat) ∧
      RangeHolds base μ
        ((B.flatRelation β st challenge μ fits).completion (B.gaps digits) base digits w.flat) ∧
      BinaryOn (B.gaps digits).support μ
        ((B.flatRelation β st challenge μ fits).completion (B.gaps digits) base digits w.flat) :=
  ⟨FlatRelation.withQuotients_holds _ radix covers _ _ _
      (fun _ large => B.scalarWeight_of_le β st μ large) (satisfies.flat μ fits),
    FlatRelation.completion_range _ radix
      (satisfies.alphabets.rangeHolds uniform radix.two radix.small μ),
    fun z _ member => by
      obtain ⟨x, low, high, rfl⟩ := (B.gaps digits).support_cases member
      rw [FlatRelation.completion_place _ _ high]
      exact satisfies.alphabets.binaryOn uniform radix.two μ x (high.trans_le fits) ⟨low, high⟩⟩

end Batch

namespace Extract

open NightstreamFPrime.Spec.Compression (Game)
open NightstreamFPrime.Spec.Checkpoint

attribute [local instance low] Classical.propDecidable

/-- The number of the draws of `offloadCheck`: the draws of `foldCheck` and the `ν_S` rounds of
Stage 3. -/
def offloadDraws (μ b : ℕ) (bin : Bool) (ν : ℕ) : ℕ :=
  ν + μ + 1 + binDraws bin + rangeDraws μ b + 1 + 1 + 1

theorem offloadDraws_eq (μ b : ℕ) (bin : Bool) (ν : ℕ) :
    offloadDraws μ b bin ν = foldDraws μ b bin + ν := by
  unfold offloadDraws foldDraws
  omega

section Checks

variable {E : Type} [Field E] [Fintype E] [DecidableEq E] [Algebra Zq E]
  {sendable : (M : Type) → M → Prop} {slots : ℕ}

/-- A game with `WinsTo` from a fresh start has `WinsAt` from no message, for an end condition
that holds for each number of messages after the last draw. -/
theorem winsAt_of_winsTo_each {α : Type} {admits : Type → Prop}
    (each : ∀ (M : Type) (message : M), admits M → sendable M message) (one : 1 ≤ slots)
    {game : Game α} {draws : ℕ} {tail : α → ℕ → ℕ → Prop}
    (wins : WinsTo admits game draws true fun result rest _ => ∀ place, tail result rest place) :
    WinsAt sendable slots game draws 0 tail :=
  (winsAt_of_winsTo each one wins fun _ => one).mono fun _ _ place ended =>
    ended.2.elim fun _ won => won place

omit [DecidableEq E] [Algebra Zq E] in
/-- `drawBin_winsAt` for each number of messages before `drawBin`, when the prover wins after
each number of messages for each coefficient `ζ_bin`. -/
theorem drawBin_winsAt_each {β : Type} (bin : Bool) {bad : Finset E} {next : E → Game β}
    {rest : ℕ} {tail : β → ℕ → ℕ → Prop}
    (each : ∀ ζ place, WinsAt sendable slots (next ζ) rest place tail) (place : ℕ) :
    WinsAt sendable slots (drawBin bin bad next) (rest + binDraws bin) place tail := by
  cases bin
  · exact each 0 place
  · exact fun ζ => each ζ 0

/-- The completeness of Stage 3 (Akita (134), (135)): for the true value
`σ_S = Σ_p S̃^♭(p)·ω̃(p)`, the prover sends the true round polynomials of degree 2 and the value
`s_{ρ_S} = S̃^♭(ρ_S)`, and Stage 3 ends in the claim `(ρ_S, S̃^♭(ρ_S))`. The first round
polynomial is message `index` after the last draw. -/
theorem setupCheck_winsAt {width cap : ℕ} (sends : Sends sendable E width cap)
    (three : 3 ≤ width) (ν : ℕ) (flat : Table) (weight : ℕ → E) {σ : E}
    (sum : σ = cubeSum ν fun p => lifted E flat p * weight p) {index : ℕ}
    (room : index < slots) :
    WinsAt sendable slots (setupCheck ν flat weight σ) ν index fun result _ place =>
      place ≤ index + 1 ∧
        ∃ ρ : List E, ρ.length = ν ∧ result = some (ρ, mle ρ (lifted E flat)) := by
  suffices wins : ∀ rest, WinsAt sendable slots (setupCheck ν flat weight σ) (rest + ν) index
      fun result _ place => place ≤ index + 1 ∧
        ∃ ρ : List E, ρ.length = ν ∧ result = some (ρ, mle ρ (lifted E flat)) by
    have zero := wins 0
    rwa [Nat.zero_add] at zero
  intro rest
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr (sumcheck_winsAt
    (combRound_truth ν 2 ![lifted E flat, weight] productComb productComb_degree)
    (fun coefficients => sends.vector _ coefficients three) rest _ ν [] σ index (by simp)
    (sum.trans (product_truth_nil ν ![lifted E flat, weight]).symm) room
    fun final length place early => ?_)
  simp only [List.length_nil, zero_add] at length
  have last : combTruth ν ![lifted E flat, weight] productComb final =
      mle final (lifted E flat) * mle final weight := by
    have value := product_truth_final final (![lifted E flat, weight] : Fin 2 → ℕ → E)
    rw [length] at value
    exact value
  refine ⟨early.trans_lt room, mle final (lifted E flat), sends.scalar _, ?_⟩
  dsimp only
  rw [ite_eq_left last]
  exact ⟨Nat.succ_le_succ early, final, length, rfl⟩

/-- The completeness of the checks of an offloaded fold (Akita Section 9 with Theorem 10.9): for a
witness with the relation of the ring reduction, the digit range statement and the binary
statement, the prover has a message for each draw sequence. After the fused sum-check it sends
`v_W = w̃(r_2)` and the value `σ = Σ_p S̃^♭(p)·ω̃(p)` of the weight of the code, with no draw
between them, so the first round polynomial of Stage 3 is message 2 after the last draw. The checks
end in the claims `(r_2, w̃(r_2))` and `(ρ_S, S̃^♭(ρ_S))`. -/
theorem offloadCheck_winsAt {width cap : ℕ} (sends : Sends sendable E width cap)
    (R : Reduction E) (b : ℕ) (wide : rangeWidth b ≤ width) (four : 4 ≤ width)
    (room : 2 < slots) (support : ℕ → Prop) [DecidablePred support] {bin : Bool} {w : Table}
    (holds : R.Holds w) (ranged : RangeHolds b R.vars w) (binary : BinaryOn support R.vars w)
    (ν : ℕ) (flat : Table) (scan : SetupScan R ν flat) :
    WinsAt sendable slots (offloadCheck R b support bin w ν flat scan)
      (offloadDraws R.vars b bin ν) 0 fun result _ _ =>
        ∃ r2 ρ : List E, r2.length = R.vars ∧ ρ.length = ν ∧
          result = some ((r2, mle r2 (lifted E w)), (ρ, mle ρ (lifted E flat))) := by
  intro α τ0 τ1
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr (winsAt_of_winsTo_each
    (admits := fun M => ∀ message : M, sendable M message) (fun _ message all => all message)
    (by omega) (rangeCheck_winsTo _ sends.scalar b R.vars
      (fun _ small message => sends.vector _ message (small.trans wide)) support ranged
      (List.ofFn τ0) List.length_ofFn _ _ fun r length place => ?_))
  dsimp only
  refine drawBin_winsAt_each bin (fun ζ place => ?_) place
  intro γ
  have zero : mle r (supportTable support w) = (0 : E) := by
    rw [mle_congr r (u := fun _ => 0) fun x small => by
      unfold supportTable
      split
      · rename_i member
        rw [derived, binary x (length ▸ small) member, map_zero]
      · rfl]
    exact mle_const r 0
  have true_claim : (R.rows α).batchTarget (List.ofFn τ1) + γ * mle r (derived E w) =
      combTruth R.vars (fusedTables w ((R.rows α).batchWeight (List.ofFn τ1))
        (virtWeight support r γ ζ)) (fusedComb E) [] := by
    have virt := fusedTruth_virt support w ((R.rows α).batchWeight (List.ofFn τ1)) r γ ζ
    rw [length] at virt
    have batched : (R.rows α).Batched R.vars w (List.ofFn τ1) := R.batched_of_holds holds α _
    rw [virt, zero, mul_zero, add_zero]
    exact congrArg (· + γ * mle r (derived E w)) batched.symm
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr (sumcheck_winsAt
    (fused_roundTruth R.vars w _ (virtWeight support r γ ζ))
    (fun coefficients => sends.vector _ coefficients four) ν _ R.vars [] _ 0 (by simp)
    true_claim (by omega) fun final long place early => ?_)
  simp only [List.length_nil, zero_add] at long
  obtain rfl : place = 0 := Nat.le_zero.mp early
  have last := fusedTruth_final final w ((R.rows α).batchWeight (List.ofFn τ1))
    (virtWeight support r γ ζ)
  rw [long, scan.split α τ1 final long] at last
  refine ⟨by omega, mle final (lifted E w), sends.scalar _, ?_⟩
  refine ⟨by omega, cubeSum ν (fun p => lifted E flat p * scan.weight α (List.ofFn τ1) final p),
    sends.scalar _, ?_⟩
  dsimp only
  rw [ite_eq_left last]
  refine (winsAt_bind sendable slots _ _ _ _ _).mpr ((setupCheck_winsAt sends (by omega) ν flat
    _ rfl room).mono fun result _ _ ended => ?_)
  obtain ⟨_, ρ, count, rfl⟩ := ended
  exact ⟨final, ρ, long, count, rfl⟩

end Checks

section Groups

variable {n : ℕ} {B : Batch (Fin n)} {E : Type} [Field E] [Fintype E] [DecidableEq E]
  [Algebra Zq E] {Wire Coord : Type} {sendable : (M : Type) → M → Prop} {slots width cap : ℕ}
  {β : ∀ g, Module.Basis (Fin (B.planes g)) Zq E} {key : (g : Fin n) → Key (B.shape g)}
  {openKey : B.OpenKey} {claim : Wire → B.Claim E} {μ : ℕ} {fits : B.cells ≤ 2 ^ μ}
  {base digits : ℕ} {room : B.quotientRoom digits ≤ 2 ^ μ} {batch : ℕ}
  {rowsFit : B.rowCount + 1 ≤ 2 ^ batch} {block : B.Block → Coord}
  {decode : (g : Fin n) → E → Option (Rq (B.carrierDegree g))} {read : Reader Wire}
  {mass : Fin n → ℕ} {start ν : ℕ} {flat : Table}
  {scan : ∀ (wire : Wire) (challenge : B.Challenge),
    SetupScan (Batch.liftedReduction β ((claim wire).statement key openKey) challenge μ fits base
      digits room batch rowsFit) ν flat}

/-- The completeness of the checks of the level `offBatch` (Akita Lemma 7.1, item 1, with
Section 9 and Theorem 10.9): for digit vectors with the exact branch relation of the claim of a
wire message, the checks end in the claim `(r_2, w̃(r_2))` for the flat witness with the digits of
the quotients and in the claim `(ρ_S, S̃^♭(ρ_S))` for the setup prefix. `space`: a wire message
has room for the two values after the fused sum-check and the first round of Stage 3. -/
theorem offBatch_winsAt (sends : Sends sendable E width cap) (wide : rangeWidth base ≤ width)
    (four : 4 ≤ width) (uniform : ∀ g, (B.shape g).certifiedBase = base) (radix : Radix base)
    (covers : fieldModulus ≤ base ^ digits) (space : 2 < slots) {wire : Wire}
    {challenge : (b : B.Block) → Rq (B.carrierDegree b.1)} {w : B.Witness}
    (satisfies : w.Satisfies β ((claim wire).statement key openKey)
      fun g i => challenge ⟨g, i⟩) :
    WinsAt sendable slots
      ((offBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
        start ν flat scan).check wire challenge
        ((B.flatRelation β ((claim wire).statement key openKey) (fun g i => challenge ⟨g, i⟩)
          μ fits).completion (B.gaps digits) base digits w.flat))
      (offloadDraws μ base B.compressed ν) 0 fun result _ _ =>
        ∃ r2 ρ : List E, r2.length = μ ∧ ρ.length = ν ∧
          result = some ((r2, mle r2 (lifted E
            ((B.flatRelation β ((claim wire).statement key openKey)
              (fun g i => challenge ⟨g, i⟩) μ fits).completion (B.gaps digits) base digits
              w.flat))), (ρ, mle ρ (lifted E flat))) :=
  have holds := Batch.liftedFold_holds uniform radix covers room batch rowsFit satisfies
  offloadCheck_winsAt sends _ base wide four space _ holds.1 holds.2.1 holds.2.2 ν flat
    (scan wire fun g i => challenge ⟨g, i⟩)

theorem offBatch_costs_length :
    (offBatch β key openKey claim μ fits base digits room batch rowsFit block decode read mass
      start ν flat scan).costs.length = offloadDraws μ base B.compressed ν := by
  show (foldCosts E _ μ batch base B.compressed ++ setupCosts E ν).length = _
  rw [List.length_append, foldCosts_length, setupCosts, List.length_replicate, offloadDraws_eq]

end Groups

end Extract

end NightstreamFPrime.Spec.Compression.Akita
