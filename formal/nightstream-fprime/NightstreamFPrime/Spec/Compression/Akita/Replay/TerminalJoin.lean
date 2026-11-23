import NightstreamFPrime.Spec.Compression.Akita.Replay.SamplerStage
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalNorm

/-! The terminal operation of the replay (`Replay/Transcript.lean`) against `terminalRun`
(`Replay/Terminal.lean`) and the terminal of the specification (`TerminalStatement.Accepts`,
Akita Section 8.2).

- `step_terminal_run`: with the setup prefix of the seed, the step of the terminal records the
  values of `terminalRun` on the recorded messages and values, or it rejects at the site of the
  check of the code that fails. This is `step_stage` for the terminal.
- `toRq_ofSparse_ring`: the dense form of a sparse challenge (`Poly.ofSparse`) is its element
  `Sparse.ring` of `R_{q,d}`. `isUnit_drawn`: each challenge that the sampler draws for a group of
  degree `2^{n+1}` with a coefficient `±1` and stride 1 is a unit (`isUnit_of_short`: it is not
  zero and `‖c‖_∞ ≤ 2`).
- `step_terminal_accepts`: for a payload that decodes and fold challenges that the sampler drew,
  the step accepts exactly when `‖z‖_2² ≤ S_max`, the challenges are valid, and some last blocks
  of `e` and `t` with the digests of the proof make the statement of the terminal accept
  (`terminalRun_ok_iff` with its unit conjunct discharged). -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita

namespace Poly

variable {d : ℕ} [NeZero d]

/-- The dense form of a sparse challenge is the element of `R_{q,d}` with its coefficients. -/
theorem toRq_ofSparse_ring (c : Sparse) : toRq d (ofSparse d c) = Sparse.ring d c := by
  refine ext_coeff fun k small => ?_
  rw [coeff_toRq _ small, Sparse.coeff_ring _ small, ofSparse, get_ofFn _ small]
  induction c with
  | nil => simp [Sparse.value, Fq.toZq_zero]
  | cons x c ih =>
      rw [List.foldr_cons, Sparse.value_cons]
      by_cases at_ : x.1 = k
      · rw [ite_eq_left at_, ite_eq_left at_, Fq.toZq_add, Fq.toZq_ofInt, ih, Int.cast_add]
      · rw [ite_eq_right at_, ite_eq_right at_, ih, zero_add]

end Poly

/-- Each challenge of the sampler is a valid sparse challenge of the degree of the group. -/
theorem sample_valid {G : FoldGroup} {bytes : ByteArray} {cs : List Sparse}
    (sampled : sample G bytes = some cs) : ∀ c ∈ cs, Sparse.Valid G.degree c := by
  cases filtered : G.filtered with
  | true =>
      obtain ⟨rfl, -⟩ := sample_filtered filtered sampled
      intro c member
      obtain ⟨j, -, decoded⟩ := List.mem_filterMap.mp (List.mem_of_mem_take member)
      exact candidate_valid (Option.filter_eq_some_iff.mp decoded).1
  | false =>
      simp only [sample, filtered, Bool.false_eq_true, ite_false, candidate_table] at sampled
      split_ifs at sampled
      cases sampled
      intro c member
      obtain ⟨j, -, found⟩ := List.mem_filterMap.mp member
      exact candidate_valid found

/-- A challenge of stride 1 in the ring of `A` is the challenge of the sampler. -/
theorem ambient_one {G : FoldGroup} (positive : 0 < G.degree) (same : G.ring = G.degree)
    (c : Sparse) : ambient G c = c := by
  rw [ambient, same, Nat.div_self positive]
  simp

/-- Each challenge that the sampler draws for a group of degree `2^{n+1}` with a coefficient
`±1` and stride 1 is a unit of `R_{q,d}` (`isUnit_of_short`). -/
theorem isUnit_drawn {G : FoldGroup} {d : ℕ} [NeZero d] (n : ℕ) (power : 2 ^ (n + 1) = d)
    (degree : G.degree = d) (ones : 0 < G.pm1) (same : G.ring = G.degree) {bytes : ByteArray}
    {cs : List Sparse} (drawn : foldChallenges G bytes = some cs) {c : Sparse} (member : c ∈ cs) :
    IsUnit (Poly.toRq d (Poly.ofSparse d c)) := by
  subst degree
  obtain ⟨cs₀, sampled, rfl⟩ := Option.map_eq_some_iff.mp drawn
  rw [List.map_congr_left fun c _ => ambient_one (Nat.pos_of_neZero _) same c, List.map_id']
    at member
  obtain ⟨⟨bounded, count, -⟩, -⟩ := (sample_mem sampled).2 c member
  rw [Poly.toRq_ofSparse_ring]
  refine isUnit_of_short n power (fun zero => ?_) (lt_of_le_of_lt
    (Nat.mul_le_mul_left 2 (Nat.pow_le_pow_left bounded 2)) (by decide))
  rw [zero] at count
  simp [coeff_zero] at count
  omega

section Join

variable {E : Type} [CommRing E] [Algebra Zq E] [Degree2 E]

namespace Replayer

/-- The state after the terminal checks passed with the values `out`. -/
def finished (s : Replayer) (level : ℕ) (out : TerminalValues) : Replayer :=
  { s with derived := { s.derived with
      terminal := some out, checks := s.derived.checks ++ terminalSites level } }

theorem step_terminal (pub : Public) (proof : ByteArray) (messages : Array Message) (i level : ℕ)
    (s : Replayer) :
    ((step pub proof messages i (.terminal level)).run).run s =
      ((terminalWith i level (terminalValue pub s level)).run).run s :=
  bind_get_run (fun x => terminalWith i level (terminalValue pub x level)) s

/-- With the setup prefix of the seed, the step of the terminal records the values of
`terminalRun` on the recorded messages and values, or it rejects at the site of the check of the
code that fails. -/
theorem step_terminal_run (pub : Public) (proof : ByteArray) (messages : Array Message)
    (i level : ℕ) (s : Replayer) {setup : Array Fq}
    (expanded : pub.setupFor pub.layout.terminal.setupCount = some setup) :
    ((step pub proof messages i (.terminal level)).run).run s =
      match terminalRun pub.layout.terminal.shape pub.layout.terminal.limit
          pub.layout.terminal.params setup (terminalMessagesIn s level)
          (terminalChallengesIn s level) with
      | .ok out => (.ok (), finished s level out)
      | .error site => (.error (.reject i site), s) := by
  rw [step_terminal]
  unfold terminalValue
  dsimp only
  rw [expanded, Option.map_some]
  cases terminalRun pub.layout.terminal.shape pub.layout.terminal.limit pub.layout.terminal.params
    setup (terminalMessagesIn s level) (terminalChallengesIn s level) <;> rfl

/-- The step of the terminal accepts exactly when the terminal of the specification accepts:
for a payload that decodes and fold challenges that the sampler drew for the terminal group
(degree `2^{n+1}`, a coefficient `±1`, stride 1), `‖z‖_2² ≤ S_max`, the challenges are valid,
and some last blocks of `e` and `t` with the digests of the proof make the statement of the
terminal accept the full partials, the challenges and `z` (Akita Section 8.2, (128) to (131)).
The last challenge is a unit (`isUnit_drawn`), so `terminalRun_ok_iff` has no unit condition
here. -/
theorem step_terminal_accepts (pub : Public) (proof : ByteArray) (messages : Array Message)
    (i level : ℕ) (s : Replayer) {setup : Array Fq}
    (expanded : pub.setupFor pub.layout.terminal.setupCount = some setup)
    (n : ℕ) (power : 2 ^ (n + 1) = pub.layout.terminal.shape.innerDegree)
    (quarter : 4 * (pub.layout.terminal.shape.innerDegree / 4) =
      pub.layout.terminal.shape.innerDegree)
    (two : ((2 * (pub.layout.terminal.shape.innerDegree / 4) : ℕ) : Zq) ≠ 0)
    (blocks : 1 ≤ pub.layout.terminal.shape.blocks)
    (degree : pub.layout.terminal.group.degree = pub.layout.terminal.shape.innerDegree)
    (ones : 0 < pub.layout.terminal.group.pm1)
    (same : pub.layout.terminal.group.ring = pub.layout.terminal.group.degree)
    {bytes : ByteArray}
    (drawn : foldChallenges pub.layout.terminal.group bytes = some (drawnIn s level 0))
    {z : Array ℤ}
    (decoded : Golomb.decode pub.layout.terminal.params (terminalMessagesIn s level).payload =
      some z) :
    (∃ s', ((step pub proof messages i (.terminal level)).run).run s = (.ok (), s')) ↔
      centeredNormSq z ≤ pub.layout.terminal.limit ∧
      ((terminalChallengesIn s level).fold.length = pub.layout.terminal.shape.blocks ∧
        (terminalChallengesIn s level).fold.all
          (sparseValid pub.layout.terminal.shape.innerDegree) = true) ∧
      ∃ (eLast : Rq pub.layout.terminal.shape.innerDegree)
        (tLast : ℕ → Rq pub.layout.terminal.shape.innerDegree),
        ringDigest eLast = (terminalMessagesIn s level).eDigest ∧
        rowsDigestOf pub.layout.terminal.shape tLast = (terminalMessagesIn s level).tDigest ∧
        (statementOf (E := E) pub.layout.terminal.shape quarter setup (terminalMessagesIn s level)
          (terminalChallengesIn s level) tLast).Accepts
          (partialsOf pub.layout.terminal.shape (terminalMessagesIn s level) eLast)
          (challengesOf pub.layout.terminal.shape (terminalChallengesIn s level))
          (zSource pub.layout.terminal.shape z) := by
  set T := pub.layout.terminal
  have accepts : (∃ s', ((step pub proof messages i (.terminal level)).run).run s = (.ok (), s')) ↔
      ∃ out, terminalRun T.shape T.limit T.params setup (terminalMessagesIn s level)
        (terminalChallengesIn s level) = .ok out := by
    rw [step_terminal_run pub proof messages i level s expanded]
    constructor
    · rintro ⟨s', ran⟩
      split at ran
      · rename_i out found
        exact ⟨out, found⟩
      · cases ran
    · rintro ⟨out, found⟩
      rw [found]
      exact ⟨_, rfl⟩
  have power' : 2 ^ Nat.log 2 T.shape.innerDegree = T.shape.innerDegree := by
    rw [← power, Nat.log_pow (by norm_num)]
  rw [accepts, terminalRun_ok_iff (E := E) power' quarter two blocks T.limit T.params setup _ _
    decoded]
  have unit : (terminalChallengesIn s level).fold.length = T.shape.blocks →
      IsUnit (Poly.toRq T.shape.innerDegree
        (challenge T.shape (terminalChallengesIn s level) (T.shape.blocks - 1))) := by
    intro length
    have fold : (terminalChallengesIn s level).fold = drawnIn s level 0 := rfl
    have member : (terminalChallengesIn s level).fold.getD (T.shape.blocks - 1) [] ∈
        drawnIn s level 0 := by
      rw [List.getD_eq_getElem _ _ (by rw [length]; omega), ← fold]
      exact List.getElem_mem _
    exact isUnit_drawn n power degree ones same drawn member
  constructor
  · rintro ⟨norm, valid, -, rest⟩
    exact ⟨norm, valid, rest⟩
  · rintro ⟨norm, valid, rest⟩
    exact ⟨norm, valid, unit valid.1, rest⟩

end Replayer

end Join

end NightstreamFPrime.Spec.Compression.Akita.Replay
