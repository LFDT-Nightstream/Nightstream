import CompressionLeanRust.Rounds
import CompressionLean.Ports.Shrink.Game
import CompressionLean.Shrink.Ccs
import CompressionLean.Game.Compose

/-! The statement that the shrink verifier (`shrink/mod.rs::verify`) must meet once it moves to
`neo-verified` (`.audit/aeneas-phase5.md`).

The verifier reads its transcript and its WHIR opening through a `Channel`, and the program's
shape run through an `Evaluate`. `VerifyFn` is the type that Aeneas gives such a function, with
each slice read as a list. The statement runs the verifier on `recorder`, a channel that records
each call and answers each draw with an arbitrary `oracle` of the calls so far. `Sound` says: if
the verifier accepts, then its channel calls are `schedule`'s, the opening succeeded, the program's
public words are the statement, and `shrinkGame` (the game of the claims) accepts the play whose
messages are the proof's and whose draws are the verifier's draws. Nothing is assumed of the proof
or of the oracle. The opening is assumed sound (`WhirSound`), as the claims assume. `Complete` says
that the verifier accepts every honest run, and that one exists. `rust_bound` turns a bound on
the claims' break event (`GameBreak`) into the same bound on `RustBreak`, where the Rust verifier
accepts. -/

namespace CompressionLeanRust.Verifier

open Aeneas Aeneas.Std Result
open neo_verified
open CompressionLean CompressionLean.Shrink CompressionLean.Lib
open NightstreamFPrime.Spec (goldilocksModulus)
open Polynomial

local notation "ι" => algebraMap Gl Cubic

/-! ## The verifier's interface -/

/-- The calls that `verify` makes on its caller's transcript and WHIR opening (the trait `Channel`). -/
structure Channel (σ : Type) where
  observeWords : σ → List field.Gl → Result σ
  commit : σ → Result σ
  observe : σ → List field.Ext → Result σ
  sample : σ → Result (field.Ext × σ)
  openAt : σ → List field.Ext → Result (Option field.Ext × σ)

/-- The program's shape run at `(rx, ry, ρ)`: the glue rows' matrix value, the counts and the
public words (the trait `ShapeRun`). -/
abbrev Evaluate : Type := List field.Ext → List field.Ext → field.Ext → Cubic × Counts × List Gl

/-- The sum-check messages of a shrink proof. The WHIR commitment and opening stay behind `Channel`. -/
structure Rounds where
  outer : List (List field.Ext)
  values : Fin 8 → field.Ext
  inner : List (List field.Ext)

/-- The moved `verify`: `true` accepts, and a Rust panic is a `fail`. -/
abbrev VerifyFn : Type 1 :=
  ∀ {σ : Type}, Channel σ → Evaluate → Counts → List field.Gl → Rounds → σ → Result (Bool × σ)

/-! ## A recording channel -/

/-- One channel call, with each word read as a field value. -/
inductive Event where
  | words (ws : List Gl)
  | commit
  | observe (vs : List Cubic)
  | sample (c : Cubic)
  | opened (point : List Cubic) (value : Option Cubic)

abbrev History : Type := List Event

/-- The channel that records each call. `oracle` answers each draw from the calls so far, and
`opening` answers each WHIR opening. -/
noncomputable def recorder (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext) :
    Channel History where
  observeWords h ws := ok (h ++ [.words (ws.map glOf)])
  commit h := ok (h ++ [.commit])
  observe h vs := ok (h ++ [.observe (vs.map cubicOf)])
  sample h := ok (oracle h, h ++ [.sample (cubicOf (oracle h))])
  openAt h point := ok (opening h (point.map cubicOf),
    h ++ [.opened (point.map cubicOf) ((opening h (point.map cubicOf)).map cubicOf)])

theorem recorder_observeWords (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext)
    (h : History) (ws : List field.Gl) : (recorder oracle opening).observeWords h ws = ok (h ++ [.words (ws.map glOf)]) :=
  rfl

theorem recorder_commit (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext)
    (h : History) : (recorder oracle opening).commit h = ok (h ++ [.commit]) :=
  rfl

theorem recorder_observe (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext)
    (h : History) (vs : List field.Ext) : (recorder oracle opening).observe h vs = ok (h ++ [.observe (vs.map cubicOf)]) :=
  rfl

theorem recorder_sample (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext)
    (h : History) : (recorder oracle opening).sample h = ok (oracle h, h ++ [.sample (cubicOf (oracle h))]) :=
  rfl

theorem recorder_openAt (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext)
    (h : History) (point : List field.Ext) :
    (recorder oracle opening).openAt h point = ok (opening h (point.map cubicOf),
      h ++ [.opened (point.map cubicOf) ((opening h (point.map cubicOf)).map cubicOf)]) :=
  rfl

/-- The draws of a history, in order. -/
def drawsIn : History → List (Σ C : Type, C)
  | [] => []
  | .sample c :: rest => ⟨Cubic, c⟩ :: drawsIn rest
  | .words _ :: rest | .commit :: rest | .observe _ :: rest | .opened _ _ :: rest => drawsIn rest

/-- The value of the first opening of a history. -/
def openedIn : History → Option Cubic
  | [] => none
  | .opened _ v :: _ => v
  | .words _ :: rest | .commit :: rest | .observe _ :: rest | .sample _ :: rest => openedIn rest

/-! ## The schedule of channel calls -/

def samples {σ : Type} (ch : Channel σ) : ℕ → σ → Result (List field.Ext × σ)
  | 0, st => ok ([], st)
  | k + 1, st => do
    let first ← ch.sample st
    let rest ← samples ch k first.2
    ok (first.1 :: rest.1, rest.2)

/-- Each round's message, then its draw. -/
def sends {σ : Type} (ch : Channel σ) : List (List field.Ext) → σ → Result (List field.Ext × σ)
  | [], st => ok ([], st)
  | m :: ms, st => do
    let st ← ch.observe st m
    let first ← ch.sample st
    let rest ← sends ch ms first.2
    ok (first.1 :: rest.1, rest.2)

/-- What the channel gives `verify`. -/
structure Draws where
  tau : List field.Ext
  rx : List field.Ext
  rho : field.Ext
  ry : List field.Ext
  opened : Option field.Ext

/-- The channel calls of `verify`, in the order of `shrink/mod.rs::verify`: the statement, the WHIR
commitment, `m` zero-check draws, each outer round's message and draw, the matrix values and the
batching draw, each inner round's message and draw, and the opening at the first `n` inner draws. -/
def schedule {σ : Type} (ch : Channel σ) (m n : ℕ) (stmt : List field.Gl) (π : Rounds) (st : σ) :
    Result (Draws × σ) := do
  let st ← ch.observeWords st stmt
  let st ← ch.commit st
  let tau ← samples ch m st
  let rx ← sends ch π.outer tau.2
  let st ← ch.observe rx.2 (List.ofFn π.values)
  let rho ← ch.sample st
  let ry ← sends ch π.inner rho.2
  let opened ← ch.openAt ry.2 (ry.1.take n)
  ok (⟨tau.1, rx.1, rho.1, ry.1, opened.1⟩, opened.2)

/-! ## The play of a run in `shrinkGame` -/

/-- Rounds as the game reads them: `d + 1` values each. -/
noncomputable def roundMsgsOf (d : ℕ) (rounds : List (List field.Ext)) : List (Σ M : Type, M) :=
  rounds.map fun m => ⟨Fin (d + 1) → Cubic, messageOf d m⟩

noncomputable def valuesOf (π : Rounds) : Fin 8 → Cubic := fun j => cubicOf (π.values j)

/-- The messages of a play: the instance, the WHIR candidates, the outer rounds, the matrix values,
the inner rounds and the opened value. -/
noncomputable def msgsOf {I : Type} (i : I) {L : ℕ} (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (π : Rounds)
    (opened : Cubic) : List (Σ M : Type, M) :=
  ⟨I, i⟩ :: ⟨{l : List (ℕ → Gl) // l.length ≤ L}, cands⟩ ::
    (roundMsgsOf 7 π.outer ++ ⟨Fin 8 → Cubic, valuesOf π⟩ :: (roundMsgsOf 1 π.inner ++ [⟨Cubic, opened⟩]))

/-- The draws of a play, in order. -/
noncomputable def drawsOf (d : Draws) : List (Σ C : Type, C) :=
  (d.tau ++ d.rx ++ d.rho :: d.ry).map fun c => ⟨Cubic, cubicOf c⟩

/-- The shape run of `run`, as `Ports/Shrink/Evaluate.lean` reads it. -/
noncomputable def evalOf (T : Template Gl) (s : Counts) (run : Run Gl) : Evaluate :=
  fun rx ry ρ => (glueTotal T s (rx.map cubicOf) (ry.map cubicOf) (cubicOf ρ) 0 run.calls, run.counts, run.publics)

/-- Assumption (`TRUST.md` §3, WHIR): the opening returns a value only when a listed candidate has
that value at the point. -/
def WhirSound (cands : List (ℕ → Gl)) (opening : History → List Cubic → Option field.Ext) : Prop :=
  ∀ h point v, opening h point = some v → ∃ w ∈ cands, cubicOf v = mle point fun c => ι (w c)

/-! ## The statement -/

/-- An accepting run of `verify` is an accepting play of `shrinkGame`, for every proof, statement,
oracle and candidate list. -/
def Sound (verify : VerifyFn) (T : Template Gl) : Prop :=
  ∀ {I : Type} [Nonempty I] (prog : I → Run Gl) (s : Counts) (L : ℕ) (i : I) (stmt : List field.Gl) (π : Rounds)
    (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
    (opening : History → List Cubic → Option field.Ext) (h : History),
    WhirSound cands.1 opening →
    verify (recorder oracle opening) (evalOf T s (prog i)) s stmt π [] = ok (true, h) →
    ∃ d w, schedule (recorder oracle opening) (Layout.rowVars T s) (Layout.cellVars T s) stmt π [] = ok (d, h) ∧
      d.opened = some w ∧ (prog i).publics = stmt.map glOf ∧
      ∃ tr, Game.Reaches (shrinkGame (K := Cubic) T s prog L) (msgsOf i cands π (cubicOf w)) (drawsOf d) (some tr)

/-- The sizes fit the Rust integers on any target: `usize` has at least 32 bits. -/
def Small (T : Template Gl) (s : Counts) (stmt : List field.Gl) : Prop :=
  partsEnd T.rows s.blocks + s.glueRows < 2 ^ 32 ∧ partsEnd T.cells s.blocks + s.glueCells < 2 ^ 32 ∧
    Layout.one T s + s.publics < 2 ^ 32 ∧ stmt.length < 2 ^ 32

/-- The rounds have the counts and the lengths that the shape fixes. -/
def RoundsFit (T : Template Gl) (s : Counts) (π : Rounds) : Prop :=
  π.outer.length = Layout.rowVars T s ∧ (∀ m ∈ π.outer, m.length = 8) ∧
    π.inner.length = Layout.cellVars T s + 1 ∧ ∀ m ∈ π.inner, m.length = 2

/-- An honest run: the sizes fit, the statement is canonical and is the program's public words,
the rounds have their shape, the opening succeeds, and `shrinkGame` accepts the play. -/
def Honest (T : Template Gl) {I : Type} [Nonempty I] (prog : I → Run Gl) (s : Counts) (L : ℕ) (i : I)
    (stmt : List field.Gl) (π : Rounds) (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
    (opening : History → List Cubic → Option field.Ext) : Prop :=
  Small T s stmt ∧ (∀ word ∈ stmt, word.val < goldilocksModulus) ∧ (prog i).publics = stmt.map glOf ∧
    RoundsFit T s π ∧
    ∃ d h w, schedule (recorder oracle opening) (Layout.rowVars T s) (Layout.cellVars T s) stmt π [] = ok (d, h) ∧
      d.opened = some w ∧
      ∃ tr, Game.Reaches (shrinkGame (K := Cubic) T s prog L) (msgsOf i cands π (cubicOf w)) (drawsOf d) (some tr)

/-- `verify` accepts every honest run, and an honest run exists. Without it, a verifier that
rejects everything would be `Sound`. -/
def Complete (verify : VerifyFn) (T : Template Gl) : Prop :=
  (∃ (prog : Unit → Run Gl) (s : Counts) (L : ℕ) (stmt : List field.Gl) (π : Rounds)
      (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
      (opening : History → List Cubic → Option field.Ext),
      WhirSound cands.1 opening ∧ Honest T prog s L () stmt π cands oracle opening) ∧
    ∀ {I : Type} [Nonempty I] (prog : I → Run Gl) (s : Counts) (L : ℕ) (i : I) (stmt : List field.Gl) (π : Rounds)
      (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (oracle : History → field.Ext)
      (opening : History → List Cubic → Option field.Ext),
      Honest T prog s L i stmt π cands oracle opening →
      ∃ h, verify (recorder oracle opening) (evalOf T s (prog i)) s stmt π [] = ok (true, h)

def Simulates (verify : VerifyFn) (T : Template Gl) : Prop := Sound verify T ∧ Complete verify T

/-! ## The recorder's history -/

theorem samples_recorder (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext) :
    ∀ (k : ℕ) (h : History), ∃ cs, samples (recorder oracle opening) k h =
      ok (cs, h ++ cs.map fun c => .sample (cubicOf c)) ∧ cs.length = k ∧ ∀ c ∈ cs, ∃ h', c = oracle h'
  | 0, h => ⟨[], by simp [samples], rfl, by simp⟩
  | k + 1, h => by
      obtain ⟨cs, run, length, drawn⟩ := samples_recorder oracle opening k (h ++ [.sample (cubicOf (oracle h))])
      refine ⟨oracle h :: cs, ?_, by simp [length], ?_⟩
      · simp only [samples, recorder_sample, bind_ok]
        rw [run]
        simp only [bind_ok, List.map_cons, List.append_assoc, List.singleton_append]
      · intro c member
        rcases List.mem_cons.mp member with rfl | inside
        · exact ⟨h, rfl⟩
        · exact drawn c inside

/-- The events of sending `ms` with draws `cs`. -/
noncomputable def sent (ms : List (List field.Ext)) (cs : List field.Ext) : History :=
  (ms.zip cs).flatMap fun p => [.observe (p.1.map cubicOf), .sample (cubicOf p.2)]

theorem sends_recorder (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext) :
    ∀ (ms : List (List field.Ext)) (h : History), ∃ cs, sends (recorder oracle opening) ms h =
      ok (cs, h ++ sent ms cs) ∧ cs.length = ms.length ∧ ∀ c ∈ cs, ∃ h', c = oracle h'
  | [], h => ⟨[], by simp [sends, sent], rfl, by simp⟩
  | m :: ms, h => by
      obtain ⟨cs, run, length, drawn⟩ := sends_recorder oracle opening ms
        (h ++ [.observe (m.map cubicOf)] ++ [.sample (cubicOf (oracle (h ++ [.observe (m.map cubicOf)])))])
      refine ⟨oracle (h ++ [.observe (m.map cubicOf)]) :: cs, ?_, by simp [length], ?_⟩
      · simp only [sends, recorder_observe, recorder_sample, bind_ok]
        rw [run]
        simp only [bind_ok, sent, List.zip_cons_cons, List.flatMap_cons, List.append_assoc, List.cons_append,
          List.nil_append]
      · intro c member
        rcases List.mem_cons.mp member with rfl | inside
        · exact ⟨_, rfl⟩
        · exact drawn c inside

/-- The history of `schedule` on `recorder` from no call. -/
noncomputable def scheduled (stmt : List field.Gl) (π : Rounds) (n : ℕ) (d : Draws) : History :=
  [.words (stmt.map glOf), .commit] ++ d.tau.map (fun c => .sample (cubicOf c)) ++ sent π.outer d.rx ++
    [.observe ((List.ofFn π.values).map cubicOf), .sample (cubicOf d.rho)] ++ sent π.inner d.ry ++
    [.opened ((d.ry.take n).map cubicOf) (d.opened.map cubicOf)]

theorem schedule_recorder (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext)
    (m n : ℕ) (stmt : List field.Gl) (π : Rounds) :
    ∃ d, schedule (recorder oracle opening) m n stmt π [] = ok (d, scheduled stmt π n d) ∧
      d.tau.length = m ∧ d.rx.length = π.outer.length ∧ d.ry.length = π.inner.length ∧
      (∀ c ∈ d.tau ++ d.rx ++ d.rho :: d.ry, ∃ h', c = oracle h') ∧
      ∃ h', d.opened = opening h' ((d.ry.take n).map cubicOf) := by
  obtain ⟨tau, run1, len1, from1⟩ := samples_recorder oracle opening m ([] ++ [.words (stmt.map glOf)] ++ [.commit])
  obtain ⟨rx, run2, len2, from2⟩ := sends_recorder oracle opening π.outer
    ([] ++ [.words (stmt.map glOf)] ++ [.commit] ++ tau.map fun c => .sample (cubicOf c))
  let h3 := [] ++ [.words (stmt.map glOf)] ++ [.commit] ++ (tau.map fun c => .sample (cubicOf c)) ++
    sent π.outer rx ++ [.observe ((List.ofFn π.values).map cubicOf)]
  obtain ⟨ry, run3, len3, from3⟩ := sends_recorder oracle opening π.inner (h3 ++ [.sample (cubicOf (oracle h3))])
  let h4 := h3 ++ [.sample (cubicOf (oracle h3))] ++ sent π.inner ry
  refine ⟨⟨tau, rx, oracle h3, ry, opening h4 ((ry.take n).map cubicOf)⟩, ?_, len1, len2, len3, ?_, h4, rfl⟩
  · simp only [schedule, recorder_observeWords, recorder_commit, bind_ok]
    rw [run1]
    simp only [bind_ok]
    rw [run2]
    simp only [bind_ok, recorder_observe, recorder_sample]
    rw [run3]
    simp only [bind_ok, recorder_openAt, scheduled, h3, h4, List.nil_append, List.append_assoc, List.cons_append]
  · intro c member
    simp only [List.mem_append, List.mem_cons] at member
    rcases member with (inTau | inRx) | rho | inRy
    · exact from1 c inTau
    · exact from2 c inRx
    · exact ⟨h3, rho⟩
    · exact from3 c inRy

theorem drawsIn_append (a b : History) : drawsIn (a ++ b) = drawsIn a ++ drawsIn b := by
  induction a with
  | nil => rfl
  | cons e rest ih => cases e <;> simp [drawsIn, ih]

theorem drawsIn_sent : ∀ (ms : List (List field.Ext)) (cs : List field.Ext), ms.length = cs.length →
    drawsIn (sent ms cs) = cs.map fun c => ⟨Cubic, cubicOf c⟩
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | m :: ms, c :: cs, h => by
      simp only [sent, List.zip_cons_cons, List.flatMap_cons]
      rw [drawsIn_append]
      simp only [drawsIn, List.map_cons, List.cons_append, List.nil_append]
      exact congrArg _ (drawsIn_sent ms cs (by simpa using h))

theorem drawsIn_scheduled (stmt : List field.Gl) (π : Rounds) (n : ℕ) (d : Draws)
    (hrx : d.rx.length = π.outer.length) (hry : d.ry.length = π.inner.length) :
    drawsIn (scheduled stmt π n d) = drawsOf d := by
  simp only [scheduled, drawsIn_append, drawsIn_sent _ _ hrx.symm, drawsIn_sent _ _ hry.symm, drawsOf, List.map_append,
    List.map_cons]
  have tau : ∀ l : List field.Ext, drawsIn (l.map fun c => Event.sample (cubicOf c)) = l.map fun c => ⟨Cubic, cubicOf c⟩ := by
    intro l
    induction l with
    | nil => rfl
    | cons c rest ih => simp [drawsIn, ih]
  simp [tau, drawsIn]

theorem openedIn_append (a b : History) (clean : ∀ p v, Event.opened p v ∉ a) : openedIn (a ++ b) = openedIn b := by
  induction a with
  | nil => rfl
  | cons e rest ih =>
      have rest_clean : ∀ p v, Event.opened p v ∉ rest := fun p v member => clean p v (List.mem_cons_of_mem _ member)
      cases e with
      | words ws => simpa [openedIn] using ih rest_clean
      | commit => simpa [openedIn] using ih rest_clean
      | observe vs => simpa [openedIn] using ih rest_clean
      | sample c => simpa [openedIn] using ih rest_clean
      | opened p v => exact absurd (List.mem_cons.mpr (Or.inl rfl)) (clean p v)

theorem openedIn_scheduled (stmt : List field.Gl) (π : Rounds) (n : ℕ) (d : Draws) :
    openedIn (scheduled stmt π n d) = d.opened.map cubicOf := by
  rw [scheduled, openedIn_append]
  · rfl
  · intro p v member
    simp [sent] at member

/-! ## Plays of `programGame` and `shrinkGame` -/

theorem ends_node {α : Type} {P : α → Prop} (g : Game α) (ends : g.Ends P) (p : Game.Pos g) (a : α)
    (node : Game.Pos.node g p = .pure a) : P a := by
  induction g with
  | pure b =>
      simp only [Game.Pos.node, Game.pure.injEq] at node
      exact node ▸ ends
  | draw C bad next ih =>
      match p, node with
      | some ⟨c, q⟩, node => exact ih c (ends c) q node
  | recv M next ih =>
      match p, node with
      | some ⟨m, q⟩, node => exact ih m (ends m) q node

theorem reaches_recv {α M : Type} [Nonempty M] {next : M → Game α} {m : M} {ms ds : List (Σ C : Type, C)} {a : α}
    (h : Game.Reaches (.recv M next) (⟨M, m⟩ :: ms) ds a) : Game.Reaches (next m) ms ds a := by
  obtain ⟨p, node, draws, msgs⟩ := h
  match p, node, draws, msgs with
  | some ⟨m', q⟩, node, draws, msgs =>
      obtain ⟨head, tail⟩ := List.cons.inj msgs
      cases eq_of_heq (Sigma.mk.inj head).2
      exact ⟨q, node, draws, tail⟩

/-- An accepting leaf of `programGame` holds the instance and the candidates that the play sent,
and it passed `Final`. -/
theorem programGame_leaf {I : Type} {i : I} {C : Ccs Gl} {shapeOk : Prop}
    {matrices : List Cubic → List Cubic → Cubic → Cubic} {L : ℕ} {cands : {l : List (ℕ → Gl) // l.length ≤ L}}
    {ms ds : List (Σ C : Type, C)} {tr : Transcript I Gl Cubic}
    (h : Game.Reaches (programGame i C shapeOk matrices L) (⟨{l : List (ℕ → Gl) // l.length ≤ L}, cands⟩ :: ms) ds
      (some tr)) :
    tr.inst = i ∧ tr.cands = cands.1 ∧
      Final C shapeOk tr.cands tr.ry (matrices tr.rx tr.ry tr.rho) tr.innerLast tr.opened := by
  unfold programGame at h
  obtain ⟨p, node, -, -⟩ := reaches_recv h
  refine ends_node (P := fun r => ∀ t : Transcript I Gl Cubic, r = some t → t.inst = i ∧ t.cands = cands.1 ∧
    Final C shapeOk t.cands t.ry (matrices t.rx t.ry t.rho) t.innerLast t.opened) _ ?_ p _ node tr rfl
  rw [Game.ends_bind]
  refine Game.Ends.of_forall (fun tau => ?_) _
  rw [Game.ends_bind]
  refine Game.Ends.of_forall (fun outer => ?_) _
  intro values
  dsimp only
  split_ifs
  · intro rho
    dsimp only
    rw [Game.ends_bind]
    refine Game.Ends.of_forall (fun inner => ?_) _
    intro opened
    dsimp only
    split_ifs with final
    · rintro t ⟨⟩
      exact ⟨rfl, rfl, final⟩
    · rintro t ⟨⟩
  · rintro t ⟨⟩

theorem shrinkGame_leaf {I : Type} [Nonempty I] {T : Template Gl} {s : Counts} {prog : I → Run Gl} {L : ℕ} {i : I}
    {cands : {l : List (ℕ → Gl) // l.length ≤ L}} {ms ds : List (Σ C : Type, C)} {tr : Transcript I Gl Cubic}
    (h : Game.Reaches (shrinkGame (K := Cubic) T s prog L)
      (⟨I, i⟩ :: ⟨{l : List (ℕ → Gl) // l.length ≤ L}, cands⟩ :: ms) ds (some tr)) :
    tr.inst = i ∧ tr.cands = cands.1 ∧
      Final ((prog i).ccs T s) ((prog i).counts = s) tr.cands tr.ry
        (verifierMatrices T s (prog i) tr.rx tr.ry tr.rho) tr.innerLast tr.opened := by
  unfold shrinkGame at h
  have h' := reaches_recv h
  exact programGame_leaf h'

/-- A play that ends at a leaf is no prefix of another play that ends at a leaf. -/
theorem pos_prefix {α : Type} (g : Game α) (p q : Game.Pos g) (a b : α) (np : Game.Pos.node g p = .pure a)
    (nq : Game.Pos.node g q = .pure b) (ms ds : List (Σ C : Type, C))
    (mq : Game.Pos.msgs g q = Game.Pos.msgs g p ++ ms) (dq : Game.Pos.draws g q = Game.Pos.draws g p ++ ds) :
    ms = [] ∧ ds = [] := by
  induction g with
  | pure x =>
      simp only [Game.Pos.msgs, Game.Pos.draws, List.nil_append] at mq dq
      exact ⟨mq.symm, dq.symm⟩
  | draw C bad next ih =>
      match p, q, np, nq, mq, dq with
      | some ⟨c, p'⟩, some ⟨c', q'⟩, np, nq, mq, dq =>
          simp only [Game.Pos.draws, List.cons_append] at dq
          obtain ⟨head, tail⟩ := List.cons.inj dq
          cases eq_of_heq (Sigma.mk.inj head).2
          exact ih c p' q' np nq mq tail
  | recv M next ih =>
      match p, q, np, nq, mq, dq with
      | some ⟨m, p'⟩, some ⟨m', q'⟩, np, nq, mq, dq =>
          simp only [Game.Pos.msgs, List.cons_append] at mq
          obtain ⟨head, tail⟩ := List.cons.inj mq
          cases eq_of_heq (Sigma.mk.inj head).2
          exact ih m p' q' np nq tail dq

theorem reaches_prefix {α : Type} {g : Game α} {ms₁ ms₂ ds₁ ds₂ : List (Σ C : Type, C)} {a b : α}
    (h₁ : Game.Reaches g ms₁ ds₁ a) (h₂ : Game.Reaches g (ms₁ ++ ms₂) (ds₁ ++ ds₂) b) : ms₂ = [] ∧ ds₂ = [] := by
  obtain ⟨p, np, dp, mp⟩ := h₁
  obtain ⟨q, nq, dq, mq⟩ := h₂
  exact pos_prefix g p q a b np nq ms₂ ds₂ (by rw [mq, mp]) (by rw [dq, dp])

/-! ## Building a play -/

theorem drawPoint_reaches {bad : List Cubic → Finset Cubic} :
    ∀ (cs point : List Cubic) (count : ℕ), cs.length = count →
      Game.Reaches (Game.drawPoint bad count point) [] (cs.map fun c => ⟨Cubic, c⟩) (point ++ cs)
  | [], point, count, h => by
      subst h
      simpa [Game.drawPoint] using Game.Reaches.pure (α := List Cubic) point
  | c :: cs, point, count, h => by
      subst h
      simp only [List.length_cons, Game.drawPoint, List.map_cons]
      apply Game.Reaches.draw c
      simpa using drawPoint_reaches cs (point ++ [c]) cs.length rfl

theorem sumcheck_reaches {W : Type} {worlds : Finset W} {d : ℕ} {round : W → List Cubic → Cubic[X]}
    {extra : List Cubic → Finset Cubic} (rs : List ((Fin (d + 1) → Cubic) × Cubic)) (rounds : ℕ)
    (h : rs.length = rounds) (point : List Cubic) (claim : Cubic) :
    Game.Reaches (Game.sumcheck worlds d round extra rounds point claim) (rs.map fun p => ⟨Fin (d + 1) → Cubic, p.1⟩)
      (rs.map fun p => ⟨Cubic, p.2⟩) (point ++ rs.map Prod.snd, claimAfter d rs claim) := by
  subst h
  exact sumcheck_path worlds d round extra rs point claim

open Classical in
/-- The play of `programGame` with these messages and draws, when the outer check passes: it ends
at the transcript if `Final` holds, and at `none` if not. -/
theorem programGame_path {I : Type} (i : I) (C : Ccs Gl) (shapeOk : Prop)
    (matrices : List Cubic → List Cubic → Cubic → Cubic) (L : ℕ) (cands : {l : List (ℕ → Gl) // l.length ≤ L})
    (tau : List Cubic) (outer : List ((Fin 8 → Cubic) × Cubic)) (values : Fin 8 → Cubic) (rho : Cubic)
    (inner : List ((Fin 2 → Cubic) × Cubic)) (opened : Cubic) (htau : tau.length = C.rowVars)
    (houter : outer.length = C.rowVars) (hinner : inner.length = C.cellVars + 1)
    (outerOk : claimAfter 7 outer 0 = eqEval tau (outer.map Prod.snd) * constraint values) :
    Game.Reaches (programGame i C shapeOk matrices L)
      (⟨{l : List (ℕ → Gl) // l.length ≤ L}, cands⟩ :: ((outer.map fun p => ⟨Fin 8 → Cubic, p.1⟩) ++
        ⟨Fin 8 → Cubic, values⟩ :: ((inner.map fun p => ⟨Fin 2 → Cubic, p.1⟩) ++ [⟨Cubic, opened⟩])))
      ((tau.map fun c => ⟨Cubic, c⟩) ++
        ((outer.map fun p => ⟨Cubic, p.2⟩) ++ ⟨Cubic, rho⟩ :: inner.map fun p => ⟨Cubic, p.2⟩))
      (if Final C shapeOk cands.1 (inner.map Prod.snd) (matrices (outer.map Prod.snd) (inner.map Prod.snd) rho)
          (claimAfter 1 inner (batch rho values)) opened then
        some ⟨i, cands.1, tau, outer.map Prod.snd, claimAfter 7 outer 0, values, rho, inner.map Prod.snd,
          claimAfter 1 inner (batch rho values), opened⟩
      else none) := by
  unfold programGame
  apply Game.Reaches.recv cands
  have point := drawPoint_reaches (bad := tauBad C cands.1) tau [] C.rowVars htau
  rw [List.nil_append] at point
  refine Game.Reaches.congr (Game.Reaches.bind point ?_) (List.nil_append _) rfl
  have outerRun := sumcheck_reaches (worlds := (Finset.univ : Finset (Fin cands.1.length))) (d := 7)
    (round := fun k => combRound C.rowVars (outerTables C.entries (C.z (cands.1.get k)) tau) outerComb)
    (extra := fun _ => ∅) outer C.rowVars houter [] 0
  rw [List.nil_append] at outerRun
  refine Game.Reaches.bind outerRun ?_
  apply Game.Reaches.recv values
  dsimp only
  rw [ite_eq_left outerOk]
  apply Game.Reaches.draw rho
  have innerRun := sumcheck_reaches (worlds := (Finset.univ : Finset (Fin cands.1.length))) (d := 1)
    (round := fun k => combRound (C.cellVars + 1)
      (innerTables C.entries (C.z (cands.1.get k)) (outer.map Prod.snd) rho) productComb)
    (extra := fun _ => ∅) inner (C.cellVars + 1) hinner [] (batch rho values)
  rw [List.nil_append] at innerRun
  refine Game.Reaches.congr (Game.Reaches.bind innerRun (Game.Reaches.recv opened ?_)) rfl (List.append_nil _)
  dsimp only
  split_ifs <;> exact Game.Reaches.pure _

open Classical in
/-- The play of `programGame` when the outer check fails: it ends at `none` after the matrix values. -/
theorem programGame_outer_fails {I : Type} (i : I) (C : Ccs Gl) (shapeOk : Prop)
    (matrices : List Cubic → List Cubic → Cubic → Cubic) (L : ℕ) (cands : {l : List (ℕ → Gl) // l.length ≤ L})
    (tau : List Cubic) (outer : List ((Fin 8 → Cubic) × Cubic)) (values : Fin 8 → Cubic)
    (htau : tau.length = C.rowVars) (houter : outer.length = C.rowVars)
    (outerBad : ¬ claimAfter 7 outer 0 = eqEval tau (outer.map Prod.snd) * constraint values) :
    Game.Reaches (programGame i C shapeOk matrices L)
      (⟨{l : List (ℕ → Gl) // l.length ≤ L}, cands⟩ :: ((outer.map fun p => ⟨Fin 8 → Cubic, p.1⟩) ++
        [⟨Fin 8 → Cubic, values⟩]))
      ((tau.map fun c => ⟨Cubic, c⟩) ++ (outer.map fun p => ⟨Cubic, p.2⟩)) none := by
  unfold programGame
  apply Game.Reaches.recv cands
  have point := drawPoint_reaches (bad := tauBad C cands.1) tau [] C.rowVars htau
  rw [List.nil_append] at point
  refine Game.Reaches.congr (Game.Reaches.bind point ?_) (List.nil_append _) rfl
  have outerRun := sumcheck_reaches (worlds := (Finset.univ : Finset (Fin cands.1.length))) (d := 7)
    (round := fun k => combRound C.rowVars (outerTables C.entries (C.z (cands.1.get k)) tau) outerComb)
    (extra := fun _ => ∅) outer C.rowVars houter [] 0
  rw [List.nil_append] at outerRun
  refine Game.Reaches.congr (Game.Reaches.bind outerRun (Game.Reaches.recv values ?_)) rfl (List.append_nil _)
  dsimp only
  rw [ite_eq_right outerBad]
  exact Game.Reaches.pure none

/-! ## The play of a run -/

noncomputable def outerPairs (π : Rounds) (d : Draws) : List ((Fin 8 → Cubic) × Cubic) :=
  (π.outer.zip d.rx).map fun p => (messageOf 7 p.1, cubicOf p.2)

noncomputable def innerPairs (π : Rounds) (d : Draws) : List ((Fin 2 → Cubic) × Cubic) :=
  (π.inner.zip d.ry).map fun p => (messageOf 1 p.1, cubicOf p.2)

/-- The claim where the outer rounds end. -/
noncomputable def outerLast (π : Rounds) (d : Draws) : Cubic := claimAfter 7 (outerPairs π d) 0

/-- The claim where the inner rounds end, from the batched matrix values. -/
noncomputable def innerLast (π : Rounds) (d : Draws) : Cubic :=
  claimAfter 1 (innerPairs π d) (batch (cubicOf d.rho) (valuesOf π))

/-- The outer check of `verify`: the zero-check ends at `eq(τ, rx)` times the constraint. -/
def OuterOk (π : Rounds) (d : Draws) : Prop :=
  outerLast π d = eqEval (d.tau.map cubicOf) (d.rx.map cubicOf) * constraint (valuesOf π)

/-- The lengths of a run's rounds and draws, as `RoundsFit` and `schedule` give them. -/
structure Lengths (T : Template Gl) (s : Counts) (π : Rounds) (d : Draws) : Prop where
  outer : π.outer.length = Layout.rowVars T s
  inner : π.inner.length = Layout.cellVars T s + 1
  tau : d.tau.length = Layout.rowVars T s
  rx : d.rx.length = π.outer.length
  ry : d.ry.length = π.inner.length

theorem map_fst_pairs {α β γ δ : Type} {ε : Type _} (f : α → γ) (g : β → δ) (u : γ → ε) :
    ∀ (l₁ : List α) (l₂ : List β), l₁.length = l₂.length →
      ((l₁.zip l₂).map fun p => (f p.1, g p.2)).map (fun q => u q.1) = l₁.map fun a => u (f a)
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: l₁, b :: l₂, h => by
      simp only [List.zip_cons_cons, List.map_cons, map_fst_pairs f g u l₁ l₂ (by simpa using h)]

theorem map_snd_pairs {α β γ δ : Type} {ε : Type _} (f : α → γ) (g : β → δ) (v : δ → ε) :
    ∀ (l₁ : List α) (l₂ : List β), l₁.length = l₂.length →
      ((l₁.zip l₂).map fun p => (f p.1, g p.2)).map (fun q => v q.2) = l₂.map fun b => v (g b)
  | [], [], _ => rfl
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: l₁, b :: l₂, h => by
      simp only [List.zip_cons_cons, List.map_cons, map_snd_pairs f g v l₁ l₂ (by simpa using h)]

theorem roundMsgs_pairs (π : Rounds) (d : Draws) (hrx : d.rx.length = π.outer.length) (hry : d.ry.length = π.inner.length) :
    ((outerPairs π d).map fun p => (⟨Fin 8 → Cubic, p.1⟩ : Σ M : Type, M)) = roundMsgsOf 7 π.outer ∧
      ((outerPairs π d).map fun p => (⟨Cubic, p.2⟩ : Σ C : Type, C)) = d.rx.map (fun c => ⟨Cubic, cubicOf c⟩) ∧
      (outerPairs π d).map Prod.snd = d.rx.map cubicOf ∧
      ((innerPairs π d).map fun p => (⟨Fin 2 → Cubic, p.1⟩ : Σ M : Type, M)) = roundMsgsOf 1 π.inner ∧
      ((innerPairs π d).map fun p => (⟨Cubic, p.2⟩ : Σ C : Type, C)) = d.ry.map (fun c => ⟨Cubic, cubicOf c⟩) ∧
      (innerPairs π d).map Prod.snd = d.ry.map cubicOf ∧
      (outerPairs π d).length = π.outer.length ∧ (innerPairs π d).length = π.inner.length :=
  ⟨map_fst_pairs (messageOf 7) cubicOf (fun m => (⟨Fin 8 → Cubic, m⟩ : Σ M : Type, M)) π.outer d.rx hrx.symm,
    map_snd_pairs (messageOf 7) cubicOf (fun c => (⟨Cubic, c⟩ : Σ C : Type, C)) π.outer d.rx hrx.symm,
    map_snd_pairs (messageOf 7) cubicOf (fun c => c) π.outer d.rx hrx.symm,
    map_fst_pairs (messageOf 1) cubicOf (fun m => (⟨Fin 2 → Cubic, m⟩ : Σ M : Type, M)) π.inner d.ry hry.symm,
    map_snd_pairs (messageOf 1) cubicOf (fun c => (⟨Cubic, c⟩ : Σ C : Type, C)) π.inner d.ry hry.symm,
    map_snd_pairs (messageOf 1) cubicOf (fun c => c) π.inner d.ry hry.symm,
    by simp [outerPairs, hrx], by simp [innerPairs, hry]⟩

open Classical in
/-- The play of a run whose outer check passes: it ends at the run's transcript if `Final` holds. -/
theorem run_path {I : Type} [Nonempty I] (T : Template Gl) (s : Counts) (prog : I → Run Gl) (L : ℕ) (i : I)
    (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (π : Rounds) (d : Draws) (opened : Cubic) (len : Lengths T s π d)
    (outerOk : OuterOk π d) :
    Game.Reaches (shrinkGame (K := Cubic) T s prog L) (msgsOf i cands π opened) (drawsOf d)
      (if Final ((prog i).ccs T s) ((prog i).counts = s) cands.1 (d.ry.map cubicOf)
          (verifierMatrices T s (prog i) (d.rx.map cubicOf) (d.ry.map cubicOf) (cubicOf d.rho)) (innerLast π d) opened then
        some ⟨i, cands.1, d.tau.map cubicOf, d.rx.map cubicOf, outerLast π d, valuesOf π, cubicOf d.rho,
          d.ry.map cubicOf, innerLast π d, opened⟩
      else none) := by
  obtain ⟨om, od, os, im, id, is, ol, il⟩ := roundMsgs_pairs π d len.rx len.ry
  have path := programGame_path i ((prog i).ccs T s) ((prog i).counts = s) (verifierMatrices T s (prog i)) L cands
    (d.tau.map cubicOf) (outerPairs π d) (valuesOf π) (cubicOf d.rho) (innerPairs π d) opened
    (by rw [List.length_map, len.tau]; rfl) (by rw [ol, len.outer]; rfl) (by rw [il, len.inner]; rfl)
    (by rw [os]; exact outerOk)
  rw [om, od, os, im, id, is] at path
  unfold shrinkGame
  refine Game.Reaches.recv i (Game.Reaches.congr path rfl ?_)
  simp [drawsOf, List.map_map, Function.comp_def]

/-- The play of a run whose outer check fails ends at `none` after the matrix values. -/
theorem run_outer_fails {I : Type} [Nonempty I] (T : Template Gl) (s : Counts) (prog : I → Run Gl) (L : ℕ) (i : I)
    (cands : {l : List (ℕ → Gl) // l.length ≤ L}) (π : Rounds) (d : Draws) (len : Lengths T s π d)
    (outerBad : ¬ OuterOk π d) :
    Game.Reaches (shrinkGame (K := Cubic) T s prog L)
      (⟨I, i⟩ :: ⟨{l : List (ℕ → Gl) // l.length ≤ L}, cands⟩ :: (roundMsgsOf 7 π.outer ++ [⟨Fin 8 → Cubic, valuesOf π⟩]))
      ((d.tau ++ d.rx).map fun c => ⟨Cubic, cubicOf c⟩) none := by
  obtain ⟨om, od, os, -, -, -, ol, -⟩ := roundMsgs_pairs π d len.rx len.ry
  have path := programGame_outer_fails i ((prog i).ccs T s) ((prog i).counts = s) (verifierMatrices T s (prog i)) L
    cands (d.tau.map cubicOf) (outerPairs π d) (valuesOf π) (by rw [List.length_map, len.tau]; rfl)
    (by rw [ol, len.outer]; rfl) (by rw [os]; exact outerBad)
  rw [om, od] at path
  unfold shrinkGame
  refine Game.Reaches.recv i (Game.Reaches.congr path rfl ?_)
  simp [List.map_map, Function.comp_def]

/-- A run that `shrinkGame` accepts passes the outer check. -/
theorem run_outer_ok {I : Type} [Nonempty I] {T : Template Gl} {s : Counts} {prog : I → Run Gl} {L : ℕ} {i : I}
    {cands : {l : List (ℕ → Gl) // l.length ≤ L}} {π : Rounds} {d : Draws} {opened : Cubic} {tr : Transcript I Gl Cubic}
    (len : Lengths T s π d)
    (h : Game.Reaches (shrinkGame (K := Cubic) T s prog L) (msgsOf i cands π opened) (drawsOf d) (some tr)) :
    OuterOk π d := by
  by_contra bad
  have fails := run_outer_fails T s prog L i cands π d len bad
  have split : msgsOf i cands π opened =
      (⟨I, i⟩ :: ⟨{l : List (ℕ → Gl) // l.length ≤ L}, cands⟩ :: (roundMsgsOf 7 π.outer ++ [⟨Fin 8 → Cubic, valuesOf π⟩])) ++
        (roundMsgsOf 1 π.inner ++ [⟨Cubic, opened⟩]) := by
    simp [msgsOf]
  have draws : drawsOf d = ((d.tau ++ d.rx).map fun c => ⟨Cubic, cubicOf c⟩) ++
      ((d.rho :: d.ry).map fun c => ⟨Cubic, cubicOf c⟩) := by
    simp [drawsOf]
  rw [split, draws] at h
  exact absurd (reaches_prefix fails h).1 (by simp)

/-! ## The claims' event and the Rust verifier's event -/

/-- `Spec.FinalRestoreBreak` for any second game: the shrink outcome is an accepting transcript whose
claim is false, and every valid candidate has an accepting reading in the second game. -/
def GameBreak {I β : Type} (Claim : I → Prop) (T : Template Gl) (s : Counts) (prog : I → Run Gl) (g : Game β)
    (Reads : I → (ℕ → Gl) → g.Pos → Prop) (r : Option (Option (Transcript I Gl Cubic)) × (g.Pos → Option β)) :
    Prop :=
  ∃ tr, r.1 = some (some tr) ∧
    (∀ w ∈ tr.cands, ((prog tr.inst).ccs T s).Relation w → ∃ p, Reads tr.inst w p ∧ ∃ b, r.2 p = some b) ∧
    ¬ Claim tr.inst

/-- The same event for the Rust verifier: the shrink outcome is the leaf of a play `p`, where `p`
sends the instance, the WHIR candidates, the proof's rounds and the value that the opening
returned, and draws the challenges that `verify` drew; `verify` accepted that run; the instance's
claim is false; and every valid candidate has an accepting reading. -/
def RustBreak (verify : VerifyFn) {I β : Type} [Nonempty I] (Claim : I → Prop) (T : Template Gl) (s : Counts)
    (prog : I → Run Gl) (L : ℕ) (g : Game β) (Reads : I → (ℕ → Gl) → g.Pos → Prop)
    (r : Option (Option (Transcript I Gl Cubic)) × (g.Pos → Option β)) : Prop :=
  ∃ (p : Game.Pos (shrinkGame (K := Cubic) T s prog L)) (a : Option (Transcript I Gl Cubic)) (i : I)
    (stmt : List field.Gl) (π : Rounds) (cands : {l : List (ℕ → Gl) // l.length ≤ L})
    (oracle : History → field.Ext) (opening : History → List Cubic → Option field.Ext) (hist : History) (v : Cubic),
    r.1 = some a ∧ Game.Pos.node _ p = .pure a ∧ WhirSound cands.1 opening ∧
    verify (recorder oracle opening) (evalOf T s (prog i)) s stmt π [] = ok (true, hist) ∧
    openedIn hist = some v ∧ Game.Pos.msgs _ p = msgsOf i cands π v ∧ Game.Pos.draws _ p = drawsIn hist ∧
    (∀ w ∈ cands.1, ((prog i).ccs T s).Relation w → ∃ q, Reads i w q ∧ ∃ b, r.2 q = some b) ∧ ¬ Claim i

theorem rustBreak_gameBreak {verify : VerifyFn} {T : Template Gl} (sound : Sound verify T) {I β : Type} [Nonempty I]
    {Claim : I → Prop} {s : Counts} {prog : I → Run Gl} {L : ℕ} {g : Game β} {Reads : I → (ℕ → Gl) → g.Pos → Prop}
    {r : Option (Option (Transcript I Gl Cubic)) × (g.Pos → Option β)}
    (rust : RustBreak verify Claim T s prog L g Reads r) : GameBreak Claim T s prog g Reads r := by
  obtain ⟨p, a, i, stmt, π, cands, oracle, opening, hist, v, hr, hnode, whir, accepts, hopened, hmsgs, hdraws,
    reads, false_claim⟩ := rust
  obtain ⟨d, w, run, hw, -, tr, reach⟩ := sound prog s L i stmt π cands oracle opening hist whir accepts
  obtain ⟨d', run', -, hrx, hry, -, -⟩ := schedule_recorder oracle opening (Layout.rowVars T s) (Layout.cellVars T s)
    stmt π
  rw [run'] at run
  simp only [Result.ok.injEq, Prod.mk.injEq] at run
  obtain ⟨rfl, rfl⟩ := run
  rw [openedIn_scheduled, hw, Option.map_some] at hopened
  cases Option.some.inj hopened
  rw [drawsIn_scheduled stmt π _ _ hrx hry] at hdraws
  obtain ⟨q, qnode, qdraws, qmsgs⟩ := reach
  cases Game.Pos.eq_of_msgs_draws (qmsgs.trans hmsgs.symm) (qdraws.trans hdraws.symm)
  rw [qnode] at hnode
  cases Game.pure.inj hnode
  obtain ⟨inst, same, -⟩ := shrinkGame_leaf ⟨_, qnode, qdraws, qmsgs⟩
  refine ⟨tr, hr, ?_, ?_⟩
  · rw [inst, same]
    exact reads
  · rw [inst]
    exact false_claim

/-- A bound on the claims' break event bounds the Rust verifier's break event, for a `Sound`
verifier, in the restoration experiment of the claims. -/
theorem rust_bound {verify : VerifyFn} {T : Template Gl} (sound : Sound verify T) {I β : Type} [Nonempty I]
    (Claim : I → Prop) (s : Counts) (prog : I → Run Gl) (L : ℕ) (g : Game β) (Reads : I → (ℕ → Gl) → g.Pos → Prop)
    (calls : ℕ) (prover : Game.Strategy (Option (Option (Transcript I Gl Cubic)) × (g.Pos → Option β))) (bound : ℝ)
    (game : Game.prob prover (Game.restorePair (shrinkGame (K := Cubic) T s prog L) g calls [] [])
      (GameBreak Claim T s prog g Reads) ≤ bound) :
    Game.prob prover (Game.restorePair (shrinkGame (K := Cubic) T s prog L) g calls [] [])
      (RustBreak verify Claim T s prog L g Reads) ≤ bound :=
  (Game.prob_mono prover _ fun _ rust => rustBreak_gameBreak sound rust).trans game

end CompressionLeanRust.Verifier
