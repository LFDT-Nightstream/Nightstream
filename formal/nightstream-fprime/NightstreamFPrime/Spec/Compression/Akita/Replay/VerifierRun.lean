import NightstreamFPrime.Spec.Compression.Akita.Replay.VerifierRunTwin
import NightstreamFPrime.Spec.Compression.Akita.Replay.SpongeModel
import NightstreamFPrime.Spec.Compression.Akita.Replay.Simulate
import NightstreamFPrime.Spec.Compression.Akita.Replay.DigestFree
import NightstreamFPrime.Spec.Compression.Akita.Replay.JoinBound
import NightstreamFPrime.Spec.Compression.Akita.Replay.Resolve

/-! The byte verifier `verifyProof` as an algorithm with the two oracles of the experiment: the
finite ideal sponge and the digest oracle (Akita revision `d50f700`; ePrint 2026/1983, revision of
2026-11-23, Section 8.2 `[KECCAK]` and Lemma 10.47; `Akita/Digest.lean`, `Digest.Run`;
`Replay/Simulate.lean`; `Replay/SpongeModel.lean`; `docs/akita-work/w21/design.md`, sections 2
and 5).

- `Digest.Run.Bound.mono`: a bound on the queries of an algorithm holds for larger counts.
  `Digest.Run.Bound.length_left`: the trace has at most the bound of queries to `H_FS`.
- `digestOf hash` is the digest oracle on byte arrays for a table on the strings of at most `D`
  bytes. An input of more than `D` bytes gets the empty array.
- `lastBlocks` gives the last blocks of `e` and of `t` that the terminal hashes, and `none` when
  a check of the terminal before its two digests fails. `terminalRun_congr`: the terminal reads
  the hash only at `eBytes` and `tBytes` of these blocks.
- `Verifier.go` runs the operations of a script. At a work or a draw it asks the sponge for the
  base of the query of the step (`asks`), when this base is in the bounds of the key types and
  has no answer yet. At the terminal it asks the digest oracle for `eBytes` and then `tBytes`.
  Each step runs with the answers of the queries.
- `verifierRun pub proof` is the verifier: the checks of the statement, the script, the decoding
  of the proof, and then `Verifier.go`. `verifierRun_eval`: for each table `fs` of the sponge and
  each table `hash` of the digest, its output is `verifyProof` with the sponge `Sponge.spongeOf fs`
  and the digest `digestOf hash`. This holds for each layout.
- `Verifier.askCount` and `Verifier.digestCount` count the queries on the operations of a script,
  and `verifierRun_bound` is the bound: one sponge query for each work, one for each draw that is
  not directly after a work or a draw, and two digest queries for each terminal.
- `verifierRun_asked` and `verifierRun_digests`: after a run with no stop of the operations
  before an operation, the base of the query of this operation is in the trace of `verifierRun`,
  and for the terminal the bytes of its two last blocks are in the trace. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Digest.Run

variable {Fs Challenge Msg Digest Output : Type}

/-- A bound on the queries of an algorithm holds for larger counts. -/
theorem Bound.mono {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) :
    ∀ {more moreDigests : ℕ}, queries ≤ more → digests ≤ moreDigests →
      run.Bound more moreDigests := by
  induction bounded with
  | done output queries digests => exact fun _ _ => .done _ _ _
  | challenge point next queries digests each induction =>
      intro more moreDigests le leDigests
      obtain ⟨r, rfl⟩ : ∃ r, more = r + 1 := ⟨more - 1, by omega⟩
      exact .challenge point next r moreDigests fun answer =>
        induction answer (by omega) leDigests
  | digest message next queries digests each induction =>
      intro more moreDigests le leDigests
      obtain ⟨r, rfl⟩ : ∃ r, moreDigests = r + 1 := ⟨moreDigests - 1, by omega⟩
      exact .digest message next more r fun answer => induction answer le (by omega)

/-- The queries of an algorithm do not change with a function on its output. -/
theorem trace_map {Other : Type} (f : Output → Other) (fs : Fs → Challenge)
    (hash : Msg → Digest) (run : Run Fs Challenge Msg Digest Output) :
    (run.map f).trace fs hash = run.trace fs hash := by
  induction run with
  | done output => rfl
  | challenge point next induction => exact congrArg (List.cons _) (induction (fs point))
  | digest message next induction => exact congrArg (List.cons _) (induction (hash message))

/-- An algorithm with at most `queries` queries to `H_FS` has at most `queries` of them in its
trace. -/
theorem Bound.length_left {run : Run Fs Challenge Msg Digest Output} {queries digests : ℕ}
    (bounded : run.Bound queries digests) (fs : Fs → Challenge) (hash : Msg → Digest) :
    ((run.trace fs hash).filterMap Sum.getLeft?).length ≤ queries := by
  induction bounded with
  | done output queries digests => exact Nat.zero_le _
  | challenge point next queries digests each induction =>
      exact Nat.succ_le_succ (induction (fs point))
  | digest message next queries digests each induction => exact induction (hash message)

end NightstreamFPrime.Spec.Compression.Akita.Digest.Run

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Lib
open NightstreamFPrime.Spec.Compression.Akita
open Sponge (split answerOf spongeOf)
open Replayer

section Oracle

variable {D : ℕ}

/-- The string of a `ByteArray` of at most `D` bytes. -/
def strOf (bytes : ByteArray) (short : bytes.size ≤ D) : Sponge.Str D :=
  Sponge.Short.ofList (bytes.data.toList.map fun u => (⟨u.toNat, u.toNat_lt⟩ : Fin 256))
    (by rw [List.length_map, Array.length_toList]; exact short)

theorem array_strOf (bytes : ByteArray) (short : bytes.size ≤ D) :
    (strOf bytes short).array = bytes := by
  show (⟨((bytes.data.toList.map fun u => (⟨u.toNat, u.toNat_lt⟩ : Fin 256)).map
    Bytes.byte).toArray⟩ : ByteArray) = bytes
  have back : (Bytes.byte ∘ fun u : UInt8 => (⟨u.toNat, u.toNat_lt⟩ : Fin 256)) = id :=
    funext fun u => UInt8.toNat_inj.mp rfl
  rw [List.map_map, back, List.map_id, Array.toArray_toList]

/-- The digest oracle on byte arrays for the table `hash` on the strings of at most `D` bytes:
the 32 bytes of the table for an input of at most `D` bytes, and the empty array for a longer
input. -/
def digestOf (hash : Sponge.Str D → Bytes 32) (input : ByteArray) : ByteArray :=
  if short : input.size ≤ D then Bytes.array (hash (strOf input short)) else ByteArray.empty

theorem digestOf_short (hash : Sponge.Str D → Bytes 32) {input : ByteArray}
    (short : input.size ≤ D) :
    digestOf hash input = Bytes.array (hash (strOf input short)) := by
  rw [digestOf, dite_eq_left short]

theorem digestOf_long (hash : Sponge.Str D → Bytes 32) {input : ByteArray}
    (long : ¬ input.size ≤ D) :
    digestOf hash input = ByteArray.empty := by
  rw [digestOf, dite_eq_right long]

end Oracle

section Terminal

variable (S : Shape)

/-- The last blocks of `e` and of `t` that the terminal recovers and hashes, and `none` when a
check of the terminal before its two digests fails. -/
def lastBlocks (cap : ℕ) (P : Golomb.Params) (setup : Array Fq) (msgs : TerminalMessages)
    (ch : TerminalChallenges) : Option (Array Fq × List (Array Fq)) :=
  match Golomb.decode P msgs.payload with
  | none => none
  | some z =>
      if centeredNormSq z ≤ cap then
        if ch.fold.length = S.blocks ∧ ch.fold.all (sparseValid S.innerDegree) = true then
          match Poly.inverse (Nat.log 2 S.innerDegree) (challenge S ch (S.blocks - 1)) with
          | none => none
          | some inv =>
              match recover S (challenge S ch (S.blocks - 1)) inv (folded S msgs ch)
                (reduced S ch z) with
              | none => none
              | some eLast =>
                  match tLastOf S msgs ch inv ((List.range S.innerRank).map (aRow S setup z)) with
                  | none => none
                  | some tLast => some (eLast, tLast)
        else none
      else none

/-- The bytes that the terminal hashes for the last block of `e`. -/
def eBytes (eLast : Array Fq) : ByteArray := digestInput (Poly.canon S.innerDegree eLast)

/-- The bytes that the terminal hashes for the last block of `t`. -/
def tBytes (tLast : List (Array Fq)) : ByteArray :=
  digestInput (tLast.foldl (fun out x => out ++ Poly.canon S.innerDegree x) #[])

/-- The terminal reads the hash only at the bytes of its two last blocks. -/
theorem terminalRun_congr {digest digest' : ByteArray → ByteArray} (cap : ℕ) (P : Golomb.Params)
    (setup : Array Fq) (msgs : TerminalMessages) (ch : TerminalChallenges)
    (same : ∀ eLast tLast, lastBlocks S cap P setup msgs ch = some (eLast, tLast) →
      digest (eBytes S eLast) = digest' (eBytes S eLast) ∧
        digest (tBytes S tLast) = digest' (tBytes S tLast)) :
    terminalRun S digest cap P setup msgs ch = terminalRun S digest' cap P setup msgs ch := by
  unfold terminalRun
  cases decoded : Golomb.decode P msgs.payload with
  | none => rfl
  | some z =>
      show terminalChecks S digest cap setup msgs ch z =
        terminalChecks S digest' cap setup msgs ch z
      unfold terminalChecks
      by_cases norm : centeredNormSq z ≤ cap
      · by_cases valid : ch.fold.length = S.blocks ∧
            ch.fold.all (sparseValid S.innerDegree) = true
        · rw [ite_eq_left norm, ite_eq_left norm, ite_eq_left valid, ite_eq_left valid]
          cases inverse : Poly.inverse (Nat.log 2 S.innerDegree)
              (challenge S ch (S.blocks - 1)) with
          | none => rfl
          | some inv =>
              dsimp only
              cases recovered : recover S (challenge S ch (S.blocks - 1)) inv (folded S msgs ch)
                  (reduced S ch z) with
              | none => rfl
              | some eLast =>
                  dsimp only
                  cases rows : tLastOf S msgs ch inv
                      ((List.range S.innerRank).map (aRow S setup z)) with
                  | none => rfl
                  | some tLast =>
                      dsimp only
                      obtain ⟨e, t⟩ := same eLast tLast (by
                        simp only [lastBlocks, decoded, norm, valid, and_self, ↓reduceIte,
                          inverse, recovered, rows])
                      have eSame : fieldDigest digest (Poly.canon S.innerDegree eLast) =
                          fieldDigest digest' (Poly.canon S.innerDegree eLast) := e
                      have tSame : rowsDigest S digest tLast = rowsDigest S digest' tLast := t
                      rw [eSame, tSame]
        · rw [ite_eq_left norm, ite_eq_left norm, ite_eq_right valid, ite_eq_right valid]
      · rw [ite_eq_right norm, ite_eq_right norm]

end Terminal

/-- The level of a terminal operation. -/
def Op.terminalLevel : Op → Option ℕ
  | .terminal level => some level
  | _ => none

theorem Op.terminalLevel_some {op : Op} {level : ℕ} (found : op.terminalLevel = some level) :
    op = .terminal level := by
  cases op <;> first | (cases found; rfl) | cases found

theorem Op.terminalLevel_none {op : Op} (missing : op.terminalLevel = none) :
    op.isTerminal = false := by
  cases op <;> first | rfl | cases missing

namespace Replayer

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The step of an operation does not read the sponge of the public input. -/
theorem step_withSponge (sponge : Duplex) (i : ℕ) (op : Op) :
    step (pub.withSponge sponge) proof messages i op = step pub proof messages i op := by
  cases op <;> rfl

/-- A run does not read the sponge of the public input. -/
theorem runFrom_withSponge (sponge : Duplex) (l : List (Op × ℕ)) (s : Replayer) :
    runFrom (pub.withSponge sponge) proof messages l s = runFrom pub proof messages l s := by
  unfold runFrom
  simp only [step_withSponge]

/-- The last blocks of the terminal of the level on a state. -/
def blocksIn (s : Replayer) (level : ℕ) : Option (Array Fq × List (Array Fq)) :=
  (pub.setupFor pub.layout.terminal.setupCount).bind fun setup =>
    lastBlocks pub.layout.terminal.shape pub.layout.terminal.limit pub.layout.terminal.params
      setup (terminalMessagesIn s level) (terminalChallengesIn s level)

/-- The step of an operation reads the hash of the public input only at the terminal, and there
only at the bytes of the two last blocks. -/
theorem step_digest_congr (digest digest' : ByteArray → ByteArray) (i : ℕ) (op : Op)
    (s : Replayer)
    (same : ∀ level, op.terminalLevel = some level → ∀ blocks, blocksIn pub s level = some blocks →
      digest (eBytes pub.layout.terminal.shape blocks.1) =
          digest' (eBytes pub.layout.terminal.shape blocks.1) ∧
        digest (tBytes pub.layout.terminal.shape blocks.2) =
          digest' (tBytes pub.layout.terminal.shape blocks.2)) :
    ((step (pub.withDigest digest) proof messages i op).run).run s =
      ((step (pub.withDigest digest') proof messages i op).run).run s := by
  cases terminal : op.terminalLevel with
  | none =>
      rw [step_withDigest pub digest proof messages i (Op.terminalLevel_none terminal),
        step_withDigest pub digest' proof messages i (Op.terminalLevel_none terminal)]
  | some level =>
      obtain rfl := Op.terminalLevel_some terminal
      have value : ∀ d, terminalValue (pub.withDigest d) s level =
          (pub.setupFor pub.layout.terminal.setupCount).map fun setup =>
            terminalRun pub.layout.terminal.shape d pub.layout.terminal.limit
              pub.layout.terminal.params setup (terminalMessagesIn s level)
              (terminalChallengesIn s level) := fun _ => rfl
      rw [step_terminal, step_terminal, value, value]
      cases expanded : pub.setupFor pub.layout.terminal.setupCount with
      | none => rfl
      | some setup =>
          rw [Option.map_some, Option.map_some, terminalRun_congr _ _ _ _ _ _
            fun eLast tLast found => same level rfl (eLast, tLast) (by
              unfold blocksIn
              rw [expanded]
              exact found)]

end Replayer

namespace Verifier

section Bases

variable {L B C S : ℕ}

/-- The base of a query, when the query is in the bounds of the key types. -/
noncomputable def baseOf (L B C S : ℕ) (query : Option (List Duplex.Event × ℕ)) :
    Option (Sponge.Base L B C) :=
  query.bind fun p => (split p.1 p.2 : Option (Sponge.Base L B C × Sponge.Sub S C)).map Prod.fst

/-- Two tables with the same answers at the base of a query give the same answer for it. -/
theorem answerOf_agree {t fs : Sponge.Base L B C → Sponge.Y S C}
    {query : Option (List Duplex.Event × ℕ)}
    (same : ∀ base, baseOf L B C S query = some base → t base = fs base) :
    ∀ p, query = some p → answerOf t p.1 p.2 = answerOf fs p.1 p.2 := by
  rintro p rfl
  cases found : (split p.1 p.2 : Option (Sponge.Base L B C × Sponge.Sub S C)) with
  | none => rw [Sponge.answerOf_none t found, Sponge.answerOf_none fs found]
  | some pair =>
      obtain ⟨base, sub⟩ := pair
      rw [Sponge.answerOf_some t found, Sponge.answerOf_some fs found,
        same base (by simp only [baseOf, Option.bind_some, found, Option.map_some])]

/-- A query after a squeeze has the base of the query of this squeeze. -/
theorem split_squeeze {events : List Duplex.Event} {c count : ℕ} {base : Sponge.Base L B C}
    {sub : Sponge.Sub S C} (found : split (.squeeze c :: events) count = some (base, sub)) :
    ∃ sub' : Sponge.Sub S C, split events c = some (base, sub') := by
  obtain ⟨parts, -⟩ := Sponge.split_eq_some.mp found
  have each : ∀ x ∈ sub.counts, x ≤ C := fun x member => by
    have listed : x ∈ sub.1.toList.map Fin.val := member
    obtain ⟨f, -, rfl⟩ := List.mem_map.mp listed
    exact Nat.le_of_lt_succ f.isLt
  have short : sub.counts.length ≤ S := by
    unfold Sponge.Sub.counts
    rw [List.length_map]
    exact sub.1.length_toList
  unfold Sponge.Sub.events at parts
  cases counts : sub.counts with
  | nil =>
      rw [counts] at parts
      exact (base.events_ne c events parts.symm).elim
  | cons first rest =>
      rw [counts] at parts each short
      rw [List.map_cons, List.cons_append] at parts
      obtain ⟨head, tail⟩ := List.cons.inj parts
      obtain rfl : c = first := Duplex.Event.squeeze.inj head
      rw [tail]
      exact ⟨_, Sponge.split_of base rest c (Nat.le_of_succ_le short)
        (fun x member => each x (List.mem_cons_of_mem _ member)) (each c List.mem_cons_self)⟩

/-- Each query on the events has its base in `known`. -/
def Covered (S : ℕ) (known : List (Sponge.Base L B C)) (events : List Duplex.Event) : Prop :=
  ∀ (count : ℕ) (base : Sponge.Base L B C) (sub : Sponge.Sub S C),
    split events count = some (base, sub) → base ∈ known

theorem covered_squeeze {known : List (Sponge.Base L B C)} {events : List Duplex.Event} {c : ℕ}
    (inside : ∀ base, baseOf L B C S (some (events, c)) = some base → base ∈ known) :
    Covered S known (.squeeze c :: events) := by
  intro count base sub found
  obtain ⟨sub', found'⟩ := split_squeeze found
  exact inside base (by simp only [baseOf, Option.bind_some, found', Option.map_some])

end Bases

section Run

variable {L B C S D : ℕ} {Fs Answer Output : Type}

/-- The algorithm after a step: it stops with the stop of the step. -/
def afterStep
    (next : Replayer → Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) (Except Stop Unit)) :
    Except Stop Unit × Replayer → Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) (Except Stop Unit)
  | (.ok _, s) => next s
  | (.error stop, _) => .done (.error stop)

/-- A query of the digest oracle for an input of at most `D` bytes. A longer input gets the
empty array with no query. -/
def askDigest (input : ByteArray)
    (next : ByteArray → Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) Output) :
    Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) Output :=
  if short : input.size ≤ D then .digest (strOf input short) fun answer =>
    next (Bytes.array answer)
  else next ByteArray.empty

theorem eval_askDigest (fs : Fs → Answer) (hash : Sponge.Str D → Bytes 32) (input : ByteArray)
    (next : ByteArray → Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) Output) :
    (askDigest input next).eval fs hash = (next (digestOf hash input)).eval fs hash := by
  unfold askDigest digestOf
  split <;> rfl

theorem bound_askDigest (input : ByteArray)
    (next : ByteArray → Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) Output) {queries digests : ℕ}
    (each : ∀ answer, (next answer).Bound queries digests) :
    (askDigest input next).Bound queries (digests + 1) := by
  unfold askDigest
  split
  · exact .digest _ _ queries digests fun answer => each _
  · exact (each _).mono (Nat.le_refl _) (Nat.le_succ _)

theorem mem_askDigest {fs : Fs → Answer} {hash : Sponge.Str D → Bytes 32} {input : ByteArray}
    {next : ByteArray → Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) Output}
    {query : Fs ⊕ Sponge.Str D}
    (later : query ∈ (next (digestOf hash input)).trace fs hash) :
    query ∈ (askDigest input next).trace fs hash := by
  unfold askDigest
  split
  · rename_i short
    rw [digestOf_short hash short] at later
    exact List.mem_cons_of_mem _ later
  · rename_i long
    rw [digestOf_long hash long] at later
    exact later

theorem askDigest_asked (fs : Fs → Answer) (hash : Sponge.Str D → Bytes 32) (input : ByteArray)
    (next : ByteArray → Digest.Run Fs Answer (Sponge.Str D) (Bytes 32) Output)
    (short : input.size ≤ D) :
    Sum.inr (strOf input short) ∈ (askDigest input next).trace fs hash := by
  unfold askDigest
  rw [dite_eq_left short]
  exact List.mem_cons_self

/-- The hash with the answers `e` and `t` at the inputs `eIn` and `tIn`. -/
def patched (eIn tIn e t : ByteArray) (input : ByteArray) : ByteArray :=
  if input = tIn then t else if input = eIn then e else ByteArray.empty

theorem patched_t (eIn tIn e t : ByteArray) : patched eIn tIn e t tIn = t := by
  rw [patched, ite_eq_left rfl]

theorem patched_e (digest : ByteArray → ByteArray) (eIn tIn : ByteArray) :
    patched eIn tIn (digest eIn) (digest tIn) eIn = digest eIn := by
  unfold patched
  by_cases same : eIn = tIn
  · rw [ite_eq_left same, same]
  · rw [ite_eq_right same, ite_eq_left rfl]

variable (pub : Public) (proof : ByteArray) (messages : Array Message)

/-- The algorithm after the step of `x` on `s` with the answers of the table `t`. -/
noncomputable def stepWith (x : Op × ℕ) (s : Replayer) (t : Sponge.Base L B C → Sponge.Y S C)
    (next : Replayer →
      Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) (Except Stop Unit)) :
    Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) (Except Stop Unit) :=
  afterStep next (((step pub proof messages x.2 x.1).run).run (reideal s (some (answerOf t))))

/-- The run of the operations `l` from the state `s` as an algorithm with the two oracles: `t`
has the answers of the sponge for the bases `known`. -/
noncomputable def go : List (Op × ℕ) → Replayer → (Sponge.Base L B C → Sponge.Y S C) →
    List (Sponge.Base L B C) →
    Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) (Except Stop Unit)
  | [], _, _, _ => .done (.ok ())
  | x :: l, s, t, known =>
      match x.1.terminalLevel with
      | some level =>
          match blocksIn pub s level with
          | some blocks =>
              askDigest (eBytes pub.layout.terminal.shape blocks.1) fun e =>
                askDigest (tBytes pub.layout.terminal.shape blocks.2) fun d =>
                  stepWith (pub.withDigest (patched (eBytes pub.layout.terminal.shape blocks.1)
                    (tBytes pub.layout.terminal.shape blocks.2) e d)) proof messages x s t
                    fun s₁ => go l s₁ t known
          | none => stepWith pub proof messages x s t fun s₁ => go l s₁ t known
      | none =>
          match baseOf L B C S (asks proof messages s x.1) with
          | some base =>
              if base ∈ known then stepWith pub proof messages x s t fun s₁ => go l s₁ t known
              else .challenge base fun answer =>
                stepWith pub proof messages x s (Function.update t base answer) fun s₁ =>
                  go l s₁ (Function.update t base answer) (base :: known)
          | none => stepWith pub proof messages x s t fun s₁ => go l s₁ t known

variable (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32)

theorem eval_stepWith (pubB pubR : Public) (x : Op × ℕ) (l : List (Op × ℕ)) (s : Replayer)
    (t : Sponge.Base L B C → Sponge.Y S C)
    (next : Replayer →
      Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) (Except Stop Unit))
    (stepped : ((step pub proof messages x.2 x.1).run).run (reideal s (some (answerOf fs))) =
      ((step pubB proof messages x.2 x.1).run).run (reideal s (some (answerOf fs))))
    (same : ∀ p, asks proof messages s x.1 = some p → answerOf t p.1 p.2 = answerOf fs p.1 p.2)
    (rest : ∀ s₁, (next s₁).eval fs hash =
      (runFrom pubR proof messages l (reideal s₁ (some (answerOf fs)))).1) :
    (stepWith pub proof messages x s t next).eval fs hash =
      (match ((step pubB proof messages x.2 x.1).run).run (reideal s (some (answerOf fs))) with
        | (.ok _, s₁) => runFrom pubR proof messages l s₁
        | (.error e, s₁) => (.error e, s₁)).1 := by
  have key := step_reideal pub proof messages x.2 x.1 (reideal s (some (answerOf t)))
    (a := answerOf t) (a' := answerOf fs) rfl
    (fun p asked => same p ((asks_reideal proof messages s _ x.1).symm.trans asked))
  unfold BlindAt at key
  have state : reideal (reideal s (some (answerOf t))) (some (answerOf fs)) =
      reideal s (some (answerOf fs)) := rfl
  rw [state] at key
  rw [← stepped, key]
  unfold stepWith
  rcases ((step pub proof messages x.2 x.1).run).run (reideal s (some (answerOf t))) with
    ⟨_ | ⟨⟩, s₁⟩
  · rfl
  · exact rest s₁

/-- The output of `go` for the tables `fs` and `hash` is the result of the run of the
operations with the answers of `fs` and the digest of `hash`. -/
theorem eval_go : ∀ (l : List (Op × ℕ)) (s : Replayer) (t : Sponge.Base L B C → Sponge.Y S C)
    (known : List (Sponge.Base L B C)), (∀ base ∈ known, t base = fs base) →
    (go pub proof messages l s t known).eval fs hash =
      (runFrom (pub.withDigest (digestOf hash)) proof messages l
        (reideal s (some (answerOf fs)))).1
  | [], _, _, _, _ => rfl
  | (op, i) :: l, s, t, known, agree => by
      rw [go, runFrom_cons]
      dsimp only
      cases terminal : op.terminalLevel with
      | some level =>
          obtain rfl := Op.terminalLevel_some terminal
          dsimp only
          have quiet : ∀ p, asks proof messages s (Op.terminal level) = some p →
              answerOf t p.1 p.2 = answerOf fs p.1 p.2 := fun p asked => by cases asked
          cases found : blocksIn pub s level with
          | none =>
              dsimp only
              exact eval_stepWith pub proof messages fs hash _ _ _ l s t _
                (step_digest_congr pub proof messages pub.digest (digestOf hash) i _ _
                  fun level' term blocks' found' => by
                    cases term
                    have other : blocksIn pub s level = some blocks' := found'
                    rw [found] at other
                    cases other)
                quiet fun s₁ => eval_go l s₁ t known agree
          | some blocks =>
              dsimp only
              rw [eval_askDigest, eval_askDigest]
              exact eval_stepWith _ proof messages fs hash _ _ _ l s t _
                (step_digest_congr pub proof messages _ (digestOf hash) i _ _
                  fun level' term blocks' found' => by
                    cases term
                    have other : blocksIn pub s level = some blocks' := found'
                    rw [found] at other
                    cases other
                    exact ⟨patched_e _ _ _, patched_t _ _ _ _⟩)
                quiet fun s₁ => eval_go l s₁ t known agree
      | none =>
          dsimp only
          have free : ((step pub proof messages i op).run).run
              (reideal s (some (answerOf fs))) =
              ((step (pub.withDigest (digestOf hash)) proof messages i op).run).run
                (reideal s (some (answerOf fs))) := by
            rw [step_withDigest pub _ proof messages i (Op.terminalLevel_none terminal)]
          cases asked : baseOf L B C S (asks proof messages s op) with
          | none =>
              dsimp only
              exact eval_stepWith pub proof messages fs hash _ _ _ l s t _ free
                (answerOf_agree fun base found => by rw [asked] at found; cases found)
                fun s₁ => eval_go l s₁ t known agree
          | some base =>
              dsimp only
              by_cases member : base ∈ known
              · rw [ite_eq_left member]
                exact eval_stepWith pub proof messages fs hash _ _ _ l s t _ free
                  (answerOf_agree fun other found => by
                    rw [asked] at found
                    cases found
                    exact agree base member)
                  fun s₁ => eval_go l s₁ t known agree
              · rw [ite_eq_right member, Digest.Run.eval_challenge]
                exact eval_stepWith pub proof messages fs hash _ _ _ l s _ _ free
                  (answerOf_agree fun other found => by
                    rw [asked] at found
                    cases found
                    exact Function.update_self _ _ _)
                  fun s₁ => eval_go l s₁ _ (base :: known) fun other inside => by
                    rcases List.mem_cons.mp inside with rfl | old
                    · exact Function.update_self _ _ _
                    · by_cases equal : other = base
                      · rw [equal]
                        exact Function.update_self _ _ _
                      · rw [Function.update_of_ne equal]
                        exact agree other old

theorem trace_stepWith (pubB : Public) (x : Op × ℕ) (s : Replayer)
    (t : Sponge.Base L B C → Sponge.Y S C)
    (next : Replayer →
      Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) (Except Stop Unit))
    (stepped : ((step pub proof messages x.2 x.1).run).run (reideal s (some (answerOf fs))) =
      ((step pubB proof messages x.2 x.1).run).run (reideal s (some (answerOf fs))))
    (same : ∀ p, asks proof messages s x.1 = some p → answerOf t p.1 p.2 = answerOf fs p.1 p.2)
    {after : Replayer}
    (ran : ((step pubB proof messages x.2 x.1).run).run (reideal s (some (answerOf fs))) =
      (.ok (), after)) :
    ∃ s₁, after = reideal s₁ (some (answerOf fs)) ∧
      (stepWith pub proof messages x s t next).trace fs hash = (next s₁).trace fs hash := by
  have key := step_reideal pub proof messages x.2 x.1 (reideal s (some (answerOf t)))
    (a := answerOf t) (a' := answerOf fs) rfl
    (fun p asked => same p ((asks_reideal proof messages s _ x.1).symm.trans asked))
  unfold BlindAt at key
  have state : reideal (reideal s (some (answerOf t))) (some (answerOf fs)) =
      reideal s (some (answerOf fs)) := rfl
  rw [state] at key
  rw [← stepped, key] at ran
  unfold stepWith
  revert ran
  rcases ((step pub proof messages x.2 x.1).run).run (reideal s (some (answerOf t))) with
    ⟨_ | ⟨⟩, s₁⟩
  · intro ran
    cases ran
  · intro ran
    cases ran
    exact ⟨s₁, rfl, rfl⟩

/-- The first operation of `go` asks for the base of its query, when `known` does not have this
base. -/
theorem go_asked (x : Op × ℕ) (l : List (Op × ℕ)) (s : Replayer)
    (t : Sponge.Base L B C → Sponge.Y S C) (known : List (Sponge.Base L B C))
    {base : Sponge.Base L B C}
    (asked : baseOf L B C S (asks proof messages s x.1) = some base) :
    base ∈ known ∨ Sum.inl base ∈ (go (D := D) pub proof messages (x :: l) s t known).trace fs
      hash := by
  obtain ⟨op, i⟩ := x
  rw [go]
  dsimp only at asked ⊢
  cases terminal : op.terminalLevel with
  | some level =>
      obtain rfl := Op.terminalLevel_some terminal
      have quiet : baseOf L B C S (asks proof messages s (Op.terminal level)) = none := rfl
      rw [quiet] at asked
      cases asked
  | none =>
      dsimp only
      rw [asked]
      dsimp only
      by_cases member : base ∈ known
      · exact Or.inl member
      · rw [ite_eq_right member]
        exact Or.inr List.mem_cons_self

/-- The terminal as the first operation of `go` asks for the bytes of the two last blocks. -/
theorem go_digests (i level : ℕ) (l : List (Op × ℕ)) (s : Replayer)
    (t : Sponge.Base L B C → Sponge.Y S C) (known : List (Sponge.Base L B C))
    {blocks : Array Fq × List (Array Fq)} (found : blocksIn pub s level = some blocks) :
    (∀ short, Sum.inr (strOf (eBytes pub.layout.terminal.shape blocks.1) short) ∈
      (go (D := D) pub proof messages ((Op.terminal level, i) :: l) s t known).trace fs hash) ∧
    ∀ short, Sum.inr (strOf (tBytes pub.layout.terminal.shape blocks.2) short) ∈
      (go (D := D) pub proof messages ((Op.terminal level, i) :: l) s t known).trace fs hash := by
  rw [go]
  dsimp only [Op.terminalLevel]
  rw [found]
  exact ⟨fun short => askDigest_asked fs hash _ _ short,
    fun short => mem_askDigest (askDigest_asked fs hash _ _ short)⟩

/-- After a step with no stop, the queries of `go` have the queries of `go` on the next
operations, and each new known base is a query. -/
theorem go_tail (x : Op × ℕ) (l : List (Op × ℕ)) (s : Replayer)
    (t : Sponge.Base L B C → Sponge.Y S C) (known : List (Sponge.Base L B C))
    (agree : ∀ base ∈ known, t base = fs base) {after : Replayer}
    (ran : ((step (pub.withDigest (digestOf hash)) proof messages x.2 x.1).run).run
      (reideal s (some (answerOf fs))) = (.ok (), after)) :
    ∃ (s₁ : Replayer) (t' : Sponge.Base L B C → Sponge.Y S C)
      (known' : List (Sponge.Base L B C)),
      after = reideal s₁ (some (answerOf fs)) ∧ (∀ base ∈ known', t' base = fs base) ∧
      (∀ query, query ∈ (go (D := D) pub proof messages l s₁ t' known').trace fs hash →
        query ∈ (go (D := D) pub proof messages (x :: l) s t known).trace fs hash) ∧
      ∀ base ∈ known', base ∈ known ∨
        Sum.inl base ∈ (go (D := D) pub proof messages (x :: l) s t known).trace fs hash := by
  obtain ⟨op, i⟩ := x
  rw [go]
  dsimp only at ran ⊢
  cases terminal : op.terminalLevel with
  | some level =>
      obtain rfl := Op.terminalLevel_some terminal
      dsimp only
      have quiet : ∀ p, asks proof messages s (Op.terminal level) = some p →
          answerOf t p.1 p.2 = answerOf fs p.1 p.2 := fun p asked => by cases asked
      cases found : blocksIn pub s level with
      | none =>
          dsimp only
          obtain ⟨s₁, rfl, trace⟩ := trace_stepWith pub proof messages fs hash _
            (Op.terminal level, i) s t (fun s₁ => go (D := D) pub proof messages l s₁ t known)
            (step_digest_congr pub proof messages pub.digest (digestOf hash) i _ _
              fun level' term blocks' found' => by
                cases term
                have other : blocksIn pub s level = some blocks' := found'
                rw [found] at other
                cases other)
            quiet ran
          exact ⟨s₁, t, known, rfl, agree, fun query later => by rw [trace]; exact later,
            fun base member => Or.inl member⟩
      | some blocks =>
          dsimp only
          obtain ⟨s₁, rfl, trace⟩ := trace_stepWith (pub.withDigest (patched
              (eBytes pub.layout.terminal.shape blocks.1)
              (tBytes pub.layout.terminal.shape blocks.2)
              (digestOf hash (eBytes pub.layout.terminal.shape blocks.1))
              (digestOf hash (tBytes pub.layout.terminal.shape blocks.2))))
            proof messages fs hash _ (Op.terminal level, i) s t
            (fun s₁ => go (D := D) pub proof messages l s₁ t known)
            (step_digest_congr pub proof messages _ (digestOf hash) i _ _
              fun level' term blocks' found' => by
                cases term
                have other : blocksIn pub s level = some blocks' := found'
                rw [found] at other
                cases other
                exact ⟨patched_e _ _ _, patched_t _ _ _ _⟩)
            quiet ran
          exact ⟨s₁, t, known, rfl, agree,
            fun query later => mem_askDigest (mem_askDigest (by rw [trace]; exact later)),
            fun base member => Or.inl member⟩
  | none =>
      dsimp only
      have free : ((step pub proof messages i op).run).run (reideal s (some (answerOf fs))) =
          ((step (pub.withDigest (digestOf hash)) proof messages i op).run).run
            (reideal s (some (answerOf fs))) := by
        rw [step_withDigest pub _ proof messages i (Op.terminalLevel_none terminal)]
      cases asked : baseOf L B C S (asks proof messages s op) with
      | none =>
          dsimp only
          obtain ⟨s₁, rfl, trace⟩ := trace_stepWith pub proof messages fs hash _ (op, i) s t
            (fun s₁ => go (D := D) pub proof messages l s₁ t known) free
            (answerOf_agree fun base found => by rw [asked] at found; cases found) ran
          exact ⟨s₁, t, known, rfl, agree, fun query later => by rw [trace]; exact later,
            fun base member => Or.inl member⟩
      | some base =>
          dsimp only
          by_cases member : base ∈ known
          · rw [ite_eq_left member]
            obtain ⟨s₁, rfl, trace⟩ := trace_stepWith pub proof messages fs hash _ (op, i) s t
              (fun s₁ => go (D := D) pub proof messages l s₁ t known) free
              (answerOf_agree fun other found => by
                rw [asked] at found
                cases found
                exact agree base member) ran
            exact ⟨s₁, t, known, rfl, agree, fun query later => by rw [trace]; exact later,
              fun base member => Or.inl member⟩
          · rw [ite_eq_right member]
            obtain ⟨s₁, rfl, trace⟩ := trace_stepWith pub proof messages fs hash _ (op, i) s
              (Function.update t base (fs base))
              (fun s₁ => go (D := D) pub proof messages l s₁ (Function.update t base (fs base))
                (base :: known)) free
              (answerOf_agree fun other found => by
                rw [asked] at found
                cases found
                exact Function.update_self _ _ _) ran
            refine ⟨s₁, Function.update t base (fs base), base :: known, rfl,
              fun other inside => ?_,
              fun query later => List.mem_cons_of_mem _ (by rw [trace]; exact later),
              fun other inside => ?_⟩
            · rcases List.mem_cons.mp inside with rfl | old
              · exact Function.update_self _ _ _
              · by_cases equal : other = base
                · rw [equal]
                  exact Function.update_self _ _ _
                · rw [Function.update_of_ne equal]
                  exact agree other old
            · rcases List.mem_cons.mp inside with rfl | old
              · exact Or.inr List.mem_cons_self
              · exact Or.inl old

/-- An operation of `go` after a run with no stop asks for the base of its query, when `known`
does not have this base. -/
theorem go_asked_later : ∀ (pre : List (Op × ℕ)) (x : Op × ℕ) (post : List (Op × ℕ))
    (s : Replayer) (t : Sponge.Base L B C → Sponge.Y S C) (known : List (Sponge.Base L B C)),
    (∀ base ∈ known, t base = fs base) → ∀ {before : Replayer},
    runFrom (pub.withDigest (digestOf hash)) proof messages pre
      (reideal s (some (answerOf fs))) = (.ok (), before) →
    ∀ {base : Sponge.Base L B C}, baseOf L B C S (asks proof messages before x.1) = some base →
    base ∈ known ∨ Sum.inl base ∈
      (go (D := D) pub proof messages (pre ++ x :: post) s t known).trace fs hash
  | [], x, post, s, t, known, _, before, ran, base, asked => by
      rw [runFrom_nil] at ran
      cases ran
      rw [asks_reideal] at asked
      exact go_asked pub proof messages fs hash x post s t known asked
  | y :: pre, x, post, s, t, known, agree, before, ran, base, asked => by
      rw [runFrom_cons] at ran
      rcases stepped : ((step (pub.withDigest (digestOf hash)) proof messages y.2 y.1).run).run
        (reideal s (some (answerOf fs))) with ⟨_ | ⟨⟩, after⟩
      · rw [stepped] at ran
        cases ran
      · rw [stepped] at ran
        obtain ⟨s₁, t', known', rfl, agree', later, old⟩ :=
          go_tail (D := D) pub proof messages fs hash y (pre ++ x :: post) s t known agree stepped
        rcases go_asked_later pre x post s₁ t' known' agree' ran asked with inside | queried
        · exact old base inside
        · exact Or.inr (later _ queried)

/-- The terminal of `go` after a run with no stop asks for the bytes of the two last blocks. -/
theorem go_digests_later : ∀ (pre : List (Op × ℕ)) (i level : ℕ) (post : List (Op × ℕ))
    (s : Replayer) (t : Sponge.Base L B C → Sponge.Y S C) (known : List (Sponge.Base L B C)),
    (∀ base ∈ known, t base = fs base) → ∀ {before : Replayer},
    runFrom (pub.withDigest (digestOf hash)) proof messages pre
      (reideal s (some (answerOf fs))) = (.ok (), before) →
    ∀ {blocks : Array Fq × List (Array Fq)}, blocksIn pub before level = some blocks →
    (∀ short, Sum.inr (strOf (eBytes pub.layout.terminal.shape blocks.1) short) ∈
      (go (D := D) pub proof messages (pre ++ (Op.terminal level, i) :: post) s t known).trace
        fs hash) ∧
    ∀ short, Sum.inr (strOf (tBytes pub.layout.terminal.shape blocks.2) short) ∈
      (go (D := D) pub proof messages (pre ++ (Op.terminal level, i) :: post) s t known).trace
        fs hash
  | [], i, level, post, s, t, known, _, before, ran, blocks, found => by
      rw [runFrom_nil] at ran
      cases ran
      exact go_digests pub proof messages fs hash i level post s t known found
  | y :: pre, i, level, post, s, t, known, agree, before, ran, blocks, found => by
      rw [runFrom_cons] at ran
      rcases stepped : ((step (pub.withDigest (digestOf hash)) proof messages y.2 y.1).run).run
        (reideal s (some (answerOf fs))) with ⟨_ | ⟨⟩, after⟩
      · rw [stepped] at ran
        cases ran
      · rw [stepped] at ran
        obtain ⟨s₁, t', known', rfl, agree', later, -⟩ :=
          go_tail (D := D) pub proof messages fs hash y (pre ++ (Op.terminal level, i) :: post)
            s t known agree stepped
        obtain ⟨first, second⟩ := go_digests_later pre i level post s₁ t' known' agree' ran found
        exact ⟨fun short => later _ (first short), fun short => later _ (second short)⟩

/-- The count of the sponge queries of `go` on the operations `l` is at most: one for each work,
and one for each draw that is not directly after a work or a draw. `covered`: the operation
before `l` is a work or a draw. -/
def askCount : List (Op × ℕ) → Bool → ℕ
  | [], _ => 0
  | x :: l, covered =>
      if isWork x.1 then askCount l true + 1
      else if isDraw x.1 then askCount l true + (if covered then 0 else 1)
      else askCount l false

/-- The count of the digest queries of `go` on the operations `l` is at most: two for each
terminal. -/
def digestCount : List (Op × ℕ) → ℕ
  | [] => 0
  | x :: l => digestCount l + (if x.1.isTerminal then 2 else 0)

theorem bound_stepWith (x : Op × ℕ) (s : Replayer) (t : Sponge.Base L B C → Sponge.Y S C)
    (next : Replayer →
      Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) (Except Stop Unit))
    {queries digests : ℕ}
    (each : ∀ s₁, ((step pub proof messages x.2 x.1).run).run
      (reideal s (some (answerOf t))) = (.ok (), s₁) → (next s₁).Bound queries digests) :
    (stepWith pub proof messages x s t next).Bound queries digests := by
  unfold stepWith
  rcases found : ((step pub proof messages x.2 x.1).run).run (reideal s (some (answerOf t)))
    with ⟨_ | ⟨⟩, s₁⟩
  · exact .done _ _ _
  · exact each s₁ found

theorem bound_ask (queries extra digests : ℕ) (known : List (Sponge.Base L B C))
    (base? : Option (Sponge.Base L B C)) (t : Sponge.Base L B C → Sponge.Y S C)
    (next : (Sponge.Base L B C → Sponge.Y S C) → List (Sponge.Base L B C) →
      Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) (Except Stop Unit))
    (need : ∀ base, base? = some base → base ∉ known → 1 ≤ extra)
    (each : ∀ t' known', (∀ base, base? = some base → base ∈ known') →
      (next t' known').Bound queries digests) :
    (match base? with
      | some base =>
          if base ∈ known then next t known
          else .challenge base fun answer => next (Function.update t base answer) (base :: known)
      | none => next t known).Bound (queries + extra) digests := by
  cases base? with
  | none =>
      exact (each t known fun _ found => by cases found).mono (Nat.le_add_right _ _)
        (Nat.le_refl _)
  | some base =>
      dsimp only
      by_cases member : base ∈ known
      · rw [ite_eq_left member]
        exact (each t known fun other found => by cases found; exact member).mono
          (Nat.le_add_right _ _) (Nat.le_refl _)
      · rw [ite_eq_right member]
        obtain ⟨e, rfl⟩ : ∃ e, extra = e + 1 :=
          ⟨extra - 1, by have := need base rfl member; omega⟩
        exact .challenge base _ (queries + e) digests fun answer =>
          (each _ _ fun other found => by cases found; exact List.mem_cons_self).mono
            (Nat.le_add_right _ _) (Nat.le_refl _)

/-- The queries of `go`: at most `askCount` queries of the sponge and `digestCount` queries of
the digest oracle, when no work of `l` has zero bits. -/
theorem bound_go : ∀ (l : List (Op × ℕ)) (s : Replayer) (t : Sponge.Base L B C → Sponge.Y S C)
    (known : List (Sponge.Base L B C)) (covered : Bool),
    (∀ x ∈ l, ∀ site level, x.1 ≠ .work site level 0) →
    (covered = true → Covered S known s.sponge.events) →
    (go (D := D) pub proof messages l s t known).Bound (askCount l covered) (digestCount l)
  | [], _, _, _, _, _, _ => .done _ _ _
  | (op, i) :: l, s, t, known, covered, positive, cover => by
      have later : ∀ x ∈ l, ∀ site level, x.1 ≠ .work site level 0 := fun x member =>
        positive x (List.mem_cons_of_mem _ member)
      rw [go]
      dsimp only
      cases terminal : op.terminalLevel with
      | some level =>
          obtain rfl := Op.terminalLevel_some terminal
          dsimp only
          have tail : ∀ pub' : Public, (stepWith pub' proof messages (Op.terminal level, i) s t
              fun s₁ => go (D := D) pub proof messages l s₁ t known).Bound (askCount l false)
                (digestCount l) := fun pub' =>
            bound_stepWith pub' proof messages _ s t _ fun s₁ _ =>
              bound_go l s₁ t known false later fun on => Bool.noConfusion on
          show Digest.Run.Bound _ (askCount l false) (digestCount l + 2)
          cases blocksIn pub s level with
          | none => exact (tail pub).mono (Nat.le_refl _) (Nat.le_add_right _ _)
          | some blocks =>
              exact bound_askDigest _ _ fun e => bound_askDigest _ _ fun d => tail _
      | none =>
          dsimp only
          have digests : digestCount ((op, i) :: l) = digestCount l := by
            show digestCount l + (if op.isTerminal then 2 else 0) = digestCount l
            rw [Op.terminalLevel_none terminal]
            rfl
          rw [digests]
          have continued : ∀ (t' : Sponge.Base L B C → Sponge.Y S C)
              (known' : List (Sponge.Base L B C)) (covered' : Bool),
              (covered' = true → ((∃ site level bits, op = .work site level bits ∧ bits ≠ 0) ∨
                isDraw op = true) ∧
                ∀ base, baseOf L B C S (asks proof messages s op) = some base → base ∈ known') →
              (stepWith pub proof messages (op, i) s t' fun s₁ =>
                go (D := D) pub proof messages l s₁ t' known').Bound (askCount l covered')
                  (digestCount l) := by
            intro t' known' covered' inside
            refine bound_stepWith pub proof messages _ s t' _ fun s₁ ran =>
              bound_go l s₁ t' known' covered' later fun on => ?_
            obtain ⟨squeezes, listed⟩ := inside on
            obtain ⟨events, count, asked, after⟩ := step_asked pub proof messages i op
              (reideal s (some (answerOf t'))) squeezes (by rw [ran])
            rw [ran] at after
            rw [asks_reideal] at asked
            have events' : s₁.sponge.events = .squeeze count :: events := after
            rw [events']
            exact covered_squeeze fun base found => listed base (by rw [asked]; exact found)
          by_cases work : isWork op = true
          · have count : askCount ((op, i) :: l) covered = askCount l true + 1 := by
              show (if isWork op then _ else _) = _
              rw [ite_eq_left work]
            rw [count]
            have squeezes : (∃ site level bits, op = .work site level bits ∧ bits ≠ 0) ∨
                isDraw op = true := by
              cases op with
              | work site level bits =>
                  refine Or.inl ⟨site, level, bits, rfl, fun zero => ?_⟩
                  subst zero
                  exact positive _ List.mem_cons_self site level rfl
              | _ => cases work
            exact bound_ask (askCount l true) 1 (digestCount l) known _ t
              (fun t' known' => stepWith pub proof messages (op, i) s t' fun s₁ =>
                go (D := D) pub proof messages l s₁ t' known')
              (fun _ _ _ => Nat.le_refl 1)
              fun t' known' listed => continued t' known' true fun _ => ⟨squeezes, listed⟩
          · by_cases draws : isDraw op = true
            · have count : askCount ((op, i) :: l) covered =
                  askCount l true + (if covered then 0 else 1) := by
                show (if isWork op then _ else _) = _
                rw [ite_eq_right work, ite_eq_left draws]
              rw [count]
              refine bound_ask (askCount l true) _ (digestCount l) known _ t
                (fun t' known' => stepWith pub proof messages (op, i) s t' fun s₁ =>
                  go (D := D) pub proof messages l s₁ t' known')
                (fun base found missing => ?_)
                fun t' known' listed => continued t' known' true fun _ => ⟨Or.inr draws, listed⟩
              cases covered with
              | false => exact Nat.le_refl 1
              | true =>
                  exfalso
                  have events : ∃ count, asks proof messages s op =
                      some (s.sponge.events, count) := by
                    cases op <;> first | exact ⟨_, rfl⟩ | cases draws
                  obtain ⟨count, asked⟩ := events
                  rw [asked] at found
                  simp only [baseOf, Option.bind_some, Option.map_eq_some_iff] at found
                  obtain ⟨⟨base', sub⟩, split_, rfl⟩ := found
                  exact missing (cover rfl count base' sub split_)
            · have count : askCount ((op, i) :: l) covered = askCount l false := by
                show (if isWork op then _ else _) = _
                rw [ite_eq_right work, ite_eq_right draws]
              rw [count, asks_quiet proof messages s (Bool.eq_false_iff.mpr work)
                (Bool.eq_false_iff.mpr draws)]
              exact continued t known false fun on => Bool.noConfusion on

end Run

end Verifier

section Verifier

open NightstreamFPrime.Spec.Compression.Akita.Replay.Verifier

variable {L B C S D : ℕ}

/-- The outcome of the verifier for the result of the replay of the decoded messages and the
count of the bytes after the last message. -/
def outcomeOf (messages : Array Message) (rest : ℕ) (result : Except Stop Unit) : Outcome :=
  match verdictOf (match result with
    | .ok () => none
    | .error stop => some stop) (fun _ => Verdict.accept) messages with
  | .accept => if rest = 0 then .verdict .accept else .trailing
  | verdict => .verdict verdict

/-- The byte verifier as an algorithm with the two oracles: the sponge table on the bases
`Base L B C`, and the digest table on the strings of at most `D` bytes. -/
noncomputable def verifierRun (pub : Public) (proof : ByteArray) :
    Digest.Run (Sponge.Base L B C) (Sponge.Y S C) (Sponge.Str D) (Bytes 32) Outcome :=
  if pub.shaped then
    match script pub.layout pub.plan with
    | .error _ => .done (.verdict (.reject 0 "plan"))
    | .ok ops =>
        match decode ops proof with
        | .error site => .done (.malformed site)
        | .ok (messages, rest) =>
            (go pub.sealed proof messages ops.zipIdx.toList
              (channelStart (pub.withSponge Duplex.empty)) (fun _ _ _ => 0) []).map
              (outcomeOf messages rest)
  else .done (.verdict (.reject 0 "statement"))

theorem channel_reideal (session descriptor : ByteArray) (d : Duplex)
    (ideal : Option (List Duplex.Event → ℕ → ByteArray)) :
    channel session descriptor (d.reideal ideal) =
      (channel session descriptor d).reideal ideal := by
  unfold channel
  rw [Duplex.absorb_reideal, Duplex.absorb_reideal, Duplex.absorb_reideal]

/-- The public input with the sponge of the table `fs` and the digest of the table `hash`. -/
noncomputable abbrev Public.withOracles (pub : Public) (fs : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) : Public :=
  (pub.withSponge (spongeOf fs)).withDigest (digestOf hash)

/-- The run of the verifier with the two oracles is the run from the start state of `go` with
the answers of `fs`. -/
theorem runFrom_spongeOf (pub : Public) (proof : ByteArray) (messages : Array Message)
    (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32)
    (l : List (Op × ℕ)) :
    runFrom (pub.withOracles fs hash).sealed proof messages l
        (channelStart (pub.withOracles fs hash).sealed) =
      runFrom (pub.sealed.withDigest (digestOf hash)) proof messages l
        (reideal (channelStart (pub.withSponge Duplex.empty)) (some (answerOf fs))) := by
  have start : channelStart (pub.withOracles fs hash).sealed =
      reideal (channelStart (pub.withSponge Duplex.empty)) (some (answerOf fs)) :=
    congrArg (fun d => (⟨d, 0, {}, {}, {}, {}⟩ : Replayer))
      (channel_reideal pub.session pub.descriptor Duplex.empty (some (answerOf fs)))
  rw [start]
  exact runFrom_withSponge (pub.sealed.withDigest (digestOf hash)) proof messages
    (spongeOf fs) _ _

/-- The output of `verifierRun` for the tables `fs` and `hash` is the outcome of `verifyProof`
with the sponge of `fs` and the digest of `hash`. -/
theorem verifierRun_eval (pub : Public) (proof : ByteArray) (fs : Sponge.Base L B C → Sponge.Y S C)
    (hash : Sponge.Str D → Bytes 32) :
    (verifierRun pub proof).eval fs hash =
      verifyProof ((pub.withSponge (spongeOf fs)).withDigest (digestOf hash)) proof := by
  unfold verifierRun verifyProof
  have shaped : ((pub.withSponge (spongeOf fs)).withDigest (digestOf hash)).shaped =
      pub.shaped := rfl
  rw [shaped]
  cases pub.shaped with
  | false => rfl
  | true =>
      simp only [↓reduceIte]
      unfold runProof
      have scripted :
          script ((pub.withSponge (spongeOf fs)).withDigest (digestOf hash)).sealed.layout
          ((pub.withSponge (spongeOf fs)).withDigest (digestOf hash)).sealed.plan =
            script pub.layout pub.plan := rfl
      rw [scripted]
      cases script pub.layout pub.plan with
      | error reason => rfl
      | ok ops =>
          dsimp only
          unfold verifyBytes
          cases decode ops proof with
          | error site => rfl
          | ok result =>
              obtain ⟨messages, rest⟩ := result
              dsimp only
              rw [Digest.Run.eval_map, eval_go pub.sealed proof messages fs hash _ _ _ []
                (fun _ member => absurd member List.not_mem_nil)]
              unfold verdict
              rw [replay_stop]
              rw [runFrom_spongeOf]
              rfl

/-- The queries of `verifierRun` on a statement with the shape of the schedule row, a script and
a proof that decodes are the queries of `go`. -/
theorem verifierRun_trace (pub : Public) (proof : ByteArray)
    (fs : Sponge.Base L B C → Sponge.Y S C) (hash : Sponge.Str D → Bytes 32) {ops : Array Op}
    {messages : Array Message} {rest : ℕ} (shaped : pub.shaped = true)
    (scripted : script pub.layout pub.plan = .ok ops)
    (decoded : decode ops proof = .ok (messages, rest)) :
    (verifierRun pub proof).trace fs hash =
      (go (D := D) pub.sealed proof messages ops.zipIdx.toList
        (channelStart (pub.withSponge Duplex.empty)) (fun _ _ _ => 0) []).trace fs hash := by
  unfold verifierRun
  rw [ite_eq_left shaped]
  simp only [scripted, decoded]
  exact Digest.Run.trace_map _ _ _ _

section Queries

variable (pub : Public) (proof : ByteArray) (fs : Sponge.Base L B C → Sponge.Y S C)
  (hash : Sponge.Str D → Bytes 32) {ops : Array Op} {messages : Array Message} {rest : ℕ}
  (shaped : pub.shaped = true) (scripted : script pub.layout pub.plan = .ok ops)
  (decoded : decode ops proof = .ok (messages, rest)) {c : ℕ}

theorem go_cut {op : Op} (at_ : ops[c]? = some op)
    (ran : (runFrom (pub.withOracles fs hash).sealed proof messages (ops.zipIdx.toList.take c)
      (channelStart (pub.withOracles fs hash).sealed)).1 = .ok ()) :
    ops.zipIdx.toList = ops.zipIdx.toList.take c ++ (op, c) :: ops.zipIdx.toList.drop (c + 1) ∧
      runFrom (pub.sealed.withDigest (digestOf hash)) proof messages (ops.zipIdx.toList.take c)
          (reideal (channelStart (pub.withSponge Duplex.empty)) (some (answerOf fs))) =
        (.ok (), stateBefore (pub.withOracles fs hash).sealed proof messages ops c) := by
  refine ⟨?_, ?_⟩
  · conv_lhs => rw [← List.take_append_drop (c + 1) ops.zipIdx.toList, take_succ_zipIdx ops at_]
    rw [List.append_assoc]
    rfl
  · rw [← runFrom_spongeOf]
    exact Prod.ext ran rfl

include shaped scripted decoded

/-- The base of the query of an operation of the script, after a run with no stop of the
operations before it, is a query of `verifierRun`. -/
theorem verifierRun_asked {op : Op} (at_ : ops[c]? = some op)
    (ran : (runFrom (pub.withOracles fs hash).sealed proof messages (ops.zipIdx.toList.take c)
      (channelStart (pub.withOracles fs hash).sealed)).1 = .ok ())
    {base : Sponge.Base L B C}
    (asked : baseOf L B C S (asks proof messages
      (stateBefore (pub.withOracles fs hash).sealed proof messages ops c) op) = some base) :
    Sum.inl base ∈ (verifierRun pub proof).trace fs hash := by
  obtain ⟨cut, before⟩ := go_cut pub proof fs hash at_ ran
  rw [verifierRun_trace pub proof fs hash shaped scripted decoded, cut]
  exact (go_asked_later pub.sealed proof messages fs hash _ (op, c) _ _ _ []
    (fun _ member => absurd member List.not_mem_nil) before asked).resolve_left
    List.not_mem_nil

/-- The bytes of the two last blocks of the terminal of the script, after a run with no stop of
the operations before it, are queries of `verifierRun` when they have at most `D` bytes. -/
theorem verifierRun_digests {level : ℕ} (at_ : ops[c]? = some (.terminal level))
    (ran : (runFrom (pub.withOracles fs hash).sealed proof messages (ops.zipIdx.toList.take c)
      (channelStart (pub.withOracles fs hash).sealed)).1 = .ok ())
    {blocks : Array Fq × List (Array Fq)}
    (found : blocksIn pub.sealed
      (stateBefore (pub.withOracles fs hash).sealed proof messages ops c) level = some blocks) :
    (∀ short, Sum.inr (strOf (eBytes pub.layout.terminal.shape blocks.1) short) ∈
      (verifierRun pub proof).trace fs hash) ∧
    ∀ short, Sum.inr (strOf (tBytes pub.layout.terminal.shape blocks.2) short) ∈
      (verifierRun pub proof).trace fs hash := by
  obtain ⟨cut, before⟩ := go_cut pub proof fs hash at_ ran
  rw [verifierRun_trace pub proof fs hash shaped scripted decoded, cut]
  exact go_digests_later pub.sealed proof messages fs hash _ c level _ _ _ []
    (fun _ member => absurd member List.not_mem_nil) before found

end Queries

/-- The bound on the queries of `verifierRun`: `queries` queries of the sponge and `digests`
queries of the digest oracle, for a script with no work of zero bits, `askCount` at most
`queries` and `digestCount` at most `digests`. -/
theorem verifierRun_bound (pub : Public) (proof : ByteArray) {queries digests : ℕ}
    (counts : ∀ ops, script pub.layout pub.plan = .ok ops →
      (∀ x ∈ ops.zipIdx.toList, ∀ site level, x.1 ≠ .work site level 0) ∧
        askCount ops.zipIdx.toList false ≤ queries ∧ digestCount ops.zipIdx.toList ≤ digests) :
    (verifierRun (L := L) (B := B) (C := C) (S := S) (D := D) pub proof).Bound queries
      digests := by
  unfold verifierRun
  split
  · cases scripted : script pub.layout pub.plan with
    | error reason => exact .done _ _ _
    | ok ops =>
        dsimp only
        cases decode ops proof with
        | error site => exact .done _ _ _
        | ok result =>
            obtain ⟨messages, rest⟩ := result
            dsimp only
            obtain ⟨positive, few, fewDigests⟩ := counts ops scripted
            exact ((bound_go pub.sealed proof messages _ _ _ [] false positive
              fun on => Bool.noConfusion on).mono few fewDigests).map _
  · exact .done _ _ _

end Verifier

end NightstreamFPrime.Spec.Compression.Akita.Replay
