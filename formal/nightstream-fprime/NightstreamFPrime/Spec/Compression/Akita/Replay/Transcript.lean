import Std.Data.HashMap
import NightstreamFPrime.Spec.Compression.Akita.Replay.Wire
import NightstreamFPrime.Spec.Compression.Akita.Replay.Keccak
import NightstreamFPrime.Spec.Compression.Akita.Replay.Relation

/-! The Fiat–Shamir transcript of the pinned Akita verifier on the decoded messages of a proof
(Akita revision `d50f700`, jolt-transcript `25d87ba`, spongefish 0.8.0).

- The channel (`new_verifier_channel`, `bind_instance`): the duplex sponge absorbs the session
  identifier, SHAKE128 of `irtf-cfrg-fiat-shamir/session-id`, 136 zero bytes and the tag
  `akita-pcs/standalone-session/v8` (`derive_session_id`); then `BytesMsg(session)`; then
  `PROTOCOL_TAG ‖ BytesMsg(descriptor)`. `BytesMsg(b)` is the `u64` length, `b` and zeros to a
  word (`jolt-transcript/src/codec.rs`).
- A public value and a prover message are absorbed as their bytes with zeros to a word
  (`ProofChannel::public`, `receive_atoms`). The payload of `z` is its length as one `u64` word,
  then its bytes (`receive_bounded_bytes`).
- A challenge vector of `len` elements of `E` is one squeeze of `poolBytes (2·len)` bytes: 8-byte
  little-endian candidates, a candidate at or above `q` rejected, the first `2·len` accepted
  candidates the coordinates in order (`jolt-transcript/src/challenge.rs`, `pool_bytes`,
  `decode_pool`).
- The work of a site with `g > 0` bits (`VerifierTranscript::check_grind`): the nonce is canonical
  and below `2^(g + 7)`; it is absorbed as one `u64` word; 8 bytes are squeezed and their
  little-endian word has `g` low bits zero. The response nonce of a fold is canonical, below
  `2^width` and absorbed as one word (`read_fold_response`, `value_fits`). The message of the fold
  challenges of a group is squeezed and not decoded here.
- `Value` of an absorb: `replay` computes it from the statement, the recorded messages and
  challenges, and the computations of `Value`. The rows of the relation of a root group are the
  last coefficients of its commitment bytes. The fused claim of a fold is `stage2`
  (`Replay/Relation.lean`); `replay` records its target, relation claim and opening claim.

The model of Akita is a random oracle; the transcript has no specification in Lean. The evidence
is the equality with the events of the vectors in `tests/AkitaVectors.lean`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

def tag (name : String) (length : ℕ) : ByteArray :=
  name.toUTF8 ++ ByteArray.mk (Array.replicate (length - name.utf8ByteSize) 0)

def zeros (count : ℕ) : ByteArray := ByteArray.mk (Array.replicate count 0)

def wordPadded (bytes : ByteArray) : ByteArray := bytes ++ zeros ((8 - bytes.size % 8) % 8)

def le64 (value : ℕ) : ByteArray :=
  ByteArray.mk ((Array.range 8).map fun i => (value / 2 ^ (8 * i) % 256).toUInt8)

/-- `BytesMsg(bytes)`. -/
def bytesMsg (bytes : ByteArray) : ByteArray := le64 bytes.size ++ wordPadded bytes

def extBytes (values : Array Ext) : ByteArray :=
  values.foldl (fun out e => out ++ le64 e.c0.word.toNat ++ le64 e.c1.word.toNat) ByteArray.empty

def sessionId : ByteArray :=
  Keccak.shake128 (tag "irtf-cfrg-fiat-shamir/session-id" 32 ++ zeros 136 ++
    tag "akita-pcs/standalone-session/v8" 32) 32

def protocolTag : ByteArray := tag "akita-pcs/transcript/v8/sha3" 32

/-- The digest of the canonical bytes of a root commitment (`CommitmentDigest`). -/
def commitmentDigest (commitment : ByteArray) : ByteArray :=
  Keccak.sha3_256 (tag "akita-pcs/commitment-digest/v8" 32 ++ commitment)

def channel (session descriptor : ByteArray) : Duplex :=
  ((Duplex.empty.absorb sessionId).absorb (bytesMsg session)).absorb
    (protocolTag ++ bytesMsg descriptor)

/-- `pool_bytes(len)` for `q = 2^64 − 59`: `R = 64 − ⌈log_2 59⌉ = 58`, the least `k` with
`(k + 1)·(R − ⌈log_2 (len + k)⌉) ≥ 161`, and 8 bytes for each of the `len + k` candidates. -/
def poolBytes (len : ℕ) : ℕ :=
  if len = 0 then 0 else
    let fits (k : ℕ) : Bool :=
      let L := Nat.size (len + k - 1)
      L < 58 && 161 ≤ (k + 1) * (58 - L)
    8 * (len + ((List.range 161).find? fits).getD 161)

def decodePool (bytes : ByteArray) (len : ℕ) : Option (Array Ext) := Id.run do
  let mut accepted : Array Fq := #[]
  for i in [0:bytes.size / 8] do
    if accepted.size = 2 * len then break
    if let some a := Fq.ofWord? (Wire.word bytes (8 * i)) then accepted := accepted.push a
  if accepted.size < 2 * len then return none
  return some ((Array.range len).map fun i => ⟨accepted[2 * i]!, accepted[2 * i + 1]!⟩)

def worked (check : ByteArray) (bits : ℕ) : Bool := (Wire.word check 0).toNat % 2 ^ bits = 0

/-- `Σ_i γ^i·c_i`. -/
def batched (children : Array Ext) (gamma : Ext) : Ext :=
  (children.foldl (fun (sum, weight) c => (sum + weight * c, weight * gamma))
    (Ext.zero, Ext.one)).1

/-- The input claim of the tensor reduction of one claim from the partials `p_0, p_1` and `η`: the
rows `(p_0.c0, p_1.c0)` and `(p_0.c1, p_1.c1)` with the weights `1 − η` and `η`. -/
def tensorClaim (partials : Array Ext) (eta : Ext) : Ext :=
  let p0 := partials[0]!
  let p1 := partials[1]!
  (Ext.one - eta) * ⟨p0.c0, p1.c0⟩ + eta * ⟨p0.c1, p1.c1⟩

/-- `ρ` with a zero coordinate at position `log_2 degree − 1`. -/
def packed (degree : ℕ) (rho : Array Ext) : Array Ext :=
  let at_ := Nat.log 2 degree - 1
  (rho.extract 0 at_).push Ext.zero ++ rho.extract at_ rho.size

/-- The coefficients of `count` elements at the end of the canonical bytes of a commitment. -/
def commitmentRows (commitment : ByteArray) (count : ℕ) : Array Fq :=
  (Array.range count).map fun j => ⟨Wire.word commitment (commitment.size - 8 * count + 8 * j)⟩

inductive Stop
  | noWork (op : ℕ) (site : String)
  | mismatch (op : ℕ)
deriving DecidableEq, Repr

structure Derived where
  challenges : Array (Key × Array Ext) := #[]
  works : Array (Site × ℕ × ℕ × Bool) := #[]
  absorbs : Array (Key × ByteArray) := #[]
  values : Array (Key × Ext) := #[]
  squeezed : ℕ := 0

structure Replayer where
  sponge : Duplex
  next : ℕ
  env : Std.HashMap Key (Array Ext)
  fields : Std.HashMap Key (Array Fq)
  derived : Derived

abbrev ReplayM := ExceptT Stop (StateM Replayer)

namespace Replayer

def recorded (key : Key) : ReplayM (Array Ext) := do
  return ((← get).env.get? key).getD #[]

def record (key : Key) (values : Array Ext) : ReplayM Unit :=
  modify fun s => { s with env := s.env.insert key (((s.env.get? key).getD #[]) ++ values) }

def absorb (bytes : ByteArray) : ReplayM Unit :=
  modify fun s => { s with sponge := s.sponge.absorb bytes }

def squeeze (count : ℕ) : ReplayM ByteArray := do
  let s ← get
  let (out, sponge) := s.sponge.squeeze count
  set { s with sponge, derived := { s.derived with squeezed := s.derived.squeezed + count } }
  return out

def message (i : ℕ) (messages : Array Message) : ReplayM Message := do
  let s ← get
  let some m := messages[s.next]? | throw (.mismatch i)
  set { s with next := s.next + 1 }
  return m

def fieldsOf (key : Key) : ReplayM (Array Fq) := do
  return ((← get).fields.get? key).getD #[]

def first (key : Key) : ReplayM Ext := do
  return ((← recorded key)[0]?).getD Ext.zero

/-- The claims of Stage 2 of the fold `F` at `level` (`fold/mod.rs`, `verify_fold`): the row
coefficients (one claim: `1`), the claims (the openings, or the final claims of the tensor
reduction at a level with evaluation trace), the public rows of the relation, and
`β·image + η·virtual` with `η·virtual` only on the Euclidean route
(`batch_l2_virtual_evaluations`). -/
def stage2Of (pub : Public) (level : ℕ) (F : Fold) : ReplayM Stage2 := do
  let at_ (site : String) : Key := ⟨level, site, none⟩
  let runs ← F.rhs.mapM fun run => do
    let data ← match run.source with
      | .root g => pure (commitmentRows ((pub.groups[g]?.map (·.commitment)).getD .empty)
          (run.degree * run.count))
      | .setup => pure pub.setupRows
      | .witness => fieldsOf ⟨level - 1, "next_witness", none⟩
      | .payload => fieldsOf (at_ "opening_payload")
    pure (run, data)
  let rows ← if 1 < F.groups.length then recorded (at_ "row_coefficients") else pure #[Ext.one]
  let claims ← recorded (at_ (if F.trace.isSome then "tensor.final_claims" else "openings"))
  let virtual ← if F.euclidean then do
      let eta ← first (at_ "norm.virtual_batch")
      pure (eta * batched (← recorded (at_ "norm.virtual_evaluations")) eta)
    else pure Ext.zero
  let rest := (← first (at_ "stage2_batch")) * (← first (at_ "range.image")) + virtual
  return stage2 (← recorded (at_ "tau1")).toList (← first (at_ "alpha")) runs F.openingRow rows
    claims rest

def value (pub : Public) (key : Key) : Value → ReplayM ByteArray
  | .digest g => return commitmentDigest ((pub.groups[g]?.map (·.commitment)).getD .empty)
  | .point g => return extBytes ((pub.groups[g]?.map (·.point)).getD #[])
  | .values => return extBytes (pub.groups.flatMap (·.value))
  | .slot => return pub.slot
  | .recorded keys => do
      return extBytes (← keys.foldlM (fun out k => return out ++ (← recorded k)) #[])
  | .zero => return extBytes #[Ext.zero]
  | .batched children gamma => do
      return extBytes #[batched (← recorded children) (← first gamma)]
  | .norm before merge energy => do
      let claim ← match before with
        | some (children, gamma) => do pure (batched (← recorded children) (← first gamma))
        | none => pure Ext.zero
      return extBytes #[claim + (← first merge) * (← first energy)]
  | .tensor partials eta => return extBytes #[tensorClaim (← recorded partials) (← first eta)]
  | .packed degree rho => return extBytes (packed degree (← recorded rho))
  | .fused F => do
      let c ← stage2Of pub key.level F
      let at_ (site : String) : Key := ⟨key.level, site, none⟩
      modify fun s => { s with derived := { s.derived with values := s.derived.values ++
        #[(at_ "opening_target", c.target), (at_ "relation_claim", c.relation),
          (at_ "opening_claim", c.opening)] } }
      return extBytes #[c.fused]

def step (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ) :
    Op → ReplayM Unit
  | .absorb key v => do
      let bytes ← value pub key v
      absorb (wordPadded bytes)
      match v with
      | .digest _ | .slot => pure ()
      | _ => record key (← (decodeExts bytes))
      modify fun s => { s with derived := { s.derived with
        absorbs := s.derived.absorbs.push (key, bytes) } }
  | .fields key _ => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
      match m.data with
      | .fields values => modify fun s => { s with fields := s.fields.insert key values }
      | _ => throw (.mismatch i)
  | .digest _ => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
  | .exts key _ => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
      match m.data with
      | .exts values => record key values
      | _ => throw (.mismatch i)
  | .integer key => do
      let m ← message i messages
      absorb (proof.extract m.offset (m.offset + m.size))
      match m.data with
      | .integer value => record key #[Ext.ofBase (Fq.ofNat value)]
      | _ => throw (.mismatch i)
  | .bounded _ _ => do
      let m ← message i messages
      absorb (le64 (m.size - 4) ++ wordPadded (proof.extract (m.offset + 4) (m.offset + m.size)))
  | .work site _ bits => do
      if bits = 0 then return
      let m ← message i messages
      let .nonce nonce canonical := m.data | throw (.mismatch i)
      unless canonical ∧ nonce < 2 ^ (bits + 7) do throw (.noWork i site.text)
      absorb (le64 nonce)
      let passed := worked (← squeeze 8) bits
      modify fun s => { s with derived := { s.derived with
        works := s.derived.works.push (site, m.offset, nonce, passed) } }
      unless passed do throw (.noWork i site.text)
  | .challenge key count => do
      let some values := decodePool (← squeeze (poolBytes (2 * count))) count
        | throw (.noWork i key.site)
      record key values
      modify fun s => { s with derived := { s.derived with
        challenges := s.derived.challenges.push (key, values) } }
  | .foldNonce level width => do
      let m ← message i messages
      let .nonce nonce canonical := m.data | throw (.mismatch i)
      unless canonical ∧ (32 ≤ width ∨ nonce < 2 ^ width) do
        throw (.noWork i s!"FoldResponse \{ level: {level} }")
      absorb (le64 nonce)
  | .squeeze _ _ bytes => discard (squeeze bytes)
where
  decodeExts (bytes : ByteArray) : ReplayM (Array Ext) :=
    return (Array.range (bytes.size / 16)).map fun j =>
      ⟨⟨Wire.word bytes (16 * j)⟩, ⟨Wire.word bytes (16 * j + 8)⟩⟩

end Replayer

/-- The replay of a decoded proof: the operations of the script from the sponge of the channel,
and what it derives. -/
def replay (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message) :
    Option Stop × Derived :=
  let run : ReplayM Unit := ops.zipIdx.forM fun (op, i) => Replayer.step pub proof messages i op
  let (result, s) := (run.run).run ⟨channel pub.session pub.descriptor, 0, {}, {}, {}⟩
  (match result with | .ok () => none | .error stop => some stop, s.derived)

/-- Each nonce of the messages is in range and each work predicate holds: the replay does not stop
at a work. -/
def works (pub : Public) (ops : Array Op) (proof : ByteArray) (messages : Array Message) : Bool :=
  match (replay pub ops proof messages).1 with
  | some (.noWork _ _) => false
  | _ => true

/-- The verdict on the decoded messages: `noWork` at the first work of the replay that fails, and
otherwise the verdict `checks` of the equations, which Lean does not have yet. -/
def verdict (pub : Public) (ops : Array Op) (proof : ByteArray) (checks : Array Message → Verdict)
    (messages : Array Message) : Verdict :=
  match (replay pub ops proof messages).1 with
  | some (.noWork op site) => .noWork op site
  | _ => checks messages

/-- `verifyBytes_noWork` of `docs/akita-work/match-design.md` section 5.1 for this layer: when the
equations never give `noWork`, a `noWork` outcome comes from a proof that decodes with no byte
left and whose messages do not have their work. -/
theorem verifyBytes_noWork (pub : Public) (ops : Array Op) (proof : ByteArray)
    (checks : Array Message → Verdict) (equations : ∀ m op site, checks m ≠ .noWork op site)
    {op : ℕ} {site : String}
    (outcome : verifyBytes (verdict pub ops proof checks) ops proof = .verdict (.noWork op site)) :
    ∃ messages, decode ops proof = .ok (messages, 0) ∧ works pub ops proof messages = false := by
  unfold verifyBytes at outcome
  rcases decoded : decode ops proof with failure | ⟨messages, rest⟩
  · simp [decoded] at outcome
  · by_cases empty : rest = 0
    · subst empty
      refine ⟨messages, rfl, ?_⟩
      simp only [decoded, ite_true, Outcome.verdict.injEq] at outcome
      unfold verdict at outcome
      unfold works
      split at outcome
      · rfl
      · exact absurd outcome (equations messages op site)
    · simp [decoded, empty] at outcome

end NightstreamFPrime.Spec.Compression.Akita.Replay
