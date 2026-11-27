import NightstreamFPrime.Spec.Compression.Akita.Replay.Golomb
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalKron
import NightstreamFPrime.Spec.Compression.Akita.Replay.TerminalSetup
import NightstreamFPrime.Spec.Compression.Akita.Terminal
import NightstreamFPrime.Spec.Compression.Akita.TracePack

/-! The checks of the terminal of the pinned Akita verifier on words (Akita revision `d50f700`,
`akita-verifier/src/terminal/direct.rs`, `verify_terminal_ring_relations` and
`verify_terminal_trace`, after `fold/terminal.rs`), and their relation to the terminal of the
specification (`Akita/Terminal.lean`, Akita Section 8.2, (128) to (131)).

- `terminalRun` decodes `z` (`Golomb.decode`), checks `‖z‖_2² ≤ S_max` (`centeredNormSq`), the
  fold challenges (`sparseValid`) and the inverse of the last one (`Poly.inverse`), recovers the
  last block of `e` from the consistency row and the last block of `t` from the rows of `A` with
  this inverse (`recover`, each checked by the product), compares their field digests with the
  digests of the proof, and checks the trace row. It returns the decoded `z`, its squared norm,
  the two recovered blocks and the opening, or the name of the first check that fails.
- `folded` is `Σ_{i<B−1} c_i·e_i`, `reduced` is `Σ_x a_x·Σ_u b^u·z_{x,u}` with `a_x = ẽq(ρ_pos, x)`
  in the fixed subfield, `aRow` is row `r` of `A·z` with `A` the prefix view of the setup
  (ring `r·M·δ + c` of the flat setup, `terminal/ntt.rs`), `tPartial` is `Σ_{i<B−1} c_i·t_{i,r}`,
  and `opening` is `Σ_i χ_blk(i)·T_{ρ_pack}(e_i)` with `T` of `recover_psi_inner_product`.
- `terminalRun_ok_iff`: for a payload that decodes, the run accepts exactly when the norm and the
  challenge checks pass, the last challenge is a unit, and some last blocks of `e` and `t` with
  the digests of the proof make the full partials, the full state and `z` pass the checks
  `TerminalStatement.Accepts` (128), (129) and (130) of the statement `statementOf`, which is
  `TerminalStatement.ofPoint` of `Extract/TraceLink.lean` for the split point
  `(ρ_pack, ρ_pos, ρ_blk)` of `prepare_opening_point`. This is the last-block variant of Akita
  Section 8.2: the blocks of (131) are the only ones with the two relations.
  `terminalRun_decode` is the rejection of a payload that does not decode. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

open NightstreamFPrime.Spec.Compression.Akita
open NightstreamFPrime.Spec.Compression.Lib (eqAt)

/-- The prover messages that the checks read: the sent blocks of `e` and of `t` (the last block
of each is omitted), the two digests and the payload of `z`. -/
structure TerminalMessages where
  e : Array Fq
  eDigest : ByteArray
  t : Array Fq
  tDigest : ByteArray
  payload : ByteArray

/-- The values of the transcript that the checks read: the point `ρ` and the pair `(θ, v)` of the
tensor reduction of the terminal (`final_relation`), and the fold challenges. -/
structure TerminalChallenges where
  point : List Ext
  scale : Ext
  target : Ext
  fold : List (List (ℕ × ℤ))

/-- What the checks compute: the decoded `z`, `‖z‖_2²`, the last block of `e`, the last block of
`t` (one ring element for each row of `A`), the rows of `A·z` and the opening
`Σ_i χ_blk(i)·T(e_i)`. -/
structure TerminalValues where
  z : Array ℤ
  normSq : ℕ
  eLast : Array Fq
  tLast : List (Array Fq)
  rows : List (Array Fq)
  opening : Ext

/-- `Σ_{i<n} f i` in `E`. -/
def sumExt (f : ℕ → Ext) : ℕ → Ext
  | 0 => Ext.zero
  | n + 1 => sumExt f n + f n

namespace Poly

/-- Element `i` of a flat array of elements of `R_{q,d}`. -/
def ring (d : ℕ) (a : Array Fq) (i : ℕ) : Array Fq := ofFn d fun k => get a (i * d + k)

/-- `toRq a = toRq b`, on the residues of the words. -/
def equal (d : ℕ) (a b : Array Fq) : Bool :=
  (List.range d).all fun k => (get a k).word.toNat % fieldModulus == (get b k).word.toNat % fieldModulus

/-- The canonical coefficients. -/
def canon (d : ℕ) (a : Array Fq) : Array Fq := ofFn d fun k => Fq.ofNat (get a k).word.toNat

end Poly

/-- `validate` of a sparse challenge: each position below `d`, no zero coefficient, no position
twice. -/
def sparseValid (d : ℕ) (c : List (ℕ × ℤ)) : Bool :=
  c.all (fun x => decide (x.1 < d) && x.2 != 0) && decide (c.map (·.1)).Nodup

/-- `2^{-1}` in `Z_q`. -/
def half : Fq := Fq.ofNat ((fieldModulus + 1) / 2)

/-- Entry `u` of `ψ^{-1}(Z)` for `d = 4·m` (`recover_psi_inner_product`): `(Z_u, (Z_{u+m} −
Z_{u+3m})/2)` for `u < m`, and `(Z_{u+m}, (Z_u + Z_{u+2m})/2)` for `m ≤ u < 2·m`. -/
def unpackAt (m : ℕ) (Z : Array Fq) (u : ℕ) : Ext :=
  if u < m then ⟨Poly.get Z u, half.mul ((Poly.get Z (u + m)).sub (Poly.get Z (u + 3 * m)))⟩
  else ⟨Poly.get Z (u + m), half.mul ((Poly.get Z u).add (Poly.get Z (u + 2 * m)))⟩

/-- `T_{ρ_pack}(Z) = Σ_u ẽq(ρ_pack, u)·ψ^{-1}(Z)_u`. -/
def functional (m : ℕ) (pack : List Ext) (Z : Array Fq) : Ext :=
  sumExt (fun u => eqIndex pack u * unpackAt m Z u) (2 * m)

section Checks

variable (S : Shape)

/-- The segments `(ρ_pack, ρ_pos, ρ_blk)` of the point of the terminal (`prepare_opening_point`:
`log_2 (d/2)` packed coordinates, `log_2 M` position coordinates, then the block coordinates). -/
def packPoint (ρ : List Ext) : List Ext := ρ.take (Nat.log 2 (S.innerDegree / 2))

def positionPoint (ρ : List Ext) : List Ext :=
  (ρ.drop (Nat.log 2 (S.innerDegree / 2))).take (Nat.log 2 S.blockLength)

def blockPoint (ρ : List Ext) : List Ext :=
  ρ.drop (Nat.log 2 (S.innerDegree / 2) + Nat.log 2 S.blockLength)

/-- Fold challenge `i` in `R_{q,d}`. -/
def challenge (ch : TerminalChallenges) (i : ℕ) : Array Fq :=
  Poly.ofSparse S.innerDegree (ch.fold.getD i [])

/-- Column `c` of `z`: coefficients `c·d` to `c·d + d − 1` of the decoded values. -/
def zRing (z : Array ℤ) (c : ℕ) : Array Fq :=
  Poly.ofFn S.innerDegree fun k => Fq.ofInt (z.getD (c * S.innerDegree + k) 0)

/-- `Σ_{i<B−1} c_i·e_i`. -/
def folded (msgs : TerminalMessages) (ch : TerminalChallenges) : Array Fq :=
  Poly.sum S.innerDegree (fun i => Poly.mul S.innerDegree (challenge S ch i)
    (Poly.ring S.innerDegree msgs.e i)) (S.blocks - 1)

/-- `Σ_x a_x·Σ_u b^u·z_{x,u}` with `a_x = ẽq(ρ_pos, x)` (`base_reduced_z` and
`accumulate_position_product`, with `gadget_row_scalars`). -/
def reduced (ch : TerminalChallenges) (z : Array ℤ) : Array Fq :=
  Poly.sum S.innerDegree (fun x => Poly.subMul S.innerDegree (S.innerDegree / 4)
    (eqIndex (positionPoint S ch.point) x)
    (Poly.sum S.innerDegree (fun u => Poly.scale S.innerDegree (Fq.ofNat (S.sourceBase ^ u))
      (zRing S z (x * S.sourceDigits + u))) S.sourceDigits)) S.blockLength

/-- Row `r` of `A·z`, with `A` the prefix view of the flat setup: entry `(r, c)` is the ring
element `r·M·δ + c` of the setup (`centered_rows`). For `M·δ·d ≤ 2^16` and `|z| ≤ 2^15` it is
`Poly.dotSmall`, and otherwise the schoolbook sum. -/
def aRow (setup : Array Fq) (z : Array ℤ) (r : ℕ) : Array Fq :=
  if S.blockLength * S.sourceDigits * S.innerDegree ≤ 2 ^ 16 ∧
      z.all (fun n => decide (n.natAbs ≤ 2 ^ 15)) = true then
    Poly.dotSmall S.innerDegree (S.blockLength * S.sourceDigits)
      (fun c => Poly.ring S.innerDegree setup (r * (S.blockLength * S.sourceDigits) + c))
      fun c j => z.getD (c * S.innerDegree + j) 0
  else
    Poly.sum S.innerDegree (fun c => Poly.mul S.innerDegree
      (Poly.ring S.innerDegree setup (r * (S.blockLength * S.sourceDigits) + c)) (zRing S z c))
      (S.blockLength * S.sourceDigits)

/-- `Σ_{i<B−1} c_i·t_{i,r}`, with `t_{i,r}` the ring element `i·n_A + r` of the sent `t`. -/
def tPartial (msgs : TerminalMessages) (ch : TerminalChallenges) (r : ℕ) : Array Fq :=
  Poly.sum S.innerDegree (fun i => Poly.mul S.innerDegree (challenge S ch i)
    (Poly.ring S.innerDegree msgs.t (i * S.innerRank + r))) (S.blocks - 1)

/-- The block `x` with `part + c·x = target`, from the inverse `inv` of `c`, checked by the
product (`LastBlockRecovery::solve`). -/
def recover (c inv part target : Array Fq) : Option (Array Fq) :=
  let x := Poly.mul S.innerDegree inv (Poly.sub S.innerDegree target part)
  if Poly.equal S.innerDegree (Poly.add S.innerDegree part (Poly.mul S.innerDegree c x)) target
  then some x else none

/-- Block `i` of the full `e`: a sent block, or the recovered last block. -/
def eFull (msgs : TerminalMessages) (eLast : Array Fq) (i : ℕ) : Array Fq :=
  if i < S.blocks - 1 then Poly.ring S.innerDegree msgs.e i else eLast

/-- `Σ_i χ_blk(i)·T_{ρ_pack}(e_i)` (`verify_terminal_trace`, subfield multiplier). -/
def opening (ch : TerminalChallenges) (msgs : TerminalMessages) (eLast : Array Fq) : Ext :=
  sumExt (fun i => eqIndex (blockPoint S ch.point) i *
    functional (S.innerDegree / 4) (packPoint S ch.point) (eFull S msgs eLast i)) S.blocks

/-- The field digest of the recovered rows of `t`, in the order of the rows. -/
def rowsDigest (digest : ByteArray → ByteArray) (rows : List (Array Fq)) : ByteArray :=
  fieldDigest digest (rows.foldl (fun out x => out ++ Poly.canon S.innerDegree x) #[])

/-- The last block of `t`: one recovered ring element for each row of `A`. -/
def tLastOf (msgs : TerminalMessages) (ch : TerminalChallenges) (inv : Array Fq)
    (rows : List (Array Fq)) : Option (List (Array Fq)) :=
  (List.range S.innerRank).mapM fun r =>
    recover S (challenge S ch (S.blocks - 1)) inv (tPartial S msgs ch r) (rows.getD r #[])

/-- The checks of the terminal on a decoded `z`, in the order of the code, with the hash `digest`
of the two last-block digests. -/
def terminalChecks (digest : ByteArray → ByteArray) (cap : ℕ) (setup : Array Fq) (msgs : TerminalMessages)
    (ch : TerminalChallenges) (z : Array ℤ) : Except String TerminalValues :=
  if centeredNormSq z ≤ cap then
    if ch.fold.length = S.blocks ∧ ch.fold.all (sparseValid S.innerDegree) = true then
      match Poly.inverse (Nat.log 2 S.innerDegree) (challenge S ch (S.blocks - 1)) with
      | none => .error "terminal.fold_challenges.unit"
      | some inv =>
          match recover S (challenge S ch (S.blocks - 1)) inv (folded S msgs ch) (reduced S ch z)
          with
          | none => .error "terminal.e_last_block"
          | some eLast =>
              let rows := (List.range S.innerRank).map (aRow S setup z)
              match tLastOf S msgs ch inv rows with
              | none => .error "terminal.a_rows"
              | some tLast =>
                  if fieldDigest digest (Poly.canon S.innerDegree eLast) = msgs.eDigest then
                    if rowsDigest S digest tLast = msgs.tDigest then
                      if (ch.scale * opening S ch msgs eLast).equals ch.target = true then
                        .ok ⟨z, centeredNormSq z, eLast, tLast, rows, opening S ch msgs eLast⟩
                      else .error "terminal.trace"
                    else .error "terminal_t_last_block_digest"
                  else .error "terminal.e_last_block.digest"
    else .error "terminal.fold_challenges"
  else .error "terminal.z.norm"

/-- The terminal checks of `verify_terminal_ring_relations` and `verify_terminal_trace`. -/
def terminalRun (digest : ByteArray → ByteArray) (cap : ℕ) (P : Golomb.Params) (setup : Array Fq)
    (msgs : TerminalMessages) (ch : TerminalChallenges) : Except String TerminalValues :=
  match Golomb.decode P msgs.payload with
  | none => .error "terminal.z.decode"
  | some z => terminalChecks S digest cap setup msgs ch z

theorem terminalRun_decode {digest : ByteArray → ByteArray} {cap : ℕ} {P : Golomb.Params} {setup : Array Fq}
    {msgs : TerminalMessages} (ch : TerminalChallenges) (fails : Golomb.decode P msgs.payload = none) :
    terminalRun S digest cap P setup msgs ch = .error "terminal.z.decode" := by
  rw [terminalRun, fails]

end Checks

end NightstreamFPrime.Spec.Compression.Akita.Replay
