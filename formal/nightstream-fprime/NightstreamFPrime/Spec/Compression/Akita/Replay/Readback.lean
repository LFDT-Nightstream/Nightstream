import NightstreamFPrime.Spec.Compression.Akita.Replay.Grows

/-! What the replay records, read back (Akita revision `d50f700`, jolt-transcript `25d87ba`,
`codec.rs`):

- `word_of_data`: `Wire.word` of the 8 bytes of `le64 n` is `n`; `decodeExts_extBytes`: the
  recorded value of an absorbed element of `E` is the element. A check that reads an absorbed
  claim reads the value that the replay computed.
- `step_ended`: the end of a sum-check records `sumcheckRun` on the claim, the round messages and
  the point recorded before it (`formIn`, `endRoundsIn`), or stops with `mismatch`. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Compression.Akita.Replay

/-- One step of `Wire.word`: a word below `2^56` shifted by a byte and a byte. -/
theorem orShift (acc : UInt64) (b : UInt8) (small : acc.toNat < 2 ^ 56) :
    ((acc <<< 8) ||| b.toUInt64).toNat = acc.toNat * 256 + b.toNat := by
  rw [UInt64.toNat_or, UInt64.toNat_shiftLeft]
  have h8 : (8 : UInt64).toNat % 64 = 8 := rfl
  rw [h8, Nat.mod_eq_of_lt (by rw [Nat.shiftLeft_eq]; omega), UInt8.toNat_toUInt64,
    ← Nat.shiftLeft_add_eq_or_of_lt (UInt8.toNat_lt b), Nat.shiftLeft_eq]

/-- `Wire.word` of the bytes `g i` is `Σ_i g i·256^i`. -/
theorem foldWord (g : ℕ → UInt8) : ∀ l : List ℕ, l.length ≤ 8 →
    (l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (g i).toUInt64) 0).toNat =
      l.foldr (fun i acc => acc * 256 + (g i).toNat) 0 ∧
    l.foldr (fun i acc => acc * 256 + (g i).toNat) 0 < 256 ^ l.length
  | [], _ => by simp
  | i :: l, short => by
      have short' : l.length ≤ 7 := by simp at short; omega
      obtain ⟨same, lt⟩ := foldWord g l (by omega)
      have bound : 256 ^ l.length ≤ 2 ^ 56 :=
        (Nat.pow_le_pow_right (by norm_num) short').trans (by norm_num)
      have small : (l.foldr (fun i (acc : UInt64) => (acc <<< 8) ||| (g i).toUInt64)
          0).toNat < 2 ^ 56 := by
        rw [same]; omega
      have byte := UInt8.toNat_lt (g i)
      simp only [List.foldr_cons, List.length_cons, pow_succ]
      refine ⟨by rw [orShift _ _ small, same], ?_⟩
      generalize 256 ^ l.length = P at lt
      omega

/-- The word of the 8 bytes of `le64 n` is `n`. -/
theorem word_of_data {bytes : ByteArray} {front rest : Array UInt8} {n : ℕ} (small : n < 2 ^ 64)
    (split : bytes.data = front ++ (le64 n).data ++ rest) :
    Wire.word bytes front.size = n.toUInt64 := by
  have size : (le64 n).data.size = 8 := by simp [le64]
  have byte : ∀ i < 8, bytes.get! (front.size + i) = (n / 2 ^ (8 * i) % 256).toUInt8 := by
    intro i inside
    show bytes.data[front.size + i]! = _
    rw [split, getElem!_pos _ _ (by simp only [Array.size_append]; omega),
      Array.getElem_append_left (by simp only [Array.size_append]; omega),
      Array.getElem_append_right (by omega)]
    simp [le64]
  apply UInt64.toNat_inj.mp
  unfold Wire.word
  rw [(foldWord _ (List.range 8) (by simp)).1]
  have r8 : List.range 8 = [0, 1, 2, 3, 4, 5, 6, 7] := rfl
  rw [r8]
  simp only [List.foldr_cons, List.foldr_nil]
  rw [byte 0 (by norm_num), byte 1 (by norm_num), byte 2 (by norm_num), byte 3 (by norm_num),
    byte 4 (by norm_num), byte 5 (by norm_num), byte 6 (by norm_num), byte 7 (by norm_num)]
  simp only [Nat.toUInt8, Nat.toUInt64]
  norm_num
  omega

/-- The replay records an absorbed element of `E` as the element: `decodeExts` reads it back from
its bytes `extBytes`. -/
theorem decodeExts_extBytes (x : Ext) : Replayer.step.decodeExts (extBytes #[x]) = pure #[x] := by
  have size : (le64 x.c0.word.toNat).data.size = 8 := by simp [le64]
  have w0 := word_of_data (front := #[]) (rest := (le64 x.c1.word.toNat).data)
    (bytes := extBytes #[x]) x.c0.word.toNat_lt (by simp [extBytes])
  have w8 := word_of_data (front := (le64 x.c0.word.toNat).data) (rest := #[])
    (bytes := extBytes #[x]) x.c1.word.toNat_lt (by simp [extBytes])
  rw [size] at w8
  simp only [List.size_toArray, List.length_nil] at w0
  have total : (extBytes #[x]).size = 16 := by
    rw [← ByteArray.size_data]
    simp [extBytes, le64]
  unfold Replayer.step.decodeExts
  rw [total]
  have range : Array.range (16 / 16) = #[0] := by decide
  simp only [range, List.map_toArray, List.map_cons, List.map_nil, Nat.mul_zero, Nat.zero_add, w0,
    w8, Nat.toUInt64, UInt64.ofNat_toNat]

namespace Replayer

/-- The message form of the sum-check with the point before `before`, on the recorded values. -/
def formIn (s : Replayer) : Before → Form
  | .plain => .plain
  | before => .equality (pointIn s before)

/-- The rounds of the sum-check `key` with `n` rounds on the recorded values: the round messages
with the challenges of the point. -/
def endRoundsIn (s : Replayer) (key : Key) (n : ℕ) : List (List Ext × Ext) :=
  ((List.range n).map fun r => (recordedIn s ⟨key.level, key.site ++ ".round", some r⟩).toList).zip
    (recordedIn s ⟨key.level, key.site ++ ".point", none⟩).toList

theorem mapM_recorded (f : ℕ → Key) : ∀ (l : List ℕ) (s : Replayer),
    ((l.mapM fun r => recorded (f r)).run).run s = (.ok (l.map fun r => recordedIn s (f r)), s)
  | [], _ => rfl
  | r :: l, s => by
      rw [List.mapM_cons, run_bind]
      show (((l.mapM fun r => recorded (f r)) >>= fun bs => pure (recordedIn s (f r) :: bs)).run).run s = _
      rw [run_bind, mapM_recorded f l s]
      rfl

theorem run_pure {α : Type} (a : α) (s : Replayer) :
    (((pure a : ReplayM α)).run).run s = (.ok a, s) := rfl

theorem run_recorded (key : Key) (s : Replayer) :
    ((recorded key).run).run s = (.ok (recordedIn s key), s) := rfl

theorem run_first (key : Key) (s : Replayer) :
    ((first key).run).run s = (.ok (firstIn s key), s) := rfl

theorem run_pointBefore (before : Before) (s : Replayer) :
    ((pointBefore before).run).run s = (.ok (pointIn s before), s) := rfl

/-- `.ended` records the end of `sumcheckRun` on the recorded values: the output, the end and the
rounds; it stops with `mismatch` when `sumcheckRun` has no end. -/
theorem step_ended (pub : Public) (proof : ByteArray) (messages : Array Message) (i : ℕ)
    (key : Key) (n : ℕ) (before : Before) (s : Replayer) :
    ((step pub proof messages i (.ended key n before)).run).run s =
      match sumcheckRun (formIn s before) (firstIn s ⟨key.level, key.site ++ ".claim", none⟩)
          (endRoundsIn s key n) with
      | none => (.error (.mismatch i), s)
      | some e => (.ok (), { s with
          env := s.env.insert ⟨key.level, key.site ++ ".output", none⟩
            ((recordedIn s ⟨key.level, key.site ++ ".output", none⟩) ++ #[e.claim])
          derived := { s.derived with
            ends := s.derived.ends.push (key, before, e)
            rounds := s.derived.rounds.push (key, endRoundsIn s key n) } }) := by
  simp only [step]
  rw [run_bind, mapM_recorded]
  cases before <;>
    simp only [formIn, endRoundsIn, run_bind, run_pure, run_recorded, run_first, run_pointBefore,
      List.map_map, Function.comp_def] <;>
    generalize sumcheckRun _ _ _ = r <;> cases r <;> rfl

end Replayer

end NightstreamFPrime.Spec.Compression.Akita.Replay
