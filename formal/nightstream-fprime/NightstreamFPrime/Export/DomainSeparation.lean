import NightstreamFPrime.Spec.Sponge
import NightstreamFPrime.Export.BoundedDepth.StepPackage
import NightstreamFPrime.Lifecycle.BoundedDepth.Keys
import NightstreamFPrime.Lifecycle.RandomOracle.Schedule
import NightstreamFPrime.Lifecycle.Nebula.Framing
import NightstreamFPrime.Lifecycle.Stage1.Poseidon2HashChainV1
import NightstreamFPrime.Lifecycle.VerifierContext

/-! Domain separation between the uses of the Poseidon2 sponge (`SECURITY_MODEL.md`, `[SPONGE]`
and section Hashed encodings).

Each use absorbs a word stream that starts with its tag (`uses`). Every two tags of the table
differ at a position of both (`tags_differ`), so two block lists whose word streams start with
the tags of two distinct uses are distinct (`separated`). A use owns the block lists that it
absorbs: the padded input of a hash use under `Spec.Sponge.Padding` (`hash_owns`), the calls of a
fold transcript (`fold_owns`), and the framed blocks of a Nebula transcript (`nebula_owns`).

The last four rows restate Rust tags that no Lean definition has: the circuit key digest, the
two compression domains and the application identity (`SPEC.md` section 4.4, Algorithm 10). No
emitted data compares them with Rust yet. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.DomainSeparation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- One use of the sponge and the words that its absorbed stream starts with. -/
structure Use where
  name : String
  tag : List F

/-- ASCII text, one word per byte. -/
def byteWords (text : String) : List F := text.toList.map fun c => Poseidon2.ofNat c.toNat

/-- The first framed block of a Nebula transcript: the tag's length, then its bytes. -/
def nebulaTag (text : String) : List F := block (Nebula.textWords text)

/-- The uses of the sponge with their tags. -/
def uses : List Use :=
  [⟨"bounded-depth state hash", BoundedDepth.stateDomainChunk⟩,
   ⟨"slot digest", BoundedDepth.slotDomainChunk⟩,
   ⟨"Stage 1 state hash", Lifecycle.stateDomainChunk⟩,
   ⟨"package identity", Export.BoundedDepth.StepPackage.boundedDepthDomain⟩,
   ⟨"relation identifier", Export.Package.identityDomain⟩,
   ⟨"verifier context", (VerifierContext.componentDomain 0).take 31⟩,
   ⟨"hash-chain application", Stage1.Poseidon2HashChainV1.domainTag⟩,
   ⟨"light fold", BoundedDepth.lightFoldDomainChunk⟩,
   ⟨"refresh fold", BoundedDepth.refreshFoldDomainChunk⟩,
   ⟨"production fold", Transcript.foldDomainChunk⟩,
   ⟨"Nebula plan", nebulaTag "Nightstream/Nebula/v3/plan"⟩,
   ⟨"Nebula ops header", nebulaTag (Nebula.headerTag .ops)⟩,
   ⟨"Nebula memory header", nebulaTag (Nebula.headerTag .mem)⟩,
   ⟨"Nebula ops chain", nebulaTag (Nebula.chainTag .ops)⟩,
   ⟨"Nebula memory chain", nebulaTag (Nebula.chainTag .mem)⟩,
   ⟨"Nebula eta", nebulaTag "Nightstream/Nebula/v3/eta"⟩,
   ⟨"Nebula state", nebulaTag "Nightstream/Nebula/v3/state"⟩,
   ⟨"circuit key digest", BoundedDepth.domainChunk "Nightstream/OptionB/circuit-key/v1"⟩,
   ⟨"compression Pi_CCS", BoundedDepth.domainChunk "Nightstream/OptionB/compress/ccs/v1"⟩,
   ⟨"compression argument", BoundedDepth.domainChunk "Nightstream/compress/akita/v2"⟩,
   ⟨"application identity",
     byteWords "Nightstream/OptionB/application/combined-native-experiment/v1"⟩]

/-- `left` and `right` differ at a position of both. -/
def differs (left right : List F) : Bool := (left.zip right).any fun pair => pair.1 != pair.2

theorem differs_spec : ∀ {left right : List F}, differs left right = true →
    ∃ position, ∃ (inLeft : position < left.length) (inRight : position < right.length),
      left[position] ≠ right[position]
  | [], _, found => by simp [differs] at found
  | _ :: _, [], found => by simp [differs] at found
  | a :: left, b :: right, found => by
      by_cases equal : a = b
      · have rest : differs left right = true := by
          simpa [differs, equal] using found
        obtain ⟨position, inLeft, inRight, ne⟩ := differs_spec rest
        exact ⟨position + 1, by simpa using inLeft, by simpa using inRight, by simpa using ne⟩
      · exact ⟨0, by simp, by simp, by simpa using equal⟩

/-- Every two tags of the table differ at a position of both. -/
theorem tags_differ : uses.Pairwise fun use use' => differs use.tag use'.tag = true := by
  decide

/-- Two block lists are separated for two uses when they are distinct whenever their word
streams start with the two tags. -/
def Separated (use use' : Use) : Prop :=
  ∀ blocks blocks' : List (List F), use.tag <+: blocks.flatten → use'.tag <+: blocks'.flatten →
    blocks ≠ blocks'

/-- Every two distinct rows of the table are separated. -/
theorem separated : uses.Pairwise Separated :=
  tags_differ.imp fun found _ _ leftPrefix rightPrefix => by
    obtain ⟨position, inLeft, inRight, ne⟩ := differs_spec found
    exact Sponge.blocks_ne_of_differ leftPrefix rightPrefix inLeft inRight ne

/-- A hash use owns the padded input of each input that starts with its tag. -/
theorem hash_owns (padding : Sponge.Padding) {tag input : List F} (starts : tag <+: input) :
    tag <+: (padding.pad input).flatten :=
  starts.trans (padding.keeps input)

theorem statePreimage_starts {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}
    (state : BoundedDepth.State logicalWidth publicFits) :
    BoundedDepth.stateDomainChunk <+: BoundedDepth.statePreimage state := by
  simp only [BoundedDepth.statePreimage, List.append_assoc]
  exact List.prefix_append _ _

theorem identityInput_starts (light refresh : Export.BoundedDepth.StepPackage.Package) :
    Export.BoundedDepth.StepPackage.boundedDepthDomain <+:
      Export.BoundedDepth.StepPackage.boundedDepthDomain ++
        Export.Package.valuePreimage
          (Export.BoundedDepth.StepPackage.identityValue light refresh) :=
  List.prefix_append _ _

private theorem ofFn_padChunk {words : List F} (full : words.length = Poseidon2.rate) :
    List.ofFn (RandomOracle.Sponge.padChunk words) = words := by
  apply List.ext_getElem (by simp [full])
  intro position _ inWords
  simp [RandomOracle.Sponge.padChunk, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem inWords]

private theorem prefix_ofFn_padChunk {words : List F} (short : words.length ≤ Poseidon2.rate) :
    words <+: List.ofFn (RandomOracle.Sponge.padChunk words) := by
  rw [List.prefix_iff_eq_take]
  apply List.ext_getElem (by simp [Nat.min_eq_left short])
  intro position inWords _
  simp [RandomOracle.Sponge.padChunk, List.getD_eq_getElem?_getD,
    List.getElem?_eq_getElem inWords]

private theorem absorbCalls_long {words : List F} (long : Poseidon2.rate < words.length) :
    RandomOracle.Sponge.absorbCalls words =
      RandomOracle.Sponge.padChunk (words.take Poseidon2.rate) ::
        RandomOracle.Sponge.absorbCalls (words.drop Poseidon2.rate) := by
  have count : (words.length + Poseidon2.rate - 1) / Poseidon2.rate =
      ((words.drop Poseidon2.rate).length + Poseidon2.rate - 1) / Poseidon2.rate + 1 := by
    simp only [List.length_drop]
    unfold Poseidon2.rate at *
    omega
  unfold RandomOracle.Sponge.absorbCalls
  rw [count, List.range_succ_eq_map, List.map_cons, Nat.zero_mul, List.drop_zero, List.map_map]
  congr 1
  apply List.map_congr_left
  intro chunk _
  simp only [Function.comp, List.drop_drop, Nat.succ_mul]
  congr 3
  omega

/-- The words of a list are a prefix of the words of its absorb calls. -/
theorem prefix_absorbCalls (words : List F) :
    words <+: ((RandomOracle.Sponge.absorbCalls words).map List.ofFn).flatten := by
  induction h : words.length using Nat.strong_induction_on generalizing words with
  | _ length inductionHypothesis =>
      by_cases long : Poseidon2.rate < words.length
      · rw [absorbCalls_long long, List.map_cons, List.flatten_cons,
          ofFn_padChunk (by simp [Nat.min_eq_left (Nat.le_of_lt long)])]
        have rest := inductionHypothesis _ (by simp; unfold Poseidon2.rate at *; omega)
          (words.drop Poseidon2.rate) rfl
        rw [← List.take_append_drop Poseidon2.rate words]
        simp only [List.take_append_drop]
        conv => lhs; rw [← List.take_append_drop Poseidon2.rate words]
        exact (List.prefix_append_right_inj _).mpr rest
      · by_cases empty : words = []
        · subst empty
          exact List.nil_prefix
        · have one : (words.length + Poseidon2.rate - 1) / Poseidon2.rate = 1 := by
            have positive : 0 < words.length := List.length_pos_iff.mpr empty
            unfold Poseidon2.rate at *
            omega
          unfold RandomOracle.Sponge.absorbCalls
          rw [one, List.range_one, List.map_cons, List.map_nil, Nat.zero_mul, List.drop_zero,
            List.take_of_length_le (Nat.le_of_not_lt long)]
          simpa using prefix_ofFn_padChunk (Nat.le_of_not_lt long)

/-- A fold transcript owns the calls before each of its challenges. -/
theorem fold_owns (schedule : RandomOracle.Schedule) {width : Nat}
    (statement : schedule.Statement) (messages : schedule.Messages width)
    (challenge : schedule.Challenge) :
    schedule.domainChunk <+:
      ((schedule.challengeCalls statement messages challenge).map List.ofFn).flatten := by
  have head := schedule.challengeCalls_head? statement messages challenge
  cases calls : schedule.challengeCalls statement messages challenge with
  | nil => simp [calls] at head
  | cons call rest =>
      rw [calls, List.head?_cons, Option.some_inj] at head
      rw [List.map_cons, List.flatten_cons, head, ofFn_padChunk schedule.domainChunk_length]
      exact List.prefix_append _ _

theorem chunks_eq_absorbCalls (words : List F) :
    Nebula.chunks words = RandomOracle.Sponge.absorbCalls words := rfl

/-- A Nebula transcript owns the chunks of its framed blocks, after any further calls. -/
theorem nebula_owns (first : List F) (rest : List (List F))
    (more : List (Fin Poseidon2.rate → F)) :
    block first <+: ((Nebula.blockChunks (first :: rest) ++ more).map List.ofFn).flatten := by
  rw [Nebula.blockChunks_cons, List.append_assoc, List.map_append, List.flatten_append]
  rw [chunks_eq_absorbCalls]
  exact (prefix_absorbCalls (block first)).trans (List.prefix_append _ _)

end NightstreamFPrime.Export.DomainSeparation
