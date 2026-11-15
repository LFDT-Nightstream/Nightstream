import NightstreamFPrime.Export.DomainSeparation
import NightstreamFPrime.Lifecycle.Nebula.EtaRound

/-! `[SPONGE]` for the Nebula `η` transcript (`SECURITY_MODEL.md`, adversary 5).

The `η` transcript reads lane 0 of the sponge state after its framed blocks and after one, two
and three more permutations (`reads`). The conditions of `[SPONGE]` for this use hold
(`conditions`): the call list of a read fixes a canonical `η` input and the read
(`reads_injective`), the calls start with the `η` tag (`reads_owned`), and that tag is separated
from every other use of the table (`separated_eta`). So, under the premise
`Spec.Sponge.Transfer` with these conditions, the bound of the random-oracle model of the memory
round (`Lifecycle.Nebula.EtaRound.EtaSampler.unbalancedChance_le`) holds in the
ideal-permutation model plus `spongeError` (`unbalanced_le`). The step circuit computes `η`
with the permutation, so this bound is for a verifier that computes `η` outside a circuit. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.BoundedDepth.EtaSponge

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Nebula
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nebula
open NightstreamFPrime.Lifecycle.Nebula.EndToEnd
open NightstreamFPrime.Export.DomainSeparation

/-- The sponge calls before read `index` of the `η` transcript: the chunks of its framed blocks,
then one permutation per earlier read. -/
def reads (input : EtaInput Nebula.Digest) (index : Fin 4) : List (Fin Poseidon2.rate → F) :=
  blockChunks (etaInputBlocks input) ++ List.replicate index.val fun _ => 0

theorem chunks_length (words : List F) :
    (chunks words).length = (words.length + Poseidon2.rate - 1) / Poseidon2.rate := by
  unfold chunks
  rw [List.length_map, List.length_range]

theorem blockChunks_eta_length (input : EtaInput Nebula.Digest) :
    (blockChunks (etaInputBlocks input)).length = 7 := by
  simp only [blockChunks, etaInputBlocks, List.flatMap_cons, List.flatMap_nil, List.append_nil,
    List.length_append, chunks_length, block, List.length_cons, digestWords, List.length_ofFn,
    textWords, List.length_map, List.length_append]
  rfl

theorem reads_injective {input input' : EtaInput Nebula.Digest} {index index' : Fin 4}
    (small : input.ts < fieldModulus) (small' : input'.ts < fieldModulus)
    (same : reads input index = reads input' index') : input = input' ∧ index = index' := by
  have lengths := congrArg List.length same
  simp only [reads, List.length_append, blockChunks_eta_length, List.length_replicate] at lengths
  have indices : index = index' := Fin.ext (by omega)
  subst indices
  unfold reads at same
  exact ⟨etaInputBlocks_injective small small' (List.append_cancel_right same), rfl⟩

/-- The `η` row of the table. -/
def etaUse : Use := ⟨"Nebula eta", nebulaTag "Nightstream/Nebula/v3/eta"⟩

theorem etaUse_mem : etaUse ∈ uses := by
  simp [etaUse, uses]

theorem reads_owned (input : EtaInput Nebula.Digest) (index : Fin 4) :
    etaUse.tag <+: ((reads input index).map List.ofFn).flatten :=
  nebula_owns _ _ _

theorem separated_eta : ∀ use ∈ uses, use ≠ etaUse → Separated etaUse use :=
  fun _ member distinct => separated_of_mem etaUse_mem member distinct

/-- The conditions of `[SPONGE]` for the `η` use. -/
structure Conditions : Prop where
  injective : ∀ {input input' : EtaInput Nebula.Digest} {index index' : Fin 4},
    input.ts < fieldModulus → input'.ts < fieldModulus → reads input index = reads input' index' →
      input = input' ∧ index = index'
  owned : ∀ input index, etaUse.tag <+: ((reads input index).map List.ofFn).flatten
  separated : ∀ use ∈ uses, use ≠ etaUse → Separated etaUse use

theorem conditions : Conditions := ⟨reads_injective, reads_owned, separated_eta⟩

/-- Adversary 5's memory round in the ideal-permutation model: under `Spec.Sponge.Transfer` with
the `η` conditions, and with the sampler's retry term removed, the wins of each `world` are at
most `(queries + 1) · 2 · m_mem/q²` plus `spongeError calls`. -/
theorem unbalanced_le {p : Plan} (valid : p.Valid) (app : PortApp p.toGeometry) {Sample : Type}
    (x : Experiment app.width Sample) {queries : ℕ} {World : Type} (winsOf : World → ℝ)
    (calls : ℕ)
    (premise : Sponge.Transfer Conditions winsOf
      (fun sampler : EtaRound.EtaSampler Sample queries =>
        sampler.unbalancedChance app x - sampler.retryCollision app x) calls)
    (world : World) :
    winsOf world ≤ ((queries + 1 : ℕ) : ℝ) * (2 * (p.maxTuples : ℝ) / (fieldModulus ^ 2 : ℕ)) +
      Sponge.spongeError calls :=
  Sponge.transfer_le premise conditions
    (fun sampler => by linarith [sampler.unbalancedChance_le (app := app) (x := x) valid]) world

end NightstreamFPrime.Export.BoundedDepth.EtaSponge
