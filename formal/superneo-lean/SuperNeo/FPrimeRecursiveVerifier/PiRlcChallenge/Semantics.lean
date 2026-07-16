import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Transcript.DigestRounds
import SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge.Sampler.Selection

/-!
Owns: composition of transcript-derived chunks, acceptance, and first-accepted
selection for one fixed Pi_RLC rho sample.

Does not own: the fifteen-sample schedule, concrete Poseidon2, or R1CS rows.

Emits constraints: no.

Authority boundary: `TranscriptRhoSampleSemantics` treats only
`rhoDigestTrace` output as authoritative; `RhoSampleSemantics` alone makes no
transcript claim.

| Predicate/theorem | Rust stage | Guarantee | Assumptions | Permits row removal? |
|---|---|---|---|---|
| `RhoSampleSemantics` | `challenge.sampler` | Exactly 64 chunks, enough accepts, and the first 54 accepted symbols | Canonical supplied chunks | No — Rust refinement open |
| `TranscriptRhoSampleSemantics` | `challenge.transcript` | Binds the sampler relation to `rhoDigestTrace` | Authoritative cursor and supplied core | No — concrete Poseidon2/Rust refinement open |
| `rhoSample_output_length`, `rhoSample_output_mem_alphabet` | `challenge.sampler` | Output has length 54 and lies in `[-2, 2]` | `RhoSampleSemantics` | No — Rust refinement open |

`RhoSampleSemantics` is the post-digest relation and makes no transcript-
authority claim by itself. `TranscriptRhoSampleSemantics` binds the same
relation to `DigestRounds.rhoDigestTrace`; the production Poseidon2 core still
requires its separate concrete refinement.
-/

namespace SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge

/-- Exact non-transcript semantics of one fixed sampler invocation. -/
def RhoSampleSemantics (chunks : List Chunk) (output : List Int) : Prop :=
  chunks.length = chunksPerSample ∧
    EnoughAccepts chunks ∧
    output = firstAcceptedSymbols chunks

/--
Sampler relation whose chunks are authoritative outputs of the exact sponge
schedule for the supplied permutation core and incoming cursor.
-/
def TranscriptRhoSampleSemantics
    (core : Poseidon2Core) (cursor : SpongeCursor) (rhoIndex : Nat)
    (output : List Int) : Prop :=
  RhoSampleSemantics (rhoDigestTrace core cursor rhoIndex).chunks output

theorem rhoSample_output_length
    {chunks : List Chunk} {output : List Int}
    (hSample : RhoSampleSemantics chunks output) :
    output.length = outputLength := by
  rcases hSample with ⟨_, hEnough, rfl⟩
  exact firstAcceptedSymbols_length chunks hEnough

theorem rhoSample_output_mem_alphabet
    {chunks : List Chunk} {output : List Int}
    (hSample : RhoSampleSemantics chunks output)
    {value : Int} (hMember : value ∈ output) :
    (-2 : Int) ≤ value ∧ value ≤ 2 := by
  rcases hSample with ⟨_, _, rfl⟩
  exact firstAcceptedSymbols_mem_alphabet chunks value hMember

theorem transcriptRhoSample_output_length
    {core : Poseidon2Core} {cursor : SpongeCursor} {rhoIndex : Nat}
    {output : List Int}
    (hSample : TranscriptRhoSampleSemantics core cursor rhoIndex output) :
    output.length = outputLength :=
  rhoSample_output_length hSample

end SuperNeo.FPrimeRecursiveVerifier.PiRlcChallenge
