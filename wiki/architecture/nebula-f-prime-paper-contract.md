# Nebula F′ paper contract

Date: 2026-08-15

## Executive summary

Nightstream must prove a bounded recursive relation that combines three
separate paper contracts. SuperNeo defines the norm-bounded CCS fold.
HyperNova defines how a fresh augmented step binds the complete prior recursive
state. Nebula defines commitment-carrying advice, commit-then-challenge memory
checking, segment finalization, and pay-per-use execution.

The required result is not one large flattened circuit. It is one fixed
relation that verifies the complete prior SuperNeo fold and one
verifier-selected phase of the Nebula transition. The carried phase state must
make the ordered phase sequence refine the complete reference transition.
Poseidon2 binds supplied values to authoritative values. A digest is never an
independent source of truth.

## Paper roles

| Paper | Local contract | Main source |
|---|---|---|
| SuperNeo | Fold norm-bounded CCS evaluation claims with PiCCS, PiRLC, and decomposition | `docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:23` |
| SuperNeo | Preserve commitment, public-input, and matrix-evaluation meaning under linear combination | `docs/superneo-paper/05-5-embedding-products-with-evaluation-homomorphism.md:50` |
| HyperNova | Bind each fresh augmented instance to the complete prior state before the fold update | `docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md:37` |
| HyperNova | Keep the recursive verifier interface fixed and compact | `docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md:52` |
| Nebula | Carry binding commitments to the advice used by each incremental step | `docs/nebula-paper/03_3-commitment-carrying-ivc.md:15` |
| Nebula | Commit to memory-operation data before sampling fingerprint challenges | `docs/nebula-paper/04_4-efficient-read-write-memory-in-ivc.md:11` |
| Nebula | Finalize segment-local operation and memory scans, then fold finalized segments | `docs/nebula-paper/04_4-efficient-read-write-memory-in-ivc.md:61` |
| Nebula | Select one R1CS subcircuit while inactive subcircuits use zero advice | `docs/nebula-paper/05_5-nivc-using-a-universal-switchboard-circuit.md:49` |

## Irreducible relation requirements

1. The recursive step checks the complete prior SuperNeo verifier relation.
   Splitting that verifier over later recursive steps would create an
   ever-growing backlog of unchecked fold obligations.
2. The recursive step checks exactly one bounded Nebula phase. The phase code,
   cursor, profile identity, prior continuation, and successor continuation are
   verifier-bound values.
3. A supplied chunk drives both its Poseidon2 replay and its local algebra. The
   relation cannot hash one value and compute with another value.
4. A different supplied frame can be accepted only through a named Poseidon2
   collision or commitment-binding failure.
5. PiCCS, PiRLC, and decomposition remain one complete SuperNeo transition.
   Phase boundaries can change execution order, but not the values, challenge
   order, or final relation.
6. Nebula memory challenges are derived only after the operation and scan
   commitments are fixed. Terminal checks enforce the four-set product
   equality and the initial-to-final memory commitment link.
7. The generated recursive and terminal CCS rows, Rust encoder, and Lean
   semantics use one exact assignment and one exact layout.
8. The larger of the recursive and terminal joint row and assignment domains
   is at most `2^24`.

## Required data flow

```text
authoritative prior carrier + phase code + supplied phase frame
    -> replay exact frame into Poseidon2
    -> check the same frame with phase-local algebra
    -> update the bounded continuation and cursor
    -> run the complete SuperNeo fold verifier
    -> bind the exact successor carrier
    -> terminal relation checks completion and finalized Nebula state
```

## Active implementation map

| Area | Current owner | Status |
|---|---|---|
| Monolithic folded relation | `crates/neo-fold-clean/src/frontends/nebula/f_prime.rs:175` | Executable reference and current lifecycle relation |
| Recursive circuit entry | `crates/neo-fold-clean/src/frontends/nebula/f_prime.rs:900` | Complete current recursive arm |
| Verifier-owned phase order | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_program.rs:17` | Exact Rust schedule and cursor authority |
| Lean schedule | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingFPrimeProgram.lean:141` | Proves the 400-item order, physical circuit map, and cursor discipline |
| Chunk replay and algebra join | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingFusedPass.lean:172` | Produces equality or a named collision |
| Claim replay | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_claim_replay.rs:1` | Exact claim-local relation; first claim cursor is 83 |
| Claim coordinate overlay | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_claim_replay/coordinate_overlay.rs:1` | Stores one no-op, two carry, and 23 active fixed-position coordinate arms; exact private links bind each selected arm to claim replay |
| Claim coordinate sequence | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingClaimReplayCoordinateSequence.lean:1` | Exact 86-phase source-row and private-link model; accepted rows imply the direct commitment of all 21,220 claim fields |
| PiCCS variable-coordinate binding | `crates/neo-fold-clean/src/paper/reductions/accumulator_sis_circuit.rs:109` and `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCcsCoordinateBindingSetup.lean:1` | Fixed-position seeded Ajtai map, exact Phi81 multiplication, additive phase masks, and collision-to-Module-SIS theorem |
| Ajtai algebra boundary | `formal/nightstream-lean/Nightstream/Protocol/Nebula/AjtaiBinding.lean:1` and `formal/nightstream-lean/Nightstream/Assurance/Nebula/AjtaiBinding.lean:1` | Protocol owns finite matrix algebra and collision extraction; Assurance owns only the V2 compact shapes and security bridges |
| PiCCS coordinate selector bridge | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCcsCoordinateBindingRows.lean:1` | Exact 21,220-word selector, 28-coordinate zero tail, and verifier-owned seeded coefficient tensor |
| PiCCS coordinate source rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCcsCoordinateBindingOpeningRows.lean:1` | Exact 41-row shared zero word, ordered 124-row active openings, and row-derived selector source authority |
| PiCCS coordinate output rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCcsCoordinateBindingOutputRows.lean:1` | Dense compact output values equal the exact masked rank-two Phi81 Ajtai commitment coordinates |
| PiCCS complete coordinate rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCcsCoordinateBindingCompleteRows.lean:1` | Exact Rust family order, fixed degree/rank shape rows, production census, and end-to-end row soundness |
| PiCCS claim-chunk partition | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCcsCoordinateBindingClaimSchedule.lean:1` | Maps every selected coordinate to its exact claim word, proves the 86 masks partition the full witness, and derives each partial commitment from accepted local rows |
| PiCCS round authority | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCCSAuthority.lean:423` | Exact 26-round authority model |
| PiRLC input authority | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCInputBindingSetup.lean:1` | Fixed-position binding of all 89,100 inputs, 110 exact additive family slices, 108 carried residual fields, and a Module-SIS failure boundary |
| PiRLC family commitment rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCInputPhaseRows.lean:1` | Exact fixed-position selector, 810 canonical openings, verifier-owned seeded coefficients, and 108 compact output fields in 100,591 rows |
| PiRLC family residual rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCInputFamilyRows.lean:1` | Reuses the exact compact output in 108 additive residual equations; all 100,699 accepted rows imply the semantic local residual transition |
| PiRLC challenge and cursor rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCFamilyCarryRows.lean:1` | Exact centered decoding of 810 arithmetic symbols, 810 carried challenge equalities, and one natural-number cursor increment in 1,621 rows |
| PiRLC family source rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCFamilySourceRows.lean:1` | One assignment across all 146,114 arithmetic, input, residual, challenge, and cursor rows; accepted rows imply the family relation except for two Poseidon2 replays |
| PiRLC family replay artifact | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_replay.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyReplayArtifact.lean:1` | Exact even and odd cursor shapes, 432 Rust-emitted Poseidon2 calls, direct input and output source columns, and extracted-SSA-to-reference refinement |
| PiRLC normalized body decoder | `crates/neo-fold-clean/src/frontends/r1cs_f_prime/selective_projected_decoder.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyBodyDecoder.lean:1` | Compact Rust-generated source-to-final slot rules; Lean checks the full even and odd source cover, no overlap, alias order, and final-slot bounds under property `FPRIME-PIRLC-FAMILY-BODY-DECODER-COVER` |
| PiRLC normalized body row ledger | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_relation/row_ledger.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyBodyRowLedger.lean:1` | The Rust compiler audit compresses to 8 fixed runs, 14 retained runs, and 26 affine batches for 3,064 rewrites; Lean independently proves exact ownership of 558,932 even source rows, 560,132 odd source rows, all rewrite identifiers, and all 279,089 emitted rows under property `FPRIME-PIRLC-FAMILY-BODY-ROW-LEDGER-COVER` |
| PiRLC retained algebra port images | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_relation/retained_algebra.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyBodyAlgebraRetained.lean:1` | Rust checks the exact source recipe and all 13 normalized ports for the first 43,794 retained rows in both parity arms; Lean cross-checks the compact receipt with the source recipe, decoder, row ledger, radix map, and independently recomputed nonzero census under property `FPRIME-PIRLC-FAMILY-BODY-ALGEBRA-RETAINED-PORT-IMAGE` |
| PiRLC normalized algebra semantics | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedAlgebraRows.lean:1` | Exact 45,415-to-2,484,972 radix substitution, parity selectors, and thirteen-port product points; active normalized acceptance implies all 43,794 source rows and `FamilyPhaseRelation` on one decoded assignment |
| PiRLC retained residual port images | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_relation/retained_residual.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyBodyResidualRetained.lean:1` | Rust checks the exact 108-row additive residual recipe and all 13 normalized ports in both parity arms; Lean checks the receipt against the direct radix-seven decoder run, row ledger, and independent nonzero census under property `FPRIME-PIRLC-FAMILY-BODY-RESIDUAL-RETAINED-PORT-IMAGE` |
| PiRLC normalized residual semantics | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedResidualRows.lean:1` | Active normalized acceptance, exact residual-state placement, and the authoritative local commitment output imply the concrete 108-field additive residual transition on the same decoded assignment |
| PiRLC retained carry port images | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_relation/retained_carry.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyBodyCarryRetained.lean:1` | Rust checks the exact 1,621-row challenge-carry recipe and all 13 normalized ports in both parity arms; Lean checks the receipt against the direct radix-seven decoder run, row ledger, and independent nonzero census under property `FPRIME-PIRLC-FAMILY-BODY-CARRY-RETAINED-PORT-IMAGE` |
| PiRLC normalized carry semantics | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedCarryRows.lean:1` | Exact 146,224-to-2,484,972 source substitution for the retained carry block; active normalized acceptance and carried strong-set membership derive the five-symbol range, centered challenge decoding, unchanged challenge carry, and one cursor increment on one decoded assignment |
| PiRLC retained family-overlay port images | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_relation/retained_overlay.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyOverlayRetained.lean:1` | Rust checks all 110 compact seeded A blocks and every retained explicit row in all 13 ports; Lean checks exact selector and row geometry, the six verifier seed chunks, and independent compact and explicit nonzero censuses under property `FPRIME-PIRLC-FAMILY-OVERLAY-RETAINED-PORT-IMAGE` |
| PiRLC normalized family-overlay semantics | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedOverlayRows.lean:1` | Exact 33,360-to-35,856 low-norm source image; an active family selector, constant one, accepted 108-row overlay, and exact physical source digits place the concrete 108-field family commitment without digest authority |
| PiRLC normalized body-overlay link receipt | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_relation/retained_links.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyNormalizedLink.lean:1` | Rust checks the exact 640-field normalization shift, both parity maps, all 33,359 links per family, and the shared body and overlay final slots; Lean validates the compact receipt under property `FPRIME-PIRLC-FAMILY-NORMALIZED-LINK-SLOT-IMAGE` |
| PiRLC normalized opening-row receipt | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_pi_rlc_family_relation/opening_rows.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyBodyOpeningRows.lean:1` | Rust checks 50,707 exact rows: 16,605 packed active-digit rows, 82 zero-digit rows, and 34,020 canonical-opening rows; Lean checks all source coordinates, final coordinates, chunk bounds, complement flags, and row images |
| PiRLC normalized opening semantics | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedOpeningRows.lean:1` | Accepted packed rows derive the active signed-digit range; accepted canonical rows and the outer borrow-coordinate range derive each canonical 41-digit opening; the exact source-slot fold then derives the body source digits used by the link and overlay proofs |
| PiRLC normalized outer-norm transfer | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedOuterNorm.lean:1` | An exact typed identity map from the 2,484,972-column body view to the same-width Phi81 assignment transfers the verifier-owned `b = 4` norm, including the norm from one fresh CCS opening, to all 32,400 borrow coordinates; the complete artifact must still prove this width and assignment identity |
| PiRLC normalized body-overlay link semantics | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedLinkRows.lean:1` and `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedOpeningRows.lean:1` | Active accepted opening, equality, and overlay rows transfer exact body digits into the seeded overlay and transfer its concrete 108-field output back into the body residual slots; body source digits are no longer an external premise |
| PiRLC normalized family composition | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCNormalizedFamilyRows.lean:1` | One model-level final assignment feeds the retained algebra, residual, and carry blocks; joint acceptance derives the residual update, challenge range, decoded challenge equality, challenge carry, cursor increment, and exact algebra output, then implies the concrete family phase subject only to two Poseidon2 replay facts and exact placement premises |
| PiRLC complete family rows | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCFamilyCompleteRows.lean:1` | 275,114 even rows or 276,314 odd rows; accepted source and replay rows imply `FamilyPhaseRelation` with no replay-equality premise |
| PiRLC family authority | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCAuthority.lean:399` | One ordered 937-field family state with algebraic input residual, input and output replay, exact carried challenges, and cursor authority |
| PiRLC family sequence | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCFamilySequence.lean:1` | One model-level 110-state chain carries the exact challenges, telescopes every concrete residual update, reaches cursor 110, and recovers the authoritative PiCCS inputs and exact PiRLC outputs or one named Module-SIS failure |
| PiRLC physical state decoding | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyState.lean:1` | Exact decoding of both 937-field semantic states from every generated even or odd family body, including all source-row placement facts |
| PiRLC public state suffix | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyPublicState.lean:1` | The accepted suffix binds four-lane before and after full-`x_out` hashes and two exact non-wrapping global cursor words. The local 937-field family digest occupies the semantic-state slots of each 32-field preimage |
| PiRLC full-`x_out` artifact | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyXOutArtifact.lean:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyXOutPreimage.lean:1` | Exact Rust-generated rows prove both nine-round Poseidon2 hashes and the structural preimage roles: domain, both counter pairs, fixed program counter, semantic-state digest, and Nebula marker. The verifier digest, PiCCS header, boundary, Construction-2 accumulator, and Nebula digest remain opaque until lifecycle circuits derive them |
| PiRLC physical family adapter | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyPhysicalState.lean:1` | One accepted body, linked overlay, and public suffix imply the exact family phase, all ten public words, and the structural meaning of both full-`x_out` preimages on the same body assignment |
| PiRLC local-digest continuity | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyContinuity.lean:1` | Inner prototype layer: equal direct family-digest words give exact family-cursor continuity and equal 937-field states, or one named collision in the exact framed Poseidon2 digest. This is not full recursive-state continuity |
| PiRLC full-`x_out` continuity | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyXOutContinuity.lean:1` | Security-reduced two-layer theorem: equal complete `x_out` values recover equal semantic-state digests or an outer binding failure; those local digests recover the exact 937-field family state or the named inner Poseidon2 collision. Phase hash rows are artifact-checked, but typed outer-state authority and shared-wire enforcement are not complete |
| PiRLC full-`x_out` family sequence | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyXOutSequence.lean:1` | Security-reduced 110-arm theorem: accepted physical family arms, pinned stateful outer states, local semantic-digest bindings, and equal adjacent full `x_out` values construct the exact semantic run or return a named outer binding failure or inner Poseidon2 collision. Phase structural fields are artifact-checked; authoritative outer states, transitions, shared equality, start, and finish remain premises |
| PiRLC physical family sequence | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPiRLCFamilyPhysicalSequence.lean:1` | Inner semantic prototype: one accepted arm for every exact ordinal and equal adjacent direct family-digest words construct the exact 110-step semantic run, or one named local Poseidon2 collision; start and finish authority then give exact outputs or a concrete binding failure or collision |
| PiCCS generated-row bridge | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiCCSRoundArtifact.lean:201` | One exact round |
| PiRLC generated-row bridge | `formal/nightstream-lean/Nightstream/Implementation/Nebula/Production/Carrier/StreamingPiRLCArtifact.lean:77` | One exact family |
| Common-plus-phase CCS composer | `crates/neo-fold-clean/src/frontends/r1cs_f_prime/grouped_phase.rs:1` | Executable shared-public composition with exact Rust-to-Lean link rows |
| Exact schedule composer | `crates/neo-fold-clean/src/frontends/r1cs_f_prime/grouped_phase.rs:1` | Stores lifecycle and phase-kind circuits once; exact selector and cursor links are artifact-checked |
| Linked-overlay composer | `crates/neo-fold-clean/src/frontends/r1cs_f_prime/linked_overlay.rs:1` and `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/SelectiveCcs/SelectorComposition/ScheduledLinkedOverlay.lean:1` | Adds one separately stored selective overlay, exact schedule selection, activation, and gated radix-decoded private links |
| Linked-overlay fixture | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/SelectiveCcs/SelectorComposition/ScheduledLinkedOverlayArtifact.lean:1` | Recomputes the generated 384-row, 540-column, 54-public-column artifact and proves row acceptance is exactly the linked semantic contract plus padding |
| Production schedule adapter | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_relation.rs:1` | Supplies the exact 2-circuit, 23-kind, 400-arm maps to the composer |
| Generated overlay schedule | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingFPrimeProgramArtifact.lean:1` | Checks the exact Rust claim-coordinate map, the combined 136-kind overlay map, all 25 claim link runs, and the three normalized PiRLC family link runs |
| 400-arm overlay refinement | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/StreamingPhasedOverlayRelation.lean:1` | Exact five-family semantic interface; accepted lifecycle, phase, schedule, overlay, and private-link rows imply the verifier-selected program step |
| Generic grouped-product row bridge | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/SelectiveCcs/Rewrite/Artifact/EvaluationRowBridge.lean:1` | Any active decoded evaluation row with exact port images is equivalent to its decoded source recurrence on values read from the same final assignment; the six-row Rust fixture now uses this theorem |
| Generic retained-row bridge | `formal/nightstream-lean/Nightstream/Implementation/R1CS/Correspondence/SelectiveCcs/Rewrite/Artifact/RetainedRowBridge.lean:1` | Exact normalized A/B/C port images and an active selector are equivalent to one decoded source R1CS row on the same reconstructed assignment; exact matrix-row action transports the result to a physical artifact row |
| Shared public layout | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_public.rs:1` | Exact 641-column logical layout padded to 648 columns at ring degree 54 |
| Streaming full-state envelope | `crates/neo-fold-clean/src/frontends/nebula/f_prime/streaming_state_envelope.rs:1` | Isolated Rust-conformant helper for the required stateful, Nebula-present envelope: it reuses the canonical `state_x_out` circuit and emits the exact 256 little-endian public bits. It is not yet connected to the large phase artifacts |
| Terminal proof path | `crates/neo-fold-clean/src/frontends/r1cs_f_prime/terminal_r1cs/lifecycle.rs:101` | Existing monolithic terminal lifecycle |

## Concrete bounded program

The verifier-owned program has 400 arms. It uses three logical lifecycle
modes: base, bootstrap, and steady. These modes need only two physical
lifecycle circuits. Arm 0 uses the base circuit. Every later arm uses the same
recursive circuit. The current control-flow artifact maps the phase-local work
to 23 physical circuit kinds. PiRLC family phases use separate even- and
odd-indexed physical kinds because their exact Poseidon replay matrices contain
215 and 217 calls, respectively. The semantic phase stays one PiRLC family
relation. The physical map also accounts for the fixed-position claim-chunk
commitment rows described below.

Claim replay starts only after the prelude and all 82 prior-state chunks. Its
first program cursor is therefore 83. Its final claim arm is selected at cursor
168. The claim frame offset is `(program cursor - 83) * 1024`. This offset is
part of the generated Rust-to-Lean artifact.

All physical circuits must use this shared public prefix:

| Columns | Meaning |
|---|---|
| `0` | Constant one |
| `1..257` | After-state Poseidon2 digest; this is `x_out` |
| `257..513` | Before-state Poseidon2 digest |
| `513..577` | Before program cursor, 64 bits |
| `577..641` | After program cursor, 64 bits |
| `641..648` | Zero padding for ring degree 54 |

The after-state digest is not copied into a second suffix. Both the lifecycle
relation and the selected phase relation must recompute their digest view from
their authoritative inputs. The shared digest can connect the two relations,
but it cannot replace either relation's local checks. The current schema and
composer fix this layout.

The current PiRLC family suffix now computes the complete before and after
`x_out` hashes from two 32-field preimages. Exact generated rows bind the
domain, both copies of the global cursor halves, the fixed program counter,
the four local semantic-state digest fields, and the Nebula-present marker.
They also prove both nine-round Poseidon2 hashes into the shared public words.

This is the required phase-side hash shape, but it is not complete lifecycle
authority. The verifier digest, PiCCS header, boundary, Construction-2
accumulator, and Nebula digest are still opaque witness values in the phase
circuit. HyperNova requires the lifecycle circuit to derive those values from
the complete prior recursive state and checked transition. The shared public
`x_out` words can join the lifecycle and selected phase only after both sides
recompute the same hash. This keeps the 648-column public layout unchanged.

## PiCCS variable-coordinate binding

Claim replay visits the prior running instances in frame order, while PiCCS
uses the same values in coefficient-major statement order. A sequential hash
state cannot make both orders authoritative without another complete pass or a
large carried state. The current binding uses the existing linear Ajtai map to
keep one fixed global position for each of the 21,220 variable fields.

Each field uses the standard 41-coordinate signed-ternary word. The complete
message has 16,112 degree-54 ring columns and 28 trailing zero coordinates.
During one phase, selected positions use their canonical words and all other
positions use one constrained zero word. Therefore partial commitments add to
the full-vector commitment in any phase order. If two different canonical
field vectors have the same commitment, the Lean model produces a nonzero
rank-two Module-SIS kernel witness with strict coefficient bound three.

For a phase with 1,024 active fields, the source R1CS has 127,127 rows and
126,104 columns. The production selective low-norm lowering snapshot has
84,401 rows and 67,068 committed and assigned coordinates. These local values
are below `2^24`. They are not a proof that the complete recursive or terminal
joint domain is below `2^24`.

This result fixes the representation and security-reduction boundary. The Lean
map now uses the exact rank-two, 16,112-column `SeededAjtai.Setup`, the fixed
Rust seed identity, the executable Phi81 quotient multiplication, and the
108-field output layout. The selector bridge now proves that every real matrix
coordinate reads the named canonical active digit or the shared constrained
zero word. It also proves that the final 28 matrix coordinates have no advice
column and that the compact row block uses the verifier-owned sampled
coefficient tensor. The source-row bridge proves that the 41 zero rows and the
ordered 124-row active openings imply `SourceColumnsExact`; the commitment
output is not a premise of that result. The output-row bridge proves that the
108 accepted compact rows equal the direct masked Module-SIS commitment. Its
proof covers sparse zero elision, the 28-coordinate zero tail, coefficient
rotation in the Phi81 quotient ring, and row-major output flattening. It does
not treat a supplied output commitment as an input premise. The complete-row
bridge then places the 41 zero rows, ordered 124-row active
openings, two fixed shape rows, and 108 output rows in the exact Rust family
order. For 1,024 active fields, it proves the exact 127,127-row census and that
accepted rows determine the fixed degree-54, rank-two shape and direct masked
commitment. It does not yet prove that Rust `rand_chacha` generates the same
matrix, prove Rust opening-cache and exact generated-artifact conformance, add
every required PiCCS pass to the 400-arm schedule, or carry the resulting
commitment through the recursive and terminal artifacts.

The claim-chunk partition now fixes where these selected fields enter the
stream. Claim chunk 0 supplies the 52 PiCCS point fields at local offsets 383
through 434. Claim chunks 60 through 81 supply the 21,168 evaluation fields.
The other 63 claim chunks supply no selected PiCCS coordinate field. Thus, 23
claim phases have nonempty coordinate rows. Their union has 2,634,753 source
rows:

`23 * (41 + 2 + 108) + 21,220 * 124 = 2,634,753`.

This is a union census, not the cost of one recursive step. One step checks
only the verifier-selected local phase. The 86 masks are disjoint and cover
the complete 21,220-field witness, so their partial commitments add to the
direct full commitment. The 400-arm schedule does not change.

Rust synthesis uses a base-plus-overlay design. The base stores the full and
final Poseidon2 replay bodies once. Its exact low-norm shape is 57,469 rows by
671,868 columns. The overlay stores one no-op arm, one full-carry arm, one
final-carry arm, and 23 active fixed-position coordinate arms. Its exact
selective-union shape is 1,308,967 rows by 67,500 columns. Gated private link
rows bind every selected overlay chunk and every before/after commitment field
to the exact replay fields. This avoids 23 copies of the 641,384-coordinate
replay body. Lean now checks the exact Rust 26-kind overlay map for all 400
arms. It also proves that the 86 linked claim-phase source witnesses start at
zero, apply every active or carry step, and finish at the direct commitment of
all 21,220 claim fields. The 400-arm model also proves that any accepted
base-plus-overlay relation selects one exact program arm and performs its
verifier-owned step, subject to exact row-family equivalences. Exact
source-field link metadata is now shared by Rust and Lean as 25 compact runs.
Rust expands these runs and checks all 26,620 source pairs against the actual
base and overlay syntheses: 5,400 before/after commitment pairs and 21,220
active claim-field pairs. Lean proves exact equality with the source schedule
and the same census. This metadata is explicit algebraic structure, not a
digest. Exact Rust conformance is now checked for the retained PiRLC algebra,
residual, and carry rows in both parity arms and for all 110 family-overlay
arms. Exact normalized body-overlay link slots are also Rust-conformant for
both parity arms. Exact conformance for the other rewrite rows, complete
assignments, and Lean-owned matrix actions remains open.

PiRLC input authority now uses one fixed-position rank-two Ajtai binding for
the complete ordered 89,100-field input vector. Every field uses 41 canonical
signed-ternary coordinates, for exactly 3,653,100 coordinates and 67,650
degree-54 message columns. The 110 family masks are disjoint and complete.
Each family phase opens only its 810 input fields at their unchanged global
positions. The 110 local commitments sum to the full input commitment. A
complete run that starts at the authoritative commitment and finishes at zero
recovers every supplied input ring or exposes one named Module-SIS binding
failure. The concrete Phi81 representation carries the rank-two commitment as
108 Goldilocks fields. The fixed production seed identity and all six derived
32-byte chunk seeds are now shared by Rust and Lean. The complete expanded
Rust coefficient action still needs a final same-matrix conformance bridge.

The exact fixed-position commitment footprint for one family is 100,591 rows
by 99,782 columns. The complete commitment-and-residual block has 100,699
rows. Centered challenge decoding, challenge carry, and cursor increment add
1,621 rows. With the 43,794 arithmetic rows, the complete current family
source block has exactly 146,114 rows. A direct full-vector source relation is
11,048,551 rows by 10,959,452 columns. Each value is below `2^24`. These are
source bounds, not normalized selective-CCS or joint recursive bounds.

PiRLC family semantics now use one ordered 937-field source state: a 9-field
input Poseidon2 replay state, the 108-field algebraic input residual, a 9-field
output Poseidon2 replay state, 810 exact coefficients for the 15 carried
challenge rings, and one family cursor. The Poseidon2 states are checked
compression and collision boundaries. They are not input authority. Each
family phase uses the same verifier-owned family label and the same 810 input
fields for the residual update, the input replay, and the PiRLC algebra. It
absorbs the 54 output coefficients into the output replay, carries the
challenges without change, and increments the cursor.

The exact 100,699-row family input block now proves its 108 residual updates
from the same 810 fields that create the local commitment. The commitment is
not a premise. The 108 compact output columns are the phase columns in the
residual equations. This closes the local residual-authority gap at the
handwritten source-row level.

A same-assignment Lean theorem now joins this block to the 43,794 PiRLC
arithmetic rows and the 1,621 challenge-and-cursor rows. The extra 810 decode
rows are necessary because the arithmetic columns contain symbols `0..4`,
while the carried challenge fields contain their centered images `-2..2`.
Accepted rows derive `FamilyPhaseRelation` with no residual, challenge-carry,
or cursor-increment premise. Exact arithmetic input placement makes the
opening and algebra rows read the same 810 values. The carry decoder reuses
the arithmetic layout, so the ring combination and carried challenge fields
also use the same symbols.

The PiRLC replay artifact stores two exact cursor-parity arms. The even arm
has 215 Poseidon2 calls and 129,000 rows. The odd arm has 217 calls and
130,200 rows. Thus, Rust emits 432 calls across both stored arms. The input
replay reads arithmetic columns 811 through 1,620. The output replay reads
arithmetic columns 1,621 through 1,674. The two eight-lane replay-before
states start at columns 146,224 and 146,232. There is no copy row or digest
indirection between the arithmetic values and the replay.

The complete accepted family relation has 275,114 rows for an even family and
276,314 rows for an odd family. Its Lean soundness theorem derives
`FamilyPhaseRelation` without a replay-equality premise. The physical replay
proof first uses the exact Rust-extracted Poseidon2 SSA. An explicit,
independent bridge then proves equality with the reference Poseidon2
permutation. This keeps the core column replay tied to the extracted circuit.

The normalized body decoder now covers source columns `1..559136` for the
even arm and `1..560336` for the odd arm. Both arms use 2,484,972 final
columns. The even compact image classifies 511,020 template columns and
48,115 residual columns. The odd image classifies 512,220 template columns
and the same 48,115 residual columns. The Rust artifact-current test expands
every rule against the prepared selective layout. The independent Lean check
expands all 559,135 or 560,335 source owners, rejects duplicate ownership, and
checks every final span and backward alias reference. This is artifact-checked
for `FPRIME-PIRLC-FAMILY-BODY-DECODER-COVER`. It does not prove that a linear
definition or an eliminated trace reconstructs the source value, and it does
not prove matrix or assignment equality.

The normalized body row ledger now covers 558,932 even source rows, 560,132
odd source rows, all 3,064 rewrite identifiers, and all 279,089 emitted rows.
The compact artifact has 8 fixed runs, 14 retained runs, and 26 affine rewrite
batches. It records 1,376 Poseidon2 rewrites, 1,620 shifted-ternary canonical
rewrites, and 68 linear definitions. Four Poseidon2 boundary rewrites emit 90
rows; the other 1,372 emit 86 rows. The independent Lean checker rejects
overlap, gaps, invalid arms, invalid rewrite widths, and out-of-range spans.
This is artifact-checked for
`FPRIME-PIRLC-FAMILY-BODY-ROW-LEDGER-COVER`. It proves row ownership and
geometry only. It does not prove port images, matrix actions, or assignment
values.

The first retained interval in each parity arm is the complete 43,794-row
PiRLC algebra. Rust now checks both source parity matrices against the
independent 15-source, 54-lane Phi81 recipe. It then checks every selected row
in all 13 final ports. Port 1 contains the exact arm selector. Ports 2, 3, and
4 contain the radix image of source A, B, and C. The other nine ports are
zero. The final field images use radix 7 at width 23 and radix 3 at width 41.
The two-arm final nonzero census is 87,588 for the selector, 2,099,628 for A,
6,343,920 for B, and 2,014,524 for C. Lean independently recomputes this
census and the 3,996 reduced product coefficients per source. It also checks
the receipt against the source recipe, decoder, and row ledger. This is
Rust-conformant for
`FPRIME-PIRLC-FAMILY-BODY-ALGEBRA-RETAINED-PORT-IMAGE`. The receipt is not
matrix authority in Lean.

The model-level normalized algebra theorem now defines the exact
45,415-to-2,484,972 radix substitution used by this receipt. It gives every
source linear combination one final `Form`, constructs the exact product
point with only ports 1 through 4 active, and reads all rows from one decoded
final assignment. If the selected parity coordinate is one and all 43,794
product points accept, the theorem derives every authoritative generated
source row. With the separate constant-one, range, transition, and state-link
premises, it derives `FamilyPhaseRelation` on that same assignment. The sparse
slot evaluator is proved equal to the dense form action, so this theorem does
not replace matrix semantics with a second assignment. Combined with the
exhaustive Rust matrix scan, this is Rust-conformant evidence for the retained
algebra recipe. Exact Lean authority for the stored matrix artifact, selector
value derivation from other accepted rows, digit range and canonicality rows,
and Rust encoder assignment equality remain open.

The next retained interval is the complete 1,621-row challenge-carry block.
Rust checks source rows 144,385 through 146,005 against the independent
centered-decode, challenge-equality, and cursor-increment recipe in both parity
matrices. It also checks all 13 final ports at emitted rows 78,113 through
79,733 and 200,471 through 202,091. The final image uses direct radix-seven
slots for the 810 challenge symbols, both 810-field carried vectors, and both
cursor fields. Lean checks the compact receipt against the exact decoder run,
the row ledger, and independently computed source and final nonzero counts.
This is Rust-conformant for
`FPRIME-PIRLC-FAMILY-BODY-CARRY-RETAINED-PORT-IMAGE`.

The model-level normalized carry theorem decodes these direct slots from the
same 2,484,972-field assignment used by the normalized algebra theorem. If
the selected parity coordinate and constant coordinate are one and all 1,621
product points accept, it derives every source challenge-carry row. Exact
state placement and membership of the carried challenge in the production
strong set then derive each decoded symbol in `0..4`. Thus, no separate
per-family challenge-range rows are necessary. With the cursor bound, the same
theorem derives exact centered challenge decoding, unchanged carried
challenges, and one cursor increment. Combined with the exhaustive Rust matrix
scan, this is Rust-conformant evidence for the retained carry recipe. Exact
Lean authority for the stored matrix artifact, selector and constant
derivation, the start-to-carry strong-set invariant, state placement, and Rust
encoder assignment equality remain open.

The retained residual interval contains exactly 108 additive equations. Rust
checks source rows 144,277 through 144,384 against their independent recipe in
both parity matrices. It also checks all 13 final ports at emitted rows 78,005
through 78,112 and 200,363 through 200,470. The image uses direct radix-seven
slots for the phase commitment output and the before and after residual
vectors. Lean checks the compact receipt against the decoder run, row ledger,
and independently computed source and final nonzero counts. This is
Rust-conformant for
`FPRIME-PIRLC-FAMILY-BODY-RESIDUAL-RETAINED-PORT-IMAGE`.

The model-level normalized residual theorem reads these direct slots from the
same 2,484,972-field assignment. Active normalized acceptance, constant-one
placement, exact before and after residual-state placement, and exact family
overlay output placement derive the concrete additive residual transition.
The local commitment is read from the authoritative overlay output. It is not
a digest or an independent semantic premise.

The separate normalized family overlay has 110 selected arms. Every source
arm has 108 rows and 33,360 columns. The normalized union has 28,627 rows and
35,856 columns. Selector columns are 1 through 110. The 110 retained row
blocks start at row 16,737 with stride 108. Physical digit columns 1 through
33,251 use one direct centered coordinate at final column `source + 110`.
The 108 physical output columns start at 33,252 and use 23 radix-seven
coordinates from final column 33,362 through 35,845. Rust checks every source
seeded block, all 110 normalized seeded A blocks, the constant-one B terms,
the radix-seven C terms, and zero in every unused port. Lean checks the exact
six 32-byte verifier seed chunks, affine row and selector geometry, and the
compact and explicit nonzero censuses. This is Rust-conformant for
`FPRIME-PIRLC-FAMILY-OVERLAY-RETAINED-PORT-IMAGE`. The receipt stores the
seed chunks directly. It does not use a seed digest as authority.

The model-level normalized overlay theorem decodes this exact 35,856-field
assignment. If one family selector and the constant coordinate are one, all
108 selected product points accept, and the physical source digits are exact,
then every radix-seven output equals the corresponding coordinate of the
concrete family commitment. Thus, the overlay output is derived from the
verifier-owned seeded map and the source witness. It is not an independent
commitment premise.

The body-overlay link audit found and corrected a same-assignment defect. The
field-lowering pass moves 640 public outputs to the normalized prefix. The old
compact link contract used the pre-lowering body fields 45,415, 45,456, and
144,278. The correct normalized body fields are 46,055, 46,096, and 144,918.
The prior honest test compared two pre-lowering witnesses, so it did not find
this error. The corrected test reads the normalized body assignment. Rust now
checks both prepared parity maps and the exact final slots for all three runs:

- 41 zero-word links join body slots starting at 1,059,804 to overlay slots
  starting at 111;
- 33,210 active-word links join body slots starting at 19,332 to overlay slots
  starting at 152;
- 108 output links join width-23 radix-seven body slots starting at 1,076,091
  to overlay slots starting at 33,362.

This gives 33,359 links per family and 3,669,490 links across all 110
families. A Rust-conformant Lean receipt checks the 640-field shift, both
parity kind codes, the affine source and final-slot geometry, and agreement
with the independent body decoder and overlay receipts. The model-level link
theorem proves that active accepted equality rows transfer exact body digits
into the overlay and transfer the concrete overlay commitment into the body
phase-binding slots. Thus, overlay output placement is not an independent
authority premise when these link rows accept. The exact 50,707-row opening
receipt checks 16,605 packed active-digit rows, 82 zero-digit rows, and 34,020
canonical-opening rows. The model-level opening theorem derives every active
body digit from those rows and from the exact 41-digit source-slot fold. Thus,
body digits and overlay output placement are not independent authority premises
when the opening, link, and overlay rows accept. A typed same-assignment
theorem now maps the full 2,484,972-column body view into a Phi81 assignment
of the same width. The verifier-owned `b = 4` norm, including the norm from a
fresh CCS opening, then supplies all 32,400 borrow-coordinate norm facts. The
remaining generated boundary is proof that the complete artifact has this
exact width and that its body view is this same outer assignment.

The model-level normalized family theorem now composes the retained algebra,
residual, and carry blocks on one final assignment. It proves that the algebra
and carry blocks decode each challenge-symbol slot from the same radix-seven
value. Joint acceptance, constant-one placement, residual and carry-state
placement, authoritative family-overlay output placement, a carried strong-set
challenge, and the family ordinal derive the residual update, challenge range,
centered decoded challenge, unchanged challenge carry, cursor increment, and
exact algebra output. The caller no longer supplies those conclusions as
independent premises. The remaining `ReplayTransition` premise contains only
the normalized input and output Poseidon2 replay facts. Selector and constant
derivation, state placement, the start-to-carry strong-set invariant, stored
Rust matrix authority, and complete Rust witness-assignment equality also
remain open. The normalized link theorem now derives the overlay output
placement from accepted opening, link, and overlay rows. The opening theorem
derives the exact body digits from the same final assignment.

The model-level family sequence now joins all 110 local transitions through
one continuous state function. Its prefix theorem proves that the starting
108-field residual is the sum of every processed family commitment plus the
current residual. At family 110 this is exactly the complete concrete residual
equation. A row-derived authoritative start and a zero terminal residual
therefore recover every supplied PiRLC input, or expose the concrete
Module-SIS failure. In the non-failure branch, every stored family output is
the exact PiRLC combination of the authoritative PiCCS inputs and the one
carried challenge array.

The generated family suffix binds each accepted body to ten public words: four
before-`x_out` words, four after-`x_out` words, and two global cursor words.
Each full hash contains the direct digest of the 937-field family state in its
four semantic-state slots. Exact rows now prove this placement and the other
structural slots. Equality of the inner family digests gives equality of the
complete 937-field semantic states, or one named collision in the exact framed
Poseidon2 digest. This remains the inner security-reduced continuity layer.

The full-state theorem states the required two-layer reduction without
treating either digest as authority. Equal complete `x_out` values give equal
stateful semantic-digest components or a named outer `x_out` binding failure.
Equal semantic components then give equal 937-field family states or the named
local Poseidon2 family-state collision. The generated family circuit now
recomputes both complete hashes. It does not yet derive the five opaque outer
components from verifier-owned lifecycle state. The grouped recursive layout
must also make adjacent full outputs equal or enforce their equality.

The full-state sequence theorem applies this two-layer reduction at every one
of the 109 adjacent family boundaries. In its no-failure branch, all 110
accepted physical arms form the exact model-level `AcceptedRun`. The theorem
then recovers the authoritative PiRLC outputs, or returns the concrete
Module-SIS binding failure. It does not infer circuit facts that do not yet
exist: pinned typed outer-state inputs, the typed-to-physical semantic-state
bridge, adjacent full-`x_out` equality, and the start and finish relations
remain explicit premises. The accepted-arm adapter now derives the phase-side
semantic slots and structural preimage fields. A bridge from those physical
fields to the typed authoritative outer state is still missing.

One inner physical sequence theorem applies the direct-family-digest result to
all 110 exact family ordinals. In the no-collision branch, it constructs the model-level
`AcceptedRun` from the exact inputs, outputs, and before and after states of
the accepted physical arms. Given the existing semantic start and finish
relations, it derives every authoritative PiRLC output or returns the concrete
Module-SIS binding failure or the named Poseidon2 continuity collision. This
theorem does not supply authoritative outer-state lifecycle rows, generated
equality, start, or finish rows.

The start rows must still derive the full input residual from the exact PiCCS
output projection and bind it to the before state of family zero. The finish
rows must bind the after state of family 109 and force its residual to zero.
Expanded matrix and assignment conformance, normalized selective slots and
matrices, and recursive and terminal integration remain open.

## Current gap

The 400-item schedule and its physical circuit map prove control flow. The
PiRLC family source and replay relation is complete at the handwritten and
generated source-row boundary. The normalized PiRLC body source-to-final slot
classification and physical row ownership are exact. The retained algebra,
residual, and carry port images are Rust-conformant for both parity arms. All
110 retained family-overlay port images are also Rust-conformant. The exact
normalized recipes now have model-level same-assignment implications to all
43,794 algebra rows, all 108 residual rows, and all 1,621 challenge-carry
rows. A separate overlay theorem proves that each accepted 108-row selected
arm computes the exact concrete family commitment when its physical source
digits are exact. The carry implication derives the five-symbol challenge
range from the carried strong-set invariant, then discharges exact challenge
decoding, challenge carry, and cursor increment when state placement holds.
The residual implication derives the additive input-residual update from the
authoritative family-overlay output. Their joint same-assignment theorem also
derives the exact algebra output and implies the concrete family phase,
subject only to the two normalized Poseidon2 replay facts in
`ReplayTransition` and exact placement premises. The normalized link theorem
discharges the family-overlay output placement from accepted opening, link,
and overlay rows. The normalized opening theorem derives the exact body source
digits, subject to outer SuperNeo norm membership for the borrow coordinates.
An exact typed bridge now derives that norm membership from one fresh
radix-four CCS opening on the same assignment. The complete generated artifact
must still prove the exact outer width and assignment identity used by this
bridge. The model-level 110-family sequence now telescopes all local residual
updates and derives authoritative outputs or the named Module-SIS failure.
Equal adjacent direct family-digest words imply exact cursor continuity and
complete local family-state continuity, or one named inner Poseidon2
collision. A separate model theorem now composes the complete `x_out` binding
reduction with this local reduction. A full 110-arm theorem now uses these
reductions to refine accepted physical arms to the exact model sequence, under
explicit outer-state and boundary premises. The current generated family
suffix now implements the full 32-field hash envelope and proves its
structural slots. It does not give authority to the verifier digest, PiCCS
header, boundary, Construction-2 accumulator, or Nebula digest. The lifecycle
circuits must derive those five components from the complete prior state and
checked transition, then recompute the same shared hash. The grouped layout
does not yet enforce adjacent full-`x_out` equality. The production layout
must add this lifecycle authority, enforce full-state continuity, and bind the
PiCCS-to-family-zero and family-109-to-finish boundaries.
Authority for the
stored matrix artifact, selector and constant derivation, the start-to-carry
strong-set invariant, state placement, and complete Rust assignment
conformance is still open. The
grouped-product evaluation-row implication is now generic:
exact final port images and an active selector imply the decoded source
recurrence on the same assignment.
The compact body decoder now supplies the exact slot classification.
Production still needs proofs that the remaining classifications supply the
required port images, plus reconstruction bridges for linear definitions,
eliminated traces, and the other rewrite families. The retained-row
implication itself is generic. The algebra, residual, carry, and overlay
intervals now have exhaustive Rust port-image certificates and model-level
connections to their source relations. Full Lean authority for the stored
matrix actions and complete Rust-to-Lean witness assignment remain open. The
schedule does not yet prove the complete F′ relation. The remaining relation
must add all
application, transcript, claim-replay, PiCCS, PiRLC, decomposition,
successor-binding, and terminal rows to one shared low-norm assignment. It
must also prove that phase-local rows compose to the monolithic Lean relation.

The smallest practical design stores the complete outer verifier once and the
phase union once. Both components use one exact shared public authority prefix;
their private regions are separate. This makes committed width proportional to
the outer verifier plus the largest phase, not the sum of all phase widths.
Checked group-sum and activation rows tie every selected phase to its lifecycle
branch. Common rows occur once. Lean proves the generic selector contract and
the exact generated link-row recipe. A generated 384-by-540 fixture proves the
three-selector schedule link, overlay activation, and one radix-three private
field link. The base and overlay relations are now executable, but the complete
400-arm matrix artifact, the remaining production phase rows, and the recursive
fixed point remain open. The source-level claim overlay schedule and its
86-phase commitment refinement are complete. The model-level 400-arm
base-plus-overlay refinement is also complete. These results do not yet
identify the generated low-norm production rows with the source model. The
compact source-link artifact and the independent normalized-link receipt now
fix the exact final slots, and the model theorem proves the equality-row
meaning. Complete Rust-to-Lean witness assignment equality and stored-matrix
authority remain open.

## Completion evidence

Completion needs all of the following current evidence:

- exact recursive and terminal artifacts;
- generated-row soundness for every row family;
- complete phase-sequence refinement to the reference F′ relation;
- Rust-to-Lean matrix, assignment, encoder, and layout equality;
- exact joint-domain bounds at or below `2^24`;
- passing `prove`, `extend`, `finish`, and `verify` lifecycle tests;
- hostile tests for wrong phase, wrong cursor, substituted frame, wrong
  transcript state, wrong memory link, and incomplete terminal state.

HyperNova states constant-step knowledge soundness, not one uniform extractor
for a polynomial number of steps. Nightstream must not claim a stronger
security theorem without a separate proof. Cryptographic collision resistance,
commitment binding, Fiat–Shamir security, and SuperNeo proof-system soundness
remain explicit assumptions.
