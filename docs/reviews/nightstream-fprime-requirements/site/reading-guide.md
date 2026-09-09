## How the requirements relate

The HTML tree groups work by subject. The `depends_on` field records a different relationship: a result that a requirement uses. The reading map below explains the main interfaces. It does not add dependency edges or certify any theorem.

The folding flow is PiCCS → PiRLC → PiDEC. NIFS owns their composition. HyperNova uses that folding interface inside its recursive computation. Circuit lowering and Rust conformance connect the mathematical statements to the selected implementation.

| Contract or boundary | Existing requirement owners | What must connect |
| --- | --- | --- |
| Statements and representations | {{F.field}}, {{F.ring}}, {{F.packing}}, {{F.transform}}, {{F.norm}} | The field, ring, witness, public inputs, dimensions and norm bounds must describe the same values. |
| Fresh and running claims → common evaluation claims | {{C.input}}, {{C.output}}, {{C.security}} | PiCCS must establish the required constraints and evaluation claims, plus its stated strong reduction. |
| Common evaluation claims → combined parent | {{R.input}}, {{R.combine}}, {{R.security}} | PiRLC must combine the same instances with the selected challenges and establish its weak reduction. |
| Combined parent → bounded children | {{D.prover}}, {{D.verify}}, {{D.security}} | PiDEC must connect recombination to valid child openings. These children remain obligations for later verification. |
| Three phases → one folding theorem | {{N.profile}}, {{N.local.wiring}}, {{N.security.same_phi}}, {{N.security.composition}} | Use one relation, commitment projection, setup and profile. Include extraction, probability loss and expected runtime in the composed claim. |
| Public coins → selected transcript | {{T}}, {{N.security.fiat_shamir}} | Bind the ordered inputs, context and messages. State the security model that justifies the selected non-interactive protocol. |
| Folding → recursive computation | {{H.compat}}, {{H.state.enc_hash}}, {{H.base}}, {{H.recursive}} | Define the running bundle, default state, bounded public encoding and faithful encoding/decoding of accepted assignments. |
| Selected circuit → actual implementation | {{N.local.actual_step}}, {{L.lowering}}, {{L.export}}, {{P.runtime}}, {{P.assurance}} | Show that the accepted rows and actual code use the same values, matrices, profile and authority. |
| Terminal acceptance → intended computation | {{H.terminal}}, {{H.security.history}}, {{P.delivery.terminal}}, {{P.delivery.scope}} | Check the running openings, newest committed witness and public-state link. State exactly what accepted verification proves. |
| Material outside selected Stage 1 | {{O}} | Track exclusions separately. Folding or terminal soundness does not, by itself, establish zero knowledge or succinctness. |

These are overlapping reading paths through the same IDs. They do not create extra requirements or extra completion credit.

## What each status means

| Axis | Meaning | Counts as finished |
| --- | --- | --- |
| Proof (`proof`) | A local Lean result for the stated obligation. | `proved` |
| Link (`connection`) | Connection to the required local consumer. Read the requirement and evidence to see which connection. | `connected` |
| Rust (`rust`) | Implementation or scoped execution evidence. | `implemented`, `tested_scoped` |

For each axis, the denominator contains applicable leaf requirements. `not_required` and `assumption` are excluded. Leaves with origin `out_of_scope` are excluded from all completion totals. Group records do not earn separate proof credit.

- `definition`: a definition exists; it is not a finished proof and stays in the Proof denominator.
- `partial`: part of the proof or connection remains open.
- `open`: the required connection or implementation remains open.
- `assumption`: an explicit assumption, not a proved theorem.
- `not_reviewed`: this review did not establish the status.
- `recorded_only`: evidence was recorded; this is not a passed conformance result.
- `implemented`: code exists; formal implementation refinement is not implied.
- `tested_scoped`: tests passed within the recorded scope; read that scope before using the result.
- `not_required`: this axis does not apply to this entry.

An entry can have Proof finished while Link is open. Link is the connection between results and the exact values used by a consumer. It can include proof composition, but a compatible interface name or an import alone is not that connection.

Phase assurance uses the owner goal's separate terms: **Compiler-closed**, **Conformance-closed**, and **Production-closed**. These local counters do not assign those statuses. Conformance needs the required current matrix, assignment, complete nonzero result, mutation and independent-review evidence. Production also needs the validated package on the sole production path.

## Local results and complete security claims

Keep these claim levels separate when reading evidence:

| Claim level | Evidence needed for that level |
| --- | --- |
| Algebraic correctness | The required identities, bounds and witness transformations. |
| Randomized security | The verifier's random choices, malicious-prover model and probability bound. |
| Knowledge extraction | An extractor with the required success guarantee and runtime. |
| Composition | The above results apply to the same protocol, data, parameters and assumptions. |
| Implementation refinement | The actual program implements the stated mathematical behavior for the claimed inputs. |

The current source has no structured `claim_kind`, per-theorem `claim_scope`, or verified transitive-closure field. Do not infer those fields from a green counter. Use the requirement text, remaining obligation and source references to identify the claim. If they do not establish the scope, report it as unverified.

A theorem can be proved locally under hypotheses whose application is still open. Following recorded dependencies helps locate those obligations. Even if all recorded dependencies are locally finished, the map alone does not certify complete security: the dependency list may be incomplete, and assumptions still need their stated justification.

## How to use the export

1. Read this index, then follow the group Markdown links. Each group file contains all its subgroups and leaf entries, fully expanded.
2. Cite the stable requirement ID and its HTML link. The same ID occurs only once as a record in this export.
3. Use **Parent / Contains** for navigation. Use **Depends on / Used by** for recorded dependency edges. A reference to a group is a group-level dependency; it is not an automatically proved connection to every child.
4. Treat “none recorded” as missing relationship information, not proof of independence. “Used by” reverses only the recorded edges.
5. Preserve the three status axes, exact assumptions, profile and source snapshot when assessing progress. Counts do not measure time remaining or the chance of completion.
6. Open the repository-relative paper and code paths in the matching checkout. The export includes citations, not copies of the cited source files. Working-tree updates are not guaranteed to exist at the base commit alone; line numbers can move after a source edit.

The supplied AI feedback motivated this reading guide and the static export. Its separate repository review is not a fresh verification of this snapshot. The guide does not reclassify entries or adopt a new proof architecture.
