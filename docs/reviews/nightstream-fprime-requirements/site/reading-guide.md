## How the requirements relate

The [proof map](proof-map.md) puts the security goal above HyperNova, NIFS and verifier semantics. Read the main chain upward from verifier acceptance. Select a result for all its direct connections, conditions and Lean evidence. **All connections** shows the complete recorded graph. Premises, assumptions and prover construction sit beside the main chain. The map selects key declarations; it does not list every Lean dependency or change completion counts. [Open the interactive map](https://nightstream-requirements.nicarq.chatgpt.site/#proof-map).

The Requirements view groups work by subject. The Tech tree places the Stage 1 goal above production acceptance, recursion and the circuit, NIFS, the three phases, and shared primitives. Its lines summarize the main assembly; they do not certify theorem dependencies or completion. The `depends_on` field records the individual results that a requirement uses. The reading map below explains the main interfaces. It does not add dependency edges or certify any theorem.

The folding flow is PiCCS → PiRLC → PiDEC. NIFS owns their composition. HyperNova uses that folding interface inside its recursive computation. Circuit lowering and Rust conformance connect the mathematical statements to the selected implementation.

Select a group in the Tech tree to open its proof graph. Unlike the main assembly view, this graph draws only recorded `depends_on` edges, with prerequisites below their consumers. Each box is one requirement with separate Proof and Link status. Outside inputs and consumers can be included without merging them into whole-group dependencies. Items with no recorded connections stay visible separately. Selecting a box highlights its recorded prerequisite and consumer chains and shows its theorem evidence.

Each group Markdown file links to a proof-graph page with both diagrams, a status table, and the complete recorded connection table. These pages work as plain text and render diagrams in viewers that support Mermaid. Every graph page is included in the Markdown ZIP. Individual requirement records link to the interactive graph with that requirement selected.

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

The page uses compact finished/applicable counters. Proof counts only proved results; Link counts connected results; Rust counts code that exists or scoped tests that passed. Assumptions and N/A are excluded from these fractions. A Rust counter does not certify full conformance. Hover over a counter for its full breakdown, or read the expanded records. The tables below and the Markdown export retain every status category.

| Axis | Meaning | Visible count categories |
| --- | --- | --- |
| Proof (`proof`) | A local Lean result for the stated obligation. | Proved, Assumed, Open, N/A |
| Link (`connection`) | Connection to the required local consumer. Read the requirement and evidence to see which connection. | Connected, Open, N/A |
| Rust (`rust`) | Implementation or scoped execution evidence. | Scoped tests, Code only, Recorded, Open, N/A |

Each axis accounts for every in-scope leaf. Assumptions and N/A remain visible. These are counts, not a completion percentage. Leaves with origin `out_of_scope` are reported separately. Group records do not earn separate proof credit. The scope filter changes the visible records; the counts still describe the full group.

- `definition`: a definition exists; it counts as Open on the Proof axis.
- `partial`: part of the proof or connection remains open.
- `open`: the required connection or implementation remains open.
- `assumption`: an explicit assumption, not a proved theorem.
- `not_reviewed`: this review did not establish the status.
- `recorded_only`: evidence was recorded; this is not a passed conformance result.
- `implemented`: code exists; formal implementation refinement is not implied.
- `tested_scoped`: tests passed within the recorded scope; read that scope before using the result.
- `not_required`: this axis does not apply to this entry.

An entry can have Proof proved while Link is open. Link is the connection between results and the exact values used by a consumer. It can include proof composition, but a compatible interface name or an import alone is not that connection. Code only is not a passed Rust test.

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

Each record has a structured `scope`: model, interactive protocol, implementation evidence, production requirement, or outside Stage 1. This is a reviewed map annotation, not a theorem extracted from Lean. It does not certify transitive proof closure. Use the exact requirement and cited result to check its premises and scope.

A theorem can be proved locally under hypotheses whose application is still open. Following recorded dependencies helps locate those obligations. Even if all recorded dependencies are locally finished, the map alone does not certify complete security: the dependency list may be incomplete, and assumptions still need their stated justification.

## Assumptions, quantitative use and readiness

The [assumption ledger](assumptions.md) separates standard cryptography, model conditions and the unproved low-norm invertibility theorem. Repeated references to one premise share a ledger entry. Each entry records its parameters, approval state and known dependent uses. An unspecified value remains open; it is not zero or an implicit approval.

The [error budget](error-budget.md) gives conditional per-test bounds and accumulation over a supplied use count. It does not give a complete production false-acceptance bound. The union bound needs the per-test result to apply at each use, including its conditioning requirements. Knowledge-extraction losses, attack-cost estimates, setup bias, Fiat–Shamir transfer and hash collision advantages are separate claims. Use count and chain depth remain parameters until deployment requirements select them.

The [readiness view](readiness.md) records what remains before constraint reduction and complete Rust validation. It reuses requirement IDs and adds no proof credit. A cvc5 candidate still needs a Lean implication proof and the required relation-identity, layout and conformance checks.

## How to use the export

1. Read this index, then follow the group Markdown links. Each group file contains all its subgroups and leaf entries, fully expanded.
2. Cite the stable requirement ID and its HTML link. The same ID occurs only once as a record in this export.
3. Use **Parent / Contains** for navigation. Use **Depends on / Used by** for recorded dependency edges. A reference to a group is a group-level dependency; it is not an automatically proved connection to every child.
4. Treat “none recorded” as missing relationship information, not proof of independence. “Used by” reverses only the recorded edges.
5. Preserve the three status axes, exact assumptions, profile and source snapshot when assessing progress. Counts do not measure time remaining or the chance of completion.
6. Open code paths at the exact protocol code commit. The map commit separately identifies the site source. Local paper references have content hashes in the reference record; the export does not contain those papers. Read [source and evidence](evidence.md) for who ran each check and its authority.

Publication must use committed site inputs and rerun source-reference checks. File and line validity does not prove mathematical meaning. A website build, source hash or publication does not replace an independent protocol review or an approved conformance record.
