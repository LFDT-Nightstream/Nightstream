# Selected F′ constraint funnel

Package checkpoint: `41aa5ef63`. Profile: `b=2`, `k_rho=16`, `B=65536`, Poseidon2.
The totals are selected Lean counts and agree with the independent Rust package checks.
The phase and leaf splits below are arithmetic accounting; they are not each separate Lean theorems.

## Whole circuit → phases

| Phase | Logical CCS rows | Row share | Committed coordinates | Coordinate share |
|---|---:|---:|---:|---:|
| Pilot: prior and next state hashes | 2,125,538 | 59.24% | 91,185,722 | 61.08% |
| PiCCS verifier | 929,147 | 25.89% | 38,957,667 | 26.09% |
| PiRLC verifier | 458,388 | 12.77% | 16,377,698 | 10.97% |
| Running transition | 49,359 | 1.38% | 2,019,210 | 1.35% |
| PiDEC verifier | 25,488 | 0.71% | 741,690 | 0.50% |
| Application | 262 | 0.01% | 10,742 | 0.01% |
| Final binding / public prefix and alignment | 9 | 0.00% | 315 | 0.00% |
| **Total** | **3,588,191** | **100%** | **149,293,044** | **100%** |

The final line combines two distinct small costs: nine binding rows, and 270 public-prefix coordinates plus 45 alignment coordinates.
Pilot input and output words are shared with the application; they are counted once.

## Pilot → hashes and framing

Each state hash has 12,350 permutations and 1,062,100 S-box rows. The two hashes use 87,092,200 S-box coordinates.
The two 49,393-word preimages use 4,050,226 coordinates. Framing and digest fields use 43,296 more.
Framing and digest pins add 1,338 rows. These sums give the Pilot phase above.

## PiCCS → row costs

| Subgroup | Rows |
|---|---:|
| Output-binding S-boxes | 592,196 |
| Terminals (E_A, E_K, CCS, norm) | 139,718 |
| Initial claim | 116,631 |
| Statement-absorption S-boxes | 32,594 |
| Round-transcript S-boxes | 21,672 |
| Transcript chaining | 15,208 |
| Challenge-derivation S-boxes | 7,482 |
| Shared SumCheck chain | 2,324 |
| Final identity, including shared gamma powers | 1,130 |
| Statement binding | 160 |
| Endpoint pins | 32 |

## PiCCS → coordinate costs

| Subgroup | Coordinates |
|---|---:|
| Output-binding S-boxes | 24,280,036 |
| Arithmetic scratch: 207,011 fields | 8,487,451 |
| Proof/input/transcript fields: 87,766 | 3,598,406 |
| Statement-absorption S-boxes | 1,336,354 |
| Round-transcript S-boxes | 888,552 |
| Challenge-derivation S-boxes | 306,762 |
| Shared boundary fields | 60,106 |

## Remaining phases

- PiRLC: 340,578 sampling/range rows, 13,158 sampler S-box rows, and 104,652 ring-product rows. Coordinates are 11,547,488, 539,478, and 4,290,732 respectively.
- Running transition: 49,248 carried words use 2,019,168 coordinates. The shared flag and inverse use 42 more.
- PiDEC: 22,680 split/range rows and 2,808 recomposition rows. Split digits use 730,620 coordinates; public-input fields use 11,070.
- Application: 258 S-box rows and four digest pins. It retains 10,578 S-box coordinates and 164 message coordinates.

## Matrix nonzeros are a separate measure

All 14 matrices contain **2,857,409,270 normalized nonzero entries**, down from 2,968,490,185 at `c9aa75b04`.
The application accounts for **147,259**, down from 1,754,410. A complete phase split of matrix entries is not recorded here.
These counts combine repeated columns and remove zero coefficients. They do not count raw list entries.

The main Poseidon S-box allocation remains 114,443,382 coordinates; the compact application adds 10,578.
Thus the S-box coordinates alone still exceed the extra-50% target of 92,179,782.
The wide sampler is merged as proved components and is not selected in this package.

Sources: `Poseidon2HashChainV1Package`, `Poseidon2HashChainV1Setup`, `PiCCSOrdinaryRetainedBlocks`,
`PiCCSOrdinaryMatrixProgram`, `ApplicationPoseidonRetainedBlock`, and the selected phase ledgers.
Full totals and conformance scope: [piccs-lowering-metrics.json](piccs-lowering-metrics.json).
