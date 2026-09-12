const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {buildProofGraph, proofTrace} = require('../proof-graph.js');
const {nodes} = JSON.parse(readFileSync(new URL('../requirements.json', 'file://' + __filename), 'utf8'));
const key = edge => edge.source + ' -> ' + edge.target;

test('PiCCS retains all 39 individual requirements and its 47 recorded internal connections', () => {
  const graph = buildProofGraph(nodes, 'C');
  assert.equal(graph.members.size, 39);
  assert.equal(graph.edges.length, 47);
  assert.equal(graph.inputs.size, 19);
  assert.deepEqual(new Set([...graph.rows.flat(), ...graph.isolated]), graph.members);
  for (const edge of graph.edges) assert(graph.rank.get(edge.source) > graph.rank.get(edge.target));
});

test('outside references show the specific ambient bound, without a whole PiRLC-to-PiCCS loop', () => {
  const graph = buildProofGraph(nodes, 'C', true);
  assert(graph.visible.has('R.security.ambient'));
  assert(!graph.visible.has('R'));
  assert.deepEqual(graph.edges.filter(edge => edge.source === 'R.security.ambient').map(edge => edge.target),
    ['C.security.sumcheck', 'C.security.extraction']);
  assert.equal(graph.cycles.length, 0);
});

test('a selected proof highlights its true chains without treating shared inputs as consumers', () => {
  const trace = proofTrace(buildProofGraph(nodes, 'C'), 'C.prover.joint');
  for (const id of ['C.prover.ccs', 'C.prover.norm', 'C.prover.eval_k', 'C.prover.eval_a', 'C.prover.constant']) assert(trace.needed.has(id));
  for (const id of ['C.sumcheck.honest_sum', 'C.security.four_obligations', 'C.security.probability']) assert(trace.consumers.has(id));
  assert(!trace.consumers.has('C.terminal.identity'));
});

test('every group preserves exactly its recorded edges, including the absence of dependency data', () => {
  const snapshot = JSON.stringify(nodes);
  for (const group of nodes.filter(node => node.kind === 'group')) {
    const graph = buildProofGraph(nodes, group.id, true);
    const expected = nodes.flatMap(node => (node.depends_on || []).map(source => ({source, target: node.id})))
      .filter(edge => graph.members.has(edge.source) || graph.members.has(edge.target));
    assert.deepEqual(graph.edges.map(key), expected.map(key));
    assert.equal(new Set([...graph.rows.flat(), ...graph.isolated]).size, graph.visible.size);
  }
  const hash = buildProofGraph(nodes, 'T', true);
  assert.equal(hash.edges.length, 0);
  assert.equal(hash.isolated.length, 18);
  assert.equal(JSON.stringify(nodes), snapshot);
});

test('mutual references stay explicit and do not cause recursion or dropped nodes', () => {
  const sample = [
    {id: 'root', parent: null, kind: 'group'}, {id: 'G', parent: 'root', kind: 'group'},
    {id: 'G.a', parent: 'G', kind: 'leaf', depends_on: ['G.b']},
    {id: 'G.b', parent: 'G', kind: 'leaf', depends_on: ['G.a']},
    {id: 'G.c', parent: 'G', kind: 'leaf', depends_on: []}
  ];
  const graph = buildProofGraph(sample, 'G');
  assert.equal(graph.edges.length, 2);
  assert.equal(graph.cycles.length, 1);
  assert.equal(graph.rank.get('G.a'), graph.rank.get('G.b'));
  assert.deepEqual(graph.isolated, ['G.c']);
  assert.equal(proofTrace(graph, 'G.a').needed.size, 2);
});
