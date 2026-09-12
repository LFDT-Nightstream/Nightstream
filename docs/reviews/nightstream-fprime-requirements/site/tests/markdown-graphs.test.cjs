const {test} = require('node:test');
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {buildProofGraph} = require('../proof-graph.js');
const {nodes} = JSON.parse(readFileSync(new URL('../requirements.json', 'file://' + __filename), 'utf8'));

test('Markdown diagrams and connection tables match both website graph modes for every group', () => {
  for (const group of nodes.filter(node => node.parent === 'root')) {
    const markdown = readFileSync(new URL('../dist/markdown/graphs/' + group.id + '.md', 'file://' + __filename), 'utf8');
    const legend = [...markdown.matchAll(/^\| `(n\d+)` \| \[[^\n]+?\]\([^\n]*?#req-([\w.]+)\) \| `([^`]+)` \| `([^`]+)` \| `([^`]+)` \|/gm)];
    const ids = new Map(legend.map(match => [match[1], match[2]]));
    const diagrams = [...markdown.matchAll(/```mermaid\n([\s\S]*?)\n```/g)].map(match => match[1]);
    const full = buildProofGraph(nodes, group.id, true);
    assert.equal(diagrams.length, full.inputs.size || full.outputs.size ? 2 : 1, group.id);
    for (const [index, diagram] of diagrams.entries()) {
      const expected = buildProofGraph(nodes, group.id, index === 1);
      const boxes = [...diagram.matchAll(/^  (n\d+)\[/gm)].map(match => ids.get(match[1]));
      assert.deepEqual(new Set(boxes), expected.visible, group.id);
      assert.equal(boxes.length, expected.visible.size);
      const edges = [...diagram.matchAll(/^  (n\d+) --> (n\d+)$/gm)]
        .map(match => ({source: ids.get(match[1]), target: ids.get(match[2])}));
      assert.deepEqual(edges, expected.edges, group.id);
      assert(markdown.includes('Internal connections: ' + buildProofGraph(nodes, group.id).edges.length + '.'));
    }
    const table = [...markdown.matchAll(/^\| \[[^\n]+?\]\([^\n]*?#req-([\w.]+)\) \| \[[^\n]+?\]\([^\n]*?#req-([\w.]+)\) \| (?:Internal|Outside input|Outside consumer) \|[^\n]*\|$/gm)]
      .map(match => ({source: match[1], target: match[2]}));
    assert.deepEqual(table, full.edges, group.id);
    for (const edge of full.edges) {
      const note = nodes.find(node => node.id === edge.target).dependency_notes?.[edge.source];
      if (note) assert(markdown.includes(note), 'Missing dependency scope note: ' + edge.target);
    }
    assert.deepEqual(new Set(ids.values()), full.visible, group.id);
    for (const match of legend) {
      const node = nodes.find(item => item.id === match[2]);
      assert.deepEqual(match.slice(3, 6), [node.proof, node.connection, node.rust]);
    }
  }
});
