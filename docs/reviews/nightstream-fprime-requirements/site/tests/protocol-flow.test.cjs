const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assurance = require('../assurance.js');
const {protocolFlowPath} = require('../protocol-flow.js');
const root = path.join(__dirname, '..');

// Match the DOM contracts used by the renderer, including append's void return.
class Element {
  constructor(tag, text, className) { this.tag = tag; this.textContent = text; this.className = className; this.children = []; this.dataset = {}; this.attributes = {}; this.listeners = {}; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  showModal() { this.open = true; }
  close() { this.open = false; this.listeners.close?.(); }
  all() { return [this, ...this.children.flatMap(child => child instanceof Element ? child.all() : [])]; }
  querySelectorAll() { return this.all().filter(node => node.dataset.flowItem); }
  getBoundingClientRect() { return {width: 0, height: 0}; }
}

test('Every operation and handoff opens its own evidence, including statement and dependency links', () => {
  const data = JSON.parse(fs.readFileSync(path.join(root, 'requirements.json')));
  const flow = JSON.parse(fs.readFileSync(path.join(root, 'dist/protocol-flow.json')));
  const context = {RequirementAssurance: assurance, location: {hash: '#protocol-flow'},
    document: {createElementNS: (_, tag) => new Element(tag)},
    ResizeObserver: class {observe() {}}, requestAnimationFrame: fn => fn()};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root, 'protocol-flow.js'), 'utf8'), context);
  const create = context.createProtocolFlow;
  const byId = new Map(data.nodes.map(node => [node.id, node]));
  const names = Object.fromEntries(data.nodes.flatMap(node => ['proof', 'connection', 'rust'].map(axis => [node[axis], node[axis]])));
  const view = create({flow, byId, names, replay: data.prover_replay, el: (tag, text, cls) => new Element(tag, text, cls),
    sourceUrl: (file, line) => 'https://example.test/' + file + '#L' + line,
    premiseLink: id => new Element('a', id)});
  const container = new Element('section');
  for (const item of flow.items) {
    context.location.hash = '#protocol-flow:' + item.id;
    view.render(container, item.id);
    const dialog = container.all().find(node => node.tag === 'dialog');
    assert.equal(dialog.open, true);
    assert.equal(dialog.all().find(node => node.attributes['id'] === 'flow-detail-title')?.textContent ||
      dialog.all().find(node => node.id === 'flow-detail-title').textContent, item.label);
    assert.equal(dialog.all().filter(node => node.className === 'flow-record').length, item.records.length);
    for (const id of item.records) {
      assert(dialog.all().some(node => node.href === '#req-' + id), id);
      for (const dependency of byId.get(id).depends_on || []) {
        assert(dialog.all().some(node => node.href === '#req-' + dependency), dependency);
      }
    }
  }
  assert.equal(container.querySelectorAll().length, flow.items.length);
  assert.equal(container.all().filter(node => node.className?.startsWith('flow-stage flow-stage-')).length, 6);
  const circuit = container.all().find(node => node.className === 'flow-circuit');
  assert.deepEqual(circuit.all().filter(node => node.dataset.flowItem).map(node => node.dataset.flowItem), flow.circuit.items);
  for (const key of ['next-witness', 'next-commitment', 'terminal']) {
    assert(!circuit.all().some(node => node.dataset.flowItem === key));
  }
  view.close();
  assert.equal(context.location.hash, 'protocol-flow');
});

test('feedback and fresh encoding travel outside the circuit; skipped cards are not crossed', () => {
  const boxes = new Map([
    ['repeat', {x: 700, y: 350, width: 180, height: 100}],
    ['fresh', {x: 40, y: 200, width: 180, height: 100}],
    ['next-state', {x: 700, y: 800, width: 180, height: 100}],
    ['next-witness', {x: 700, y: 100, width: 180, height: 100}],
    ['rho', {x: 400, y: 100, width: 180, height: 100}],
    ['public-fold', {x: 400, y: 250, width: 180, height: 100}],
    ['private-fold', {x: 400, y: 400, width: 180, height: 100}],
  ]);
  const bounds = {left: 12, right: 940, top: 20, bottom: 950};
  const route = (from, to, kind) => protocolFlowPath({from, to, kind}, boxes, bounds);
  assert.equal(route('repeat', 'fresh', 'feedback'), 'M 880 400 L 940 400 L 940 20 L 12 20 L 12 250 L 40 250');
  assert.equal(route('next-state', 'next-witness', 'encode'), 'M 880 850 L 928 850 L 928 150 L 880 150');
  assert.equal(route('rho', 'private-fold', 'data'), 'M 400 150 L 392 150 L 392 450 L 400 450');
});

test('Replay progress distinguishes complete execution from open proof links', () => {
  const data = JSON.parse(fs.readFileSync(path.join(root, 'requirements.json')));
  const flow = JSON.parse(fs.readFileSync(path.join(root, 'dist/protocol-flow.json')));
  const byId = new Map(data.nodes.map(node => [node.id, node]));
  const context = {RequirementAssurance: assurance, location: {hash: '#protocol-flow'}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(root, 'protocol-flow.js'), 'utf8'), context);
  assert.equal(context.replayProofComplete(byId.get('R.replay.witness')), true);
  assert.equal(context.replayProofComplete(byId.get('D.replay.witnesses')), true);
  assert.equal(context.replayProofComplete(byId.get('D.replay.commitments')), true);
  assert.equal(context.replayProofComplete(byId.get('D.replay.evaluations')), true);
  assert.equal(context.replayProofComplete(byId.get('C.replay.rounds')), false);
  assert.equal(context.replayProofComplete(byId.get('N.replay.composed')), false);
  assert.equal(data.prover_replay.next, 'N.replay.composed');
  const panel = context.createReplayProgress({data, byId, el: (tag, text, cls) => new Element(tag, text, cls)});
  assert.equal(panel.all().filter(node => node.className?.startsWith('replay-phase')).length, data.prover_replay.phases.length);
  assert(panel.all().some(node => node.href === '#req-N.replay.composed'));
  assert(panel.all().some(node => node.textContent === '11/11 execution checks passed'));
  const piCcs = panel.all().find(node => node.href === '#req-C.replay');
  assert(piCcs.all().some(node => node.textContent === '3/3 execution checks passed'));
  assert(piCcs.all().some(node => node.textContent === 'Polynomial proof: complete'));
  assert(!piCcs.all().some(node => node.textContent === 'Formal proof connection: complete'));
  assert(panel.all().some(node => node.textContent?.includes('Python/shell assembly and runtime orchestration are outside a single complete Lean proof.')));
  assert(panel.all().some(node => node.href === '#req-C.replay.polynomial_spec'));
  assert(!piCcs.all().some(node => node.textContent?.includes('0/3')));
  assert.equal(context.replayProofComplete(byId.get('C.replay.polynomial_spec')), true);
  assert.deepEqual(data.prover_replay.phases.map(phase => phase.id), ['C', 'R', 'D', 'H', 'N']);
  const hypernova = panel.all().find(node => node.href === '#req-H.replay');
  assert(hypernova.all().some(node => node.textContent === 'Conditional successor proof: complete'));
  const composed = panel.all().find(node => node.href === '#req-N.replay');
  assert(composed.all().some(node => node.textContent === 'Whole-program runtime proof: open'));
  assert(composed.all().some(node => node.textContent?.includes('outside this completion record')));
  assert.equal(flow.owners['D.replay.witnesses'], 'split');
  assert.equal(flow.owners['D.replay.commitments'], 'child-commitments');
  assert.equal(flow.owners['D.replay.evaluations'], 'child-evaluations');
  assert.equal(flow.owners['H.replay.commitment'], 'next-commitment');
  // A scoped Rust comparison alone must not count as complete replay execution.
  byId.get('C.replay.evaluations').replay_execution = 'open';
  const incomplete = context.createReplayProgress({data, byId, el: (tag, text, cls) => new Element(tag, text, cls)});
  const incompletePiCcs = incomplete.all().find(node => node.href === '#req-C.replay');
  assert(incompletePiCcs.all().some(node => node.textContent === '2/3 execution checks passed'));
});
