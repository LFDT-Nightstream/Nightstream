const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');
const assurance = require('../assurance.js');
const root = path.join(__dirname, '..');

// Exercise navigation and record rendering without the graph layout or browser.
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.dataset = {};
    this.attributes = {}; this.listeners = {}; this.style = {}; this.className = '';
    this.classList = {add: name => this.classList.toggle(name, true), toggle: (name, enabled) => {
      const classes = new Set(this.className.split(' ').filter(Boolean));
      enabled ? classes.add(name) : classes.delete(name); this.className = [...classes].join(' ');
    }};
  }
  append(...nodes) {
    for (const node of nodes) {
      if (node.parentElement) node.parentElement.children = node.parentElement.children.filter(child => child !== node);
      node.parentElement = this; this.children.push(node);
    }
  }
  replaceChildren(...nodes) { this.children.forEach(node => node.parentElement = null); this.children = []; this.append(...nodes); }
  setAttribute(key, value) { this.attributes[key] = value; }
  removeAttribute(key) { delete this.attributes[key]; }
  addEventListener(key, action) { this.listeners[key] = action; }
  focus() { this.focused = true; }
  scrollIntoView() { this.scrolled = true; }
  getBoundingClientRect() { return {width: 0, height: 0}; }
  all() { return [this, ...this.children.flatMap(node => node.all())]; }
  querySelectorAll(selector) {
    return this.all().slice(1).filter(node => selector === 'li[data-record]' ? node.tagName === 'LI' && node.dataset.record :
      selector.startsWith('.') ? node.className.split(' ').includes(selector.slice(1)) : node.tagName.toLowerCase() === selector);
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

test('one tree contains full evidence and opens saved records after scope filtering', () => {
  const body = new Element('body'), events = {};
  const html = fs.readFileSync(path.join(root, 'page.html'), 'utf8');
  for (const [, tag, id] of html.matchAll(/<([a-z0-9]+)[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const node = new Element(tag); node.id = id; body.append(node);
  }
  const document = {body, getElementById: id => body.all().find(node => node.id === id),
    createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag),
    querySelectorAll: () => [], addEventListener: () => {}};
  const data = JSON.parse(fs.readFileSync(path.join(root, 'requirements.json')));
  for (const [id, value] of [['requirements-data', data], ['publication-data', {map_commit: 'test'}],
    ['reference-data', {}], ['protocol-flow-data', require('../dist/protocol-flow.json')], ['proof-map-data', {}]]) {
    document.getElementById(id).textContent = JSON.stringify(value);
  }
  document.getElementById('claim-scope').value = 'all';
  const context = {document, location: {hash: ''}, navigator: {},
    window: {addEventListener: (name, fn) => events[name] = fn, scrollTo: () => {}},
    requestAnimationFrame: fn => { fn(); return 1; }, cancelAnimationFrame: () => {},
    ResizeObserver: class {observe() {} disconnect() {}},
    RequirementAssurance: {...assurance, build: () => ({views: {},
      premiseLink: id => { const node = new Element('a'); node.href = '#assumption-' + id; return node; },
      sourceUrl: (file, line) => 'https://source.test/' + file + '#L' + line})},
    createProofGraph: () => ({destroy() {}, render() {}}),
    createReplayProgress: () => new Element('section'),
    createProtocolFlow: () => ({close() {}, render() {}}),
    createProofStructureMap: () => ({destroy() {}, render() {}})};
  vm.createContext(context);
  for (const file of ['tech-tree.js', 'app.js']) vm.runInContext(fs.readFileSync(path.join(root, file), 'utf8'), context);
  const navigate = fragment => { context.location.hash = '#' + fragment; events.hashchange(); };
  const tree = document.getElementById('tech-tree');
  assert.equal(document.getElementById('requirements-view-link'), undefined);
  assert.equal(document.getElementById('requirements-tree'), undefined);
  assert.equal(document.getElementById('protocol-view-link').attributes['aria-current'], 'page');
  assert.equal(document.getElementById('protocol-flow').hidden, false);
  assert.equal(tree.hidden, true);
  const nav = html.match(/<nav class="view-nav"[\s\S]*?<\/nav>/)[0];
  assert.deepEqual([...nav.matchAll(/href="#([^"]+)"/g)].map(match => match[1]),
    ['protocol-flow', 'tech-tree', 'proof-map']);
  navigate('tech-tree');
  assert.equal(document.getElementById('tech-view-link').attributes['aria-current'], 'page');
  assert.equal(tree.hidden, false);
  assert.equal(tree.querySelectorAll('li[data-record]').length, data.nodes.length);
  assert.equal(document.getElementById('prover-replay').hidden, false);

  const targetNode = data.nodes.find(node => node.id === 'C.security.probability');
  navigate('group-C');
  const scope = document.getElementById('claim-scope');
  scope.value = 'outside_stage_1'; scope.listeners.change();
  assert.equal(document.getElementById('req-' + targetNode.id).parentElement.hidden, true);
  navigate('req-' + targetNode.id);
  const target = document.getElementById('req-' + targetNode.id);
  assert.equal(scope.value, 'all');
  assert.equal(target.querySelector('.req-anchor').focused, true);
  assert.equal(target.scrolled, true);
  for (let parent = target; parent !== tree; parent = parent.parentElement) {
    assert.notEqual(parent.hidden, true);
    if (parent.tagName === 'DETAILS') assert.equal(parent.open, true);
  }
  assert(target.all().some(node => node.textContent === targetNode.requirement));
  for (const ref of targetNode.code) assert(target.all().some(node => node.href === 'https://source.test/' + ref.path + '#L' + ref.line));
  for (const id of targetNode.assumption_ids) assert(target.all().some(node => node.href === '#assumption-' + id));
  for (const id of targetNode.depends_on) assert(target.all().some(node => node.href === '#req-' + id));
  navigate('group-O');
  assert(document.getElementById('req-O'));
  navigate('proof-map');
  assert.equal(tree.hidden, true);
  assert.equal(document.getElementById('proof-map-view-link').attributes['aria-current'], 'page');
  navigate('group-all');
  assert.equal(tree.querySelectorAll('li[data-record]').length, data.nodes.length);
});
