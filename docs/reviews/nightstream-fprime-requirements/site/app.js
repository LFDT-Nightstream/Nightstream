(() => {
  const data = JSON.parse(document.getElementById('requirements-data').textContent);
  const publication = JSON.parse(document.getElementById('publication-data').textContent);
  const references = JSON.parse(document.getElementById('reference-data').textContent);
  const assurance = RequirementAssurance.build(data, publication, references);
  const nodes = data.nodes;
  const byId = new Map(nodes.map(node => [node.id, node]));
  const children = new Map();
  for (const node of nodes) {
    if (node.id === 'root') continue;
    if (!children.has(node.parent)) children.set(node.parent, []);
    children.get(node.parent).push(node);
  }
  const names = {
    proved: 'local proof exists', partial: 'partial', definition: 'defined',
    assumption: 'explicit assumption', not_required: 'not required here',
    not_reviewed: 'not verified by this review', connected: 'connected locally',
    open: 'open', tested_scoped: 'scoped tests passed', recorded_only: 'recorded evidence only',
    implemented: 'code exists'
  };
  const origins = {
    paper: 'Paper requirement', profile: 'Nightstream profile',
    implementation: 'Implementation requirement', assumption: 'External assumption',
    out_of_scope: 'Outside selected Stage 1'
  };
  const navNames = {
    F: 'Foundations', T: 'Hash & transcript', C: 'PiCCS', R: 'PiRLC', D: 'PiDEC',
    N: 'NIFS composition', H: 'HyperNova recursion', L: 'Circuit & export',
    P: 'Production acceptance', O: 'Outside Stage 1'
  };
  const el = (tag, text, className) => {
    const result = document.createElement(tag);
    if (text !== undefined) result.textContent = text;
    if (className) result.className = className;
    return result;
  };
  const descendants = id => (children.get(id) || []).flatMap(node => node.kind === 'leaf' ? [node] : descendants(node.id));
  const unresolved = node => node.connection === 'open' || node.connection === 'partial' || node.proof === 'partial';
  const tone = status => ['open', 'partial'].includes(status) ? 'open' : status === 'assumption' ? 'assumption' : ['proved', 'connected', 'tested_scoped'].includes(status) ? 'good' : 'neutral';
  const groups = children.get('root');
  const included = nodes.filter(node => node.kind === 'leaf' && node.origin !== 'out_of_scope').length;
  const excluded = nodes.filter(node => node.kind === 'leaf' && node.origin === 'out_of_scope').length;
  const axes = [{key: 'proof', label: 'Proof'}, {key: 'connection', label: 'Link'}, {key: 'rust', label: 'Rust'}];
  const progress = new Map(nodes.filter(node => node.kind === 'group').map(node => {
    return [node.id, axes.map(axis => ({...axis, counts: RequirementAssurance.counts(descendants(node.id), axis.key)}))];
  }));
  const breakdown = axis => RequirementAssurance.categories[axis.key].map(([key, label]) => label + ' ' + axis.counts[key]).join(' · ');
  function countBadges(id) {
    const counts = el('span', undefined, 'axis-counts');
    for (const axis of progress.get(id)) {
      const badge = el('span', undefined, 'axis-count axis-' + axis.key);
      const {finished, total} = RequirementAssurance.progress(axis.key, axis.counts);
      const fraction = total ? finished + '/' + total : 'n/a';
      badge.append(el('span', axis.label), el('strong', fraction));
      badge.title = axis.label + ': ' + fraction + ' · ' + breakdown(axis);
      badge.setAttribute('aria-label', badge.title);
      counts.append(badge);
    }
    return counts;
  }
  const stats = node => {
    const leaves = descendants(node.id);
    if (node.origin === 'out_of_scope') return leaves.length + ' scope exclusions';
    const open = leaves.filter(unresolved).length;
    const assumptions = leaves.filter(item => item.proof === 'assumption').length;
    return leaves.length + ' requirements' + (open ? ' · ' + open + ' unresolved proof/link ' + (open === 1 ? 'item' : 'items') : '') + (assumptions ? ' · ' + assumptions + ' explicit ' + (assumptions === 1 ? 'assumption' : 'assumptions') : '');
  };
  function pair(box, label, value, className = '') {
    if (!value) return;
    const dl = el('dl', undefined, 'req-pair ' + className);
    dl.append(el('dt', label), el('dd', value));
    box.append(dl);
  }
  function anchorActions(node) {
    const row = el('div', undefined, 'req-anchor-row');
    const anchor = el('a', '#req-' + node.id, 'req-anchor');
    anchor.href = '#req-' + node.id;
    anchor.setAttribute('aria-label', 'Permanent link to ' + node.label);
    const copy = el('button', 'Copy link', 'copy-link');
    copy.type = 'button';
    copy.setAttribute('aria-label', 'Copy link to ' + node.label);
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(anchor.href);
        copy.textContent = 'Copied';
        document.getElementById('view-announcement').textContent = 'Link copied for ' + node.label;
      } catch {
        let field = row.querySelector('input');
        if (!field) {
          field = el('input', undefined, 'copy-link-fallback');
          field.type = 'text';
          field.readOnly = true;
          field.setAttribute('aria-label', 'Select and copy this item link');
          row.append(field);
        }
        field.value = anchor.href;
        field.focus();
        field.select();
        document.getElementById('view-announcement').textContent = 'Select and copy the displayed link.';
      }
    });
    const techLink = el('a', 'View in tech tree');
    techLink.href = node.id === 'root' ? '#tech-tree' : '#tech-' + node.id;
    row.append(el('span', 'Linked item', 'target-indicator'), anchor, copy, techLink);
    return row;
  }
  function render(node) {
    const li = el('li');
    li.dataset.record = node.id;
    if (node.parent === 'root') li.dataset.root = node.id;
    const details = el('details');
    details.id = 'req-' + node.id;
    const summary = el('summary');
    summary.append(el('span', node.label, 'req-title'));
    if (node.kind === 'group' && node.origin !== 'out_of_scope') summary.append(countBadges(node.id));
    const state = el('span', undefined, 'req-state');
    if (node.kind === 'group') state.textContent = stats(node);
    else if (node.origin === 'out_of_scope') state.textContent = 'Outside selected Stage 1';
    else for (const [axis, label] of [['proof', 'Proof'], ['connection', 'Link'], ['rust', 'Rust']]) {
      state.append(el('span', label + ': ' + names[node[axis]], 'status status-' + tone(node[axis])));
    }
    summary.append(state);
    details.append(summary);
    const content = el('div', undefined, 'req-detail');
    content.append(anchorActions(node), el('span', origins[node.origin], 'req-id'));
    pair(content, 'Scope', node.scope.join(' · ').replaceAll('_', ' '));
    pair(content, 'Requirement', node.requirement);
    pair(content, 'Remaining', node.remaining, 'remaining' + (unresolved(node) ? ' remaining-open' : ''));
    const sources = [...(node.paper || []), ...(node.code || [])];
    if (sources.length) {
      const evidence = el('details', undefined, 'req-sources');
      evidence.append(el('summary', 'Paper & code references (' + sources.length + ')'));
      const refs = el('div', undefined, 'req-evidence');
      for (const source of sources) {
        refs.append(el('div', source.section || source.symbol || source.path, 'source-name'));
        const url = node.origin === 'out_of_scope' ? null : assurance.sourceUrl(source.path, source.line);
        const path = el(url ? 'a' : 'div', source.path + ':' + source.line, 'source-path');
        if (url) path.href = url;
        else path.append(el('span', node.origin === 'out_of_scope' ? ' (outside this reference check)' : ' (local paper corpus; hash in reference record)'));
        refs.append(path);
      }
      evidence.append(refs);
      content.append(evidence);
    }
    if (node.depends_on?.length) {
      const uses = el('div', undefined, 'uses');
      uses.append(el('strong', 'Uses'));
      for (const id of node.depends_on) {
        const link = el('a', byId.get(id).label);
        link.href = '#req-' + id;
        uses.append(link);
      }
      content.append(uses);
    }
    if (node.assumption_ids?.length) {
      const premises = el('div', undefined, 'uses');
      premises.append(el('strong', 'Assumptions'));
      for (const id of node.assumption_ids) premises.append(assurance.premiseLink(id));
      content.append(premises);
    }
    for (const [id, note] of Object.entries(node.dependency_notes || {})) pair(content, 'Dependency ' + id, note);
    details.append(content);
    if (children.has(node.id)) {
      const list = el('ul', undefined, 'req-children');
      children.get(node.id).forEach(child => list.append(render(child)));
      details.append(list);
    }
    li.append(details);
    return li;
  }
  const tree = document.getElementById('requirements-tree');
  const list = el('ul', undefined, 'req-roots');
  groups.forEach(node => list.append(render(node)));
  tree.append(list);
  const navigation = document.getElementById('group-nav');
  const select = document.getElementById('group-select');
  const techOption = el('option', 'Tech tree');
  techOption.value = 'tech-tree';
  select.append(techOption);
  const navigationItems = [{id: 'all', label: 'All groups'}, ...groups];
  for (const node of navigationItems) {
    const label = navNames[node.id] || node.label;
    const link = el('a', undefined, 'nav-link');
    link.href = '#group-' + node.id;
    link.dataset.group = node.id;
    link.append(el('span', label));
    link.append(node.origin === 'out_of_scope' ? el('span', descendants(node.id).length + ' excluded', 'nav-count') : countBadges(node.id === 'all' ? 'root' : node.id));
    navigation.append(link);
    const option = el('option', label);
    option.value = node.id;
    select.append(option);
  }
  document.getElementById('source-meta').textContent = 'SuperNeo v1.1 + HyperNova · Code ' + data.provenance.code_commit.slice(0, 8) + ' · Map ' + (publication.map_commit?.slice(0, 8) || 'uncommitted preview') + ' · Production gates remain open.';
  const scope = document.getElementById('claim-scope');
  const viewNames = {readiness: 'Readiness', assumptions: 'Assumption ledger', risk: 'Error budget', evidence: 'Source and evidence'};
  for (const [id, view] of Object.entries(assurance.views)) {
    view.id = 'view-' + id;
    document.getElementById('assurance-views').append(view);
  }
  const techTree = document.getElementById('tech-tree');
  const renderTechTree = createTechTree({nodes, byId, children, groups, navNames, el, names,
    countBadges, premiseLink: assurance.premiseLink, sourceUrl: assurance.sourceUrl});
  const requirementsViewLink = document.getElementById('requirements-view-link');
  const techViewLink = document.getElementById('tech-view-link');
  const mapViewLink = document.getElementById('proof-map-view-link');
  const structureView = document.getElementById('proof-structure');
  const structureMap = createProofStructureMap({
    diagram: JSON.parse(document.getElementById('proof-map-data').textContent),
    byId, el, sourceUrl: assurance.sourceUrl, premiseLink: assurance.premiseLink});
  function selectView(id) {
    const tech = id === 'tech';
    document.body.classList.toggle('tech-mode', tech);
    document.body.classList.toggle('map-mode', id === 'map');
    structureView.hidden = id !== 'map';
    if (id !== 'map') structureMap.destroy();
    techTree.hidden = !tech;
    for (const [view, link] of [['requirements', requirementsViewLink], ['tech', techViewLink], ['map', mapViewLink]]) {
      if (id === view) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    tree.hidden = id !== 'requirements';
    document.getElementById('scope-controls').hidden = id !== 'requirements';
    for (const [name, view] of Object.entries(assurance.views)) view.hidden = name !== id;
    for (const link of document.querySelectorAll('.review-nav a')) {
      if (link.hash === '#view-' + id) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
  }
  function applyScope() {
    const group = select.value;
    for (const item of list.querySelectorAll('li[data-record]')) {
      const node = byId.get(item.dataset.record);
      const matches = scope.value === 'all' || (node.kind === 'leaf' ? node.scope.includes(scope.value) : descendants(node.id).some(n => n.scope.includes(scope.value)));
      item.hidden = !matches || (item.dataset.root && group !== 'all' && node.id !== group);
    }
  }
  function showGroup(id) {
    selectView('requirements');
    const node = byId.get(id);
    const all = id === 'all' || !groups.includes(node);
    const selected = all ? 'all' : id;
    tree.classList.toggle('single-group', !all);
    for (const item of list.children) {
      item.hidden = !all && item.dataset.root !== id;
      if (!all && item.dataset.root === id) item.querySelector('details').open = true;
    }
    for (const link of navigation.children) {
      if (link.dataset.group === selected) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    select.value = selected;
    applyScope();
    requirementsViewLink.href = '#group-' + selected;
    const title = all ? 'Requirements map' : node.label;
    document.getElementById('view-title').textContent = title;
    document.getElementById('view-progress').replaceChildren(...(!all && node.origin === 'out_of_scope' ? [] : [countBadges(all ? 'root' : id)]));
    document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / ' + (all ? 'All groups' : navNames[id]);
    document.getElementById('view-summary').textContent = all ? included + ' indexed requirements · ' + excluded + ' scope exclusions. Expand a group to inspect its primitives and proof connections.' : stats(node);
    document.getElementById('view-announcement').textContent = title;
    document.title = title + ' — Nightstream';
  }
  function showAssurance(id) {
    selectView(id);
    for (const link of navigation.children) link.removeAttribute('aria-current');
    document.getElementById('view-title').textContent = viewNames[id];
    document.getElementById('view-progress').replaceChildren();
    document.getElementById('view-summary').textContent = '';
    document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / ' + viewNames[id];
    document.getElementById('view-announcement').textContent = viewNames[id];
    document.title = viewNames[id] + ' — Nightstream';
  }
  function navigate() {
    const fragment = decodeURIComponent(location.hash.slice(1));
    const proofParts = fragment.startsWith('proof-') ? fragment.slice(6).split(':') : [];
    const proofScope = byId.get(proofParts[0]);
    const proofItem = byId.get(proofParts[1] || proofParts[0]);
    const isProof = proofScope?.kind === 'group' && proofItem && proofParts.length <= 2;
    if (fragment === 'proof-map' || fragment.startsWith('proof-map:')) {
      selectView('map');
      structureMap.render(structureView, fragment.split(':')[1]);
      for (const link of navigation.children) link.removeAttribute('aria-current');
      document.getElementById('view-title').textContent = 'Nightstream proof map';
      document.getElementById('view-progress').replaceChildren();
      document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / Proof map';
      document.getElementById('view-summary').textContent = 'Read upward from verifier acceptance to the security goal. Select a result for its connections and Lean proof.';
      document.getElementById('view-announcement').textContent = 'Nightstream proof map';
      document.title = 'Proof structure — Nightstream';
      window.scrollTo({top: 0});
    } else if (isProof || fragment === 'tech-tree' || (fragment.startsWith('tech-') && byId.has(fragment.slice(5)))) {
      const id = isProof ? proofItem.id : fragment === 'tech-tree' ? 'root' : fragment.slice(5);
      const node = isProof ? proofScope : byId.get(id);
      selectView('tech');
      renderTechTree(techTree, id, isProof ? proofScope.id : null);
      select.value = 'tech-tree';
      for (const link of navigation.children) link.removeAttribute('aria-current');
      document.getElementById('view-title').textContent = 'Tech tree';
      document.getElementById('view-progress').replaceChildren();
      document.getElementById('breadcrumb').textContent = 'Selected Stage 1 / Dependencies';
      document.getElementById('view-summary').textContent = 'The end goal is at the top. The groups needed to reach it are below, down to the shared primitives. Select a group to see all its smaller parts.';
      document.getElementById('view-announcement').textContent = 'Tech tree: ' + node.label;
      const heading = document.getElementById(isProof ? 'tech-detail-title' : 'view-title');
      heading.tabIndex = -1;
      heading.focus({preventScroll: true});
      document.title = 'Tech tree: ' + node.label + ' — Nightstream';
      if (isProof) requestAnimationFrame(() => heading.scrollIntoView({block: 'start'}));
      else window.scrollTo({top: 0});
    } else if (fragment.startsWith('view-') && assurance.views[fragment.slice(5)]) {
      showAssurance(fragment.slice(5));
      window.scrollTo({top: 0});
    } else if (fragment.startsWith('assumption-') && document.getElementById(fragment)) {
      showAssurance('assumptions');
      const target = document.getElementById(fragment);
      target.open = true;
      target.querySelector('summary').focus({preventScroll: true});
      target.scrollIntoView({block: 'start'});
    } else if (fragment.startsWith('req-') && byId.has(fragment.slice(4))) {
      scope.value = 'all';
      const id = fragment.slice(4);
      let owner = byId.get(id);
      while (owner.parent && owner.parent !== 'root') owner = byId.get(owner.parent);
      showGroup(owner.id);
      const target = document.getElementById('req-' + id);
      if (!target) return;
      let current = target;
      while (current && current !== tree) {
        if (current.tagName === 'DETAILS') current.open = true;
        current = current.parentElement;
      }
      target.querySelector('.req-anchor').focus({preventScroll: true});
      target.scrollIntoView({block: 'start'});
    } else {
      showGroup(fragment.startsWith('group-') ? fragment.slice(6) : 'all');
      if (fragment.startsWith('group-')) window.scrollTo({top: 0});
    }
  }
  select.addEventListener('change', () => { location.hash = select.value === 'tech-tree' ? 'tech-tree' : 'group-' + select.value; });
  scope.addEventListener('change', applyScope);
  window.addEventListener('hashchange', navigate);
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#req-"]');
    if (anchor && anchor.hash === location.hash) { event.preventDefault(); navigate(); }
  });
  navigate();
})();
