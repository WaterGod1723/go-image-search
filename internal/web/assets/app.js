'use strict';

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

const TITLES = {
  search: ['搜索', '按图像内容检索图片库'],
  library: ['图库', '浏览已索引的图片'],
  index: ['索引', '构建与加载图片索引'],
  debug: ['调试', '查看图像区域划分'],
};

const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function base(path) {
  const parts = String(path).replace(/\\/g, '/').split('/');
  return parts[parts.length - 1];
}

function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove('show'), 2600);
}

/* ---------------- 主题切换 ---------------- */
const THEME_META = { kawaii: '#ff8fb0', mature: '#0e1220' };
function applyTheme(name) {
  document.body.dataset.theme = name;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = THEME_META[name] || THEME_META.kawaii;
  try { localStorage.setItem('gis-theme', name); } catch (e) { /* ignore */ }
}
$('#theme-toggle').addEventListener('click', () => {
  applyTheme(document.body.dataset.theme === 'mature' ? 'kawaii' : 'mature');
});
(function initTheme() {
  let saved = 'kawaii';
  try { saved = localStorage.getItem('gis-theme') || 'kawaii'; } catch (e) { /* ignore */ }
  applyTheme(saved);
})();

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); if (j.error) msg = j.error; } catch (e) { /* ignore */ }
    throw new Error(msg);
  }
  return res.json();
}

/* ---------------- 标签切换 ---------------- */
function switchTab(name) {
  $$('.page').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
  $$('.tabbar button').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  const t = TITLES[name];
  $('#nav-title').textContent = t[0];
  $('#nav-sub').textContent = t[1];
  window.scrollTo(0, 0);
  if (name === 'library') loadLibrary();
  if (name === 'index') refreshStatus();
}
$$('.tabbar button').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));

/* ---------------- 分段控件 ---------------- */
$$('.seg').forEach(seg => {
  seg.addEventListener('click', e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    $$('button', seg).forEach(b => b.classList.toggle('active', b === btn));
  });
});

/* ---------------- 查询图片选择 ---------------- */
let queryFile = null;
const dropzone = $('#dropzone');
const pastezone = $('#pastezone');
const queryInput = $('#query-input');

function resetPickers() {
  ['#dz-placeholder', '#pz-placeholder'].forEach(s => $(s).style.display = '');
  ['#query-preview', '#paste-preview'].forEach(s => {
    const el = $(s);
    el.style.display = 'none';
    el.removeAttribute('src');
  });
}

function setQueryFile(f, source) {
  queryFile = f;
  resetPickers();
  if (source === 'paste') {
    $('#pz-placeholder').style.display = 'none';
    const img = $('#paste-preview');
    img.src = URL.createObjectURL(f);
    img.style.display = 'block';
  } else {
    $('#dz-placeholder').style.display = 'none';
    const img = $('#query-preview');
    img.src = URL.createObjectURL(f);
    img.style.display = 'block';
  }
  $('#query-name').textContent = f.name;
  $('#query-name').style.display = '';
  $('#btn-search').disabled = false;
  $('#results-empty').hidden = true;
  $('#results').innerHTML = '';
  doSearch();
}

dropzone.addEventListener('click', () => queryInput.click());
pastezone.addEventListener('click', () => queryInput.click());
queryInput.addEventListener('change', () => { if (queryInput.files[0]) setQueryFile(queryInput.files[0], 'drop'); });

dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) setQueryFile(e.dataTransfer.files[0], 'drop');
});

document.addEventListener('paste', e => {
  const item = Array.from((e.clipboardData && e.clipboardData.items) || []).find(i => i.type.startsWith('image/'));
  if (!item) return;
  e.preventDefault();
  const f = item.getAsFile();
  if (f) setQueryFile(f, 'paste');
});

/* ---------------- 搜索 ---------------- */
const colorRange = $('#color-range');
const maxdistRange = $('#maxdist-range');
$('#color-val').textContent = Number(colorRange.value).toFixed(2);
$('#maxdist-val').textContent = maxdistRange.value;
colorRange.addEventListener('input', () => $('#color-val').textContent = Number(colorRange.value).toFixed(2));
maxdistRange.addEventListener('input', () => $('#maxdist-val').textContent = maxdistRange.value);

function segVal(segSel) {
  const active = $(segSel + ' button.active');
  return active ? active.dataset.val : '5';
}

async function doSearch() {
  if (!queryFile) return;
  const fd = new FormData();
  fd.append('image', queryFile);
  fd.append('top', segVal('#seg-top'));
  fd.append('maxdist', maxdistRange.value);
  fd.append('colorWeight', colorRange.value);

  $('#search-busy').hidden = false;
  $('#btn-search').disabled = true;
  $('#results').innerHTML = '';
  $('#results-empty').hidden = true;
  try {
    const data = await fetchJSON('/api/query', { method: 'POST', body: fd });
    renderResults(data);
  } catch (e) {
    toast(e.message);
    $('#results-empty').hidden = false;
    $('#results-empty p').textContent = '检索失败';
    $('#results-empty span').textContent = e.message;
  } finally {
    $('#search-busy').hidden = true;
    $('#btn-search').disabled = false;
  }
}

$('#btn-search').addEventListener('click', doSearch);

function renderResults(data) {
  const box = $('#results');
  box.innerHTML = '';
  $('#results-empty p').textContent = '暂无结果';
  $('#results-empty span').textContent = `查询区域 ${data.regions} 个，无匹配图片`;
  if (!data.matches || data.matches.length === 0) {
    $('#results-empty').hidden = false;
    return;
  }
  $('#results-empty').hidden = true;
  data.matches.forEach((m, i) => box.appendChild(resultCard(m, i)));
}

function resultCard(m, i) {
  const score = Math.round(m.score * 100);
  const cover = Math.round(m.cover * 100);
  const el = document.createElement('div');
  el.className = 'card result-card';
  const src = '/api/image?path=' + encodeURIComponent(m.imageId);
  const rel = relOf(m.imageId);
  el.innerHTML =
    '<img class="thumb" src="' + src + '" alt="">' +
    '<div class="result-body">' +
      '<div class="result-top"><span class="result-name">' + esc(base(m.imageId)) + '</span><span class="rank">#' + (i + 1) + '</span></div>' +
      '<div class="scorebar"><div style="width:' + score + '%"></div></div>' +
      '<div class="result-meta">相似度 ' + score + '% · 覆盖 ' + cover + '% · 命中 ' + m.count + ' 区域</div>' +
      '<div class="result-actions">' +
        '<button class="copy-btn"><span class="copy-label">复制路径</span></button>' +
        '<button class="copy-btn"><span class="copy-label">复制相对路径</span></button>' +
      '</div>' +
      '<div class="result-regions">' +
        (m.regions || []).map(r =>
          '<div class="rr"><span><span class="dot" style="background:' + esc(r.color) + '"></span>区域 #' + r.regionId + '</span><span>dist ' + r.dist + '</span></div>'
        ).join('') +
      '</div>' +
    '</div>';
  el.querySelector('.thumb').addEventListener('click', () => viewImage(src, m.imageId));
  const btns = el.querySelectorAll('.copy-btn');
  btns[0].addEventListener('click', () => copyText(m.imageId));
  btns[1].addEventListener('click', () => copyText(rel));
  return el;
}

let libRoot = '';
function relOf(p) {
  const norm = p.replace(/\\/g, '/');
  if (!libRoot) return norm;
  const root = libRoot.replace(/\\/g, '/');
  if (norm.startsWith(root)) return norm.slice(root.length).replace(/^\/+/, '');
  return norm;
}
function copyText(t) {
  const done = msg => toast(msg);
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(t).then(() => done('已复制：' + t)).catch(() => fallbackCopy(t, done));
  } else {
    fallbackCopy(t, done);
  }
}
function fallbackCopy(t, done) {
  const ta = document.createElement('textarea');
  ta.value = t;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); done('已复制：' + t); } catch (e) { done('复制失败'); }
  document.body.removeChild(ta);
}

/* ---------------- 图片查看 ---------------- */
function viewImage(src, name) {
  $('#modal-img').src = src;
  $('#modal-name').textContent = name;
  $('#modal').classList.add('open');
}
function closeModal() { $('#modal').classList.remove('open'); }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
$('#modal').addEventListener('click', e => { if (e.target.id === 'modal') closeModal(); });

/* ---------------- 图库 ---------------- */
async function loadLibrary() {
  const grid = $('#lib-grid');
  try {
    const images = await fetchJSON('/api/images');
    grid.innerHTML = '';
    if (!images || images.length === 0) {
      $('#lib-empty').hidden = false;
      return;
    }
    $('#lib-empty').hidden = true;
    images.forEach(p => {
      const img = document.createElement('img');
      const src = '/api/image?path=' + encodeURIComponent(p);
      img.src = src;
      img.loading = 'lazy';
      img.addEventListener('click', () => viewImage(src, p));
      grid.appendChild(img);
    });
  } catch (e) {
    $('#lib-empty').hidden = false;
    $('#lib-empty p').textContent = '加载失败';
    $('#lib-empty span').textContent = e.message;
  }
}

/* ---------------- 构建索引 ---------------- */
let buildBusy = false;
function setBuildBusy(v) {
  buildBusy = v;
  $('#btn-build').disabled = v;
  $('#btn-load').disabled = v;
}
function updateBuildUI(job) {
  const card = $('#build-card');
  const pct = job.total > 0 ? Math.round(job.done / job.total * 100) : 0;
  $('#build-bar').style.width = pct + '%';
  $('#build-pct').textContent = pct + '%';
  if (job.state === 'done') {
    $('#build-label').textContent = '构建完成 · ' + job.images + ' 图像 · ' + job.regions + ' 区域';
    $('#build-file').textContent = '索引已保存至 ' + job.index;
  } else if (job.state === 'error') {
    $('#build-label').textContent = '构建失败';
    $('#build-pct').textContent = '';
    $('#build-file').textContent = job.error;
    card.classList.add('error');
  } else {
    $('#build-label').textContent = '正在构建 ' + job.done + '/' + job.total;
    $('#build-file').textContent = job.current || '';
  }
}

$('#btn-build').addEventListener('click', async () => {
  const body = {
    dir: $('#cfg-dir').value.trim(),
    out: $('#cfg-out').value.trim(),
    thresholdPct: parseFloat($('#cfg-thr').value),
    thresholdFactor: parseFloat($('#cfg-factor').value),
    minAreaRatio: parseFloat($('#cfg-minarea').value),
    medianFilterK: parseInt($('#cfg-median').value, 10),
    connectivity: parseInt(segVal('#seg-conn'), 10),
  };
  if (!body.dir) { toast('请输入图像库目录'); return; }

  $('#build-card').hidden = false;
  $('#build-label').textContent = '准备中…';
  $('#build-bar').style.width = '0%';
  setBuildBusy(true);
  try {
    const job = await fetchJSON('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    pollBuild(job.id);
  } catch (e) {
    setBuildBusy(false);
    toast(e.message);
  }
});

function pollBuild(id) {
  clearInterval(window._buildTimer);
  window._buildTimer = setInterval(async () => {
    try {
      const st = await fetchJSON('/api/build/status');
      updateBuildUI(st);
      if (st.state === 'done' || st.state === 'error') {
        clearInterval(window._buildTimer);
        setBuildBusy(false);
        refreshStatus();
        if (st.state === 'done') loadLibrary();
      }
    } catch (e) {
      clearInterval(window._buildTimer);
      setBuildBusy(false);
      toast('轮询失败: ' + e.message);
    }
  }, 500);
}

$('#btn-load').addEventListener('click', async () => {
  const idx = $('#cfg-load').value.trim();
  setBuildBusy(true);
  try {
    const res = await fetchJSON('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: idx }),
    });
    toast('已加载：' + res.images + ' 图像 / ' + res.regions + ' 区域');
    $('#cfg-load').value = '';
    refreshStatus();
    loadLibrary();
  } catch (e) {
    toast(e.message);
  } finally {
    setBuildBusy(false);
  }
});

/* ---------------- 索引状态 ---------------- */
async function refreshStatus() {
  try {
    const st = await fetchJSON('/api/status');
    libRoot = st.root || libRoot;
    $('#st-load').textContent = st.loaded ? '已加载' : '未加载';
    $('#st-images').textContent = st.images;
    $('#st-regions').textContent = st.regions;
    $('#st-index').textContent = st.indexPath || '-';
    $('#st-root').textContent = st.root || '-';
    if (!st.build || st.build.state === 'idle') return;
    if (st.build.state === 'running') {
      $('#build-card').hidden = false;
      pollBuild(st.build.id);
      setBuildBusy(true);
    }
  } catch (e) { /* ignore */ }
}

/* ---------------- 调试 ---------------- */
let segQuery = null;
let segLabelMap = null;     // { data: Uint8ClampedArray, width, height }
let segRegionsById = new Map();

function refreshSegImage() {
  if (!segQuery) return;
  $('#seg-img').src = '/api/segments.png?' + segQuery.toString();
}

// 加载标签图（每个像素编码区域 ID），用于鼠标悬停反查区域。
async function loadSegLabelMap(query) {
  const url = '/api/segments.map.png?' + query.toString();
  const res = await fetch(url);
  if (!res.ok) return null;
  const blob = await res.blob();
  const urlObj = URL.createObjectURL(blob);
  try {
    const img = await new Promise((resolve, reject) => {
      const i = new Image();
      i.onload = () => resolve(i);
      i.onerror = reject;
      i.src = urlObj;
    });
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return { data: imgData.data, width: canvas.width, height: canvas.height };
  } finally {
    URL.revokeObjectURL(urlObj);
  }
}

function labelAt(map, x, y) {
  if (!map || x < 0 || y < 0 || x >= map.width || y >= map.height) return 0;
  const i = (y * map.width + x) * 4;
  return (map.data[i] << 16) | (map.data[i + 1] << 8) | map.data[i + 2];
}

function showSegTip(clientX, clientY, html) {
  const tip = $('#seg-tip');
  tip.innerHTML = html;
  tip.hidden = false;
  // 默认放在光标右下，靠近视口边缘则翻转
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  const vw = window.innerWidth, vh = window.innerHeight;
  let left = clientX + 14;
  let top = clientY + 14;
  if (left + tw > vw - 8) left = clientX - tw - 14;
  if (top + th > vh - 8) top = clientY - th - 14;
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}

function hideSegTip() { $('#seg-tip').hidden = true; }

$('#btn-seg').addEventListener('click', async () => {
  const path = $('#seg-path').value.trim();
  if (!path) { toast('请输入图像路径'); return; }
  segQuery = new URLSearchParams({
    path,
    thresholdPct: $('#seg-thr').value,
    median: $('#seg-median').value,
    connectivity: segVal('#seg-conn2'),
    mode: segVal('#seg-mode'),
  });
  $('#seg-busy').hidden = false;
  $('#btn-seg').disabled = true;
  try {
    const [data, labelMap] = await Promise.all([
      fetchJSON('/api/segments?' + segQuery.toString()),
      loadSegLabelMap(segQuery).catch(() => null),
    ]);
    $('#seg-info').textContent = '图像 ' + data.width + '×' + data.height + ' · ' + data.regions.length + ' 个区域';
    segLabelMap = labelMap;
    segRegionsById = new Map();
    data.regions.forEach(r => segRegionsById.set(r.id, r));
    refreshSegImage();
    $('#seg-img').hidden = false;
    $('#seg-mode').hidden = false;
    const box = $('#seg-regions');
    box.hidden = false;
    box.innerHTML = '';
    data.regions.forEach(r => {
      const div = document.createElement('div');
      div.className = 'rr';
      const tag = r.whole ? '<span class="tag tag-whole">整图辅助</span>' : '';
      div.innerHTML =
        '<span><span class="dot" style="background:' + esc(r.color) + '"></span>区域 #' + r.id + tag + '</span>' +
        '<span>面积 ' + r.area + ' · bbox ' + r.bbox.join(',') + '</span>';
      box.appendChild(div);
    });
  } catch (e) {
    toast(e.message);
  } finally {
    $('#seg-busy').hidden = true;
    $('#btn-seg').disabled = false;
  }
});

// 切换区域可视化模式（边框 / 填色），仅刷新预览图。
$('#seg-mode').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn || !segQuery) return;
  segQuery.set('mode', btn.dataset.val);
  refreshSegImage();
});

// 鼠标悬停预览图时，按像素反查区域并展示信息。
const segImg = $('#seg-img');
segImg.addEventListener('mousemove', e => {
  if (!segLabelMap) return;
  const rect = segImg.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  const nx = Math.floor((e.clientX - rect.left) / rect.width * segLabelMap.width);
  const ny = Math.floor((e.clientY - rect.top) / rect.height * segLabelMap.height);
  const id = labelAt(segLabelMap, nx, ny);
  if (id > 0 && segRegionsById.has(id)) {
    const r = segRegionsById.get(id);
    showSegTip(e.clientX, e.clientY,
      '<span class="dot" style="background:' + esc(r.color) + '"></span>' +
      '区域 #' + r.id + (r.whole ? ' <span class="tag tag-whole">整图辅助</span>' : '') +
      ' · 面积 ' + r.area + ' · bbox ' + r.bbox.join(','));
  } else {
    hideSegTip();
  }
});
segImg.addEventListener('mouseleave', hideSegTip);

/* ---------------- 启动 ---------------- */
switchTab('search');
refreshStatus();
