'use strict';

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

/* ---------------- Wails 绑定 ---------------- */
const App = window.go.main.App;

// Wails 绑定调用统一出错处理：Go 侧返回错误时 Promise reject 的是字符串。
const api = {
  status: () => App.Status(),
  getAlgorithm: () => App.GetAlgorithm(),
  setAlgorithm: a => App.SetAlgorithm(a),
  build: req => App.Build(req),
  buildStatus: () => App.BuildStatus(),
  load: idx => App.Load(idx),
  queryData: (u, top, md, cw) => App.QueryData(u, top, md, cw),
  queryPath: (p, top, md, cw) => App.QueryPath(p, top, md, cw),
  images: () => App.ImageList(),
  imageData: p => App.ImageDataURI(p),
  segments: (p, params) => App.Segments(p, params),
  segmentsPNG: (p, params) => App.SegmentsPNG(p, params),
  segmentsMapPNG: (p, params) => App.SegmentsMapPNG(p, params),
  pickImage: () => App.PickImage(),
  pickDir: () => App.PickDirectory(),
  pickIndex: () => App.PickIndex(),
};

// 原生文件选择对话框（经 Go 绑定暴露）
function pickImage() {
  return App.PickImage();
}
function pickDir() {
  return App.PickDirectory();
}
function pickIndex() {
  return App.PickIndex();
}

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

/* ---------------- 输入历史（localStorage 持久化，datalist 下拉） ---------------- */
const HISTORY_MAX = 8;
function loadHistory(key) {
  try { return JSON.parse(localStorage.getItem(key) || '[]'); } catch (e) { return []; }
}
function saveHistory(key, value) {
  const v = String(value || '').trim();
  if (!v) return;
  let list = loadHistory(key).filter(x => x !== v);
  list.unshift(v);
  if (list.length > HISTORY_MAX) list = list.slice(0, HISTORY_MAX);
  try { localStorage.setItem(key, JSON.stringify(list)); } catch (e) { /* ignore */ }
  fillDatalist(key);
}
function fillDatalist(key) {
  const dl = document.getElementById(key);
  if (!dl) return;
  const list = loadHistory(key);
  dl.innerHTML = list.map(v => '<option value="' + esc(v) + '">').join('');
}
// 启动时填充历史下拉
['dir-history', 'index-history'].forEach(fillDatalist);

/* ---------------- 检索策略切换 ---------------- */
let currentAlgo = 'sczl';
const algoLabels = { region: '区域', sczl: 'SCZL' };
async function refreshAlgorithm() {
  try {
    currentAlgo = await api.getAlgorithm();
  } catch (e) { /* ignore，使用默认 */ }
  updateAlgoUI();
}
function updateAlgoUI() {
  $('#algo-label').textContent = algoLabels[currentAlgo] || currentAlgo;
  $('#algo-toggle').classList.toggle('sczl', currentAlgo === 'sczl');
  // 同步算法到 body，CSS 据此控制仅适用于某算法的参数显隐
  document.body.dataset.algo = currentAlgo;
}
$('#algo-toggle').addEventListener('click', async () => {
  const next = currentAlgo === 'region' ? 'sczl' : 'region';
  try {
    await api.setAlgorithm(next);
    currentAlgo = next;
    updateAlgoUI();
    resetQueryState();
    toast(`已切换为 ${algoLabels[next]} 策略`);
    refreshStatus();
    loadLibrary();
  } catch (e) {
    toast(e);
  }
});

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
let queryPath = null;
const dropzone = $('#dropzone');
const pastezone = $('#pastezone');
const queryInput = $('#query-input');

function resetPickers() {
  $('#dz-placeholder').style.display = '';
  ['#query-preview', '#paste-preview'].forEach(s => {
    const el = $(s);
    el.style.display = 'none';
    el.removeAttribute('src');
  });
}

function resetQueryState() {
  queryFile = null;
  queryPath = null;
  resetPickers();
  $('#query-name').textContent = '';
  $('#query-name').style.display = 'none';
  $('#btn-search').disabled = true;
  $('#results').innerHTML = '';
  $('#results-empty').hidden = true;
}

function setQueryPreview(source, src) {
  resetPickers();
  const sel = source === 'paste' ? '#paste-preview' : '#query-preview';
  $(sel).style.display = 'none'; // 先隐藏
  const img = $(sel);
  img.src = src;
  img.style.display = 'block';
  $('#query-name').style.display = '';
  $('#btn-search').disabled = false;
  $('#results-empty').hidden = true;
  $('#results').innerHTML = '';
}

function setQueryFile(f, source) {
  queryPath = null;
  queryFile = f;
  setQueryPreview(source, URL.createObjectURL(f));
  $('#query-name').textContent = f.name;
  doSearch();
}

async function setQueryPath(p, source) {
  queryFile = null;
  queryPath = p;
  let uri = '';
  try { uri = await api.imageData(p); } catch (e) { uri = ''; }
  setQueryPreview(source, uri);
  $('#query-name').textContent = base(p) || p;
  doSearch();
}

function setQueryPreview(source, uri) {
  $('#dz-placeholder').style.display = 'none';
  const img = source === 'paste' ? $('#paste-preview') : $('#query-preview');
  img.src = uri;
  img.style.display = 'block';
  $('#query-name').style.display = '';
  $('#btn-search').disabled = false;
  $('#results-empty').hidden = true;
  $('#results').innerHTML = '';
}

// 点击选择区弹出原生文件对话框
dropzone.addEventListener('click', async () => {
  try {
    const p = await pickImage();
    if (p) setQueryPath(p, 'drop');
  } catch (e) { /* 取消 */ }
});

pastezone.addEventListener('click', () => {
  pastezone.classList.add('pulse');
  setTimeout(() => pastezone.classList.remove('pulse'), 600);
  toast('在此处按 Ctrl+V 粘贴图片');
  pastezone.focus();
});
pastezone.tabIndex = 0;
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

// File → base64 data URL
function fileToDataURL(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.readAsDataURL(file);
  });
}

/* ---------------- 搜索 ---------------- */
const colorRange = $('#color-range');
const maxdistRange = $('#maxdist-range');
$('#color-val').textContent = Number(colorRange.value).toFixed(2);
$('#maxdist-val').textContent = maxdistRange.value;
colorRange.addEventListener('input', () => $('#color-val').textContent = Number(colorRange.value).toFixed(2));
maxdistRange.addEventListener('input', () => $('#maxdist-val').textContent = maxdistRange.value);
colorRange.addEventListener('change', () => { if (queryFile || queryPath) doSearch(); });
maxdistRange.addEventListener('change', () => { if (queryFile || queryPath) doSearch(); });
$('#seg-top').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn) return;
  if (queryFile || queryPath) doSearch();
});

function segVal(segSel) {
  const active = $(segSel + ' button.active');
  return active ? active.dataset.val : '5';
}

async function doSearch() {
  const top = parseInt(segVal('#seg-top'), 10);
  const maxdist = parseInt(maxdistRange.value, 10);
  const colorWeight = parseFloat(colorRange.value);

  $('#search-busy').hidden = false;
  $('#btn-search').disabled = true;
  $('#results').innerHTML = '';
  $('#results-empty').hidden = true;
  try {
    let data;
    if (queryPath) {
      data = await api.queryPath(queryPath, top, maxdist, colorWeight);
    } else if (queryFile) {
      const dataURL = await fileToDataURL(queryFile);
      data = await api.queryData(dataURL, top, maxdist, colorWeight);
    } else {
      return;
    }
    renderResults(data);
  } catch (e) {
    toast(e);
    $('#results-empty').hidden = false;
    $('#results-empty p').textContent = '检索失败';
    $('#results-empty span').textContent = e;
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
  const rel = relOf(m.imageId);
  el.innerHTML =
    '<img class="thumb" alt="">' +
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
  const thumb = el.querySelector('.thumb');
  const src = '';
  // 异步加载本地图片为 data URI
  api.imageData(m.imageId).then(uri => {
    thumb.src = uri;
    thumb._uri = uri;
  }).catch(() => { thumb.src = ''; });
  thumb.addEventListener('click', () => viewImage(thumb._uri || '', m.imageId));
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
    const images = await api.images();
    grid.innerHTML = '';
    if (!images || images.length === 0) {
      $('#lib-empty').hidden = false;
      return;
    }
    $('#lib-empty').hidden = true;
    images.forEach(p => {
      const img = document.createElement('img');
      img.loading = 'lazy';
      api.imageData(p).then(uri => {
        img.src = uri;
        img._uri = uri;
      }).catch(() => { img.src = ''; });
      img.addEventListener('click', () => viewImage(img._uri || '', p));
      grid.appendChild(img);
    });
  } catch (e) {
    $('#lib-empty').hidden = false;
    $('#lib-empty p').textContent = '加载失败';
    $('#lib-empty span').textContent = e;
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
  const etaEl = $('#build-eta');
  if (job.state === 'done') {
    $('#build-label').textContent = '构建完成 · ' + job.images + ' 图像 · ' + job.regions + ' 区域';
    $('#build-file').textContent = '索引已保存至 ' + job.index;
    etaEl.hidden = true;
  } else if (job.state === 'error') {
    $('#build-label').textContent = '构建失败';
    $('#build-pct').textContent = '';
    $('#build-file').textContent = job.error;
    etaEl.hidden = true;
    card.classList.add('error');
  } else {
    $('#build-label').textContent = '正在构建 ' + job.done + '/' + job.total;
    $('#build-file').textContent = job.current || '';
    const eta = estimateETA(job);
    if (eta) {
      etaEl.textContent = eta;
      etaEl.hidden = false;
    } else {
      etaEl.hidden = true;
    }
  }
}

function estimateETA(job) {
  if (!job.started || !job.total || job.done <= 0) return '';
  const start = new Date(job.started).getTime();
  if (!start) return '';
  const elapsedSec = (Date.now() - start) / 1000;
  if (elapsedSec < 1) return '';
  const remaining = (job.total - job.done) / (job.done / elapsedSec);
  if (!isFinite(remaining) || remaining < 0) return '';
  if (remaining < 60) return '预计剩余 ' + Math.ceil(remaining) + ' 秒';
  const min = Math.floor(remaining / 60);
  const sec = Math.ceil(remaining % 60);
  return '预计剩余 ' + min + ' 分 ' + sec + ' 秒';
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
    await api.build(body);
    saveHistory('dir-history', body.dir);
    pollBuild();
  } catch (e) {
    setBuildBusy(false);
    if (!$('#cfg-dir').value.trim() && e) toast(e);
    toast(e);
  }
});

function pollBuild() {
  clearInterval(window._buildTimer);
  window._buildTimer = setInterval(async () => {
    try {
      const st = await api.buildStatus();
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
      toast('轮询失败: ' + e);
    }
  }, 500);
}

$('#btn-load').addEventListener('click', async () => {
  const idx = $('#cfg-load').value.trim();
  setBuildBusy(true);
  try {
    const res = await api.load(idx);
    toast('已加载：' + res.images + ' 图像 / ' + res.regions + ' 区域');
    if (idx) saveHistory('index-history', idx);
    $('#cfg-load').value = '';
    refreshStatus();
    loadLibrary();
  } catch (e) {
    toast(e);
  } finally {
    setBuildBusy(false);
  }
});

/* ---------------- 索引状态 ---------------- */
async function refreshStatus() {
  try {
    const st = await api.status();
    libRoot = st.root || libRoot;
    $('#st-load').textContent = st.loaded ? '已加载' : '未加载';
    $('#st-images').textContent = st.images;
    $('#st-regions').textContent = st.regions;
    $('#st-index').textContent = st.indexPath || '-';
    $('#st-root').textContent = st.root || '-';
    if (st.root && !$('#cfg-dir').value.trim()) $('#cfg-dir').value = st.root;
    if (st.indexPath && !$('#cfg-load').value.trim()) $('#cfg-load').value = st.indexPath;
    if (!st.build || st.build.state === 'idle') return;
    if (st.build.state === 'running') {
      $('#build-card').hidden = false;
      pollBuild();
      setBuildBusy(true);
    }
  } catch (e) { /* ignore */ }
}

/* ---------------- 调试 ---------------- */
let segParams = null;
let segLabelMap = null;
let segRegionsById = new Map();

function currentSegParams() {
  return {
    path: $('#seg-path').value.trim(),
    thresholdPct: parseFloat($('#seg-thr').value),
    thresholdFactor: 1.0,
    minAreaRatio: 0.0006,
    medianFilterK: parseInt($('#seg-median').value, 10),
    connectivity: parseInt(segVal('#seg-conn2'), 10),
  };
}

function refreshSegImage() {
  if (!segParams) return;
  api.segmentsPNG(segParams.path, segParams, segVal('#seg-mode'))
    .then(uri => { $('#seg-img').src = uri; })
    .catch(() => {});
}

// 加载标签图（data URI），用于鼠标悬停反查区域。
async function loadSegLabelMap(params) {
  const uri = await api.segmentsMapPNG(params.path, params);
  const img = await new Promise((resolve, reject) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = reject;
    i.src = uri;
  });
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return { data: imgData.data, width: canvas.width, height: canvas.height };
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

// 原生「浏览」按钮
$('#btn-browse-seg').addEventListener('click', async () => {
  try {
    const p = await pickImage();
    if (p) $('#seg-path').value = p;
  } catch (e) { /* 取消 */ }
});

$('#btn-seg').addEventListener('click', async () => {
  segParams = currentSegParams();
  if (!segParams.path) { toast('请输入图像路径'); return; }
  $('#seg-busy').hidden = false;
  $('#btn-seg').disabled = true;
  try {
    const [data, labelMap] = await Promise.all([
      api.segments(segParams.path, segParams),
      loadSegLabelMap(segParams).catch(() => null),
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
    toast(e);
  } finally {
    $('#seg-busy').hidden = true;
    $('#btn-seg').disabled = false;
  }
});

$('#seg-mode').addEventListener('click', e => {
  const btn = e.target.closest('button');
  if (!btn || !segParams) return;
  refreshSegImage();
});

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

/* ---------------- 索引页原生按钮 ---------------- */
$('#btn-browse-dir').addEventListener('click', async () => {
  try {
    const p = await pickDir();
    if (p) $('#cfg-dir').value = p;
  } catch (e) { /* 取消 */ }
});
$('#btn-browse-out').addEventListener('click', async () => {
  try {
    const p = await pickImage();
    if (p) $('#cfg-out').value = p;
  } catch (e) { /* 取消 */ }
});
$('#btn-browse-load').addEventListener('click', async () => {
  try {
    const p = await pickIndex();
    if (p) $('#cfg-load').value = p;
  } catch (e) { /* 取消 */ }
});

/* ---------------- 启动 ---------------- */
switchTab('search');
refreshAlgorithm();
refreshStatus();