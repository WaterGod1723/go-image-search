'use strict';

const $ = sel => document.querySelector(sel);
const $$ = sel => Array.from(document.querySelectorAll(sel));

/* ---------------- Wails 绑定 ---------------- */
const App = window.go.main.App;

// Wails 绑定调用统一出错处理：Go 侧返回错误时 Promise reject 的是字符串。
const api = {
  status: () => App.Status(),
  build: req => App.Build(req),
  buildStatus: () => App.BuildStatus(),
  load: idx => App.Load(idx),
  queryData: (u, top) => App.QueryData(u, top),
  queryPath: (p, top) => App.QueryPath(p, top),
  images: () => App.ImageList(),
  imageData: p => App.ImageDataURI(p),
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
const THEME_META = { kawaii: '#ff8fb0', mature: '#0e1220', pink: '#ff4fa3', matcha: '#7ba05b' };
const THEME_ORDER = ['kawaii', 'mature', 'pink', 'matcha'];
function applyTheme(name) {
  document.body.dataset.theme = name;
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = THEME_META[name] || THEME_META.kawaii;
  try { localStorage.setItem('gis-theme', name); } catch (e) { /* ignore */ }
}
$('#theme-toggle').addEventListener('click', () => {
  const cur = document.body.dataset.theme;
  const next = THEME_ORDER[(THEME_ORDER.indexOf(cur) + 1) % THEME_ORDER.length];
  applyTheme(next);
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

/* ---------------- 用户输入持久化（localStorage） ---------------- */
const SETTINGS_KEY = 'gis-settings';
const SETTING_FIELDS = ['cfg-dir', 'cfg-out', 'cfg-load'];

function loadSettings() {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'); } catch (e) { return {}; }
}
function saveSettings() {
  const s = {};
  SETTING_FIELDS.forEach(id => { const el = document.getElementById(id); if (el) s[id] = el.value; });
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(s)); } catch (e) { /* ignore */ }
}
function restoreSettings() {
  const s = loadSettings();
  SETTING_FIELDS.forEach(id => {
    const el = document.getElementById(id);
    if (el && s[id] != null) el.value = s[id];
  });
}
SETTING_FIELDS.forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', saveSettings);
});
restoreSettings();

/* ---------------- 上次索引路径持久化 ---------------- */
const LAST_INDEX_KEY = 'gis-last-index';
function getLastIndex() { return (localStorage.getItem(LAST_INDEX_KEY) || '').trim(); }
function setLastIndex(p) {
  p = (p || '').trim();
  try {
    if (p) localStorage.setItem(LAST_INDEX_KEY, p);
    else localStorage.removeItem(LAST_INDEX_KEY);
  } catch (e) { /* ignore */ }
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

  $('#search-busy').hidden = false;
  $('#btn-search').disabled = true;
  $('#results').innerHTML = '';
  $('#results-empty').hidden = true;
  try {
    let data;
    if (queryPath) {
      data = await api.queryPath(queryPath, top);
    } else if (queryFile) {
      const dataURL = await fileToDataURL(queryFile);
      data = await api.queryData(dataURL, top);
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
  $('#results-empty span').textContent = '无匹配图片';
  if (!data.matches || data.matches.length === 0) {
    $('#results-empty').hidden = false;
    return;
  }
  $('#results-empty').hidden = true;
  data.matches.forEach((m, i) => box.appendChild(resultCard(m, i)));
}

function resultCard(m, i) {
  const score = Math.round(m.score * 100);
  const el = document.createElement('div');
  el.className = 'card result-card';
  const rel = relOf(m.imageId);
  el.innerHTML =
    '<img class="thumb" alt="">' +
    '<div class="result-body">' +
      '<div class="result-top"><span class="result-name">' + esc(base(m.imageId)) + '</span><span class="rank">#' + (i + 1) + '</span></div>' +
      '<div class="scorebar"><div style="width:' + score + '%"></div></div>' +
      '<div class="result-meta">相似度 ' + score + '%</div>' +
      '<div class="result-actions">' +
        '<button class="copy-btn"><span class="copy-label">复制路径</span></button>' +
        '<button class="copy-btn"><span class="copy-label">复制相对路径</span></button>' +
      '</div>' +
    '</div>';
  const thumb = el.querySelector('.thumb');
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
    $('#build-label').textContent = '构建完成 · ' + job.images + ' 图像';
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
        if (st.state === 'done') {
          setLastIndex(st.index);
          loadLibrary();
        }
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
    toast('已加载：' + res.images + ' 图像');
    if (idx) { saveHistory('index-history', idx); setLastIndex(idx); }
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
refreshStatus();

// 自动加载上次使用的索引（库目录字段已由 restoreSettings 还原）。
(async function autoLoadLastIndex() {
  const p = getLastIndex();
  if (!p) return;
  try {
    const res = await api.load(p);
    toast('已恢复上次索引：' + res.images + ' 图像');
  } catch (e) { /* 静默：索引文件可能已移动或删除 */ }
  refreshStatus();
  loadLibrary();
})();

/* ---------------- 格式转换桥（webview 多线程 Worker） ---------------- */
// Go 侧解码失败时经事件 img:convert 将原始字节(base64)交给浏览器，
// 由 Worker 池并行解码并转成 PNG，再经 App.Converted 回传 Go。
(function webviewConverter() {
  const rt = window.runtime;
  if (!rt || !App.Converted) return; // 无 Wails 环境（如纯网页）则跳过

  function base64ToBytes(b64) {
    const bin = atob(b64);
    const u8 = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return u8.buffer;
  }
  function base64FromBytes(u8) {
    let bin = '';
    for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]);
    return btoa(bin);
  }

  const WORKER_COUNT = Math.max(1, Math.min(4, navigator.hardwareConcurrency || 4));
  const workers = [];
  const idle = [];
  const queue = [];
  const pending = new Map(); // id -> {resolve, reject}
  let nextId = 1;

  function spawn() {
    const w = new Worker('./worker.js');
    let activeJob = null;
    w.onmessage = (e) => {
      const { id, png, error } = e.data || {};
      const job = pending.get(id);
      pending.delete(id);
      if (job) {
        if (error) job.reject(error);
        else job.resolve(base64FromBytes(new Uint8Array(png)));
      }
      activeJob = null;
      idle.push(w);
      pump();
    };
    w.onerror = (ev) => {
      // worker 加载失败或运行期崩溃：拒绝其正在处理的任务，避免 Go 侧死等超时。
      if (activeJob) {
        const job = pending.get(activeJob);
        pending.delete(activeJob);
        if (job) job.reject('worker error: ' + (ev && ev.message || 'unknown'));
        activeJob = null;
      }
      const idx = workers.indexOf(w);
      if (idx >= 0) workers.splice(idx, 1);
      pump();
    };
    w._take = (job) => { activeJob = job.msg.id; };
    workers.push(w);
    idle.push(w);
  }
  for (let i = 0; i < WORKER_COUNT; i++) spawn();

  function pump() {
    while (idle.length && queue.length) {
      const w = idle.pop();
      const job = queue.shift();
      if (w._take) w._take(job);
      w.postMessage(job.msg, [job.msg.data]);
    }
  }

  function dispatch(msg) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      msg.id = id;
      pending.set(id, { resolve, reject });
      queue.push({ msg });
      pump();
    });
  }

  // 主线程兜底：栅格化 SVG（HTMLImageElement 是唯一可靠的 SVG 光栅化器）
  function rasterizeSvg(bytes, format) {
    return new Promise((resolve, reject) => {
      const blob = new Blob([bytes], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        try {
          let w = img.naturalWidth || 0;
          let h = img.naturalHeight || 0;
          if (!w || !h) {
            // 仅 viewBox、无 width/height 的 SVG：解析 viewBox 取尺寸。
            const vb = parseViewBox(bytes);
            if (vb) { w = w || vb.w; h = h || vb.h; }
          }
          if (!w || !h) { w = w || 128; h = h || 128; }
          const canvas = document.createElement('canvas');
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, w, h);
          canvas.toBlob(b => {
            URL.revokeObjectURL(url);
            if (!b) return reject('svg rasterize failed');
            b.arrayBuffer().then(ab => resolve(base64FromBytes(new Uint8Array(ab))), reject);
          }, 'image/png');
        } catch (err) { URL.revokeObjectURL(url); reject(err); }
      };
      img.onerror = () => { URL.revokeObjectURL(url); reject('svg load failed'); };
      img.src = url;
    });
  }

  // 从 SVG 字节中解析 viewBox 的宽高（无 width/height 时兜底）。
  function parseViewBox(bytes) {
    try {
      const text = new TextDecoder().decode(bytes).slice(0, 2048);
      const m = text.match(/viewBox=["'\s]+([\d.\d-]+)\s+([\d.\d-]+)\s+([\d.]+)\s+([\d.]+)/i);
      if (m) return { w: Math.max(1, Math.round(parseFloat(m[3]))), h: Math.max(1, Math.round(parseFloat(m[4]))) };
    } catch (e) { /* ignore */ }
    return null;
  }

  rt.EventsOn('img:convert', (task) => {
    // Wails 把 Go 端 EventsEmit 的单个参数作为回调的首个入参传入。
    if (!task || !task.id) return;
    const fmt = String(task.format || '').replace(/^\./, '').toLowerCase();
    const bytes = base64ToBytes(task.data);
    const p = (fmt === 'svg')
      ? rasterizeSvg(bytes, fmt)
      : dispatch({ data: bytes, format: fmt });
    p.then(pngB64 => App.Converted(String(task.id), pngB64))
      .catch(err => { if (App.ConvertError) App.ConvertError(String(task.id), String(err)); });
  });
})();