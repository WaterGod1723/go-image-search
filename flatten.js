const fs = require('fs');
const path = require('path');

const usage = `用法:
  node flatten.js pack <输入目录> <输出文件>  将文件夹平铺为单个文件
  node flatten.js unpack <输入文件> <输出目录>  将单文件还原为文件夹`;

// 打包时忽略的文件/目录（名称或相对路径匹配）
const IGNORE = ['.git', 'node_modules', 'flatten.js', 'package.json'];

function shouldIgnore(rel) {
  return IGNORE.some((it) => rel === it || rel.startsWith(it + path.sep));
}

function pack(srcDir, outFile) {
  const entries = [];
  function walk(dir, rel) {
    for (const name of fs.readdirSync(dir)) {
      const full = path.join(dir, name);
      const rp = rel ? path.join(rel, name) : name;
      if (shouldIgnore(rp)) continue;
      const stat = fs.lstatSync(full);
      if (stat.isDirectory()) {
        walk(full, rp);
      } else if (stat.isFile()) {
        entries.push({ path: rp.replace(/\\/g, '/'), size: stat.size, content: fs.readFileSync(full, 'utf8') });
      }
    }
  }
  walk(srcDir, '');
  const data = { manifest: entries.map((e) => ({ path: e.path, size: e.size })), files: entries.map((e) => e.content) };
  fs.writeFileSync(outFile, JSON.stringify(data, null, 2), 'utf8');
  console.log(`打包完成: ${entries.length} 个文件 -> ${outFile}`);
}

function unpack(inFile, outDir) {
  const data = JSON.parse(fs.readFileSync(inFile, 'utf8'));
  for (let i = 0; i < data.manifest.length; i++) {
    const rel = data.manifest[i].path;
    const full = path.join(outDir, rel);
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, data.files[i], 'utf8');
  }
  console.log(`还原完成: ${data.manifest.length} 个文件 -> ${outDir}`);
}

const [, , cmd, arg1, arg2] = process.argv;
if (!cmd || !arg1 || !arg2) {
  console.log(usage);
  process.exit(1);
}

if (cmd === 'pack') pack(path.resolve(arg1), path.resolve(arg2));
else if (cmd === 'unpack') unpack(path.resolve(arg1), path.resolve(arg2));
else {
  console.log(usage);
  process.exit(1);
}