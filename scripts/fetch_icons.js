// fetch_icons.js — paginate iconfont.cn collections API, extract SVG icons,
// convert to 40×40 transparent PNG with sharp, save to data/test_pngs/.
// Run: node scripts/fetch_icons.js [targetCount] [startPage] [maxPages]
const fs = require('fs');
const path = require('path');
const https = require('https');
const sharp = require('sharp');

const OUT_DIR = path.join(__dirname, '..', 'data', 'test_pngs');
const TARGET = parseInt(process.argv[2] || '800', 10);
const START_PAGE = parseInt(process.argv[3] || '1', 10);
const MAX_PAGES = parseInt(process.argv[4] || '30', 10);

const COOKIE = 'cna=007SIdDs8yYCAXBQItb/Avqg; EGG_SESS_ICONFONT=Hu68kBY7XO7C6Udp3T99M1asKmUZ0gxjps8xjTrjx4ZtNCIR_nFu9Li15nxoPAWLbthjLC37Ly-f0_NjnxTrRLw4t8gpJsyMYKiJiKGq_OGW0-55SphW3JdiRdi7gPjg6V_ezROv-R7kjrK_cI8OUA5YVvS6qZ7nPzzdstHcN8gE_DFX47hroQzp7KftNy-C; u=11670537; u.sig=w9KQh6kDkrRCh-XAI_Vcfk6voW-uFXpzLVDQx4R5ZQ4; tfstk=gWbjE1gi0OQzqGPYBxPzPz88XtL157zFDfOOt13q6ELxBACpUs72u1o1CT1pXZ5wkd61ETIVudSVCG19Es-aISctia6I7qy0m1367FeULyzFijYMWJycwa3HnCfJB7-YibdvWFeULzorwn22W1rapv5RwLA66j3ABbhJEBJ9WKKte0dpedLOBhp-yCAIWc3vBbFWsLp9WFBOw7OwedL9WOC-6dDW_oOOG5RkzBjZxpfvFV3OyPxXdsnZWVQWGn1dML1PaaOXcpK1LRCGR6CA-C_g9c91tiB2ZOaQFUQfhZKW5Pgku6sRh3sbC4tl26b9Vg2Ix6fAhMKdkzNJVZfekL_gsVvA01QwfZeK_KQc3athSxwyQM5Pk36LU46MfiIpkNwQygyi8pt8tcGWxVOWL7NSjcmYafueK3PmpndkMyP7NYiMDQA527NSYqxvZILTN7MSj';

const HEADERS = {
  'accept': 'application/json, text/javascript, */*; q=0.01',
  'bx-v': '2.5.37',
  'cookie': COOKIE,
  'referer': 'https://www.iconfont.cn/collections/index?type=2',
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36',
  'x-requested-with': 'XMLHttpRequest',
};

function fetchPage(page) {
  return new Promise((resolve, reject) => {
    const ts = Date.now();
    const url = `https://www.iconfont.cn/api/collections.json?type=2&sort=time&limit=9&page=${page}&keyword=&t=${ts}&ctoken=null`;
    https.get(url, { headers: HEADERS }, (res) => {
      let body = '';
      res.on('data', (chunk) => body += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(body)); }
        catch (e) { reject(new Error(`JSON parse error page ${page}: ${e.message}`)); }
      });
    }).on('error', reject);
  });
}

function svgToPng(svgStr, outPath) {
  // Replace currentColor with a dark grey so icons are visible on any bg
  let svg = svgStr.replace(/currentColor/g, '#333333');
  // Ensure SVG has explicit width/height for sharp density
  if (!/width=/.test(svg)) {
    svg = svg.replace('<svg ', '<svg width="1024" height="1024" ');
  }
  return sharp(Buffer.from(svg), { density: 300 })
    .resize(40, 40, { fit: 'contain', background: { r: 0, g: 0, b: 0, alpha: 0 } })
    .png()
    .toFile(outPath);
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const seen = new Set();
  let saved = 0;
  // check existing files
  for (const f of fs.readdirSync(OUT_DIR)) {
    if (f.endsWith('.png')) { seen.add(f.replace('.png', '')); saved++; }
  }
  console.log(`start: ${saved} existing icons, target ${TARGET}`);

  for (let page = START_PAGE; page < START_PAGE + MAX_PAGES && saved < TARGET; page++) {
    let data;
    try { data = await fetchPage(page); }
    catch (e) { console.error(`page ${page}: ${e.message}`); continue; }
    if (data.code !== 200 || !data.data || !data.data.lists) {
      console.error(`page ${page}: code=${data.code}, stopping`);
      break;
    }
    const lists = data.data.lists;
    for (const col of lists) {
      if (!col.icons) continue;
      for (const icon of col.icons) {
        if (saved >= TARGET) break;
        if (!icon.show_svg || icon.show_svg.length < 50) continue;
        const id = String(icon.id);
        if (seen.has(id)) continue;
        seen.add(id);
        const name = `${id}.png`;
        const outPath = path.join(OUT_DIR, name);
        try {
          await svgToPng(icon.show_svg, outPath);
          saved++;
          if (saved % 50 === 0) console.log(`  saved ${saved}/${TARGET} (page ${page})`);
        } catch (e) {
          console.error(`  icon ${id}: ${e.message}`);
          seen.delete(id); // allow retry later
        }
      }
      if (saved >= TARGET) break;
    }
    console.log(`page ${page} done: ${saved}/${TARGET}`);
    // small delay to avoid rate limit
    await new Promise(r => setTimeout(r, 300));
  }
  console.log(`done: ${saved} icons in ${OUT_DIR}`);
}

main().catch(e => { console.error(e); process.exit(1); });
