import { mkdirSync, copyFileSync, cpSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, "dist");
const src = join(root, "src");

mkdirSync(dist, { recursive: true });

copyFileSync(join(root, "index.html"), join(dist, "index.html"));
for (const f of ["app.js", "styles.css", "worker.js"]) {
  copyFileSync(join(src, f), join(dist, f));
}
if (existsSync(join(src, "assets"))) {
  cpSync(join(src, "assets"), join(dist, "assets"), { recursive: true });
}

console.log("frontend build: copied assets to dist/");
