import { mkdirSync, copyFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(fileURLToPath(import.meta.url));
const dist = join(root, "dist");
const src = join(root, "src");

mkdirSync(dist, { recursive: true });

copyFileSync(join(root, "index.html"), join(dist, "index.html"));
for (const f of ["app.js", "styles.css"]) {
  copyFileSync(join(src, f), join(dist, f));
}

console.log("frontend build: copied assets to dist/");
