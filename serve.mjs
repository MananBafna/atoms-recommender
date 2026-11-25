// Static file server for the project root on http://localhost:3000 (Node stdlib only).
// If port 3000 is already in use, it assumes a server is running and exits quietly.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = 3000;
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png",
  ".jpg": "image/jpeg", ".ico": "image/x-icon", ".md": "text/plain; charset=utf-8",
  ".csv": "text/csv; charset=utf-8", ".txt": "text/plain; charset=utf-8",
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  let rel = decodeURIComponent(url.pathname);
  let file = path.normalize(path.join(ROOT, rel));
  if (!file.startsWith(ROOT)) { res.writeHead(403); return res.end("Forbidden"); }
  fs.stat(file, (err, st) => {
    if (!err && st.isDirectory()) {
      if (!rel.endsWith("/")) { res.writeHead(301, { Location: rel + "/" + url.search }); return res.end(); }
      file = path.join(file, "index.html");
    }
    fs.readFile(file, (e, buf) => {
      if (e) { res.writeHead(404, { "Content-Type": "text/plain" }); return res.end("Not found"); }
      res.writeHead(200, { "Content-Type": TYPES[path.extname(file).toLowerCase()] || "application/octet-stream",
                           "Cache-Control": "no-store" });
      res.end(buf);
    });
  });
});

server.on("error", (e) => {
  if (e.code === "EADDRINUSE") { console.log(`Port ${PORT} is in use; a server is already running.`); process.exit(0); }
  throw e;
});
server.listen(PORT, () => console.log(`Serving ${ROOT} at http://localhost:${PORT}`));
