// Minimal service worker: caches the app shell so it opens fast and works as an
// installed PWA. Generated media (/outputs) and API calls are always fetched
// fresh from the network.

const CACHE = "ai-studio-shell-v1";
const SHELL = ["/", "/index.html", "/style.css", "/app.js", "/manifest.json", "/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Never cache API responses or generated media.
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/outputs/")) {
    return; // let the browser handle it (network)
  }

  // App shell: cache-first, fall back to network.
  event.respondWith(
    caches.match(event.request).then((hit) => hit || fetch(event.request))
  );
});
