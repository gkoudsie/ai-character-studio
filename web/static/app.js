// AI Character Studio - mobile PWA client logic.
// Talks to the FastAPI backend running on the PC.

const $ = (id) => document.getElementById(id);

const els = {
  reference: $("reference"),
  refStatus: $("refStatus"),
  downloadAll: $("downloadAll"),
  clearBtn: $("clearBtn"),
  expiryNote: $("expiryNote"),
  theme: $("theme"),
  presets: $("theme-presets"),
  count: $("count"),
  countLabel: $("countLabel"),
  makeVideo: $("makeVideo"),
  makeReel: $("makeReel"),
  captions: $("captions"),
  useLlm: $("useLlm"),
  previewBtn: $("previewBtn"),
  createBtn: $("createBtn"),
  ideasCard: $("ideasCard"),
  ideasList: $("ideasList"),
  progressCard: $("progressCard"),
  stageText: $("stageText"),
  barFill: $("barFill"),
  progressMsg: $("progressMsg"),
  resultsCard: $("resultsCard"),
  reelWrap: $("reelWrap"),
  reelVideo: $("reelVideo"),
  reelDownload: $("reelDownload"),
  clipsWrap: $("clipsWrap"),
  clipsGrid: $("clipsGrid"),
  imagesWrap: $("imagesWrap"),
  imagesGrid: $("imagesGrid"),
  errorCard: $("errorCard"),
  errorText: $("errorText"),
};

let pollTimer = null;
let uploadedReference = null;   // filename returned by /api/reference
let currentJobId = null;
let isEphemeral = false;

// --- helpers ---
async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return res.json();
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

function reqBody() {
  return {
    theme: els.theme.value.trim() || "lifestyle",
    num_images: parseInt(els.count.value, 10),
    make_video: els.makeVideo.checked,
    make_reel: els.makeReel.checked,
    captions: els.captions.checked,
    use_llm: els.useLlm.checked,
    reference: uploadedReference,
  };
}

// Upload the chosen reference photo as soon as it's picked.
async function uploadReference() {
  const file = els.reference.files && els.reference.files[0];
  if (!file) { uploadedReference = null; els.refStatus.textContent = ""; return; }
  els.refStatus.textContent = "Uploading…";
  try {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/reference", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || "upload failed");
    uploadedReference = (await res.json()).reference;
    els.refStatus.textContent = "Reference ready ✓";
  } catch (e) {
    uploadedReference = null;
    els.refStatus.textContent = "Upload failed: " + e.message;
  }
}

function renderIdeas(scenes) {
  els.ideasList.innerHTML = "";
  scenes.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="cap">${escapeHtml(s.caption)}</span>` +
                   `<span class="prm">${escapeHtml(s.prompt)}</span>`;
    els.ideasList.appendChild(li);
  });
  show(els.ideasCard);
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function fillGrid(grid, urls, kind) {
  grid.innerHTML = "";
  urls.forEach((u) => {
    const el = document.createElement(kind === "video" ? "video" : "img");
    el.src = u;
    if (kind === "video") { el.controls = true; el.playsInline = true; }
    grid.appendChild(el);
  });
}

// --- theme presets ---
async function loadThemes() {
  try {
    const { themes } = await api("/api/themes");
    els.presets.innerHTML = "";
    themes.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t;
      els.presets.appendChild(opt);
    });
  } catch (_) { /* presets are optional */ }
}

// --- preview (no media) ---
async function preview() {
  els.previewBtn.disabled = true;
  hide(els.errorCard);
  try {
    const body = reqBody();
    const { scenes } = await api("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        theme: body.theme, num_images: body.num_images, use_llm: body.use_llm,
      }),
    });
    renderIdeas(scenes);
  } catch (e) {
    showError(e.message);
  } finally {
    els.previewBtn.disabled = false;
  }
}

// --- full generation ---
async function create() {
  els.createBtn.disabled = true;
  hide(els.errorCard);
  hide(els.resultsCard);
  els.barFill.style.width = "0%";
  els.stageText.textContent = "Starting…";
  show(els.progressCard);

  try {
    const { job_id } = await api("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(reqBody()),
    });
    currentJobId = job_id;
    poll(job_id);
  } catch (e) {
    showError(e.message);
    els.createBtn.disabled = false;
    hide(els.progressCard);
  }
}

function poll(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(async () => {
    let s;
    try {
      s = await api(`/api/status/${jobId}`);
    } catch (e) {
      return; // transient; keep polling
    }

    els.stageText.textContent = s.stage || "Working…";
    els.barFill.style.width = Math.round((s.progress || 0) * 100) + "%";

    if (s.scenes && s.scenes.length) renderIdeas(s.scenes);

    if (s.status === "done") {
      clearInterval(pollTimer);
      els.createBtn.disabled = false;
      hide(els.progressCard);
      renderResults(s);
    } else if (s.status === "error") {
      clearInterval(pollTimer);
      els.createBtn.disabled = false;
      hide(els.progressCard);
      showError(s.error || "Generation failed.");
    }
  }, 1500);
}

function renderResults(s) {
  let any = false;

  // Download-all + delete wiring.
  if (currentJobId) {
    els.downloadAll.href = `/api/download/${currentJobId}`;
  }
  if (isEphemeral && s.expires_at) {
    const mins = Math.max(0, Math.round((s.expires_at * 1000 - Date.now()) / 60000));
    els.expiryNote.textContent =
      `Nothing is saved on your device. These results live on the server for about ${mins} min, then auto-delete. Download what you want.`;
  } else {
    els.expiryNote.textContent = "";
  }

  if (s.reel) {
    els.reelVideo.src = s.reel;
    els.reelDownload.href = s.reel;
    show(els.reelWrap);
    any = true;
  } else { hide(els.reelWrap); }

  if (s.clips && s.clips.length) {
    fillGrid(els.clipsGrid, s.clips, "video");
    show(els.clipsWrap);
    any = true;
  } else { hide(els.clipsWrap); }

  if (s.images && s.images.length) {
    fillGrid(els.imagesGrid, s.images, "img");
    show(els.imagesWrap);
    any = true;
  } else { hide(els.imagesWrap); }

  if (any) show(els.resultsCard);
}

function showError(msg) {
  els.errorText.textContent = msg;
  show(els.errorCard);
}

// --- wire up ---
els.count.addEventListener("input", () => {
  els.countLabel.textContent = els.count.value;
});
els.reference.addEventListener("change", uploadReference);
els.previewBtn.addEventListener("click", preview);
els.createBtn.addEventListener("click", create);

els.clearBtn.addEventListener("click", async () => {
  if (!currentJobId) return;
  try { await fetch(`/api/results/${currentJobId}`, { method: "DELETE" }); } catch (_) {}
  hide(els.resultsCard);
  currentJobId = null;
});

// Learn whether the server is in ephemeral (cloud) mode.
async function loadConfig() {
  try {
    const cfg = await api("/api/config");
    isEphemeral = !!cfg.ephemeral;
  } catch (_) { /* default false */ }
}

loadConfig();
loadThemes();

// Register service worker for add-to-home-screen / offline shell.
if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
