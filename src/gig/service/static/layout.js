/* Command Center splitter. Width is stored locally. No animation. */
"use strict";
(() => {
  const handle = document.getElementById("analysis-split");
  const grid = document.getElementById("analysis-grid");
  if (!handle || !grid) return;
  const key = "gig-context-width";
  let saved = null;
  try { saved = localStorage.getItem(key); } catch { /* Optional UI preference. */ }
  const initial = saved !== null && Number.isFinite(Number(saved)) ? Math.min(520, Math.max(220, Number(saved))) : 320;
  grid.style.setProperty("--context-w", `${initial}px`);
  handle.setAttribute("aria-valuenow", String(Math.round(initial)));

  function setWidth(px) {
    const width = Math.min(520, Math.max(220, px));
    grid.style.setProperty("--context-w", `${width}px`);
    handle.setAttribute("aria-valuenow", String(Math.round(width)));
    try { localStorage.setItem(key, String(Math.round(width))); } catch (error) { /* private mode */ }
  }
  handle.addEventListener("pointerdown", (event) => {
    if (document.getElementById("content").dataset.section !== "command-center") return;
    handle.setPointerCapture(event.pointerId);
    handle.dataset.dragging = "1";
  });
  handle.addEventListener("pointermove", (event) => {
    if (handle.dataset.dragging !== "1") return;
    setWidth(grid.getBoundingClientRect().right - event.clientX);
  });
  const end = () => { delete handle.dataset.dragging; };
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  handle.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const current = parseFloat(getComputedStyle(grid).getPropertyValue("--context-w")) || 320;
    setWidth(current + (event.key === "ArrowLeft" ? 24 : -24));
    event.preventDefault();
  });
})();
