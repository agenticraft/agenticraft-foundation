/**
 * Mermaid diagram zoom controls for MkDocs Material.
 *
 * Material for MkDocs renders mermaid diagrams inside a closed shadow DOM:
 *   <pre class="mermaid"> → <div class="mermaid"> + closed shadowRoot with SVG
 *
 * Since the SVG is inaccessible, we wrap the <div class="mermaid"> host
 * element and apply zoom transforms to it directly.
 */

(function () {
  "use strict";

  var MIN_SCALE = 0.5;
  var MAX_SCALE = 3;
  var STEP = 0.2;

  /** Create a single control button via DOM API */
  function createButton(action, title, label) {
    var btn = document.createElement("button");
    btn.className = "diagram-btn";
    btn.setAttribute("data-action", action);
    btn.setAttribute("title", title);
    btn.setAttribute("aria-label", title);
    btn.textContent = label;
    return btn;
  }

  /** Build the control toolbar */
  function createControls() {
    var bar = document.createElement("div");
    bar.className = "diagram-controls";
    bar.appendChild(createButton("zoom-in", "Zoom in", "+"));
    bar.appendChild(createButton("zoom-out", "Zoom out", "\u2212"));
    bar.appendChild(createButton("reset", "Reset zoom", "\u21BA"));
    bar.appendChild(createButton("fullscreen", "Fullscreen", "\u26F6"));
    return bar;
  }

  /** Wrap a rendered mermaid div (shadow host) with zoom container + controls */
  function wrapDiagram(mermaidDiv) {
    // Avoid wrapping twice
    if (mermaidDiv.parentElement && mermaidDiv.parentElement.classList.contains("diagram-inner")) return;
    // Only target rendered divs, not source <pre> blocks
    if (mermaidDiv.tagName !== "DIV") return;

    var container = document.createElement("div");
    container.className = "diagram-container";

    var inner = document.createElement("div");
    inner.className = "diagram-inner";

    mermaidDiv.parentElement.insertBefore(container, mermaidDiv);
    inner.appendChild(mermaidDiv);
    container.appendChild(inner);
    container.appendChild(createControls());

    var scale = 1;
    var panX = 0;
    var panY = 0;
    var isPanning = false;
    var startX = 0;
    var startY = 0;

    function applyTransform() {
      mermaidDiv.style.transform = "translate(" + panX + "px, " + panY + "px) scale(" + scale + ")";
      mermaidDiv.style.transformOrigin = "center center";
    }

    function setScale(s) {
      scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, Math.round(s * 100) / 100));
      if (scale <= 1) {
        panX = 0;
        panY = 0;
      }
      applyTransform();
    }

    // Button clicks
    container.addEventListener("click", function (e) {
      var btn = e.target.closest("[data-action]");
      if (!btn) return;
      var action = btn.getAttribute("data-action");

      if (action === "zoom-in") setScale(scale + STEP);
      else if (action === "zoom-out") setScale(scale - STEP);
      else if (action === "reset") { scale = 1; panX = 0; panY = 0; applyTransform(); }
      else if (action === "fullscreen") toggleFullscreen(container);
    });

    // Pan via pointer events
    inner.addEventListener("pointerdown", function (e) {
      if (scale <= 1) return;
      isPanning = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
      inner.setPointerCapture(e.pointerId);
      inner.style.cursor = "grabbing";
    });

    inner.addEventListener("pointermove", function (e) {
      if (!isPanning) return;
      panX = e.clientX - startX;
      panY = e.clientY - startY;
      applyTransform();
    });

    inner.addEventListener("pointerup", function () {
      isPanning = false;
      inner.style.cursor = "";
    });

    inner.addEventListener("pointercancel", function () {
      isPanning = false;
      inner.style.cursor = "";
    });

    // Mouse wheel zoom
    inner.addEventListener("wheel", function (e) {
      e.preventDefault();
      var delta = e.deltaY > 0 ? -STEP : STEP;
      setScale(scale + delta);
    }, { passive: false });
  }

  function toggleFullscreen(el) {
    if (!document.fullscreenElement) {
      (el.requestFullscreen || el.webkitRequestFullscreen || el.msRequestFullscreen).call(el);
    } else {
      (document.exitFullscreen || document.webkitExitFullscreen || document.msExitFullscreen).call(document);
    }
  }

  /** Scan the page and wrap any unwrapped mermaid divs */
  function initDiagrams() {
    // Material replaces <pre class="mermaid"> with <div class="mermaid">
    // which contains a closed shadow root with the SVG
    document.querySelectorAll("div.mermaid").forEach(function (div) {
      wrapDiagram(div);
    });
  }

  // Material for MkDocs exposes document$ (RxJS observable) that fires
  // on each page load (including instant navigation). Mermaid rendering
  // is async so we use a MutationObserver to detect when <pre> is
  // replaced with <div>.
  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      // Delay to let mermaid finish rendering
      setTimeout(initDiagrams, 800);
    });
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      setTimeout(initDiagrams, 800);
    });
  }

  // MutationObserver catches the <pre> → <div> replacement by mermaid
  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      for (var j = 0; j < mutations[i].addedNodes.length; j++) {
        var node = mutations[i].addedNodes[j];
        if (node.nodeType === 1 && node.tagName === "DIV" && node.classList.contains("mermaid")) {
          // Small delay to ensure shadow DOM is attached
          setTimeout(function (n) { wrapDiagram(n); }, 50, node);
        }
      }
    }
  });

  observer.observe(document.body || document.documentElement, {
    childList: true,
    subtree: true,
  });
})();
