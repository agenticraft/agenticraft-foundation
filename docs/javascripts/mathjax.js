window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
    macros: {
      // CSP operators
      Ref:    ["\\sqsubseteq", 0],
      Traces: ["\\operatorname{traces}", 0],
      Caps:   ["\\operatorname{caps}", 0],
      Protos: ["\\operatorname{protos}", 0],

      // Common sets
      R:      ["\\mathbb{R}", 0],
      N:      ["\\mathbb{N}", 0],

      // Consensus
      Correct:    ["\\operatorname{Correct}", 0],
      decision:   ["\\operatorname{decision}", 0],
      proposal:   ["\\operatorname{proposal}", 0],
    },
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
