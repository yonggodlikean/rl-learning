import { basicSetup, EditorView } from "codemirror";
import { python } from "@codemirror/lang-python";
import { autocompletion } from "@codemirror/autocomplete";
import { indentWithTab } from "@codemirror/commands";
import { keymap } from "@codemirror/view";
import { setDiagnostics } from "@codemirror/lint";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";

const editorTheme = EditorView.theme({
  "&": { color: "#e2f0e8", backgroundColor: "#172421" },
  ".cm-content": { caretColor: "#f4dfaf", padding: "18px 0", fontFamily: "var(--mono)" },
  ".cm-gutters": { backgroundColor: "#172421", color: "#8aa79c", borderRight: "1px solid #344943" },
  ".cm-activeLineGutter, .cm-activeLine": { backgroundColor: "#25362f" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "#395a51" },
  ".cm-cursor": { borderLeftColor: "#f4dfaf" },
  ".cm-tooltip": { backgroundColor: "#24362f", color: "#eef3e9", border: "1px solid #678678" },
  ".cm-tooltip-autocomplete > ul > li[aria-selected]": { backgroundColor: "#3a584c", color: "#fff" },
}, { dark: true });

const editorHighlight = HighlightStyle.define([
  { tag: tags.keyword, color: "#f7ae90" },
  { tag: tags.string, color: "#c9db97" },
  { tag: tags.number, color: "#f4d19a" },
  { tag: tags.comment, color: "#91ab9a", fontStyle: "italic" },
  { tag: tags.definition(tags.variableName), color: "#bdd7ee" },
  { tag: tags.function(tags.variableName), color: "#9edacc" },
]);

const words = [
  "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
  "len", "list", "max", "min", "range", "reversed", "round", "set",
  "sorted", "str", "sum", "tuple", "zip",
].map((label) => ({ label, type: "function" }));

function pythonWords(context) {
  const word = context.matchBefore(/[a-zA-Z_]\w*/);
  if (!word || (word.from === word.to && !context.explicit)) return null;
  return { from: word.from, options: words, validFor: /^[a-zA-Z_]\w*$/ };
}

function mount(parent, initial, onChange, onRun, label, id) {
  const view = new EditorView({
    doc: initial,
    parent,
    extensions: [
      basicSetup,
      python(),
      editorTheme,
      syntaxHighlighting(editorHighlight),
      autocompletion({ override: [pythonWords] }),
      keymap.of([
        indentWithTab,
        { key: "Mod-Enter", run: () => { onRun(); return true; } },
      ]),
      EditorView.lineWrapping,
      EditorView.updateListener.of((update) => {
        if (update.docChanged) onChange(update.state.doc.toString());
      }),
    ],
  });
  view.contentDOM.setAttribute("aria-label", label);
  view.contentDOM.id = id;
  view.contentDOM.setAttribute("spellcheck", "false");
  return {
    getValue: () => view.state.doc.toString(),
    focus: () => view.focus(),
    destroy: () => view.destroy(),
    indent: () => {
      const selection = view.state.selection.main;
      view.dispatch({ changes: { from: selection.from, to: selection.to, insert: "    " } });
      view.focus();
    },
    setDiagnostic: (diagnostic) => {
      if (!diagnostic || !Number.isInteger(diagnostic.line)) {
        view.dispatch(setDiagnostics(view.state, []));
        return;
      }
      const lineNumber = Math.max(1, Math.min(view.state.doc.lines, diagnostic.line));
      const line = view.state.doc.line(lineNumber);
      const column = Math.max(0, Number(diagnostic.column || 1) - 1);
      const from = Math.min(line.to, line.from + column);
      view.dispatch(setDiagnostics(view.state, [{
        from,
        to: Math.min(view.state.doc.length, from + 1),
        severity: "error",
        message: diagnostic.message || "此处有语法错误",
      }]));
    },
  };
}

window.RLCodeEditor = { mount };
