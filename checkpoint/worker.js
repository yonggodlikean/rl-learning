"use strict";

// Every run gets a new Worker and a new Python namespace. The main thread
// terminates this Worker if initialization or user code takes too long.
(async () => {
  try {
    importScripts("./vendor/pyodide/pyodide.js");
    self.pyodide = await loadPyodide({
      indexURL: new URL("./vendor/pyodide/", self.location.href).href,
    });
    self.postMessage({ type: "ready" });
  } catch (error) {
    self.postMessage({ type: "error", error: `Pyodide 初始化失败：${error.message || error}` });
  }
})();

self.onmessage = async ({ data }) => {
  if (!data || data.type !== "run" || !self.pyodide) return;
  try {
    const payload = JSON.stringify({
      code: String(data.code || ""),
      tests: Array.isArray(data.tests) ? data.tests : [],
    });
    self.pyodide.globals.set("__checkpoint_payload", payload);
    const output = await self.pyodide.runPythonAsync(`
import contextlib
import io
import json
import traceback

payload = json.loads(__checkpoint_payload)
namespace = {"__name__": "__main__"}
captured = io.StringIO()
results = []
setup_error = None
diagnostic = None

try:
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        exec(compile(payload["code"], "<student-code>", "exec"), namespace)
except SyntaxError as exc:
    setup_error = traceback.format_exc(limit=4)[-1800:]
    diagnostic = {"line": exc.lineno or 1, "column": exc.offset or 1, "message": exc.msg}
except BaseException:
    setup_error = traceback.format_exc(limit=4)[-1800:]

for test in payload["tests"]:
    if setup_error:
        results.append({"name": test["name"], "passed": False, "error": setup_error})
        continue
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            exec(compile(test["code"], "<public-test>", "exec"), namespace)
        results.append({"name": test["name"], "passed": True, "error": ""})
    except BaseException:
        results.append({"name": test["name"], "passed": False,
                        "error": traceback.format_exc(limit=3)[-1300:]})

json.dumps({"results": results, "stdout": captured.getvalue()[-1800:],
            "diagnostic": diagnostic}, ensure_ascii=False)
`);
    self.postMessage({ type: "result", result: JSON.parse(output) });
  } catch (error) {
    self.postMessage({ type: "error", error: `代码运行失败：${error.message || error}` });
  }
};
