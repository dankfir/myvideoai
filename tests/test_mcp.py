"""MCP protocol integration test — spawn the stdio server, speak JSON-RPC 2.0.

Run from the repository root (the directory that contains the ``myvideoai``
package):
  python -m myvideoai.tests.test_mcp
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # repository root (parent of the package)
ENV = os.environ.copy()
# Vendored dependency roots are derived from the checkout, never hard-coded.
_VENDORED = ROOT / ".vendored"
_VENDORED_PATHS = [p for p in (_VENDORED / "torch_pkgs", _VENDORED / "py_pkgs") if p.is_dir()]
_INHERITED = [p for p in ENV.get("PYTHONPATH", "").split(os.pathsep) if p]
ENV["PYTHONPATH"] = os.pathsep.join([str(p) for p in _VENDORED_PATHS] + _INHERITED)
ENV["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ENV["PYTHONIOENCODING"] = "utf-8"

PROC = subprocess.Popen(
    [sys.executable, "-m", "myvideoai.mcp_server"],
    cwd=str(ROOT), env=ENV,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, encoding="utf-8", bufsize=1,
)


def rpc(method: str, params: dict | None = None, rid: int = 1) -> dict:
    msg = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        msg["params"] = params
    PROC.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
    PROC.stdin.flush()
    line = PROC.stdout.readline()
    if not line:
        err = PROC.stderr.read()
        raise RuntimeError(f"server exited, stderr:\n{err[-2000:]}")
    return json.loads(line)


def call(name: str, args: dict, rid: int = 1) -> dict:
    return rpc("tools/call", {"name": name, "arguments": args}, rid)


def text_of(resp: dict) -> str:
    return resp.get("result", {}).get("content", [{}])[0].get("text", "")


def main() -> int:
    results = []
    def check(label, cond, detail=""):
        results.append((label, cond, detail))
        print(f"[{'PASS' if cond else 'FAIL'}] {label}{' :: ' + detail if detail else ''}")

    # 1. initialize handshake
    init = rpc("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test-client", "version": "1.0"},
    })
    check("initialize", init.get("result", {}).get("serverInfo", {}).get("name") == "myvideoai",
          str(init.get("result", {}).get("protocolVersion")))
    PROC.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n")
    PROC.stdin.flush()

    # 2. tools/list
    tl = rpc("tools/list", rid=2)
    names = [t["name"] for t in tl.get("result", {}).get("tools", [])]
    expect = {"get_info", "probe_video", "list_backends", "list_plugins", "enhance_video",
              "preview_frame", "write_plugin", "register_model", "validate_plugin"}
    check("tools/list covers all capabilities", set(names) == expect, str(names))

    # 3. resources/list + read
    rl = rpc("resources/list", rid=3)
    check("resources/list", len(rl.get("result", {}).get("resources", [])) == 3)
    rr = rpc("resources/read", {"uri": "myvideoai://docs/plugin-spec"}, rid=4)
    rtext = rr.get("result", {}).get("contents", [{}])[0].get("text", "")
    check("resources/read plugin-spec", "PluginManifest" in rtext or "plugin" in rtext.lower(),
          f"{len(rtext)} chars")

    # 4. list_backends
    be = call("list_backends", {}, rid=5)
    check("list_backends auto pick", "auto_pick" in text_of(be), text_of(be)[:120].replace("\n", " "))

    # 5. list_plugins
    pl = call("list_plugins", {}, rid=6)
    check("list_plugins", "plugins" in text_of(pl))

    # 6. probe_video
    pv = call("probe_video", {"video": "myvideoai/samples/sample_lowres.mp4"}, rid=7)
    check("probe_video", "width" in text_of(pv), text_of(pv).replace("\n", " ")[:120])

    # 7. get_info
    gi = call("get_info", {}, rid=8)
    check("get_info", "MCP tools" in text_of(gi))

    # 8. preview_frame (real model chain) — skip if too slow? it's one frame, fast
    pf = call("preview_frame", {"input": "myvideoai/samples/sample_lowres.mp4",
                                "time": 0.5, "backend": "esrgan", "scale": 2}, rid=9)
    check("preview_frame", "mcp_preview" in text_of(pf) and "enhanced" in text_of(pf),
          text_of(pf).replace("\n", " ")[:160])

    # 9. write_plugin (frame scaffold) into USER_PLUGINS_DIR, then validate, then clean up
    wp = call("write_plugin", {"id": "examples.mcp_test_hook", "kind": "frame",
                               "name": "MCP Test Hook", "description": "agent-authored via MCP"}, rid=10)
    check("write_plugin", "scaffolded" in text_of(wp), text_of(wp).replace("\n", " ")[:120])
    vp = call("validate_plugin", {"id": "examples.mcp_test_hook"}, rid=11)
    check("validate_plugin", "VALID" in text_of(vp), text_of(vp).replace("\n", " ")[:120])
    # path traversal guard
    bad = call("write_plugin", {"id": "../../evil", "kind": "frame"}, rid=12)
    check("write_plugin rejects traversal", bad.get("result", {}).get("isError") is True,
          text_of(bad)[:120])

    # 10. unknown tool -> isError
    unk = call("no_such_tool", {}, rid=13)
    check("unknown tool isError", unk.get("result", {}).get("isError") is True)

    # 11. ping
    pg = rpc("ping", rid=14)
    check("ping", "result" in pg)

    # 12. enhance_video drop-in path (fast tiny ONNX model, 3s clip would be slow — use 1 tiny file via -t not supported,
    #     so just validate the tool errors gracefully on a bogus backend instead of running full clip)
    bad_be = call("enhance_video", {"input": "myvideoai/samples/sample_lowres.mp4", "backend": "bogus"}, rid=15)
    check("enhance_video validates backend", bad_be.get("result", {}).get("isError") is True,
          text_of(bad_be)[:100])

    PROC.stdin.close()
    try:
        PROC.wait(timeout=5)
    except subprocess.TimeoutExpired:
        PROC.kill()
    err = PROC.stderr.read()[-1500:]
    if err:
        print(f"[server stderr tail]\n{err}")

    # Self-cleanup: the scaffold created by write_plugin must not linger as an
    # active user plugin (otherwise it silently ran on every future job).
    scaffold = ROOT / "myvideoai" / "models" / "plugins" / "examples_mcp_test_hook"
    if scaffold.exists():
        import shutil
        shutil.rmtree(scaffold, ignore_errors=True)
        print(f"[cleanup] removed test scaffold {scaffold}")

    passed = sum(1 for _, c, _ in results if c)
    print(f"\nMCP PROTOCOL: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
