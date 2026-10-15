import threading
import time
import urllib.request
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "OneDrive" / "Documents" / "flutter1" / "kiara"
if not REPO_ROOT.exists():
    REPO_ROOT = Path("c:/Users/varun/OneDrive/Documents/flutter1/kiara")
sys.path.insert(0, str(REPO_ROOT))

from web.server import ThreadingHTTPServer, KiaraHTTPHandler, init_default_model

print("Initializing model...")
init_default_model()

PORT = 8991
server = ThreadingHTTPServer(("127.0.0.1", PORT), KiaraHTTPHandler)
server_thread = threading.Thread(target=server.serve_forever, daemon=True)
server_thread.start()
print(f"Server started on port {PORT}")

time.sleep(1)

# Test 1: GET /index.html
with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/") as res:
    assert res.status == 200
    html_content = res.read().decode("utf-8")
    assert "Kiara" in html_content
    print("Test 1 PASS: /index.html served correctly")

# Test 2: GET /style.css
with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/style.css") as res:
    assert res.status == 200
    css_content = res.read().decode("utf-8")
    assert "--accent-purple" in css_content
    print("Test 2 PASS: /style.css served correctly")

# Test 3: GET /app.js
with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/app.js") as res:
    assert res.status == 200
    js_content = res.read().decode("utf-8")
    assert "renderMarkdown" in js_content
    print("Test 3 PASS: /app.js served correctly")

# Test 4: GET /api/status
with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status") as res:
    assert res.status == 200
    status_data = json.loads(res.read().decode("utf-8"))
    assert status_data["status"] == "online"
    print(f"Test 4 PASS: /api/status returned: {status_data['model_name']} ({status_data['parameters']:,} params, {len(status_data['available_checkpoints'])} checkpoints)")

# Test 5: POST /api/chat (SSE stream smoke test)
req = urllib.request.Request(
    f"http://127.0.0.1:{PORT}/api/chat",
    data=json.dumps({"prompt": "Hello Kiara", "max_tokens": 10, "temperature": 0.7, "instruct": True}).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req) as res:
    assert res.status == 200
    assert "text/event-stream" in res.headers.get("Content-Type", "")
    lines = [res.readline().decode("utf-8") for _ in range(5)]
    print(f"Test 5 PASS: /api/chat SSE streaming operational, first lines: {lines[:3]}")

server.shutdown()
print("\nALL 5 WEB TESTS PASSED SUCCESSFULLY!")
