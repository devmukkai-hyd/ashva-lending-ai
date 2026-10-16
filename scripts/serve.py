"""
serve.py - start api.py, check /health, optionally open a public ngrok URL for Postman.

    python serve.py             local only:  http://127.0.0.1:8000/docs
    python serve.py public      + public ngrok URL (needs NGROK_AUTHTOKEN and LENDING_API_KEY in .env)

Ctrl+C stops the server (and closes the tunnel).
"""
import asyncio
import os
import sys
import threading
import time

import requests
import uvicorn

from api import api_app        # also loads your .env (through risk_agent)

# 1. Start the server in a background thread (its own event loop), like the notebook
_server = uvicorn.Server(uvicorn.Config(api_app, host="0.0.0.0", port=8000, log_level="warning"))


def _run_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_server.serve())


threading.Thread(target=_run_server, daemon=True).start()
time.sleep(1.5)

# 2. Self-test: a real HTTP round trip
health = requests.get("http://127.0.0.1:8000/health", timeout=5)
print("Local health check:", health.status_code, health.json())
URL = "http://127.0.0.1:8000"

# 3. Optional: public URL with ngrok
tunnel = None
if "public" in sys.argv:
    if not os.getenv("LENDING_API_KEY"):
        sys.exit("Set LENDING_API_KEY in .env first: a public URL without it lets anyone record loan decisions.")
    from pyngrok import ngrok
    ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
    tunnel = ngrok.connect(8000, "http")
    URL = tunnel.public_url

print(f"""
URL: {URL}        (Swagger UI: {URL}/docs)

Postman
  Method : POST
  URL    : {URL}/policy/ask
  Headers: Content-Type: application/json   and   X-API-Key: <your LENDING_API_KEY>
  Body   : {{"question": "What is the minimum DSCR for Retail Trade?", "sector": "Retail Trade"}}
  200 = answer;  401 = key missing/wrong;  422 = bad request or guardrail (see 'flags').

Ctrl+C to stop.""")

# 4. Keep running until Ctrl+C, then clean up
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    _server.should_exit = True
    if tunnel:
        ngrok.disconnect(tunnel.public_url)
    print("\nServer stopped" + (", tunnel closed." if tunnel else "."))
