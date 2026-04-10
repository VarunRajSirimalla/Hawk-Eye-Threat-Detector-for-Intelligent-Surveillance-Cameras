import http.server
import socketserver
import os

PORT = int(os.getenv('FRONTEND_PORT', '5173'))
ROOT = os.path.join(os.path.dirname(__file__), 'frontend')
os.chdir(ROOT)

class Handler(http.server.SimpleHTTPRequestHandler):
    pass

def serve_on_available_port(port: int) -> None:
    last_error = None
    for p in (port, port + 1, port + 2):
        try:
            with socketserver.TCPServer(('0.0.0.0', p), Handler) as httpd:
                print(f"Frontend served at http://localhost:{p}")
                httpd.serve_forever()
                return
        except OSError as e:
            last_error = e
            continue
    raise SystemExit(f"Failed to bind ports {port}-{port+2}: {last_error}")

if __name__ == '__main__':
    serve_on_available_port(PORT)
