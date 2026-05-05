"""HTTP server that disables browser caching — useful while iterating on the player."""
from http.server import SimpleHTTPRequestHandler, test
import sys


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    test(HandlerClass=NoCacheHandler, port=port, bind="0.0.0.0")
