"""HTTP server that disables browser caching AND supports HTTP Range requests.

The default SimpleHTTPRequestHandler does not implement Range requests, which
breaks seeking inside HTML5 <video> elements (the browser fetches the file
in one go and cannot jump to an arbitrary byte offset). This subclass adds
HTTP 206 Partial Content support so the timeline scrub bar works.
"""
from __future__ import annotations

import os
import re
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, test


_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class NoCacheRangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        """Override to handle Range requests with HTTP 206 Partial Content."""
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()

        range_header = self.headers.get("Range")
        if not range_header:
            return super().send_head()

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        try:
            fs = os.fstat(f.fileno())
            file_size = fs.st_size

            m = _RANGE_RE.match(range_header)
            if not m:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Range header")
                f.close()
                return None

            start_str, end_str = m.group(1), m.group(2)
            if start_str == "" and end_str == "":
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Range header")
                f.close()
                return None

            if start_str == "":
                # Suffix range: last N bytes.
                length = int(end_str)
                if length <= 0:
                    self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                    f.close()
                    return None
                start = max(0, file_size - length)
                end = file_size - 1
            else:
                start = int(start_str)
                end = int(end_str) if end_str else file_size - 1

            if start >= file_size or end >= file_size or start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                f.close()
                return None

            f.seek(start)
            content_length = end - start + 1

            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Last-Modified", self.date_time_string(int(fs.st_mtime)))
            self.end_headers()

            self._range_remaining = content_length
            return f
        except Exception:
            f.close()
            raise

    def copyfile(self, source, outputfile):
        """Honor Content-Length set by send_head() during Range responses.

        Swallow client-disconnect errors quietly: when the user scrubs the
        timeline rapidly, the browser cancels in-flight Range requests, which
        surfaces as ConnectionResetError / BrokenPipeError on Windows. These
        are expected and not server bugs.
        """
        remaining = getattr(self, "_range_remaining", None)
        try:
            if remaining is None:
                return super().copyfile(source, outputfile)
            chunk_size = 64 * 1024
            while remaining > 0:
                chunk = source.read(min(chunk_size, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass  # client cancelled (typical during fast seeking)
        finally:
            self._range_remaining = None

    def log_message(self, format, *args):
        """Skip cancelled-request noise in the access log."""
        try:
            super().log_message(format, *args)
        except Exception:
            pass

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    test(HandlerClass=NoCacheRangeHandler, port=port, bind="0.0.0.0")
