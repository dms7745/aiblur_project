#!/usr/bin/env python3
import http.server
import socketserver
import os
from pathlib import Path
import urllib.request
import json

os.chdir('/opt/ai/frontend')

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # API 요청을 포트 8000으로 프록시
        if self.path.startswith('/api/'):
            try:
                api_url = f'http://localhost:8000{self.path}'
                with urllib.request.urlopen(api_url) as response:
                    self.send_response(response.status)
                    for header, value in response.headers.items():
                        self.send_header(header, value)
                    self.end_headers()
                    self.wfile.write(response.read())
                return
            except Exception as e:
                self.send_error(500, str(e))
                return
        
        # SPA 라우팅: 존재하지 않는 파일은 index.html로
        path = self.translate_path(self.path)
        if not Path(path).exists() and not self.path.startswith('/static'):
            self.path = '/index.html'
        return super().do_GET()

PORT = 8003
Handler = MyHTTPRequestHandler

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"AIBlur Frontend (with API proxy) serving on port {PORT}...")
    httpd.serve_forever()
