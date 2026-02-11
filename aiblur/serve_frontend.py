#!/usr/bin/env python3
import http.server
import socketserver
import os
from pathlib import Path
import urllib.request
import urllib.error

os.chdir('/opt/ai/frontend')

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # /api/ 요청은 FastAPI 백엔드로 프록시
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        # 정적 파일 제공
        path = self.translate_path(self.path)
        if Path(path).exists() and Path(path).is_file():
            return super().do_GET()
        
        # SPA 라우팅: 없는 경로는 index.html로
        self.path = '/index.html'
        return super().do_GET()
    
    def do_POST(self):
        # /api/ 요청은 FastAPI 백엔드로 프록시
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        self.send_error(404)
    
    def proxy_to_backend(self):
        """FastAPI 백엔드 (포트 9000)로 요청 프록시"""
        try:
            # 요청 본문 읽기
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            # 백엔드 URL (포트 9000)
            backend_url = f'http://127.0.0.1:9000{self.path}'
            
            # 백엔드로 요청
            req = urllib.request.Request(backend_url, data=body, method=self.command)
            
            # 헤더 복사
            for header, value in self.headers.items():
                if header.lower() not in ['host', 'connection']:
                    req.add_header(header, value)
            
            with urllib.request.urlopen(req) as response:
                self.send_response(response.status)
                for header, value in response.headers.items():
                    self.send_header(header, value)
                self.end_headers()
                self.wfile.write(response.read())
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")

PORT = 8003
Handler = MyHTTPRequestHandler

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"AIBlur Frontend serving on port {PORT}...")
    print(f"API proxied to http://127.0.0.1:9000")
    httpd.serve_forever()
