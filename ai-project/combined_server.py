#!/usr/bin/env python3
import http.server
import socketserver
import os
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

# 프론트엔드 디렉토리
FRONTEND_DIR = '/opt/ai/frontend'
PORT = 8000

class CombinedRequestHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # /api/ 요청은 무시하고 일반 경로만 번역
        if path.startswith('/api/'):
            return path
        return super().translate_path(path)
    
    def do_GET(self):
        # API 요청은 FastAPI 백엔드로 라우팅
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        # 정적 파일 제공
        path = self.translate_path(self.path)
        if Path(path).exists() and Path(path).is_file():
            self.send_file(path)
            return
        
        # SPA 라우팅: 없는 경로는 index.html로
        index_path = os.path.join(FRONTEND_DIR, 'index.html')
        if Path(index_path).exists():
            self.send_file(index_path)
        else:
            self.send_error(404)
    
    def do_POST(self):
        # POST 요청도 API 라우팅
        if self.path.startswith('/api/'):
            self.proxy_to_backend()
            return
        
        self.send_error(404)
    
    def proxy_to_backend(self):
        """FastAPI 백엔드로 요청 프록시"""
        try:
            # 요청 본문 읽기
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length) if content_length > 0 else b''
            
            # 백엔드 URL
            backend_url = f'http://127.0.0.1:8001{self.path}'
            
            # 백엔드로 요청
            import urllib.request
            req = urllib.request.Request(backend_url, data=body, method=self.command)
            
            # 헤더 복사
            for header, value in self.headers.items():
                if header.lower() not in ['host', 'connection']:
                    req.add_header(header, value)
            
            with urlopen(req) as response:
                self.send_response(response.status)
                for header, value in response.headers.items():
                    self.send_header(header, value)
                self.end_headers()
                self.wfile.write(response.read())
        except Exception as e:
            self.send_error(502, f"Bad Gateway: {str(e)}")
    
    def send_file(self, filepath):
        """파일을 클라이언트로 전송"""
        try:
            with open(filepath, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', self.guess_type(filepath))
                self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self.end_headers()
                self.wfile.write(f.read())
        except Exception as e:
            self.send_error(404)

# 작업 디렉토리 설정
os.chdir(FRONTEND_DIR)

# 서버 시작
with socketserver.TCPServer(("0.0.0.0", PORT), CombinedRequestHandler) as httpd:
    print(f"AIBlur Combined Server running on port {PORT}...")
    print(f"Frontend: http://0.0.0.0:{PORT}")
    print(f"API proxied to: http://127.0.0.1:8001")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
