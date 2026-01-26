import os
import json
import shutil
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="AI Blur API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 업로드 디렉터리
UPLOAD_DIR = "/opt/ai/frontend/video"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 인메모리 데이터베이스
posts_db = []
post_id_counter = 1

class PasswordVerify(BaseModel):
    password: str

# 비밀번호 확인 API
@app.post("/api/verify-password")
async def verify_password(data: PasswordVerify):
    """게시글 비밀번호 확인"""
    admin_password = "admin1234"
    if data.password == admin_password:
        return {"status": "success", "valid": True}
    return {"status": "success", "valid": False}

# 요청 분석 API
@app.post("/request-analysis/")
async def request_analysis(
    name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    title: str = Form(...),
    content: str = Form(default=""),
    password: str = Form(...),
    videos: List[UploadFile] = File(...)
):
    """분석 요청 생성"""
    global posts_db, post_id_counter
    
    # 비디오 파일 저장
    saved_videos = []
    for video in videos:
        ext = os.path.splitext(video.filename)[1] or '.mp4'
        filename = f"original_{post_id_counter}_{uuid.uuid4().hex[:8]}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, "wb") as f:
            shutil.copyfileobj(video.file, f)
        
        saved_videos.append(f"/video/{filename}")
    
    new_post = {
        "id": post_id_counter,
        "name": name,
        "phone": phone,
        "email": email,
        "title": title,
        "content": content,
        "password": password,
        "status": "PENDING",
        "created_at": datetime.now().isoformat(),
        "videos": json.dumps(saved_videos),
        "original_video_filename": ", ".join([v.filename for v in videos]),
        "analyzed_video_path": None,
        "author": name,
        "target_address": ""
    }
    
    posts_db.append(new_post)
    post_id_counter += 1
    
    return {"status": "success", "post_id": new_post["id"]}

# 게시글 목록 조회
@app.get("/api/posts")
async def get_posts(page: int = 1, search: str = "", status_filter: str = ""):
    """게시글 목록 조회"""
    filtered = posts_db
    if search:
        filtered = [p for p in filtered if search.lower() in p.get("title", "").lower()]
    if status_filter:
        filtered = [p for p in filtered if p["status"] == status_filter]
    
    filtered = sorted(filtered, key=lambda x: x["id"], reverse=True)
    
    per_page = 10
    total_pages = (len(filtered) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    
    return {
        "posts": [
            {
                "id": p["id"],
                "author": p.get("name", p.get("author", "Unknown")),
                "name": p.get("name", ""),
                "title": p["title"],
                "email": p["email"],
                "status": p["status"],
                "created_at": p["created_at"]
            }
            for p in filtered[start:end]
        ],
        "total_posts": len(filtered),
        "total_pages": total_pages,
        "current_page": page
    }

# 게시글 상세 조회
@app.get("/api/posts/{post_id}")
async def get_post(post_id: int):
    """게시글 상세 조회"""
    for post in posts_db:
        if post["id"] == post_id:
            return post
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# 비밀번호 검증
@app.post("/api/posts/{post_id}/verify")
async def verify_post(post_id: int, data: PasswordVerify):
    """게시글 비밀번호 검증"""
    for post in posts_db:
        if post["id"] == post_id:
            if post["password"] == data.password:
                return {"status": "success", "valid": True}
            else:
                return {"status": "success", "valid": False}
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# 게시글 수정
@app.put("/api/posts/{post_id}")
async def update_post(post_id: int, data: dict):
    """게시글 수정"""
    for post in posts_db:
        if post["id"] == post_id:
            if post["password"] != data.get("password"):
                raise HTTPException(status_code=403, detail="비밀번호가 일치하지 않습니다.")
            
            post["title"] = data.get("title", post["title"])
            post["content"] = data.get("content", post["content"])
            post["target_address"] = data.get("target_address", post.get("target_address", ""))
            
            return {"status": "success"}
    
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# 게시글 삭제
@app.delete("/api/posts/{post_id}")
async def delete_post(post_id: int):
    """게시글 삭제"""
    global posts_db
    
    for i, post in enumerate(posts_db):
        if post["id"] == post_id:
            posts_db.pop(i)
            return {"status": "success"}
    
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# 영상 분석 API
@app.post("/admin/analyze/{post_id}")
async def admin_analyze(post_id: int, videos: List[UploadFile] = File(...)):
    """어드민 영상 분석 요청 - 영상 저장 후 완료 처리"""
    global posts_db
    
    for post in posts_db:
        if post["id"] == post_id:
            post["status"] = "IN_PROGRESS"
            
            # 비디오 파일 저장
            analyzed_videos = []
            for i, video in enumerate(videos):
                ext = os.path.splitext(video.filename)[1] or '.mp4'
                filename = f"analyzed_{post_id}_{i+1}{ext}"
                filepath = os.path.join(UPLOAD_DIR, filename)
                
                # 파일 저장
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(video.file, f)
                
                analyzed_videos.append(f"/video/{filename}")
            
            # 완료 처리
            post["status"] = "COMPLETED"
            post["analyzed_video_path"] = json.dumps(analyzed_videos)
            
            return {"status": "success", "post_id": post_id, "videos": analyzed_videos}
    
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# 분석 중지
@app.post("/admin/stop/{post_id}")
async def stop_analysis(post_id: int):
    """분석 중지"""
    global posts_db
    
    for post in posts_db:
        if post["id"] == post_id:
            if post["status"] == "IN_PROGRESS":
                post["status"] = "PENDING"
            return {"status": "success"}
    
    raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다.")

# 비디오 파일 서빙
@app.get("/video/{filename}")
async def serve_video(filename: str):
    """비디오 파일 제공"""
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="비디오를 찾을 수 없습니다.")

# 정적 파일 마운트
frontend_dir = "/opt/ai/frontend"
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")

# SPA 라우팅: root 경로
@app.get("/")
async def root():
    index_path = "/opt/ai/frontend/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "index.html not found"}

# SPA 라우팅: 모든 다른 GET 요청
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith("api") or full_path.startswith("admin") or full_path.startswith("static") or full_path.startswith("video"):
        raise HTTPException(status_code=404, detail="Not found")
    
    index_path = "/opt/ai/frontend/index.html"
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
