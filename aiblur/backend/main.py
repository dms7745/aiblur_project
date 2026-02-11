import os
import json
import shutil
import uuid
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import hashlib
import secrets

# ============================================================
# 🚀 MediaPipe 최적화 + MCP 표준 AI 영상 분석 시스템
# ============================================================
# 성능: MediaPipe Face Detection + 프레임 스킵 + 해상도 다운사이징
# 보안: MCP(Model Context Protocol) 표준 인터페이스
# ============================================================

print("=" * 60)
print("🚀 AI Blur System v3.0 - MediaPipe + MCP")
print("=" * 60)

# ============== MediaPipe 초기화 (가벼운 모델) ==============
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FACE_DETECTOR = None
FACE_CASCADE = None  # Fallback

def init_face_detector():
    global FACE_DETECTOR, FACE_CASCADE
    try:
        model_path = "/opt/ai/backend/blaze_face_short_range.tflite"
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            print("📥 Downloading MediaPipe model...")
            urllib.request.urlretrieve(url, model_path)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5
        )
        FACE_DETECTOR = vision.FaceDetector.create_from_options(options)
        print("✅ MediaPipe Face Detector loaded")
    except Exception as e:
        print(f"⚠️ MediaPipe failed: {e}, using Haar Cascade")
        FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

init_face_detector()

# ============== YOLO 번호판 모델 (ByteTrack 추적) ==============
LICENSE_PLATE_MODEL = None
BYTETRACK_CONFIG = "/opt/ai/backend/bytetrack.yaml"
try:
    from ultralytics import YOLO
    # 번호판 전용 모델 사용 (더 정확)
    LICENSE_PLATE_MODEL = YOLO("/opt/ai/backend/yolov8n-license-plate.pt")
    LICENSE_PLATE_MODEL.fuse()  # 모델 최적화
    print("✅ YOLO License Plate model loaded & fused")
    print("✅ ByteTrack tracking enabled")
except Exception as e:
    print(f"⚠️ License plate model: {e}")

# ============== MCP (Model Context Protocol) ==============
# API Key 기반 인증 + 데이터 샌드박싱
MCP_API_KEYS = {
    "mcp_admin_key_2024": {"role": "admin", "permissions": ["read", "write", "analyze"]},
    "mcp_viewer_key_2024": {"role": "viewer", "permissions": ["read"]},
}

# 분석 로그 저장소 (샌드박싱된 리소스)
MCP_ANALYSIS_LOGS: List[Dict[str, Any]] = []
MCP_ALLOWED_RESOURCES = ["/video/", "/logs/", "/analysis/"]

class MCPAuthError(Exception):
    pass

def verify_mcp_key(api_key: str, required_permission: str = "read") -> Dict:
    """MCP API Key 검증 및 권한 확인"""
    if api_key not in MCP_API_KEYS:
        raise MCPAuthError("Invalid MCP API Key")
    
    key_info = MCP_API_KEYS[api_key]
    if required_permission not in key_info["permissions"]:
        raise MCPAuthError(f"Permission denied: {required_permission}")
    
    return key_info

def log_mcp_access(api_key: str, action: str, resource: str, details: Dict = None):
    """MCP 접근 로그 기록 (보안 감사용)"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "api_key_hash": hashlib.sha256(api_key.encode()).hexdigest()[:16],
        "action": action,
        "resource": resource,
        "details": details or {}
    }
    MCP_ANALYSIS_LOGS.append(log_entry)
    # 최근 1000개만 유지
    if len(MCP_ANALYSIS_LOGS) > 1000:
        MCP_ANALYSIS_LOGS.pop(0)

# ============== FastAPI 설정 ==============
app = FastAPI(
    title="AI Blur API + MCP",
    version="3.0.0",
    description="MediaPipe 최적화 + MCP 표준 보안 프로토콜"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/opt/ai/frontend/video"
os.makedirs(UPLOAD_DIR, exist_ok=True)

posts_db = []
post_id_counter = 1
POSTS_DB_FILE = "/opt/ai/backend/posts_data.json"

def load_posts_db():
    """JSON 파일에서 게시글 로드"""
    global posts_db, post_id_counter
    try:
        if os.path.exists(POSTS_DB_FILE):
            with open(POSTS_DB_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                posts_db = data.get("posts", [])
                post_id_counter = data.get("next_id", 1)
                print(f"✅ Loaded {len(posts_db)} posts from {POSTS_DB_FILE}")
        else:
            posts_db = []
            post_id_counter = 1
            print(f"📝 No existing posts file, starting fresh")
    except Exception as e:
        print(f"⚠️ Error loading posts: {e}")
        posts_db = []
        post_id_counter = 1

def save_posts_db():
    print(f"💾 Saving posts to {POSTS_DB_FILE}...")
    """게시글을 JSON 파일에 저장"""
    try:
        with open(POSTS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump({"posts": posts_db, "next_id": post_id_counter}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Error saving posts: {e}")

# 서버 시작 시 로드
load_posts_db()
analysis_tasks = {}
performance_stats = {"total_frames": 0, "total_time": 0, "avg_fps": 0}

class PasswordVerify(BaseModel):
    password: str

# ============== 🚀 최적화된 블러 처리 함수 ==============

def blur_faces_optimized(frame, scale=0.5):
    """
    MediaPipe 최적화 얼굴 블러
    - 해상도 다운사이징으로 속도 향상
    - 타원형 블러 적용
    """
    global FACE_DETECTOR, FACE_CASCADE
    
    h, w = frame.shape[:2]
    face_count = 0
    
    # 🚀 해상도 다운사이징 (속도 4배 향상)
    small_h, small_w = int(h * scale), int(w * scale)
    small_frame = cv2.resize(frame, (small_w, small_h))
    
    if FACE_DETECTOR is not None:
        # MediaPipe 사용
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
        results = FACE_DETECTOR.detect(mp_image)
        
        for detection in results.detections:
            bbox = detection.bounding_box
            # 원본 좌표로 변환
            x = int(bbox.origin_x / scale)
            y = int(bbox.origin_y / scale)
            bw = int(bbox.width / scale)
            bh = int(bbox.height / scale)
            
            # 경계 체크
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            
            if x2 > x and y2 > y:
                apply_ellipse_blur(frame, x, y, x2, y2)
                face_count += 1
    
    elif FACE_CASCADE is not None:
        # Haar Cascade Fallback
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray_small, 1.1, 4, minSize=(20, 20))
        
        for (fx, fy, fw, fh) in faces:
            # 원본 좌표로 변환
            x, y = int(fx / scale), int(fy / scale)
            x2, y2 = int((fx + fw) / scale), int((fy + fh) / scale)
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 > x and y2 > y:
                apply_ellipse_blur(frame, x, y, x2, y2)
                face_count += 1
    
    return frame, face_count

def apply_ellipse_blur(frame, x, y, x2, y2):
    """타원형 블러 적용 (최적화)"""
    roi = frame[y:y2, x:x2]
    if roi.size == 0:
        return
    
    # 강화된 블러: 픽셀화 + 가우시안 이중 처리
    h_roi, w_roi = roi.shape[:2]
    if h_roi > 0 and w_roi > 0:
        # 1차: 픽셀화 (모자이크)
        temp = cv2.resize(roi, (max(1, w_roi//12), max(1, h_roi//12)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
        # 2차: 강한 가우시안 블러
        blurred = cv2.GaussianBlur(pixelated, (99, 99), 30)
    else:
        blurred = roi
    
    # 타원 마스크
    h_roi, w_roi = roi.shape[:2]
    mask = np.zeros((h_roi, w_roi), dtype=np.uint8)
    cv2.ellipse(mask, (w_roi//2, h_roi//2), (w_roi//2, h_roi//2), 0, 0, 360, 255, -1)
    
    # 마스크 적용
    mask_3ch = mask[:, :, np.newaxis] / 255.0
    frame[y:y2, x:x2] = (blurred * mask_3ch + roi * (1 - mask_3ch)).astype(np.uint8)

def blur_plates_optimized(frame, cached_boxes=None):
    """
    YOLO 번호판 블러 (캐시 활용 + ByteTrack)
    - 픽셀화 + 가우시안 이중 블러
    - 10% 영역 확장
    """
    if LICENSE_PLATE_MODEL is None:
        return frame, 0, []
    
    # 캐시된 박스 사용 (프레임 스킵 시)
    if cached_boxes is not None:
        for (x1, y1, x2, y2) in cached_boxes:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                h_roi, w_roi = roi.shape[:2]
                if h_roi > 4 and w_roi > 4:
                    temp = cv2.resize(roi, (max(1, w_roi//10), max(1, h_roi//10)), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = cv2.GaussianBlur(pixelated, (99, 99), 30)
        return frame, len(cached_boxes), cached_boxes
    
    # 새로 감지 (번호판 클래스만 - class 0) + ByteTrack 추적
    try:
        results = LICENSE_PLATE_MODEL.track(frame, verbose=False, conf=0.25, imgsz=640, 
                                            classes=[0], persist=True, 
                                            tracker=BYTETRACK_CONFIG)
    except:
        results = LICENSE_PLATE_MODEL(frame, verbose=False, conf=0.25, imgsz=640, classes=[0])
    
    boxes = []
    
    for result in results:
        for box in result.boxes:
            # 번호판(class 0)만 처리
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 번호판 영역 10% 확장
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(frame.shape[1], x2 + pad_x), min(frame.shape[0], y2 + pad_y)
            
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    h_roi, w_roi = roi.shape[:2]
                    if h_roi > 4 and w_roi > 4:
                        temp = cv2.resize(roi, (max(1, w_roi//10), max(1, h_roi//10)), interpolation=cv2.INTER_LINEAR)
                        pixelated = cv2.resize(temp, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(pixelated, (99, 99), 30)
    
    return frame, len(boxes), boxes

# ============== 🚀 최적화된 영상 처리 ==============

def process_video_v3(input_path: str, output_path: str, post_id: int):
    """
    v3.0 최적화 영상 처리
    - 프레임 스킵: 2배 속도 향상
    - 해상도 다운사이징: 4배 속도 향상
    - 번호판 캐싱: 추가 속도 향상
    - 목표: 15fps → 30fps+
    """
    global analysis_tasks, performance_stats
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open: {input_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 출력 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 🚀 최적화 파라미터
    FRAME_SKIP = 2          # 번호판은 2프레임마다 감지
    DETECT_SCALE = 0.5      # 얼굴 감지는 50% 해상도
    
    frame_count = 0
    total_faces = 0
    total_plates = 0
    cached_plate_boxes = []
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"🎬 Processing: {os.path.basename(input_path)}")
    print(f"   Input: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"   Optimization: SKIP={FRAME_SKIP}, SCALE={DETECT_SCALE}")
    print(f"{'='*50}")
    
    try:
        while True:
            if analysis_tasks.get(post_id) == "STOP":
                print(f"⏹️ Stopped by user")
                return None
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # 🚀 얼굴 블러 (매 프레임, 다운스케일링으로 빠름)
            frame, faces = blur_faces_optimized(frame, scale=DETECT_SCALE)
            total_faces += faces
            
            # 🚀 번호판 블러 (프레임 스킵 + 캐싱)
            if frame_count % FRAME_SKIP == 0:
                frame, plates, cached_plate_boxes = blur_plates_optimized(frame)
                total_plates += plates
            else:
                frame, _, _ = blur_plates_optimized(frame, cached_boxes=cached_plate_boxes)
            
            out.write(frame)
            frame_count += 1
            
            # 진행률 표시
            if frame_count % max(1, total_frames // 5) == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                print(f"   📊 {progress:.0f}% | {current_fps:.1f} fps | Faces: {total_faces}")
        
        # 완료 통계
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        # 성능 통계 업데이트
        performance_stats["total_frames"] += frame_count
        performance_stats["total_time"] += elapsed
        performance_stats["avg_fps"] = performance_stats["total_frames"] / performance_stats["total_time"]
        
        print(f"\n{'='*50}")
        print(f"✅ COMPLETED!")
        print(f"   Frames: {frame_count} in {elapsed:.1f}s")
        print(f"   Speed: {avg_fps:.1f} fps (Target: 30fps)")
        print(f"   Faces: {total_faces}, Plates: {total_plates}")
        print(f"{'='*50}\n")
        
        return output_path
        
    finally:
        cap.release()
        out.release()

# ============== 백그라운드 분석 ==============

def run_analysis(post_id: int, input_videos: List[str]):
    global posts_db, analysis_tasks
    
    try:
        analyzed_videos = []
        
        for i, video_url in enumerate(input_videos):
            # URL 경로를 실제 파일 경로로 변환
            if video_url.startswith("/video/"):
                input_path = os.path.join(UPLOAD_DIR, video_url.replace("/video/", ""))
            else:
                input_path = video_url
            
            print(f"📁 Input path: {input_path}")
            if not os.path.exists(input_path):
                print(f"❌ File not found: {input_path}")
                continue
            output_filename = f"analyzed_{post_id}_{i+1}.mp4"
            output_path = os.path.join(UPLOAD_DIR, output_filename)
            
            result = process_video_v3(input_path, output_path, post_id)
            
            if result is None:
                for post in posts_db:
                    if post["id"] == post_id:
                        post["status"] = "PENDING"
                return
            
            analyzed_videos.append(f"/video/{output_filename}")
            
            # MCP 로그 기록
            log_mcp_access("system", "analyze_complete", f"/video/{output_filename}", {
                "post_id": post_id,
                "input": os.path.basename(input_path)
            })
        
        for post in posts_db:
            if post["id"] == post_id:
                post["status"] = "COMPLETED"
                post["analyzed_video_path"] = json.dumps(analyzed_videos)
        
        print(f"🎉 Analysis completed for post {post_id}")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        for post in posts_db:
            if post["id"] == post_id:
                post["status"] = "PENDING"
    finally:
        analysis_tasks.pop(post_id, None)

executor = ThreadPoolExecutor(max_workers=2)

# ============== MCP API 엔드포인트 ==============

@app.get("/mcp/status")
async def mcp_status():
    """MCP 서버 상태 확인"""
    return {
        "protocol": "MCP",
        "version": "1.0",
        "status": "online",
        "capabilities": ["face_blur", "plate_blur", "video_analysis"],
        "performance": performance_stats
    }

@app.get("/mcp/logs")
async def mcp_get_logs(x_api_key: str = Header(None)):
    """MCP 분석 로그 조회 (인증 필요)"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="MCP API Key required")
    
    try:
        key_info = verify_mcp_key(x_api_key, "read")
        log_mcp_access(x_api_key, "read_logs", "/mcp/logs")
        return {"logs": MCP_ANALYSIS_LOGS[-100:], "total": len(MCP_ANALYSIS_LOGS)}
    except MCPAuthError as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.get("/mcp/analysis/{post_id}")
async def mcp_get_analysis(post_id: int, x_api_key: str = Header(None)):
    """MCP 분석 결과 조회 (샌드박싱)"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="MCP API Key required")
    
    try:
        key_info = verify_mcp_key(x_api_key, "read")
        
        for post in posts_db:
            if post["id"] == post_id:
                # 샌드박싱: 민감 정보 제외
                safe_data = {
                    "id": post["id"],
                    "title": post["title"],
                    "status": post["status"],
                    "created_at": post["created_at"],
                    "analyzed_video_path": post.get("analyzed_video_path")
                }
                log_mcp_access(x_api_key, "read_analysis", f"/mcp/analysis/{post_id}")
                return safe_data
        
        raise HTTPException(status_code=404, detail="Not found")
    except MCPAuthError as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.post("/mcp/analyze")
async def mcp_trigger_analysis(post_id: int, x_api_key: str = Header(None)):
    """MCP를 통한 분석 트리거 (권한 필요)"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="MCP API Key required")
    
    try:
        key_info = verify_mcp_key(x_api_key, "analyze")
        log_mcp_access(x_api_key, "trigger_analysis", f"/mcp/analyze/{post_id}")
        
        # 분석 트리거 로직
        for post in posts_db:
            if post["id"] == post_id and post["status"] == "PENDING":
                return {"status": "ready", "message": "Use /admin/analyze endpoint with video"}
        
        return {"status": "not_available"}
    except MCPAuthError as e:
        raise HTTPException(status_code=403, detail=str(e))

# ============== 기존 API ==============

@app.post("/api/verify-password")
async def verify_password(data: PasswordVerify):
    return {"status": "success", "valid": data.password == "admin1234"}


@app.post("/request-analysis/")
async def request_analysis(
    title: str = Form(...),
    author: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    content: str = Form(default=""),
    target_address: str = Form(default=""),
    phone: str = Form(default=""),
):
    """민원 접수 (프론트엔드 폼 전용 - 영상 없이 접수만)"""
    global posts_db, post_id_counter
    
    new_post = {
        "id": post_id_counter, "name": author, "author": author,
        "phone": phone, "email": email, "title": title, "content": content,
        "password": password, "status": "PENDING",
        "created_at": datetime.now().isoformat(),
        "videos": "[]",
        "original_video_filename": "",
        "analyzed_video_path": None, "target_address": target_address
    }
    posts_db.append(new_post)
    post_id = post_id_counter
    post_id_counter += 1
    save_posts_db()
    
    return {"status": "success", "post_id": post_id}

@app.post("/api/posts")
async def create_post(
    title: str = Form(...),
    author: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    content: str = Form(default=""),
    target_address: str = Form(default=""),
    phone: str = Form(default=""),
    videos: Optional[List[UploadFile]] = File(default=None)
):
    """민원 글 등록 + 자동 AI 분석"""
    global posts_db, post_id_counter
    
    saved_videos, original_filenames = [], []
    if videos:
        for video in videos:
            if video.filename:
                ext = os.path.splitext(video.filename)[1] or '.mp4'
                filename = f"original_{post_id_counter}_{uuid.uuid4().hex[:8]}{ext}"
                filepath = os.path.join(UPLOAD_DIR, filename)
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(video.file, f)
                saved_videos.append(f"/video/{filename}")
                original_filenames.append(video.filename)
    
    new_post = {
        "id": post_id_counter, "name": author, "author": author,
        "phone": phone, "email": email, "title": title, "content": content,
        "password": password, "status": "IN_PROGRESS" if saved_videos else "PENDING",
        "created_at": datetime.now().isoformat(),
        "videos": json.dumps(saved_videos) if saved_videos else "[]",
        "original_video_filename": ", ".join(original_filenames),
        "analyzed_video_path": None, "target_address": target_address
    }
    posts_db.append(new_post)
    post_id = post_id_counter
    post_id_counter += 1
    save_posts_db()
    
    # 🚀 영상이 있으면 자동으로 AI 분석 시작
    if saved_videos:
        import threading
        def auto_analyze():
            try:
                print(f"🚀 Auto-analysis starting for post {post_id}")
                run_analysis(post_id, saved_videos)
                print(f"✅ Auto-analysis completed for post {post_id}")
            except Exception as e:
                print(f"❌ Auto analysis error for post {post_id}: {e}")
                for p in posts_db:
                    if p["id"] == post_id:
                        p["status"] = "ERROR"
                        break
        threading.Thread(target=auto_analyze, daemon=True).start()
    
    return {"status": "success", "post_id": post_id}



@app.get("/api/posts")
async def get_posts(page: int = 1, search: str = "", status_filter: str = ""):
    filtered = [p for p in posts_db if (not search or search.lower() in p.get("title", "").lower())
                and (not status_filter or p["status"] == status_filter)]
    filtered = sorted(filtered, key=lambda x: x["id"], reverse=True)
    per_page, start = 10, (page - 1) * 10
    import json
    data = {
        "posts": [{"id": p["id"], "author": p.get("author", "Unknown"), "name": p.get("name", ""),
                   "title": p["title"], "email": p["email"], "status": p["status"],
                   "created_at": p["created_at"]} for p in filtered[start:start+per_page]],
        "total_posts": len(filtered), "total_pages": (len(filtered) + 9) // 10, "current_page": page
    }
    return Response(content=json.dumps(data, ensure_ascii=False), media_type="application/json; charset=utf-8")

@app.get("/api/posts/{post_id}")
async def get_post(post_id: int):
    for post in posts_db:
        if post["id"] == post_id:
            return post
    raise HTTPException(status_code=404, detail="Not found")

@app.post("/api/posts/{post_id}/verify")
async def verify_post(post_id: int, data: PasswordVerify):
    for post in posts_db:
        if post["id"] == post_id:
            return {"status": "success", "valid": post["password"] == data.password}
    raise HTTPException(status_code=404, detail="Not found")

@app.put("/api/posts/{post_id}")
async def update_post(post_id: int, data: dict):
    for post in posts_db:
        if post["id"] == post_id:
            if post["password"] != data.get("password"):
                raise HTTPException(status_code=403, detail="Password mismatch")
            post.update({k: data.get(k, post.get(k)) for k in ["title", "content", "target_address"]})
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="Not found")

@app.delete("/api/posts/{post_id}")
async def delete_post(post_id: int):
    global posts_db
    for i, post in enumerate(posts_db):
        if post["id"] == post_id:
            posts_db.pop(i)
            save_posts_db()
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="Not found")

@app.post("/admin/analyze/{post_id}")
async def admin_analyze(post_id: int, videos: List[UploadFile] = File(...)):
    global posts_db, analysis_tasks
    for post in posts_db:
        if post["id"] == post_id:
            post["status"] = "IN_PROGRESS"
            input_videos = []
            for i, video in enumerate(videos):
                ext = os.path.splitext(video.filename)[1] or '.mp4'
                filepath = os.path.join(UPLOAD_DIR, f"input_{post_id}_{i+1}{ext}")
                with open(filepath, "wb") as f:
                    shutil.copyfileobj(video.file, f)
                input_videos.append(filepath)
            analysis_tasks[post_id] = "RUNNING"
            executor.submit(run_analysis, post_id, input_videos)
            return {"status": "success", "message": "Analysis started", "post_id": post_id}
    raise HTTPException(status_code=404, detail="Not found")

@app.post("/admin/stop/{post_id}")
async def stop_analysis(post_id: int):
    for post in posts_db:
        if post["id"] == post_id:
            if post["status"] == "IN_PROGRESS":
                analysis_tasks[post_id] = "STOP"
            return {"status": "success"}
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/admin/status/{post_id}")
async def get_analysis_status(post_id: int):
    for post in posts_db:
        if post["id"] == post_id:
            return {"status": post["status"], "analyzed_video_path": post.get("analyzed_video_path")}
    raise HTTPException(status_code=404, detail="Not found")

@app.get("/video/{filename}")
async def serve_video(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Not found")

# 정적 파일
frontend_dir = "/opt/ai/frontend"
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_dir, "static")), name="static")

@app.get("/")
async def root():
    index_path = "/opt/ai/frontend/index.html"
    return FileResponse(index_path) if os.path.exists(index_path) else {"message": "Not found"}

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    if full_path.startswith(("api", "admin", "static", "video", "mcp")):
        raise HTTPException(status_code=404, detail="Not found")
    index_path = "/opt/ai/frontend/index.html"
    return FileResponse(index_path) if os.path.exists(index_path) else HTTPException(404)

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting AI Blur Server with MCP...")
    print("   MCP Endpoint: /mcp/status")
    print("   API Keys: mcp_admin_key_2024, mcp_viewer_key_2024\n")
    uvicorn.run(app, host="0.0.0.0", port=8003)
