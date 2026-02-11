# 클라우드 서버 기반 보행영상 AI 프로젝트
본 프로젝트는 CCTV 영상 내 포함된 얼굴(Face) 및 번호판(License Plate)을 실시간으로 탐지하여 자동으로 비식별화(모자이크/블러) 처리하는 클라우드 기반 AI 솔루션입니다.

## URL https://aiblur.da2un.store

## 1. 프로젝트 개요
- 개발 기간: 2026.01.08 ~ 2026.01.31
- 주요 목적: 영상 데이터 내 개인정보(얼굴, 번호판) 노출 차단을 통한 법적 규제 준수 및 비식별화 자동화
- 활용 방안 및 기대 효과:
  기존 CCTV분석 과정에서 수작업으로 진행되던 모자이크 처리 업무를 자동화하여 행정 효율성 증대
  YOLOv8 기반 객체 탐지를 통해 영상 속 특정 정보 (예: 얼굴, 번호판) 를 빠르게 식별하여 범죄 예방 및 사후 처리에 기여  

## 2. 주요 기능
- 얼굴 탐지 및 비식별화: YOLO 기반 고정밀 안면 인식 및 가우시안 블러(Gaussian Blur) 적용
- 번호판 탐지 및 비식별화: 다양한 각도의 차량 번호판 영역 탐지 및 실시간 마스킹 처리
- 객체 추적 (Object Tracking): ByteTrack 알고리즘을 적용하여 이동 중인 객체를 끊김 없이 추적 및 모자이크 유지
- 클라우드 스토리지 통합: 분석 완료된 영상의 AWS 서버 내 자동 저장 및 Nginx 기반 웹 스트리밍 제공

## 3. 주요 페이지 소개
1. 메인페이지<br>
<img width="1691" height="897" alt="메인페이지" src="https://github.com/user-attachments/assets/b31e526e-243c-4b81-847c-a43bba599ef0" />

2. 분석요청 글 등록<br>
<img width="1233" height="812" alt="분석요청글등록" src="https://github.com/user-attachments/assets/bbb53236-596a-44fa-a834-b71e3cea65bd" />
<img width="1231" height="697" alt="글등록완료" src="https://github.com/user-attachments/assets/9159e3d2-9549-4ee4-a671-26af96b872c9" />

3. 분석 시작<br>
<img width="1237" height="736" alt="분석시작" src="https://github.com/user-attachments/assets/99ecdca3-7fb6-4d20-b73e-e90d2bc1759f" />

4. 분석 완료<br>
<img width="1232" height="703" alt="분석완료글" src="https://github.com/user-attachments/assets/e6a22607-4bae-44be-8af5-504194570df2" />
<img width="1230" height="873" alt="분석영상보기" src="https://github.com/user-attachments/assets/106607c2-414a-4430-9d0d-3a12b5185cdb" />

5. 관리자 페이지<br>
<img width="1273" height="903" alt="관리자페이지" src="https://github.com/user-attachments/assets/7c28e106-8ef4-4fd5-aad0-98c1a465f29a" />


## 4. 간트 차트
<img width="889" height="290" alt="image" src="https://github.com/user-attachments/assets/e1660703-f7f1-4826-af15-6bb99ee2c854" />
<img width="890" height="327" alt="image" src="https://github.com/user-attachments/assets/e8dd3c2f-a66a-4bf5-82a4-61294f802cb4" />

## 5. 플로우 차트
<img width="531" height="935" alt="image" src="https://github.com/user-attachments/assets/1c413050-394e-4136-a117-88735195c9e4" />

## 6. 기술 스택 (Tech Stack)
- Backend<br>
  Framework: FastAPI<br>
  Database: MySQL, H2DB<br>
  Environment: Miniconda3

- Frontend<br>
  Library: React.js<br>
  Environment: Node.js

- Development Tools<br>
  IDE: VSCode<br>

- AI<br>
  Object Detection: YOLOv8 (객체 탐지 및 비식별화 핵심 모델)<br>
  Face Detection: Haar Cascade (기본 얼굴 탐지 알고리즘 활용)
