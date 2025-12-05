"""
Camera Handler with Multi-threaded Capture

키오스크 환경에서 멀티스레드로 여러 장의 사진을 촬영하는 모듈입니다.
"""

import cv2
import threading
import time
from typing import List, Optional, Callable
import numpy as np
from PIL import Image
import queue
from datetime import datetime


class CameraHandler:
    """
    카메라 핸들러 클래스
    
    멀티스레드를 사용하여 빠르게 여러 장의 사진을 촬영합니다.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        buffer_size: int = 1
    ):
        """
        Args:
            camera_id: 카메라 ID (일반적으로 0)
            width: 촬영 해상도 너비
            height: 촬영 해상도 높이
            fps: FPS 설정
            buffer_size: 카메라 버퍼 크기
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.lock = threading.Lock()
        self.capture_thread: Optional[threading.Thread] = None
        
        # 프레임 버퍼 (최신 프레임만 유지)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
    
    def start(self) -> bool:
        """
        카메라 시작
        
        Returns:
            성공 여부
        """
        if self.is_running:
            return True
        
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                return False
            
            # 카메라 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
            self.is_running = True
            
            # 프레임 캡처 스레드 시작
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            # 카메라 초기화 대기
            time.sleep(0.5)
            
            return True
        
        except Exception as e:
            print(f"Camera start error: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """카메라 중지"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def _capture_loop(self):
        """백그라운드 프레임 캡처 루프"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                
                if ret:
                    # RGB로 변환
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    with self.frame_lock:
                        self.latest_frame = frame_rgb.copy()
                else:
                    time.sleep(0.01)  # 읽기 실패 시 짧은 대기
            else:
                time.sleep(0.1)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        최신 프레임 가져오기
        
        Returns:
            최신 프레임 (RGB) 또는 None
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None
    
    def capture_single(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        단일 프레임 캡처
        
        Args:
            timeout: 타임아웃 (초)
        
        Returns:
            캡처된 프레임 (RGB) 또는 None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            frame = self.get_latest_frame()
            if frame is not None:
                return frame
            time.sleep(0.01)
        
        return None
    
    def capture_multiple(
        self,
        num_images: int = 10,
        interval: float = 0.1,
        timeout: float = 5.0
    ) -> List[np.ndarray]:
        """
        여러 장의 사진을 빠르게 연속 촬영 (멀티스레드)
        
        Args:
            num_images: 촬영할 이미지 개수
            interval: 촬영 간격 (초)
            timeout: 전체 타임아웃 (초)
        
        Returns:
            캡처된 이미지 리스트
        """
        if not self.is_running:
            return []
        
        captured_images = []
        start_time = time.time()
        
        # 멀티스레드로 빠르게 촬영
        threads = []
        image_queue = queue.Queue()
        
        def capture_worker(worker_id: int):
            """개별 캡처 워커"""
            time.sleep(worker_id * interval)  # 간격을 두고 촬영
            
            frame = self.capture_single(timeout=0.5)
            if frame is not None:
                image_queue.put((worker_id, frame))
        
        # 여러 스레드 생성 (병렬 촬영)
        for i in range(num_images):
            thread = threading.Thread(target=capture_worker, args=(i,), daemon=True)
            thread.start()
            threads.append(thread)
            
            # 타임아웃 체크
            if time.time() - start_time > timeout:
                break
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join(timeout=interval * num_images + 1.0)
        
        # 결과 수집
        while not image_queue.empty():
            try:
                worker_id, frame = image_queue.get_nowait()
                captured_images.append(frame)
            except queue.Empty:
                break
        
        # worker_id 순서로 정렬
        if len(captured_images) > 0:
            # 실제로는 촬영 시간순으로 정렬
            pass
        
        return captured_images
    
    def capture_multiple_simple(
        self,
        num_images: int = 10,
        interval: float = 0.1
    ) -> List[np.ndarray]:
        """
        단순한 방법으로 여러 장 촬영 (순차적)
        
        Args:
            num_images: 촬영할 이미지 개수
            interval: 촬영 간격 (초)
        
        Returns:
            캡처된 이미지 리스트
        """
        if not self.is_running:
            return []
        
        captured_images = []
        
        for i in range(num_images):
            frame = self.capture_single(timeout=1.0)
            if frame is not None:
                captured_images.append(frame)
            
            if i < num_images - 1:  # 마지막 이미지가 아니면 대기
                time.sleep(interval)
        
        return captured_images
    
    def is_available(self) -> bool:
        """
        카메라가 사용 가능한지 확인
        
        Returns:
            사용 가능 여부
        """
        return self.is_running and self.latest_frame is not None


def simulate_distance_sensor() -> bool:
    """
    거리 센서 시뮬레이션 (20m 이내 감지)
    
    실제 환경에서는 하드웨어 센서를 연결하여 사용합니다.
    
    Returns:
        20m 이내 접근 여부
    """
    # 실제로는 센서 값 읽기
    # 예: distance = read_distance_sensor()
    # return distance < 20.0
    
    # 시뮬레이션: 항상 True 반환 (테스트용)
    return True


if __name__ == "__main__":
    # 테스트 코드
    camera = CameraHandler(camera_id=0)
    
    if camera.start():
        print("Camera started successfully")
        
        # 단일 캡처 테스트
        frame = camera.capture_single()
        if frame is not None:
            print(f"Captured frame shape: {frame.shape}")
        
        # 여러 장 캡처 테스트
        images = camera.capture_multiple_simple(num_images=5, interval=0.1)
        print(f"Captured {len(images)} images")
        
        camera.stop()
    else:
        print("Failed to start camera")

