"""
Database Module for Kiosk Service

키오스크 서비스에서 추론 결과를 저장하는 데이터베이스 모듈입니다.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import threading


class DatabaseManager:
    """
    데이터베이스 관리 클래스
    
    SQLite를 기본으로 사용하며, 다른 DB로 확장 가능합니다.
    """
    
    def __init__(
        self,
        db_path: str = "data/kiosk_results.db",
        auto_commit: bool = True
    ):
        """
        Args:
            db_path: 데이터베이스 파일 경로
            auto_commit: 자동 커밋 여부
        """
        self.db_path = Path(db_path)
        self.auto_commit = auto_commit
        self.lock = threading.Lock()
        
        # 데이터베이스 초기화
        self._initialize_database()
    
    def _initialize_database(self):
        """데이터베이스 테이블 초기화"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 결과 저장 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inference_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    gender_confidence REAL,
                    age_confidence REAL,
                    head_bbox TEXT,
                    detection_confidence REAL,
                    image_path TEXT,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 인덱스 생성
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_image_id 
                ON inference_results(image_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON inference_results(timestamp)
            ''')
            
            if self.auto_commit:
                conn.commit()
            else:
                conn.commit()
            
            conn.close()
    
    def save_result(
        self,
        image_id: str,
        age: Optional[int] = None,
        gender: Optional[str] = None,
        gender_confidence: Optional[float] = None,
        age_confidence: Optional[float] = None,
        head_bbox: Optional[List[float]] = None,
        detection_confidence: Optional[float] = None,
        image_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        추론 결과를 데이터베이스에 저장
        
        Args:
            image_id: 이미지 ID (고유 식별자)
            age: 추정된 나이
            gender: 추정된 성별 ('Male' 또는 'Female')
            gender_confidence: 성별 신뢰도
            age_confidence: 나이 신뢰도
            head_bbox: Head bounding box [x1, y1, x2, y2]
            detection_confidence: Detection 신뢰도
            image_path: 이미지 파일 경로
            metadata: 추가 메타데이터
        
        Returns:
            저장 성공 여부
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                timestamp = datetime.now().isoformat()
                
                # Bounding box를 JSON 문자열로 변환
                head_bbox_str = json.dumps(head_bbox) if head_bbox else None
                
                # Metadata를 JSON 문자열로 변환
                metadata_str = json.dumps(metadata) if metadata else None
                
                cursor.execute('''
                    INSERT INTO inference_results (
                        image_id, timestamp, age, gender, gender_confidence,
                        age_confidence, head_bbox, detection_confidence,
                        image_path, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id, timestamp, age, gender, gender_confidence,
                    age_confidence, head_bbox_str, detection_confidence,
                    image_path, metadata_str
                ))
                
                if self.auto_commit:
                    conn.commit()
                
                conn.close()
                return True
        
        except Exception as e:
            print(f"Database save error: {e}")
            return False
    
    def get_result(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        이미지 ID로 결과 조회
        
        Args:
            image_id: 이미지 ID
        
        Returns:
            결과 딕셔너리 또는 None
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM inference_results 
                    WHERE image_id = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                ''', (image_id,))
                
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    return self._row_to_dict(row, cursor.description)
                return None
        
        except Exception as e:
            print(f"Database get error: {e}")
            return None
    
    def get_results(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        시간 범위로 결과 조회
        
        Args:
            start_time: 시작 시간 (ISO format)
            end_time: 종료 시간 (ISO format)
            limit: 최대 결과 수
        
        Returns:
            결과 딕셔너리 리스트
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                query = 'SELECT * FROM inference_results WHERE 1=1'
                params = []
                
                if start_time:
                    query += ' AND timestamp >= ?'
                    params.append(start_time)
                
                if end_time:
                    query += ' AND timestamp <= ?'
                    params.append(end_time)
                
                query += ' ORDER BY created_at DESC'
                
                if limit:
                    query += ' LIMIT ?'
                    params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                conn.close()
                
                results = []
                for row in rows:
                    results.append(self._row_to_dict(row, cursor.description))
                
                return results
        
        except Exception as e:
            print(f"Database get_results error: {e}")
            return []
    
    def _row_to_dict(self, row: tuple, columns: tuple) -> Dict[str, Any]:
        """
        DB row를 딕셔너리로 변환
        
        Args:
            row: DB row
            columns: 컬럼 정보
        
        Returns:
            딕셔너리
        """
        result = {}
        
        for i, col in enumerate(columns):
            col_name = col[0]
            value = row[i]
            
            # JSON 문자열 파싱
            if col_name in ['head_bbox', 'metadata'] and value:
                try:
                    value = json.loads(value)
                except:
                    pass
            
            result[col_name] = value
        
        return result
    
    def delete_result(self, image_id: str) -> bool:
        """
        결과 삭제
        
        Args:
            image_id: 이미지 ID
        
        Returns:
            삭제 성공 여부
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM inference_results 
                    WHERE image_id = ?
                ''', (image_id,))
                
                if self.auto_commit:
                    conn.commit()
                
                conn.close()
                return True
        
        except Exception as e:
            print(f"Database delete error: {e}")
            return False
    
if __name__ == "__main__":
    # 테스트 코드
    db = DatabaseManager(db_path=":memory:")  # 메모리 DB 사용
    
    # 결과 저장
    db.save_result(
        image_id="test_001",
        age=25,
        gender="Male",
        gender_confidence=0.95,
        age_confidence=0.88,
        head_bbox=[100, 100, 200, 200],
        detection_confidence=0.92
    )
    
    # 결과 조회
    result = db.get_result("test_001")
    print("Saved result:")
    print(result)

