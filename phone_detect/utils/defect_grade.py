"""
Defect Grade Determination

Segmentation pixel을 분석하여 결함 등급을 결정하고, 가장 심한 결함을 선정합니다.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2


class DefectGradeAnalyzer:
    """
    결함 등급 분석 및 결정 클래스
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.grade_priority = config['defect_grading'].get('grade_priority', ['D', 'C', 'B', 'A'])
        self.select_top_n = config['defect_grading'].get('select_top_n', 2)
    
    def analyze_defect_pixels(
        self,
        mask: np.ndarray,
        section: str = "display"
    ) -> Dict[str, Any]:
        """
        Segmentation mask에서 결함 pixel을 분석합니다.
        
        Args:
            mask: Defect mask [H, W] (각 pixel의 등급)
            section: 섹션 이름
        
        Returns:
            Dictionary containing:
                - 'grade_counts': 각 등급별 pixel 수
                - 'grade_areas': 각 등급별 영역 수
                - 'defects': 각 결함 영역 정보
        """
        unique_values, counts = np.unique(mask, return_counts=True)
        
        grade_counts = {}
        grade_areas = {}
        defects = []
        
        # 등급 매핑 (A=0, B=1, C=2, D=3 또는 설정에 따라)
        grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
        for value, count in zip(unique_values, counts):
            if value == 0:  # 배경
                continue
            
            grade = grade_map.get(int(value), 'Unknown')
            grade_counts[grade] = count
            grade_areas[grade] = self._count_regions(mask, int(value))
            
            # 각 결함 영역 정보 추출
            defect_regions = self._extract_defect_regions(mask, int(value))
            for region in defect_regions:
                defects.append({
                    'grade': grade,
                    'pixel_count': region['pixel_count'],
                    'area': region['area'],
                    'bbox': region['bbox'],
                    'contour': region['contour']
                })
        
        return {
            'grade_counts': grade_counts,
            'grade_areas': grade_areas,
            'defects': defects
        }
    
    def _count_regions(self, mask: np.ndarray, class_value: int) -> int:
        """
        특정 클래스의 연결된 영역 개수를 계산합니다.
        
        Args:
            mask: Defect mask [H, W]
            class_value: 클래스 값
        
        Returns:
            연결된 영역 개수
        """
        class_mask = (mask == class_value).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)
    
    def _extract_defect_regions(
        self,
        mask: np.ndarray,
        class_value: int
    ) -> List[Dict]:
        """
        특정 클래스의 모든 결함 영역을 추출합니다.
        
        Args:
            mask: Defect mask [H, W]
            class_value: 클래스 값
        
        Returns:
            결함 영역 정보 리스트
        """
        class_mask = (mask == class_value).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area == 0:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Pixel count 계산
            pixel_count = np.sum(mask[y:y+h, x:x+w] == class_value)
            
            regions.append({
                'pixel_count': pixel_count,
                'area': area,
                'bbox': [x, y, x+w, y+h],
                'contour': contour
            })
        
        return regions
    
    def determine_defect_grade(
        self,
        analysis_result: Dict[str, Any],
        section: str = "display"
    ) -> Dict[str, Any]:
        """
        결함 분석 결과를 바탕으로 최종 등급을 결정합니다.
        
        Args:
            analysis_result: analyze_defect_pixels의 결과
            section: 섹션 이름
        
        Returns:
            Dictionary containing:
                - 'final_grade': 최종 등급 (가장 심한 등급)
                - 'grade_details': 각 등급별 상세 정보
        """
        grade_counts = analysis_result['grade_counts']
        defects = analysis_result['defects']
        
        # 가장 심한 등급 찾기
        final_grade = 'A'  # 기본값 (정상)
        
        for grade in self.grade_priority:
            if grade in grade_counts and grade_counts[grade] > 0:
                final_grade = grade
                break
        
        # 등급별 상세 정보
        grade_details = {}
        for grade in ['A', 'B', 'C', 'D']:
            grade_details[grade] = {
                'pixel_count': grade_counts.get(grade, 0),
                'region_count': analysis_result['grade_areas'].get(grade, 0),
                'has_defect': grade_counts.get(grade, 0) > 0
            }
        
        return {
            'final_grade': final_grade,
            'grade_details': grade_details,
            'all_defects': defects
        }
    
    def select_top_defects(
        self,
        defects: List[Dict],
        top_n: int = 2
    ) -> List[Dict]:
        """
        가장 심한 결함 N개를 선정합니다.
        
        Args:
            defects: 결함 리스트
            top_n: 선정할 개수
        
        Returns:
            가장 심한 결함 N개
        """
        if len(defects) == 0:
            return []
        
        # 등급 우선순위에 따라 정렬
        grade_scores = {grade: idx for idx, grade in enumerate(self.grade_priority)}
        
        # 등급 우선순위, pixel_count 순으로 정렬
        sorted_defects = sorted(
            defects,
            key=lambda x: (
                grade_scores.get(x['grade'], 999),  # 등급 우선순위
                -x['pixel_count']  # pixel_count 내림차순
            )
        )
        
        return sorted_defects[:top_n]
    
    def check_thresholds(
        self,
        analysis_result: Dict[str, Any],
        section: str = "display"
    ) -> Dict[str, bool]:
        """
        각 threshold를 확인하여 결함 여부를 결정합니다.
        
        Args:
            analysis_result: analyze_defect_pixels의 결과
            section: 섹션 이름
        
        Returns:
            Dictionary containing threshold 체크 결과
        """
        thresholds_config = self.config['defect_grading']
        grade_counts = analysis_result['grade_counts']
        
        results = {}
        
        # Pixel count threshold 체크
        pixel_thresholds = thresholds_config.get('pixel_count_thresholds', {})
        for grade in ['B', 'C', 'D']:
            threshold = pixel_thresholds.get(grade, 0)
            count = grade_counts.get(grade, 0)
            results[f'{grade}_pixel_threshold'] = count >= threshold
        
        # Area threshold 체크
        area_thresholds = thresholds_config.get('area_thresholds', {})
        for grade in ['B', 'C', 'D']:
            threshold = area_thresholds.get(grade, 0)
            total_area = sum(
                d['area'] for d in analysis_result['defects']
                if d['grade'] == grade
            )
            results[f'{grade}_area_threshold'] = total_area >= threshold
        
        return results

