"""
Phone Defect Detection Service

Front와 Side 영역의 결함 검출 서비스를 제공합니다.
"""

from .front_service import FrontService
from .side_service import SideService

__all__ = ['FrontService', 'SideService']

