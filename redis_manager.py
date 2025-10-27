"""Redis 상태 관리 모듈"""
import redis
import json
from typing import Any, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class RedisManager:
    """Redis를 통한 상태 관리"""
    
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # 연결 테스트
            self.client.ping()
            logger.info("Redis 연결 성공")
        except redis.ConnectionError as e:
            logger.error(f"Redis 연결 실패: {e}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Redis 연결 상태 확인"""
        if self.client is None:
            return False
        try:
            self.client.ping()
            return True
        except:
            return False
    
    def save_state(self, session_id: str, key: str, value: Any) -> bool:
        """세션 상태 저장"""
        if not self.is_connected():
            logger.warning("Redis 연결 없음 - 상태 저장 실패")
            return False
        
        try:
            redis_key = f"streamlit:state:{session_id}:{key}"
            self.client.set(redis_key, json.dumps(value))
            self.client.expire(redis_key, 3600)
            return True
        except Exception as e:
            logger.error(f"상태 저장 실패: {e}")
            return False
    
    def get_state(self, session_id: str, key: str) -> Optional[Any]:
        """세션 상태 조회"""
        if not self.is_connected():
            return None
        
        try:
            redis_key = f"streamlit:state:{session_id}:{key}"
            value = self.client.get(redis_key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"상태 조회 실패: {e}")
            return None
    
    def delete_state(self, session_id: str, key: str) -> bool:
        """세션 상태 삭제"""
        if not self.is_connected():
            return False
        
        try:
            redis_key = f"streamlit:state:{session_id}:{key}"
            self.client.delete(redis_key)
            return True
        except Exception as e:
            logger.error(f"상태 삭제 실패: {e}")
            return False
    
    def get_all_states(self, session_id: str) -> Dict[str, Any]:
        """세션의 모든 상태 조회"""
        if not self.is_connected():
            return {}
        
        try:
            pattern = f"streamlit:state:{session_id}:*"
            keys = self.client.keys(pattern)
            result = {}
            for key in keys:
                field_name = key.split(":")[-1]
                value = self.client.get(key)
                result[field_name] = json.loads(value) if value else None
            return result
        except Exception as e:
            logger.error(f"전체 상태 조회 실패: {e}")
            return {}
    
    def save_output(self, session_id: str, output_data: Dict[str, Any]) -> bool:
        """화면 출력 결과 저장"""
        if not self.is_connected():
            return False
        
        try:
            redis_key = f"streamlit:output:{session_id}"
            self.client.set(redis_key, json.dumps(output_data))
            self.client.expire(redis_key, 3600)
            return True
        except Exception as e:
            logger.error(f"출력 저장 실패: {e}")
            return False
    
    def get_output(self, session_id: str) -> Optional[Dict[str, Any]]:
        """화면 출력 결과 조회"""
        if not self.is_connected():
            return None
        
        try:
            redis_key = f"streamlit:output:{session_id}"
            value = self.client.get(redis_key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"출력 조회 실패: {e}")
            return None
    
    def save_command(self, session_id: str, command: Dict[str, Any]) -> bool:
        """외부 명령 저장 (큐 방식)"""
        if not self.is_connected():
            return False
        
        try:
            redis_key = f"streamlit:command:{session_id}"
            self.client.lpush(redis_key, json.dumps(command))
            self.client.expire(redis_key, 300)
            return True
        except Exception as e:
            logger.error(f"명령 저장 실패: {e}")
            return False
    
    def get_command(self, session_id: str) -> Optional[Dict[str, Any]]:
        """외부 명령 조회 및 제거 (FIFO)"""
        if not self.is_connected():
            return None
        
        try:
            redis_key = f"streamlit:command:{session_id}"
            value = self.client.rpop(redis_key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"명령 조회 실패: {e}")
            return None


_redis_manager = None


def get_redis_manager() -> RedisManager:
    """RedisManager 싱글톤 인스턴스 반환"""
    global _redis_manager
    if _redis_manager is None:
        _redis_manager = RedisManager()
    return _redis_manager