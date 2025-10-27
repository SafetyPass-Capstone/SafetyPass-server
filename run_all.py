"""모든 서버를 한 번에 실행하는 통합 스크립트"""
import subprocess
import sys
import time
import signal
import os
import webbrowser
from pathlib import Path

# 프로세스 저장용
processes = []


def check_redis():
    """Redis 서버 실행 여부 확인"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
        client.ping()
        print("✅ Redis 서버가 이미 실행 중입니다.")
        return True
    except:
        return False


def start_redis():
    """Redis 서버 시작"""
    if check_redis():
        return None
    
    print("🚀 Redis 서버 시작 중...")
    try:
        # Windows
        if sys.platform == "win32":
            process = subprocess.Popen(
                ["redis-server"],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        # Unix/Linux/Mac
        else:
            process = subprocess.Popen(
                ["redis-server"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setpgrp  # 새 프로세스 그룹 생성
            )
        
        # Redis 시작 대기
        time.sleep(2)
        if check_redis():
            print("✅ Redis 서버 시작 완료")
            return process
        else:
            print("❌ Redis 서버 시작 실패")
            if process:
                process.terminate()
            return None
    except FileNotFoundError:
        print("❌ Redis가 설치되어 있지 않습니다.")
        print("\n설치 방법:")
        print("  • Ubuntu/Debian: sudo apt-get install redis-server")
        print("  • macOS: brew install redis")
        print("  • Windows: https://redis.io/download")
        return None
    except Exception as e:
        print(f"❌ Redis 시작 오류: {e}")
        return None


def start_fastapi():
    """FastAPI 서버 시작"""
    print("🚀 FastAPI 서버 시작 중 (포트 8000)...")
    
    # Windows
    if sys.platform == "win32":
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api_server:app", 
             "--host", "0.0.0.0", "--port", "8000", "--reload"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    # Unix/Linux/Mac
    else:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api_server:app", 
             "--host", "0.0.0.0", "--port", "8000", "--reload"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setpgrp
        )
    
    time.sleep(3)
    print("✅ FastAPI 서버 시작 완료: http://localhost:8000")
    return process


def start_streamlit():
    """Streamlit 앱 시작 (독립 프로세스)"""
    print("🚀 Streamlit 앱 시작 중 (포트 8501)...")
    
    # Windows
    if sys.platform == "win32":
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "main.py",
             "--server.port", "8501"],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    # Unix/Linux/Mac
    else:
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "main.py",
             "--server.port", "8501"],
            preexec_fn=os.setpgrp
        )
    
    time.sleep(5)
    print("✅ Streamlit 앱 시작 완료: http://localhost:8501")
    return process


def cleanup(signum=None, frame=None):
    """모든 프로세스 종료"""
    print("\n\n🛑 서버 종료 중...")
    
    for i, process in enumerate(processes):
        if process and process.poll() is None:
            try:
                # 프로세스 그룹 전체 종료 (자식 프로세스 포함)
                if sys.platform != "win32":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                
                # 정상 종료 대기
                process.wait(timeout=3)
                print(f"✅ 프로세스 {i+1} 종료 완료")
            except subprocess.TimeoutExpired:
                # 강제 종료
                if sys.platform != "win32":
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
                print(f"⚠️  프로세스 {i+1} 강제 종료")
            except Exception as e:
                print(f"⚠️  프로세스 {i+1} 종료 오류: {e}")
    
    print("✅ 모든 서버가 종료되었습니다.")
    sys.exit(0)


def check_port_available(port):
    """포트 사용 가능 여부 확인"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except:
            return False


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 70)
    print(" 🎯 대피 시뮬레이션 통합 서버 시작")
    print("=" * 70 + "\n")
    
    # 포트 체크
    if not check_port_available(8000):
        print("❌ 포트 8000이 이미 사용 중입니다.")
        print("   다른 FastAPI 서버를 종료하거나 포트를 변경하세요.\n")
        return
    
    if not check_port_available(8501):
        print("❌ 포트 8501이 이미 사용 중입니다.")
        print("   다른 Streamlit 앱을 종료하거나 포트를 변경하세요.\n")
        return
    
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # 1. Redis 시작
        print("[ 1/3 ] Redis 서버 시작")
        print("-" * 70)
        redis_process = start_redis()
        if redis_process:
            processes.append(redis_process)
        elif not check_redis():
            print("\n⚠️  경고: Redis 없이 계속 진행합니다.")
            print("   (일부 기능이 제한될 수 있습니다)\n")
            time.sleep(2)
        print()
        
        # 2. FastAPI 시작
        print("[ 2/3 ] FastAPI 서버 시작")
        print("-" * 70)
        fastapi_process = start_fastapi()
        processes.append(fastapi_process)
        print()
        
        # 3. Streamlit 시작
        print("[ 3/3 ] Streamlit 앱 시작")
        print("-" * 70)
        streamlit_process = start_streamlit()
        processes.append(streamlit_process)
        print()
        
        print("=" * 70)
        print(" ✅ 모든 서버가 성공적으로 시작되었습니다!")
        print("=" * 70)
        print("\n📍 접속 주소:")
        print(f"   • Streamlit 대시보드: http://localhost:8501")
        print(f"   • FastAPI 문서: http://localhost:8000/docs")
        print(f"   • FastAPI Swagger: http://localhost:8000/redoc")
        if check_redis():
            print(f"   • Redis: localhost:6379")
        print("\n💡 팁:")
        print("   • I형/T형 변경은 Streamlit 대시보드에서 하세요")
        print("   • API 테스트는 http://localhost:8000/docs 에서 하세요")
        print("=" * 70)
        print("\n⏹️  종료하려면 Ctrl+C를 누르세요...\n")
        
        # 브라우저 자동 열기 (선택)
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:8501')
        except:
            pass
        
        # 프로세스 모니터링
        while True:
            time.sleep(2)
            
            # 프로세스 상태 체크
            for i, process in enumerate(processes):
                if process and process.poll() is not None:
                    print(f"\n⚠️  프로세스 {i+1}이(가) 예기치 않게 종료되었습니다.")
                    print(f"   종료 코드: {process.returncode}")
                    cleanup()
    
    except KeyboardInterrupt:
        cleanup()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        cleanup()


if __name__ == "__main__":
    # 현재 디렉토리 확인
    if not Path("main.py").exists():
        print("❌ main.py 파일을 찾을 수 없습니다.")
        print("   Dashboard-V2-Control 폴더에서 실행해주세요.\n")
        sys.exit(1)
    
    if not Path("api_server.py").exists():
        print("❌ api_server.py 파일을 찾을 수 없습니다.")
        print("   api_server.py를 먼저 생성해주세요.\n")
        sys.exit(1)
    
    main()