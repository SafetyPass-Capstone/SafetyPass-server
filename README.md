# 기술 스택

| 분류 | 기술 |
|------|------|
| 프레임워크 | Flutter (Dart 3.4.3+) |
| 상태관리 | Provider 6.0.5 (ChangeNotifier 패턴) |
| 라우팅 | GoRouter 13.1.0 |
| HTTP 클라이언트 | Dio 5.3.2 |
| 의존성 주입 | GetIt 7.7.0 (Service Locator) |
| 모델 코드 생성 | Freezed + json_serializable |
| WebView | webview_flutter 4.4.0 |
| QR 스캔 | mobile_scanner 3.5.0 |
| 환경변수 | flutter_dotenv |
| SVG | flutter_svg |
| CSV 파싱 | csv (스타디움 좌석 데이터) |
| 폰트 | Pretendard, Alkatra |

  
  # 아키텍처

```
  lib/
  ├── constants/       # 색상, API 경로, 라우트 상수
  ├── data/
  │   ├── di/          # ApiClient 싱글톤 (Dio)
  │   ├── dto/         # Freezed 기반 Response 모델
  │   ├── service/     # API 호출 서비스
  │   └── repository/  # (미사용에 가까움)
  ├── models/          # 도메인 모델 (TicketInfo, StadiumMap 등)
  ├── stores/          # 전역 상태 (EventStore)
  ├── screens/         # 화면 + Provider 로컬 상태
  ├── widgets/
  │   ├── atoms/       # 최소 단위 UI (텍스트 등)
  │   ├── cards/       # 복합 컴포넌트
  │   └── organisms/   # 복잡한 컴포넌트 (StadiumMapWidget 등)
  └── route/           # GoRouter 설정
```

  # 주요 패턴

  - 상태관리: 전역 상태는 EventStore(Provider), 화면별 로컬 상태는 각
  *Provider로 분리
  - 데이터 계층: Service → Dto → Model 흐름, Freezed로 불변 모델 보장
  - 실시간 폴링: 긴급 모드 진입 시 1초 간격으로 대피 정보 API 호출
  (Timer.periodic)
  - 3D 시각화: 외부 웹페이지를 WebView로 임베드, JavaScript DOM 조작으로
  불필요한 UI 제거
  - 스타디움 데이터: CSV 파일을 런타임에 파싱해 좌석/출구 노드 구성
