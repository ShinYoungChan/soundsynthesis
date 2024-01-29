# 파도 시뮬레이션 소리 합성
* 거품 입자의 물리적 속성을 활용하여 가상 시뮬레이션 장면에 맞는 거품 사운드 합성
* 사운드의 물리적 현상을 기반으로 사운드의 크기를 제어
* 복잡한 3차원 유체의 움직임 대신 2차원으로 투영하여 소리를 합성하고 제어하는 방법

[dbpia 논문](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11140488)

[dbpia 논문](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11213481)

# 프로젝트 기간
* 2022.01 ~ 2022.12

# 역할 및 성과
* 팀 구성: 1명
* 역할: 파도 소리 합성 방법 개발
* 연구 성과: 2022년 한국컴퓨터정보학회 하계학술대회 우수 논문상 수상

# 입자 클러스터링
![image](https://user-images.githubusercontent.com/40080826/230782870-1a3959a6-911f-4097-9612-bbf148d316ff.png)

* 입자의 개수가 많은 경우 정규 격자의 수를 늘리지 않으면 인접 격자를 탐색할때 모든 격자가 묶이는 현상 발생

![image](https://user-images.githubusercontent.com/40080826/230782894-1f462c1c-f86a-4fd8-9add-d38c8983d642.png)

* 일정 기준치를 두고 입자의 개수가 많은 격자를 우선적으로 묶은 뒤 인접 격자를 탐색하는 방식으로 방법을 변경

![image](https://user-images.githubusercontent.com/40080826/230782936-8cddbdaf-7b7b-4ae9-867f-45663fc6ffbd.png)

* 기존 방법으로는 분리되지 않은 파도 입자의 격자들이 분리되어 클러스터링 된 것을 확인할 수 있음.

# 파도 소리 매칭
![image](https://user-images.githubusercontent.com/40080826/230783045-baf780cf-8410-4421-ba91-40ede6eab9d1.png)

![image](https://user-images.githubusercontent.com/40080826/230783070-a5afe8bf-fc1f-41fa-ad84-1250b9b25b9e.png)

* L_map: 파도 입자의 클러스터링 된 구역의 평균 속도
* L_j: 0.2초씩 분할된 파도 소리 데이터의 평균 음량
* period: 0부터 시작하여 요구사항을 충족하는 클립이 없을시 1씩 증가시켜 재탐색

# 소리 감쇠 적용
![image](https://user-images.githubusercontent.com/40080826/230783229-ce93e642-713b-4b38-8425-b71587e9623a.png)

* 클러스터링 된 구역의 속도 방향 -> 소리의 방향을 나타냄.
* 소리의 방향에 따라 소리의 감쇠가 각 구역에서 다르게 적용될 수 있도록 함.

# Time delay
![image](https://user-images.githubusercontent.com/40080826/230783337-87dba5e2-26f5-4a95-8a28-3488f311ace3.png)

![image](https://user-images.githubusercontent.com/40080826/230783374-4f36c96d-2524-4d2f-8fac-04a35730d83a.png)

* 소리가 0.2초씩 진행하고 끊기는 것이 아니라 자연스러운 소리의 합성을 위해 0.2초 구간이 지나도 소리의 감쇠를 적용해 자연스럽게 소리가 사라질 수 있도록 적용.
* (a) 사진은 Time delay를 적용하지 않았을때 사진, (b) 사진은 Time delay를 적용한 사진
