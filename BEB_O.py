import numpy as np


def simulate_slotted_aloha_beb(G, num_slots, num_nodes):
    successful_transmissions = 0  # 성공적인 전송의 수
    backoff_counters = np.zeros(num_nodes)  # 각 노드의 backoff 카운터

    for slot in range(num_slots):
        transmitting_nodes = []  # 이 슬롯에서 전송을 시도하는 노드들의 리스트

        for i in range(num_nodes):
            # 만약 backoff 카운터가 0이라면 전송 시도
            if backoff_counters[i] == 0:
                # G 값을 이용해 전송 시도를 확률적으로 결정
                if np.random.rand() < G:
                    transmitting_nodes.append(i)

        # 전송 시도하는 노드가 1개일 때만 성공적인 전송
        if len(transmitting_nodes) == 1:
            successful_transmissions += 1
        elif len(transmitting_nodes) > 1:
            # 충돌이 발생한 노드는 backoff를 증가시킴
            for node in transmitting_nodes:
                # BEB 적용: backoff counter를 2배로 늘린 랜덤 지연 적용
                backoff_counters[node] = np.random.randint(1, 2 ** (int(backoff_counters[node]) + 1))

        # 슬롯이 지날 때마다 모든 노드의 backoff 카운터를 1씩 감소
        backoff_counters = np.maximum(0, backoff_counters - 1)

    # 효율성 계산
    efficiency = successful_transmissions / num_slots
    return efficiency


# 시뮬레이션 실행
G =1  # G값 설정 (예: 0.5)
num_slots = 100000  # 슬롯 수 설정
num_nodes = 70  # 노드 수 설정

efficiency = simulate_slotted_aloha_beb(G, num_slots, num_nodes)
print(f"Efficiency: {efficiency:.4f}")
