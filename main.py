import random
import numpy as np
import pandas as pd

TSLOTS = 100000
MAX_COLLISIONS = 6  # 최대 충돌 횟수

class classNode:
    def __init__(self):
        self.ttl = 0  # 초기 대기 시간 설정 없음 (즉시 전송)
        self.collision_count = 0
        self.packet_active = False  # 처음에는 패킷 비활성화 상태

    def tick(self):
        if self.ttl > 0:
            self.ttl -= 1

    def reset_ttl(self):
        if self.collision_count < MAX_COLLISIONS:
            self.collision_count += 1
            window_size = 2 ** self.collision_count  # 지수적으로 증가하는 window_size
            self.ttl = random.randrange(0, window_size)
        else:
            # 충돌 횟수가 최대치에 도달하면 패킷 비활성화 및 재생성 대기
            self.packet_active = False

    def generate_new_packet(self):
        self.ttl = 0  # 즉시 전송하도록 설정
        self.collision_count = 0
        self.packet_active = True  # 패킷 활성화 상태로 전환

def main():
    random.seed()
    np.random.seed()  # Numpy 랜덤 시드 설정

    # 결과를 저장할 DataFrame 생성
    results_df = pd.DataFrame(columns=['N', 'G', 'Throughput', 'Total_Packets', 'Total_Collisions', 'Total_Successes'])

    for N in [70]:  # 노드 수 설정
        for G in [0.1, 0.3, 0.5, 0.7, 1.0]:  # G 값 설정
            successful_slots = 0
            total_packets = 0
            total_collisions = 0

            for _ in range(10):  # 각 설정에 대해 여러 번 시뮬레이션 수행
                snode = [classNode() for _ in range(N)]

                for slot in range(TSLOTS):
                    transmitted_nodes = []
                    packets_to_transmit = np.random.poisson(G)  # 평균 G를 가진 포아송 분포에서 패킷 생성 수 결정

                    # 패킷 생성 수를 노드 수로 제한
                    nodes_to_transmit = np.random.choice(range(N), size=min(packets_to_transmit, N), replace=False)

                    for i in range(N):
                        if i in nodes_to_transmit:
                            if snode[i].ttl == 0:  # 즉시 전송
                                if snode[i].packet_active:
                                    transmitted_nodes.append(i)
                                else:
                                    snode[i].generate_new_packet()
                            else:
                                snode[i].tick()

                    total_packets += len(nodes_to_transmit)  # 생성된 패킷 수 누적

                    if len(transmitted_nodes) == 1:
                        successful_slots += 1
                        snode[transmitted_nodes[0]].packet_active = False  # 성공한 패킷 비활성화
                    elif len(transmitted_nodes) > 1:
                        for j in transmitted_nodes:
                            snode[j].reset_ttl()
                            total_collisions += 1  # 충돌 횟수 누적

            # 총 슬롯 수에 대한 성공한 슬롯 비율로 처리량 계산
            throughput = successful_slots / float(TSLOTS) if TSLOTS > 0 else 0
            print(f"N = {N:2d}, G = {G:.1f}: Throughput = {throughput:.4f}, Total Packets = {total_packets}, Total Collisions = {total_collisions}, Total Successes = {successful_slots}")

            # DataFrame에 결과 추가
            results_df = results_df.append({'N': N, 'G': G, 'Throughput': throughput,
                                             'Total_Packets': total_packets,
                                             'Total_Collisions': total_collisions,
                                             'Total_Successes': successful_slots}, ignore_index=True)

    # DataFrame을 CSV 파일로 저장
    results_df.to_csv('throughput_results_detailed.csv', index=False)

if __name__ == "__main__":
    main()
