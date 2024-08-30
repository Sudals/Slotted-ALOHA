import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TSLOTS = 100000
MAX_COLLISIONS = 8  # 최대 충돌 횟수
MIN_COLLISIONS = 0
class packet:
    def __init__(self):
        self.ttl = random.randrange(0, 4)  # 전송 대기 시간 (슬롯 수)
        self.collision_count = 2
        self.cw = 2
    def BEB(self):
        if self.collision_count <= MAX_COLLISIONS:
            self.collision_count += 1
            self.ttl = random.randrange(0, (self.cw**self.collision_count)-1)
            return self.ttl
        else:
            # 충돌 횟수가 최대치에 도달하면 패킷 폐기 및 재생성
            self.collision_count = 2
            return -1
class classNode:
    def __init__(self, ttl):
        self.ttl =random.randrange(0,4)  # 전송 대기 시간 (슬롯 수)
        self.collision_count = 2
        self.cw = 2
        self.packet_active = False  # 패킷 활성화 상태
        self.count = 0
        self.pq = []
    def tick(self):
        self.pq[0].ttl -= 1

    def send(self):
        self.pq.pop(0)
        if len(self.pq)==0:
            self.packet_active=False

    def generate_new_packet(self):
        self.pq.append(packet())
        self.count +=1
        self.packet_active = True  # 패킷 활성화 상태로 전환
    def remove(self):
        self.pq.pop(0)
        if len(self.pq)==0:
            self.packet_active=False
def main():
    random.seed()
    np.random.seed()  # Numpy 랜덤 시드 설정

    # 결과를 저장할 DataFrame 생성
    results_df = pd.DataFrame(columns=['N', 'G', 'Slot Efficiency', '4', '8', '16', '32','64','128','256'])

    for window_size in [256]:
        Nlist = []
        glist = []
        selist = []
        print(f"Window size: {window_size}")

        for G in [0.1,0.38,0.5,1]:
            successful_slots = 0
            collison_slots = 0
            total_packets = 0
            remove_packets=0
            for N in [70]:
                # 각 G 값에 대해 노드 생성
                snode = [classNode(random.randrange(0, N)) for _ in range(N)]


                for slot in range(TSLOTS):
                    transmitted_nodes = []
                    packets_to_transmit = np.random.poisson(G)  # 포아송 분포를 사용하여 패킷 생성 수 결정

                    if packets_to_transmit > N:
                        packets_to_transmit = N  # 노드 수를 초과할 수 없으므로 제한
                    total_packets +=packets_to_transmit

                    nodes_to_transmit = random.sample(range(N), packets_to_transmit)

                    for i in range(N):
                        if i in nodes_to_transmit:
                            snode[i].generate_new_packet()
                        elif snode[i].packet_active:
                            if 0==snode[i].pq[0].ttl:
                                transmitted_nodes.append(i)
                            else:
                                snode[i].tick()

                    if len(transmitted_nodes) == 1:
                        successful_slots += 1
                        snode[transmitted_nodes[0]].send()

                    elif len(transmitted_nodes) > 1:
                        collison_slots += 1
                        for j in transmitted_nodes:
                            if snode[j].pq[0].BEB() == -1:
                                snode[j].remove()
                                remove_packets+=1
                slot_efficiency = successful_slots / TSLOTS
                collison_efficiency = collison_slots / TSLOTS
                print(f"{successful_slots} , {total_packets} , {collison_slots} , {remove_packets}")
                print(f"N = {N:2d}, G = {G:.1f}: {slot_efficiency:.4f} : {collison_efficiency:.4f}")


    return

if __name__ == "__main__":
    main()
