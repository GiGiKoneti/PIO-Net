"""
AHB-Lite 5-layer Crossbar Model for RP2040.
4 masters (C0, C1, DMA_R, DMA_W) x 6 slaves (SRAM0-5).
Zero wait-state when distinct banks accessed.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class MasterPort:
    name: str
    # current request
    addr: int = 0
    write: bool = False
    wdata: int = 0
    req: bool = False
    # response
    rdata: int = 0
    ready: bool = False

@dataclass
class SlavePort:
    name: str
    base: int
    size: int
    memory: List[int] = field(default_factory=list)
    busy: bool = False

class AHBCrossbar:
    def __init__(self):
        # Masters
        self.masters = {
            'C0': MasterPort('C0'),
            'C1': MasterPort('C1'),
            'DMA_R': MasterPort('DMA_R'),
            'DMA_W': MasterPort('DMA_W'),
        }
        # Slaves: SRAM banks
        self.slaves = {}
        # SRAM0-3: 64KB each = 16384 words
        for i in range(4):
            base = i * 0x10000  # 64KB
            size = 0x10000
            mem = [0] * (size // 4)
            self.slaves[f'SRAM{i}'] = SlavePort(f'SRAM{i}', base, size, mem)
        # SRAM4,5: 4KB each = 1024 words
        for i in range(4,6):
            base = 0x40000 + (i-4)*0x1000
            size = 0x1000
            mem = [0] * (size // 4)
            self.slaves[f'SRAM{i}'] = SlavePort(f'SRAM{i}', base, size, mem)
        self.contention_cycles = 0

    def _find_slave(self, addr: int) -> Optional[SlavePort]:
        for sl in self.slaves.values():
            if sl.base <= addr < sl.base + sl.size:
                return sl
        return None

    def tick(self):
        # Collect requests per slave
        req_map: Dict[str, List[MasterPort]] = {}
        for m in self.masters.values():
            if m.req:
                sl = self._find_slave(m.addr)
                if sl:
                    req_map.setdefault(sl.name, []).append(m)
        # Resolve contention
        for sl_name, masters in req_map.items():
            if len(masters) > 1:
                self.contention_cycles += 1
            # Service first master
            m = masters[0]
            sl = self.slaves[sl_name]
            sl.busy = True
            if m.write:
                idx = (m.addr - sl.base)//4
                sl.memory[idx] = m.wdata
            else:
                idx = (m.addr - sl.base)//4
                m.rdata = sl.memory[idx]
            m.ready = True
            m.req = False
        # Others not serviced keep req asserted (would retry next cycle)

    # Helper for masters to issue request
    def request(self, master: str, addr: int, write: bool, wdata: int = 0):
        m = self.masters[master]
        m.addr = addr
        m.write = write
        m.wdata = wdata
        m.req = True
        m.ready = False

    def get_response(self, master: str):
        m = self.masters[master]
        return m.ready, m.rdata

if __name__ == "__main__":
    bus = AHBCrossbar()
    # Test zero contention: DMA_R reads SRAM4, C0 reads SRAM0 simultaneously
    bus.request('DMA_R', 0x40000, False)
    bus.request('C0', 0x0, False)
    bus.tick()
    print("Contention cycles:", bus.contention_cycles)  # expect 0
    # Test contention: both to SRAM4
    bus.request('DMA_R', 0x40000, False)
    bus.request('C0', 0x40000, False)
    bus.tick()
    print("Contention cycles:", bus.contention_cycles)  # expect 1