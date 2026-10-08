"""
DMA Ping-Pong Simulator for RP2040.
Models two channels chained together, DREQ pacing from PIO.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from pio_sim.pio_sim import PIOStateMachine

@dataclass
class DMAChannel:
    chan_id: int
    read_addr: int
    write_addr: int
    transfer_count: int
    read_inc: bool
    write_inc: bool
    dreq_callback: Optional[Callable[[], bool]] = None  # returns True when ready
    chain_to: Optional[int] = None
    active: bool = False
    _pos: int = 0

    def can_transfer(self) -> bool:
        if self.dreq_callback:
            return self.dreq_callback()
        return True

    def step(self, memory: List[int]) -> bool:
        """Perform one 32-bit transfer if possible. Returns True if transfer done."""
        if not self.active or self._pos >= self.transfer_count:
            return False
        if not self.can_transfer():
            return False
        # read
        val = memory[self.read_addr // 4]
        # write
        memory[self.write_addr // 4] = val
        # increment
        if self.read_inc:
            self.read_addr += 4
        if self.write_inc:
            self.write_addr += 4
        self._pos += 1
        if self._pos >= self.transfer_count:
            self.active = False
            # trigger chained channel
            return True
        return False

class DMASimulator:
    def __init__(self, memory_size_words: int = 200000):
        self.memory = [0] * memory_size_words  # ~800KB
        self.channels: List[Optional[DMAChannel]] = [None] * 12  # RP2040 has 12 channels
        self.pio: Optional[PIOStateMachine] = None

    def claim_channel(self, chan_id: int, config: DMAChannel):
        self.channels[chan_id] = config

    def set_pio(self, pio: PIOStateMachine):
        self.pio = pio

    def tick(self):
        for ch in self.channels:
            if ch and ch.active:
                done = ch.step(self.memory)
                if done and ch.chain_to is not None:
                    nxt = self.channels[ch.chain_to]
                    if nxt:
                        nxt.active = True
                        nxt._pos = 0

    # Helper to create ping-pong config matching blueprint
    def setup_pionet_pingpong(self, weights_base: int, activations_base: int, length: int = 1024):
        # Channel 0: weights from SRAM4 -> PIO TX FIFO (write_addr = PIO TX register)
        # Channel 1: PIO RX FIFO -> SRAM5
        # We'll simulate DREQ by checking PIO FIFO space
        def dreq_tx_ready():
            return len(self.pio.tx_fifo) < self.pio.fifo_depth if self.pio else False
        def dreq_rx_ready():
            return len(self.pio.rx_fifo) > 0 if self.pio else False

        # Use memory-mapped buffers for TX/RX in simulation
        tx_buf_addr = 0x80000
        rx_buf_addr = 0x81000
        ch0 = DMAChannel(
            chan_id=0,
            read_addr=weights_base,
            write_addr=tx_buf_addr,
            transfer_count=length,
            read_inc=True,
            write_inc=False,
            dreq_callback=dreq_tx_ready,
            chain_to=1,
            active=False
        )
        ch1 = DMAChannel(
            chan_id=1,
            read_addr=rx_buf_addr,
            write_addr=activations_base,
            transfer_count=length,
            read_inc=False,
            write_inc=True,
            dreq_callback=dreq_rx_ready,
            chain_to=0,
            active=True
        )
        # Pre-fill RX buffer with dummy data so channel 1 can read
        for i in range(length):
            self.memory[rx_buf_addr//4 + i] = 0xdeadbeef + i
        self.claim_channel(0, ch0)
        self.claim_channel(1, ch1)

if __name__ == "__main__":
    dma = DMASimulator()
    # quick sanity
    dma.memory[0x1000//4:0x1000//4+4] = [0xdeadbeef, 0xcafebabe, 0x12345678, 0x0]
    ch = DMAChannel(0, 0x1000, 0x2000, 4, True, True)
    dma.claim_channel(0, ch)
    ch.active = True
    for _ in range(5):
        dma.tick()
    print(dma.memory[0x2000//4:0x2000//4+4])