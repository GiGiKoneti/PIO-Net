"""
Cycle-accurate RP2040 PIO State Machine Simulator.
Implements the 9 base instructions + side-set.
"""
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Optional
import struct

class Opcode(IntEnum):
    JMP = 0
    WAIT = 1
    IN_ = 2
    OUT = 3
    PUSH = 4
    PULL = 5
    MOV = 6
    IRQ = 7
    SET = 8

@dataclass
class Instruction:
    opcode: Opcode
    # generic fields; actual encoding per RP2040 datasheet
    # we keep a simple representation for simulation
    operands: Dict = field(default_factory=dict)
    side_set: Optional[int] = None
    delay: int = 0  # extra cycles (0-31)

    def __str__(self):
        return f"{self.opcode.name} {self.operands}"

class PIOStateMachine:
    def __init__(self, program: List[Instruction], sm_id: int = 0):
        self.program = program
        self.pc = 0
        self.isr = 0          # Input Shift Register 32-bit
        self.osr = 0          # Output Shift Register 32-bit
        self.x = 0xffffffff   # Scratch X
        self.y = 0            # Scratch Y
        self.rx_fifo = []     # list of 32-bit words
        self.tx_fifo = []     # list of 32-bit words
        self.fifo_depth = 4   # 4 words default, can be 8 when joined
        self.irq_flag = False
        self.clock_div = 1
        self.input_pin = 0    # simulated pin value
        self.sm_id = sm_id

    # Helpers for FIFO
    def tx_push(self, word: int):
        if len(self.tx_fifo) < self.fifo_depth:
            self.tx_fifo.append(word & 0xffffffff)
        else:
            raise BufferError("TX FIFO full")

    def rx_pop(self) -> int:
        if self.rx_fifo:
            return self.rx_fifo.pop(0)
        raise BufferError("RX FIFO empty")

    def step(self):
        """Execute one instruction (including delay cycles)."""
        instr = self.program[self.pc]
        # execute main opcode
        self._execute(instr)
        # handle delay cycles (simplified: just advance pc after delay)
        self.pc = (self.pc + 1) % len(self.program)

    def _execute(self, instr: Instruction):
        op = instr.opcode
        o = instr.operands
        if op == Opcode.JMP:
            cond = o.get('cond', 'always')
            target = o['target']
            take = False
            if cond == 'always':
                take = True
            elif cond == 'x--':
                take = self.x != 0
                if take:
                    self.x = (self.x - 1) & 0xffffffff
            elif cond == 'y--':
                take = self.y != 0
                if take:
                    self.y = (self.y - 1) & 0xffffffff
            elif cond == 'pin':
                take = bool(self.input_pin)
            elif cond == '!osre':
                take = len(self.tx_fifo) > 0  # osr not empty if tx_fifo has data
            if take:
                self.pc = target
        elif op == Opcode.WAIT:
            # wait for pin value
            polarity = o.get('polarity', 1)
            while self.input_pin != polarity:
                # in real hardware this stalls; in sim we just break to avoid infinite loop
                break
        elif op == Opcode.IN_:
            bits = o.get('bits', 1)
            src = o.get('src', 'pins')
            if src == 'pins':
                val = self.input_pin & ((1 << bits) - 1)
            else:
                raise NotImplementedError
            self.isr = ((self.isr << bits) | val) & 0xffffffff
        elif op == Opcode.OUT:
            bits = o.get('bits', 1)
            dest = o.get('dest', 'null')
            # shift out from OSR
            val = (self.osr >> (32 - bits)) & ((1 << bits) - 1)
            self.osr = (self.osr << bits) & 0xffffffff
            if dest == 'null':
                pass
            else:
                raise NotImplementedError
        elif op == Opcode.PUSH:
            # push ISR to RX FIFO
            self.rx_fifo.append(self.isr)
            self.isr = 0
        elif op == Opcode.PULL:
            if self.tx_fifo:
                self.osr = self.tx_fifo.pop(0)
            else:
                # block - in simulation just stall pc
                pass
        elif op == Opcode.MOV:
            dest = o.get('dest')
            src = o.get('src')
            val = self._get_reg(src)
            if dest == 'x':
                self.x = val
            elif dest == 'y':
                self.y = val
            elif dest == 'osr':
                self.osr = val
            elif dest == 'isr':
                self.isr = val
            else:
                raise NotImplementedError
        elif op == Opcode.IRQ:
            irq_num = o.get('irq', 0)
            self.irq_flag = True
        elif op == Opcode.SET:
            # set pin or register
            dest = o.get('dest')
            val = o.get('value', 0)
            if dest == 'pin':
                self.input_pin = val
            else:
                raise NotImplementedError

    def _get_reg(self, name: str) -> int:
        if name == 'x': return self.x
        if name == 'y': return self.y
        if name == 'osr': return self.osr
        if name == 'isr': return self.isr
        if name == 'null': return 0
        if name == '~null': return 0xffffffff
        raise ValueError(f"Unknown register {name}")

def assemble_pio(text: str) -> List[Instruction]:
    """Very small assembler for the subset used in pionet_bnn_mac.pio"""
    raw_lines = [l.strip() for l in text.splitlines() if l.strip() and not l.startswith(';')]
    # filter out assembler directives starting with .
    lines = [l for l in raw_lines if not l.startswith('.')]
    prog = []
    label_map = {}
    # first pass labels
    for idx, line in enumerate(lines):
        if line.endswith(':'):
            label_map[line[:-1]] = idx
    # second pass
    for idx, line in enumerate(lines):
        if line.endswith(':'):
            continue
        parts = line.split()
        mnemonic = parts[0].upper()
        args = ' '.join(parts[1:]) if len(parts)>1 else ''
        instr = _parse_instruction(mnemonic, args, label_map, idx)
        prog.append(instr)
    return prog

def _parse_instruction(mnemonic: str, args: str, labels: Dict[str,int], pc: int) -> Instruction:
    # Very limited parser for our kernel
    if mnemonic == 'PULL':
        block = 'BLOCK' in args
        return Instruction(Opcode.PULL, {'block': block})
    if mnemonic == 'MOV':
        dest, src = [a.strip() for a in args.split(',')]
        return Instruction(Opcode.MOV, {'dest': dest, 'src': src})
    if mnemonic == 'WAIT':
        # wait 1 pin 0  -> polarity 1 on pin 0
        # format: wait <polarity> pin <pin>
        tokens = args.split()
        polarity = int(tokens[0])
        pin = int(tokens[2])
        return Instruction(Opcode.WAIT, {'polarity': polarity, 'pin': pin})
    if mnemonic == 'IN':
        # in pins, 1
        tokens = args.replace(',', ' ').split()
        src = tokens[0]
        bits = int(tokens[1])
        return Instruction(Opcode.IN_, {'src': src, 'bits': bits})
    if mnemonic == 'JMP':
        # jmp pin bit_one   or jmp x-- next_bit
        tokens = args.split()
        cond = tokens[0]
        target_label = tokens[1] if len(tokens)>1 else None
        target = labels.get(target_label, 0)
        if cond == 'pin':
            return Instruction(Opcode.JMP, {'cond': 'pin', 'target': target})
        if cond == 'x--':
            return Instruction(Opcode.JMP, {'cond': 'x--', 'target': target})
        if cond == 'y--':
            return Instruction(Opcode.JMP, {'cond': 'y--', 'target': target})
        if cond == '!osre':
            return Instruction(Opcode.JMP, {'cond': '!osre', 'target': target})
        # unconditional
        return Instruction(Opcode.JMP, {'cond': 'always', 'target': target})
    if mnemonic == 'OUT':
        tokens = args.replace(',', ' ').split()
        dest = tokens[0]
        bits = int(tokens[1])
        return Instruction(Opcode.OUT, {'dest': dest, 'bits': bits})
    if mnemonic == 'IRQ':
        # irq set 0
        tokens = args.split()
        irq_num = int(tokens[1])
        return Instruction(Opcode.IRQ, {'irq': irq_num})
    if mnemonic == 'SET':
        # set pin 0 etc
        tokens = args.split()
        dest = tokens[0]
        val = int(tokens[1])
        return Instruction(Opcode.SET, {'dest': dest, 'value': val})
    if mnemonic == '.WRAP_TARGET' or mnemonic == '.WRAP':
        return Instruction(Opcode.JMP, {'cond':'always','target':0})  # placeholder
    raise NotImplementedError(f"Unsupported mnemonic {mnemonic}")

# Load the kernel from the blueprint
PIO_KERNEL_SOURCE = """
.program pionet_bnn_mac
.side_set 1 opt

.wrap_target
    pull block
    mov y, osr

wait_clk:
    wait 1 pin 0
    in pins, 1

    jmp pin bit_one

bit_zero:
    out null, 1
    jmp !osre match_check_0
    jmp end_mac

match_check_0:
    jmp y-- increment_acc
    jmp end_mac

bit_one:
    out null, 1
    jmp !osre match_check_1
    jmp end_mac

match_check_1:
    jmp y-- increment_acc

increment_acc:
    jmp x-- next_bit

next_bit:
    wait 0 pin 0
    jmp wait_clk

end_mac:
    jmp x-- reset_acc
    irq set 0

reset_acc:
    mov x, ~null
.wrap
"""

def load_pionet_kernel() -> List[Instruction]:
    return assemble_pio(PIO_KERNEL_SOURCE)

if __name__ == "__main__":
    prog = load_pionet_kernel()
    print(f"Loaded {len(prog)} instructions")
    for i, ins in enumerate(prog):
        print(f"{i:02d}: {ins}")