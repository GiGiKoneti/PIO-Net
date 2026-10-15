//
// Created by Adarsha Udupa on 14-10-2026.
#include "main.h"

#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "pdm_reader.pio.h"

#define BUF_WORDS 512   // 512*32=16384 PDM bits
uint32_t pdm_buf[BUF_WORDS];

int main(void)
{
    stdio_init_all();
    uint offset = pio_add_program(pio0, &pdm_reader_program);
    pdm_reader_program_init(pio0, 0, offset, 11, 10);
    while(1) {
        for (int i = 0; i < BUF_WORDS; i++)
            pdm_buf[i] = pio_sm_get_blocking(pio0, 0);
        for (int i = 0; i < BUF_WORDS; i++)
            printf("0x%08X\n", pdm_buf[i]);
    }
}
