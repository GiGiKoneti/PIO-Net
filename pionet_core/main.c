#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/pio.h"
#include "pdm_reader.pio.h"

#define BUF_WORDS       512
#define LED_PIN         25
#define ACOUSTIC_THRESH 18      // avg ones/32>18 → sound detected

static uint32_t pdm_buf[BUF_WORDS];

int main(void)
{
    stdio_init_all();
    sleep_ms(2000);             // wait for USB CDC to enumerate
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    uint offset = pio_add_program(pio0, &pdm_reader_program);
    pdm_reader_program_init(pio0, 0, offset, 11, 10);
    uint32_t report_timer = 0;
    uint32_t triggered_count = 0;
    uint32_t total_words = 0;

    while (1)
    {
        uint32_t word = pio_sm_get_blocking(pio0, 0);
        pdm_buf[total_words % BUF_WORDS] = word;
        total_words++; // Count set bits for estiamting energy
        uint32_t ones = __builtin_popcount(word);
        if (ones > ACOUSTIC_THRESH)
        {
            gpio_put(LED_PIN, 1);
            triggered_count++;
        }
        else
        {
            gpio_put(LED_PIN, 0);
        }

        // Print summary about every 1 second (32kHz words * 32 bits = 1024000 bits/s)
        if (total_words % 32000 == 0)
        {
            printf("words=%lu triggered=%lu last=0x%08X ones=%lu\n",
                   total_words, triggered_count,
                   pdm_buf[(total_words - 1) % BUF_WORDS], ones);
        }
    }
}