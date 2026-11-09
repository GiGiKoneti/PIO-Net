// ==============================================================================
// Pure Baseline WFI Power Benchmark for Raspberry Pi Pico (RP2040)
// - Core 1: Held in hardware reset (never launched)
// - Core 0: Sleeping in __wfi()
// - USB CDC: Disabled (pico_enable_stdio_usb 0)
// - Telemetry / UART: Disabled (pico_enable_stdio_uart 0)
// - Clocks: clk_usb stopped, pll_usb deinitialized
// - Expected typical current: ~1.3 mA - 1.5 mA
// ==============================================================================

#include "pico/stdlib.h"
#include "hardware/sync.h"
#include "hardware/clocks.h"
#include "hardware/pll.h"
#include "hardware/gpio.h"

int main(void) {
    // Disable USB clock and deinit USB PLL to eliminate 48 MHz VCO power
    clock_stop(clk_usb);
    pll_deinit(pll_usb);

    // Ensure status LED is off
#ifdef PICO_DEFAULT_LED_PIN
    gpio_init(PICO_DEFAULT_LED_PIN);
    gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
    gpio_put(PICO_DEFAULT_LED_PIN, 0);
#endif

    // Core 0 enters permanent low-power WFI sleep
    // Core 1 remains in hardware reset (multicore_launch_core1 was never called)
    while (1) {
        __wfi();
    }

    return 0;
}
