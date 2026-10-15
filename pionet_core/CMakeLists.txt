cmake_minimum_required(VERSION 3.13)

include(D:/pico/sdk/external/pico_sdk_import.cmake)

project(pionet_core C CXX ASM)

pico_sdk_init()

add_executable(pionet_core)

pico_generate_pio_header(pionet_core ${CMAKE_CURRENT_LIST_DIR}/pdm_reader.pio)

target_sources(pionet_core PRIVATE main.c)

target_include_directories(pionet_core PRIVATE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(pionet_core PRIVATE
        pico_stdlib
        hardware_pio
)

pico_add_extra_outputs(pionet_core)
pico_enable_stdio_usb(pionet_core 1)
pico_enable_stdio_uart(pionet_core 0)