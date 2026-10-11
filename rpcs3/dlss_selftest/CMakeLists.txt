# DLSS test: headless self-test of the motion vector path (see dlss_selftest.cpp). Linux/X11 only.
add_executable(dlss_selftest
    dlss_selftest.cpp
    dlss_selftest_stubs.cpp
    ../Input/ps_move_config.cpp
    ../rpcs3_version.cpp)
target_include_directories(dlss_selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(dlss_selftest PRIVATE rpcs3_emu 3rdparty::stblib 3rdparty::fusion X11::X11 ${CMAKE_DL_LIBS} rt)
