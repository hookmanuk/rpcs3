# DLSS test (fork): VKDLSS's sources, the NVIDIA DLSS SDK and the headless self-test. Included by rpcs3/Emu/CMakeLists.txt
# (its one line for all of this).
#   -DBUILD_DLSS_SELFTEST=ON                               rpcs3/dlss_selftest (Linux/X11, no Qt)
if(TARGET 3rdparty_vulkan)
    target_sources(rpcs3_emu PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/VKDLSS.cpp
        ${CMAKE_CURRENT_LIST_DIR}/VKGSRenderDLSS.cpp)

    # NVIDIA DLSS itself is the optional plugin rpcs3_ngx.dll (rpcs3/ngx_plugin, Windows, MSBuild): nothing of NVIDIA's is
    # linked here.

    option(BUILD_DLSS_SELFTEST "Build the DLSS motion vector self-test (dlss_selftest)" OFF)
    if(BUILD_DLSS_SELFTEST AND X11_FOUND)
        add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../dlss_selftest ${CMAKE_BINARY_DIR}/rpcs3/dlss_selftest)
    endif()
endif()
