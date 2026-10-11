# DLSS test (fork): VKDLSS's sources, the NVIDIA DLSS SDK and the headless self-test. Included by rpcs3/Emu/CMakeLists.txt
# (its one line for all of this).
#   -DDLSS_SDK=<folder of https://github.com/NVIDIA/DLSS>  builds VKDLSS.cpp against NGX (else the motion debug views only)
#   -DBUILD_DLSS_SELFTEST=ON                               rpcs3/dlss_selftest (Linux/X11, no Qt)
if(TARGET 3rdparty_vulkan)
    target_sources(rpcs3_emu PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/VKDLSS.cpp
        ${CMAKE_CURRENT_LIST_DIR}/VKGSRenderDLSS.cpp)

    set(DLSS_SDK "" CACHE PATH "NVIDIA DLSS SDK folder (include/, lib/)")
    if(DLSS_SDK)
        target_compile_definitions(rpcs3_emu PRIVATE HAVE_DLSS)
        target_include_directories(rpcs3_emu PRIVATE "${DLSS_SDK}/include")
        if(WIN32)
            target_link_libraries(rpcs3_emu PRIVATE "${DLSS_SDK}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib")
        else()
            target_link_libraries(rpcs3_emu PRIVATE "${DLSS_SDK}/lib/Linux_x86_64/libnvsdk_ngx.a" ${CMAKE_DL_LIBS})
        endif()
    endif()

    option(BUILD_DLSS_SELFTEST "Build the DLSS motion vector self-test (dlss_selftest)" OFF)
    if(BUILD_DLSS_SELFTEST AND X11_FOUND)
        add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../../dlss_selftest ${CMAKE_BINARY_DIR}/rpcs3/dlss_selftest)
    endif()
endif()
