add_library(timer
    timer.cc
)

add_library(disp
    secfunc.cc
    disp.cc
    toms748.cc
    utils.cc
    swegn96.cc
)

target_link_libraries(disp
    PRIVATE
    fmt::fmt-header-only
    segn96_fortran
)

add_executable(secfunc
    main_secfunc.cc
)

target_link_libraries(secfunc
    PRIVATE
    disp
    HighFive
    fmt::fmt-header-only
    toml11
)

add_executable(forward
    main_forward.cc
)

target_link_libraries(forward
    PRIVATE
    disp
    HighFive
    fmt::fmt-header-only
    toml11
)

add_executable(step4_batch_forward
    main_step4_batch.cc
)

target_link_libraries(step4_batch_forward
    PRIVATE
    disp
)

add_executable(inversion
    main_inversion.cc
    problem.cc
    model.cc
)

target_link_libraries(inversion
    PRIVATE
    disp
    fmt::fmt-header-only
    toml11
    HighFive
    lbfgspp
    timer
)

if(ENABLE_DEBUG)
    target_compile_definitions(inversion PRIVATE DEBUG)
endif()

if (USE_OPENMP)
target_link_libraries(inversion
    PRIVATE
    OpenMP::OpenMP_CXX
)
target_link_libraries(step4_batch_forward
    PRIVATE
    OpenMP::OpenMP_CXX
)
endif()

if(BUILD_TESTING)
add_executable(qedisp_kernel_numerics_tests ${CMAKE_SOURCE_DIR}/tests/test_kernel_numerics.cc)
target_link_libraries(qedisp_kernel_numerics_tests PRIVATE disp)
add_test(NAME qedisp_kernel_numerics_tests COMMAND qedisp_kernel_numerics_tests)
endif()

if(BUILD_TESTING)
add_executable(qedisp_group_numerics_tests ${CMAKE_SOURCE_DIR}/tests/test_group_numerics.cc)
target_link_libraries(qedisp_group_numerics_tests PRIVATE disp)
add_test(NAME qedisp_group_numerics_tests COMMAND qedisp_group_numerics_tests)
endif()

if(BUILD_TESTING AND USE_OPENMP)
add_executable(qedisp_kernel_threads_tests ${CMAKE_SOURCE_DIR}/tests/test_kernel_threads.cc)
target_link_libraries(qedisp_kernel_threads_tests PRIVATE disp OpenMP::OpenMP_CXX)
add_test(NAME qedisp_kernel_threads_tests COMMAND qedisp_kernel_threads_tests)
endif()
