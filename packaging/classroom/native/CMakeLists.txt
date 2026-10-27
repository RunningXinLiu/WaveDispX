cmake_minimum_required(VERSION 3.16)
project(ClassroomSolvers LANGUAGES C CXX Fortran)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/solvers")
set(CMAKE_Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/fortran_modules")
set(VENDOR "${CMAKE_CURRENT_LIST_DIR}/../vendor")
set(QED "${VENDOR}/qed")
set(CPS "${VENDOR}/cps")
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS C)
add_library(qed_fortran STATIC "${QED}/fortran/sregn96.f90" "${QED}/fortran/slegn96.f90")
add_library(qed_disp STATIC
    "${QED}/src/secfunc.cc" "${QED}/src/disp.cc" "${QED}/src/toms748.cc"
    "${QED}/src/utils.cc" "${QED}/src/swegn96.cc")
target_include_directories(qed_disp PUBLIC "${QED}/include" "${QED}/external/fmt/include")
target_compile_definitions(qed_disp PUBLIC FMT_HEADER_ONLY _USE_MATH_DEFINES)
target_link_libraries(qed_disp PUBLIC Eigen3::Eigen qed_fortran)
add_executable(forward "${QED}/src/main_forward.cc")
target_include_directories(forward PRIVATE "${QED}/external/highfive/include" "${QED}/external/toml11/include" ${HDF5_INCLUDE_DIRS})
target_link_libraries(forward PRIVATE qed_disp ${HDF5_LIBRARIES})

# Native CPS implementations, including its real CALPLOT file writer. No X11,
# plotting stubs, replacement dispersion algorithm or changed numeric source.
set(common "${CPS}/SUBS/igetmod.f" "${CPS}/SUBS/lgstr.f" "${CPS}/SUBS/mchdep.f"
           "${CPS}/SUBS/mnmarg.f" "${CPS}/SUBS/mgtarg.f")
add_library(cps_common STATIC ${common} "${CPS}/VOLIII/src/sio.f")
file(GLOB calplot_fortran CONFIGURE_DEPENDS "${CPS}/CALPLOT/src/flib/*.f")
add_library(calplot STATIC ${calplot_fortran}
    "${CPS}/CALPLOT/src/flib/dfsubs.c" "${CPS}/CALPLOT/src/cmd/dpsubs.c"
    "${CPS}/CALPLOT/src/clib/clcplt.c" "${CPS}/CALPLOT/src/clib/grd.c"
    "${CPS}/CALPLOT/src/clib/ponintr.c")
target_include_directories(calplot PRIVATE "${CPS}/CALPLOT/src/include")
if(WIN32)
    # Only the file writer needs the existing Windows signal/binary-I/O branch.
    # Global MSDOS would select obsolete 16-bit DOS types and remove gfortran's
    # trailing underscores in dfsubs.c; MinGW uses the normal Fortran ABI.
    set_source_files_properties("${CPS}/CALPLOT/src/clib/clcplt.c"
        PROPERTIES COMPILE_DEFINITIONS MSDOS)
endif()
foreach(name sprep96 sdisp96)
    add_executable(${name} "${CPS}/VOLIII/src/${name}.f")
    target_link_libraries(${name} PRIVATE cps_common)
endforeach()
add_executable(sdpsrf96 "${CPS}/VOLIII/src/sdpsrf96.f" "${CPS}/SUBS/grphsubf.f")
target_link_libraries(sdpsrf96 PRIVATE cps_common calplot)
foreach(target cps_common calplot sprep96 sdisp96 sdpsrf96)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-std=legacy;-frecord-marker=4>)
    if(CMAKE_Fortran_COMPILER_VERSION VERSION_GREATER_EQUAL 10)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:Fortran>:-fallow-argument-mismatch>)
    endif()
endforeach()
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    # CPS C callbacks use legacy K&R prototypes; retain semantics on modern GCC.
    target_compile_options(calplot PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wno-incompatible-pointer-types;-Wno-implicit-function-declaration>)
endif()
install(TARGETS forward sprep96 sdisp96 sdpsrf96 RUNTIME DESTINATION solvers)
