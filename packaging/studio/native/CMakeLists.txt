cmake_minimum_required(VERSION 3.16)
project(StudioSolvers LANGUAGES C CXX Fortran)
# Reuse the frozen upstream sources and existing platform adaptations.
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../classroom/native" classroom)
set(QED "${CMAKE_CURRENT_LIST_DIR}/../../classroom/vendor/qed")
add_library(qedisp_group SHARED "${QED}/fortran/sregn96.f90" "${QED}/fortran/slegn96.f90")
set_target_properties(qedisp_group PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/solvers"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/solvers"
  Fortran_MODULE_DIRECTORY "${CMAKE_BINARY_DIR}/studio_modules"
  WINDOWS_EXPORT_ALL_SYMBOLS ON)
if(WIN32)
  set_target_properties(qedisp_group PROPERTIES PREFIX "")
endif()
