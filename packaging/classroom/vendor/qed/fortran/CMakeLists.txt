add_library(segn96_fortran
    sregn96.f90
    slegn96.f90
)

if(USE_OPENMP)
target_link_libraries(segn96_fortran PUBLIC OpenMP::OpenMP_Fortran)
endif()
