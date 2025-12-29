add_subdirectory(timevar)

target_sources(astra
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/vtk.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/vtk.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/dump.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/dump.cpp
  )