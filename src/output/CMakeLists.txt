add_subdirectory(timevar)

target_sources(astra
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/vtk.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/vtk.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/dump.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/dump.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/dumpVariables.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/dumpVariables.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/output.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/output.cpp
  )