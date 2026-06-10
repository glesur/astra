add_subdirectory(utils)
add_subdirectory(initFlow)
add_subdirectory(timeIntegrator)
add_subdirectory(rightHandSide)
add_subdirectory(output)

target_sources(astra
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/astra.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/arrays.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/checkNan.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/global.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/global.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/fft.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/fft.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/field.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/grid.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/grid.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/gridhost.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/input.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/input.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/logger.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/loop.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/profiler.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/profiler.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/subgrid.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/subgrid.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/transpose.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/test.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/test.cpp
  )

#add_subdirectory("${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/local")
