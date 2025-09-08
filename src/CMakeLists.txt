add_subdirectory(utils)

target_sources(astra
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/astra.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/main.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/arrays.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/global.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/global.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/loop.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/profiler.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/profiler.cpp

  )

#add_subdirectory("${PROJECT_BINARY_DIR}" "${PROJECT_BINARY_DIR}/local")
