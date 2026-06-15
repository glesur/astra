target_sources(astra
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/advection.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/burgers.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/hydro.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/mhd.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/rightHandSide.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/rightHandSideFactory.hpp
  )
