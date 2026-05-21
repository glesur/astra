target_sources(astra
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevar.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevar.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarEnergy.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarEnergy.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarEnstrophy.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarEnstrophy.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarMinMax.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarMinMax.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarStress.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarStress.cpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarSpectrum.hpp
  PUBLIC ${CMAKE_CURRENT_LIST_DIR}/timevarSpectrum.cpp
  )
