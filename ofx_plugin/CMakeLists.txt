# Compilation de l'effet OFX « TAPNext Shapes » (MSVC, Xcode/clang, GCC).
#   cmake -S . -B build && cmake --build build --config Release
# Le résultat est copié dans dist/TAPNextShapes.ofx.bundle/Contents/<arch>/.
cmake_minimum_required(VERSION 3.15)
project(TAPNextShapes CXX)
set(CMAKE_CXX_STANDARD 17)
add_library(TAPNextShapes MODULE src/TAPNextShapes.cpp)
target_include_directories(TAPNextShapes PRIVATE third_party/openfx third_party/stb)
set_target_properties(TAPNextShapes PROPERTIES PREFIX "" SUFFIX ".ofx")
if(WIN32)
  set(ARCH Win64)
  if(MSVC)
    target_compile_definitions(TAPNextShapes PRIVATE _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS)
    set_property(TARGET TAPNextShapes PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  endif()
elseif(APPLE)
  set(ARCH MacOS)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
else()
  set(ARCH Linux-x86-64)
endif()
set(OUT ${CMAKE_CURRENT_SOURCE_DIR}/dist/TAPNextShapes.ofx.bundle/Contents/${ARCH})
add_custom_command(TARGET TAPNextShapes POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT}
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:TAPNextShapes> ${OUT}/)
