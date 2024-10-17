project(sycl-playground)
set(CMAKE_CXX_COMPILER icpx)


set(COMPILE_FLAGS "-fsycl -Wall")
set(LINK_FLAGS "-fsycl -lze_loader")

set(TARGET_NAME sycl-playground)
add_executable(${TARGET_NAME} sycl-playground.cpp)
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_FLAGS "${COMPILE_FLAGS}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "${LINK_FLAGS}")
