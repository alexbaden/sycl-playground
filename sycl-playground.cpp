#include <iostream>

#include <sycl/sycl.hpp>


int main(int argc, char* argv[]) {
 	sycl::queue q = sycl::queue(sycl::gpu_selector_v);

	std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
	const int64_t global_mem_size = q.get_device().get_info<sycl::info::device::global_mem_size>();
	std::cout << "Global memory size: " << global_mem_size << std::endl;

	const size_t iters = argc > 1 ? static_cast<size_t>(std::stoi(argv[1])) : 41;
	std::cout << "Using " << iters << std::endl;
	const size_t bytes = 268435456;
	std::vector<char> host_buf(bytes, 'a');


	size_t allocated_mem_size = 0;
	std::vector<char*> ptrs(iters);
	for (size_t i = 0; i < iters; i++) {
		ptrs[i] = sycl::malloc_device<char>(bytes, q);
		allocated_mem_size += bytes;
		std::cout << "Total allocated mem size after iteration " << i << " : " << allocated_mem_size << " (remaining: " << global_mem_size - static_cast<int64_t>(allocated_mem_size) << ")" << std::endl;
		if (!ptrs[i]) {
			std::cout << "Failed to allocate device memory!" << std::endl;
			return 1;
		}
		q.memcpy(ptrs[i], host_buf.data(), bytes).wait_and_throw();
	}

	return 0;
}
