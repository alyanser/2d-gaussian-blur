#pragma once

#include <cmath>

namespace cuda {

namespace details {
	__global__ void add(const double * a, const double * b, double * res){
		*res = *a + *b;
	}

	__global__ void sub(const double * a, const double * b, double * res){
		*res = *a - *b;
	}

	__global__ void mul(const double * a, const double * b, double * res){
		*res = *a * *b;
	}

	__global__ void div(const double * a, const double * b, double * res){
		*res = *a / *b;
	}

	__global__ void exp(const double * arg, double * res){
		*res = std::exp(*arg);
	}
} // namespace details

inline double * alloc_gpu_memory_for_double(){
	double * ptr;
	cudaMalloc(&ptr, sizeof(double));
	return ptr;
}

inline void write_double_to_gpu_memory(double * ptr, const double val){
	cudaMemcpy(ptr, &val, sizeof(double), cudaMemcpyHostToDevice);
}

inline double read_double_from_gpu_mem(const double * ptr){
	double res;
	cudaMemcpy(&res, ptr, sizeof(double), cudaMemcpyDeviceToHost);
	return res;
}

class Ops {
public:
	Ops(double val) : val_(val)
	{}

	Ops operator + (Ops & rhs) const noexcept {
		auto * a_ptr = alloc_gpu_memory_for_double();
		auto * b_ptr = alloc_gpu_memory_for_double();
		auto * res_ptr = alloc_gpu_memory_for_double();

		write_double_to_gpu_memory(a_ptr, val_);
		write_double_to_gpu_memory(b_ptr, rhs.val_);

		details::add<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		auto res = read_double_from_gpu_mem(res_ptr);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops operator - (Ops & rhs) const noexcept {
		auto * a_ptr = alloc_gpu_memory_for_double();
		auto * b_ptr = alloc_gpu_memory_for_double();
		auto * res_ptr = alloc_gpu_memory_for_double();

		write_double_to_gpu_memory(a_ptr, val_);
		write_double_to_gpu_memory(b_ptr, rhs.val_);

		details::sub<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		const auto res = read_double_from_gpu_mem(res_ptr);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops operator * (Ops & rhs) const noexcept {
		auto * a_ptr = alloc_gpu_memory_for_double();
		auto * b_ptr = alloc_gpu_memory_for_double();
		auto * res_ptr = alloc_gpu_memory_for_double();

		write_double_to_gpu_memory(a_ptr, val_);
		write_double_to_gpu_memory(b_ptr, rhs.val_);

		details::mul<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		const auto res = read_double_from_gpu_mem(res_ptr);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops operator / (Ops & rhs) const noexcept {
		auto * a_ptr = alloc_gpu_memory_for_double();
		auto * b_ptr = alloc_gpu_memory_for_double();
		auto * res_ptr = alloc_gpu_memory_for_double();

		write_double_to_gpu_memory(a_ptr, val_);
		write_double_to_gpu_memory(b_ptr, rhs.val_);

		details::mul<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		const auto res = read_double_from_gpu_mem(res_ptr);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops exp() const noexcept {
		auto * a_ptr = alloc_gpu_memory_for_double();
		auto * res_ptr = alloc_gpu_memory_for_double();

		write_double_to_gpu_memory(a_ptr, val_);

		details::exp<<<1, 1>>>(a_ptr, res_ptr);

		const auto res = read_double_from_gpu_mem(res_ptr);

		cudaFree(a_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops operator - () const noexcept {
		return -val_;
	}

	operator double () const noexcept {
		return val_;
	}

private:
	double val_;
};

} // namespace cuda
