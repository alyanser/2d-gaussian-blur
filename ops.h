#pragma once

#include <cmath>

namespace cuda {

namespace details {
	__global__ void add(double * a, double * b, double * res){
		*res = *a + *b;
	}

	__global__ void sub(double * a, double * b, double * res){
		*res = *a - *b;
	}

	__global__ void mul(double * a, double * b, double * res){
		*res = *a * *b;
	}

	__global__ void div(double * a, double * b, double * res){
		*res = *a / *b;
	}

	__global__ void exp(double * arg, double * res){
		*res = std::exp(*arg);
	}
} // namespace details

class Ops {
public:
	Ops(double val) : val_(val)
	{}

	Ops operator + (Ops & rhs) const noexcept {
		double * a_ptr;
		double * b_ptr;
		double * res_ptr;

		cudaMalloc(&a_ptr, sizeof(double));
		cudaMalloc(&b_ptr, sizeof(double));
		cudaMalloc((&res_ptr), sizeof(double));

		cudaMemcpy(a_ptr, &val_, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(b_ptr, &rhs.val_, sizeof(double), cudaMemcpyHostToDevice);
		details::add<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		double res;
		cudaMemcpy(&res, res_ptr, sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops operator - (Ops rhs) const noexcept {
		double * a_ptr;
		double * b_ptr;
		double * res_ptr;

		cudaMalloc(&a_ptr, sizeof(double));
		cudaMalloc(&b_ptr, sizeof(double));
		cudaMalloc((&res_ptr), sizeof(double));

		cudaMemcpy(a_ptr, &val_, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(b_ptr, &rhs.val_, sizeof(double), cudaMemcpyHostToDevice);
		details::sub<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		double res;
		cudaMemcpy(&res, res_ptr, sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops operator * (Ops rhs) const noexcept {
		double * a_ptr;
		double * b_ptr;
		double * res_ptr;

		cudaMalloc(&a_ptr, sizeof(double));
		cudaMalloc(&b_ptr, sizeof(double));
		cudaMalloc((&res_ptr), sizeof(double));

		cudaMemcpy(a_ptr, &val_, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(b_ptr, &rhs.val_, sizeof(double), cudaMemcpyHostToDevice);
		details::mul<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		double res;
		cudaMemcpy(&res, res_ptr, sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops operator / (Ops rhs) const noexcept {
		double * a_ptr;
		double * b_ptr;
		double * res_ptr;

		cudaMalloc(&a_ptr, sizeof(double));
		cudaMalloc(&b_ptr, sizeof(double));
		cudaMalloc((&res_ptr), sizeof(double));

		cudaMemcpy(a_ptr, &val_, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(b_ptr, &rhs.val_, sizeof(double), cudaMemcpyHostToDevice);
		details::div<<<1, 1>>>(a_ptr, b_ptr, res_ptr);

		double res;
		cudaMemcpy(&res, res_ptr, sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(a_ptr);
		cudaFree(b_ptr);
		cudaFree(res_ptr);

		return res;
	}

	Ops exp() const noexcept {
		double * a_ptr;
		double * res_ptr;

		cudaMalloc(&a_ptr, sizeof(double));
		cudaMalloc((&res_ptr), sizeof(double));

		cudaMemcpy(a_ptr, &val_, sizeof(double), cudaMemcpyHostToDevice);
		details::exp<<<1, 1>>>(a_ptr, res_ptr);

		double res;
		cudaMemcpy(&res, res_ptr, sizeof(double), cudaMemcpyDeviceToHost);

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
