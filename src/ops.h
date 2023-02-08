#pragma once

#include <cmath>

namespace cuda {

namespace gpu_ops {

template<typename T>
__global__ void add(const T * a, const T * b, T * res){
	*res = *a + *b;
}

template<typename T>
__global__ void sub(const T * a, const T * b, T * res){
	*res = *a - *b;
}

template<typename T>
__global__ void mul(const T * a, const T * b, T * res){
	*res = *a * *b;
}

template<typename T>
__global__ void div(const T * a, const T * b, T * res){
	*res = *a / *b;
}

template<typename T>
__global__ void exp(const T * arg, T * res){
	*res = std::exp(*arg);
}

} // namespace gpu_ops

template<typename T>
class Ops {
public:
	Ops(const T val)
		: val_(val)
	{}

	Ops operator + (Ops & rhs) const noexcept {
		Gpu_memory x(val_);
		Gpu_memory y(rhs.val_);
		Gpu_memory res;

		gpu_ops::add<<<1, 1>>>(x.get(), y.get(), res.get());

		return *res;
	}

	Ops operator * (Ops & rhs) const noexcept {
		Gpu_memory x(val_);
		Gpu_memory y(rhs.val_);
		Gpu_memory res;

		gpu_ops::mul<<<1, 1>>>(x.get(), y.get(), res.get());

		return *res;
	}

	Ops operator / (Ops & rhs) const noexcept {
		Gpu_memory x(val_);
		Gpu_memory y(rhs.val_);
		Gpu_memory res;

		gpu_ops::div<<<1, 1>>>(x.get(), y.get(), res.get());

		return *res;
	}

	Ops operator - (Ops & rhs) const noexcept {
		Gpu_memory x(val_);
		Gpu_memory y(rhs.val_);
		Gpu_memory res;

		gpu_ops::sub<<<1, 1>>>(x.get(), y.get(), res.get());

		return *res;
	}

	Ops exp() const noexcept {
		Gpu_memory x(val_);
		Gpu_memory res;

		gpu_ops::exp<<<1, 1>>>(x.get(), res.get());

		return *res;
	}

	Ops operator - () const noexcept {
		return -val_;
	}

	operator T () const noexcept {
		return val_;
	}

private:

	class Gpu_memory {
	public:
		Gpu_memory() noexcept {
			cudaMalloc(&ptr_, sizeof(T));
		}

		Gpu_memory(const T value) noexcept
			: Gpu_memory()
		{
			set_value(value);
		}

		~Gpu_memory() noexcept {
			cudaFree(ptr_);
		}

		Gpu_memory(const Gpu_memory & rhs) = delete;
		Gpu_memory & operator = (const Gpu_memory & rhs) = delete;
		Gpu_memory(Gpu_memory && rhs) = delete;
		Gpu_memory & operator = (Gpu_memory && rhs) = delete;

		void set_value(const T & val) noexcept {
			cudaMemcpy(ptr_, &val, sizeof(T), cudaMemcpyHostToDevice);
		}

		T * get() noexcept {
			return ptr_;
		}

		const T * get() const noexcept {
			return ptr_;
		}

		T operator * () const noexcept {
			T val;
			cudaMemcpy(&val, ptr_, sizeof(T), cudaMemcpyDeviceToHost);
			return val;
		}

	private:
		T * ptr_;
	};

	T val_;
};

} // namespace cuda
