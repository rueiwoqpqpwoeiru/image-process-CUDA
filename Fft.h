#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include "Basic.h"
#include "Mem.h"
#include "Util.h"

template<typename T_in> class Fft {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		int _size[2] = { h, w };
		cudaMalloc((void**)&_in_fft, sizeof(cufftComplex) * _size[0] * _size[1]);
		cudaMalloc((void**)&_out_fft, sizeof(cufftComplex) * _size[0] * _size[1]);
		cufftPlanMany(&_plan, 2, _size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 1);
	}
	~Fft() {
		cudaFree(_in_fft);
		cudaFree(_out_fft);
		cufftDestroy(_plan);
	}
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const dim3& block = dim3(32, 8)) {
		Proc_unit p(in.d_data.w, in.d_data.h, block);
		Stopwatch sw;
		K_func::generic_kernel<Fft> << < p.grid, p.block >> > (this, in.d_data, ch_in, _in_fft);
		cufftExecC2C(_plan, _in_fft, _out_fft, CUFFT_FORWARD);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const D_data<T_in> in, const unsigned int ch_in, cufftComplex* c) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (in.w > x && in.h > y) {
			(c + in.w * y + x)->x = static_cast<float>(in.tex(x, y, ch_in));
			(c + in.w * y + x)->y = 0.0;
		}
	}
	cufftComplex* get_out() const { return _out_fft; }
private:
	cufftComplex* _in_fft;
	cufftComplex* _out_fft;
	cufftHandle _plan;
};

template<typename T_out> class Ifft {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_out.open_2d(w, h, ch);
		int _size[2] = { h, w };
		cudaMalloc((void**)&_out_fft, sizeof(cufftComplex) * _size[0] * _size[1]);
		cufftPlanMany(&_plan, 2, _size, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 1);
	}
	~Ifft() {
		cudaFree(_out_fft);
		cufftDestroy(_plan);		
	}
	void do_proc(cufftComplex* in_fft, const unsigned int& ch_out, const dim3& block = dim3(32, 8)) {
		Proc_unit p(_out.d_data.w, _out.d_data.h, block);
		Stopwatch sw;
		cufftExecC2C(_plan, in_fft, _out_fft, CUFFT_INVERSE);
		K_func::generic_kernel<Ifft> << < p.grid, p.block >> > (this, _out_fft, _out.d_data, ch_out);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const cufftComplex* c, D_data<T_out> out, unsigned int ch_out) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		float d = 1.0 / static_cast<float>(out.w * out.h);
		if (out.w > x && out.h > y) {
			out.val(x, y, ch_out) = static_cast<T_out>((sqrtf((c + out.w * y + x)->x * (c + out.w * y + x)->x + (c + out.w * y + x)->y * (c + out.w * y + x)->y)) * d);
		}
	}
	const D_mem<T_out>& get_out() const { return _out; }
private:
	cufftComplex* _out_fft;
	cufftHandle _plan;
	D_mem<T_out> _out;
};

class Mul_amp {
public:
	void alloc_out(unsigned int w, unsigned int h) {
		_w = w;
		_h = h;
		cudaMalloc((void**)&_out, sizeof(cufftComplex) * _w * _h);
	}
	~Mul_amp() {
		cudaFree(_out);
	}
	void do_proc(cufftComplex* in0, cufftComplex* in1, const dim3& block = dim3(32, 8)) {
		Proc_unit p(_w, _h, block);
		Stopwatch sw;
		K_func::generic_kernel<Mul_amp> << < p.grid, p.block >> > (this, in0, in1, _out, _w, _h);
		sw.print_time(__FUNCTION__);
	}
	// amp is calc by in1
	__device__ __forceinline__ void kernel(cufftComplex* in0, cufftComplex* in1, cufftComplex* out, unsigned int w, unsigned int h) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (w > x && h > y) {
			float a = sqrt(((in1 + w * y + x)->x) * ((in1 + w * y + x)->x) + ((in1 + w * y + x)->y) * ((in1 + w * y + x)->y));
			(out + w * y + x)->x = ((in0 + w * y + x)->x) * a;
			(out + w * y + x)->y = ((in0 + w * y + x)->y) * a;
		}
	}
	cufftComplex* get_out() const { return _out; }
private:
	unsigned int _w;
	unsigned int _h;
	cufftComplex* _out;
};

class Freq_flt {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_in.open_2d(w, h, ch);
		_out.resize(ch);
		for (unsigned int k = 0; k < ch; k++) {
			_out[k].alloc_out(w, h, 1);
		}
	}
	// gaussian (w, h are odd)
	void set_flt_blur(const float sigma, const unsigned int ch) {
		// set host data
		float sum = 0.0;
		if (1e-8 < sigma) {
			for (unsigned int i = 0; i < _in.d_data.h; i++) {
				for (unsigned int j = 0; j < _in.d_data.w; j++) {
					int dx = ((j + _in.d_data.w / 2) % _in.d_data.w) - _in.d_data.w / 2;
					int dy = ((i + _in.d_data.h / 2) % _in.d_data.h) - _in.d_data.h / 2;
					unsigned int d2 = pow(static_cast<int>(dx), 2) + pow(static_cast<int>(dy), 2);
					double val = (1.0 / (2.0 * 3.141592653589793 * pow(sigma, 2))) * exp(-(static_cast<float>(d2)) / (sigma * sigma));
					_in.h_data.val(j, i, ch) = static_cast<float>(val);
					sum = sum + _in.h_data.val(j, i, ch);
				}
			}
			// normalize
			for (unsigned int i = 0; i < _in.d_data.h; i++) {
				for (unsigned int j = 0; j < _in.d_data.w; j++) {
					_in.h_data.val(j, i, ch) = static_cast<float>(_in.h_data.val(j, i, ch) / sum);
				}
			}
		}
		else {
			for (unsigned int i = 0; i < _in.d_data.h; i++) {
				for (unsigned int j = 0; j < _in.d_data.w; j++) {
					_in.h_data.val(j, i, ch) = 0.0;
				}
			}
			_in.h_data.val(0, 0, ch) = 1.0;
		}
		// trans host to device (trans other channels)
		_in.h2d();
		// fft
		_out[ch].do_proc(_in, ch);
	}
	cufftComplex* get_out(unsigned int ch) const { return _out[ch].get_out(); }
private:
	D_mem<float> _in;
	std::vector<Fft<float> > _out;
};

// fir filter (shift invariant only)
template<typename T_in, typename T_out> class Filtered_img_fft {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_fft.alloc_out(w, h, ch);
		_ifft.alloc_out(w, h, ch);
		_mul_amp.alloc_out(w, h);
	}
	void alloc_flt(unsigned int w, unsigned int h, unsigned int ch) {
		_freq_flt.alloc_out(w, h, ch);
	}
	void set_flt_blur(const float sigma, const unsigned int ch) {
		_freq_flt.set_flt_blur(sigma, ch);
	}
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const unsigned int& ch_out, const unsigned int& ch_flt, const dim3& block = dim3(32, 8)) {
		_fft.do_proc(in, ch_in);
		_mul_amp.do_proc(_fft.get_out(), _freq_flt.get_out(ch_flt));
		_ifft.do_proc(_mul_amp.get_out(), ch_out);
	}
	const D_mem<T_out>& get_out() const { return _ifft.get_out(); }
private:
	Fft<T_in> _fft;
	Ifft<T_out> _ifft;
	Freq_flt _freq_flt;
	Mul_amp _mul_amp;
};

void unit_test_Fft(int argc, char* argv[]) {
	const std::string in_file = argv[1];
	const std::string out_dir = argv[2];
	// input img
	D_mem<uint16> in_img;
	in_img.imread(in_file);
	// ++++++++++++++++
	// test for Fft
	// ++++++++++++++++
	Filtered_img_fft<uint16, uint16> Filtered_img_fft;
	Filtered_img_fft.alloc_out(in_img.d_data.w, in_img.d_data.h, 5);
	Filtered_img_fft.alloc_flt(in_img.d_data.w, in_img.d_data.h, 5);
	// set filter
	Filtered_img_fft.set_flt_blur(0.0, 0);
	Filtered_img_fft.set_flt_blur(1.0, 1);
	Filtered_img_fft.set_flt_blur(3.0, 2);
	Filtered_img_fft.set_flt_blur(7.0, 3);
	Filtered_img_fft.set_flt_blur(16.0, 4);
	// do process
	Filtered_img_fft.do_proc(in_img, 2, 0, 0);
	Filtered_img_fft.do_proc(in_img, 2, 1, 1);
	Filtered_img_fft.do_proc(in_img, 2, 2, 2);
	Filtered_img_fft.do_proc(in_img, 2, 3, 3);
	Filtered_img_fft.do_proc(in_img, 2, 4, 4);
	// output
	Filtered_img_fft.get_out().imwrite(out_dir + "/out_04_fft.tif");
}
