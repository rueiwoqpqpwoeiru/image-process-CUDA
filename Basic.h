#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include "Mem.h"
#include "Util.h"

// kernel func
namespace K_func {
	// call a fixed-name method that named "kernel" in each class
	template<typename T, class... Args> __global__ void generic_kernel(T* obj, Args... args) {
		(obj->kernel)(args...);
	}
	__device__ __forceinline__ uint16 clip_uint16(const float& val) {
		uint16 ret = 0;
		if (0.0 > val) {
			ret = 0;
		}
		else if (65535.0 < val) {
			ret = 65535;
		}
		else {
			ret = static_cast<uint16>(val + 0.5);
		}
		return ret;
	}
};

struct Proc_unit {
	dim3 block;
	dim3 grid;
	Proc_unit(unsigned int w, unsigned int h, dim3 block_) {
		set(w, h, block_);
	}
	void set(unsigned int w, unsigned int h, dim3 block_) {
		block = block_;
		grid = dim3((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
	};
};

// convert bit depth
template<typename T_in, typename T_out> class Cast_img {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_out.open_2d(w, h, ch);
	}
	void do_proc(const D_mem<T_in>& in, const dim3& block = dim3(32, 8)) {
		Proc_unit p(in.d_data.w, in.d_data.h, block);
		Stopwatch sw;
		K_func::generic_kernel<Cast_img> << < p.grid, p.block >> > (this, in.d_data, _out.d_data);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const D_data<T_in> in, D_data<T_out> out) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (in.w > x && in.h > y) {
			for (unsigned int k = 0; k < in.ch; k++) {
				out.val(x, y, k) = static_cast<T_out>(in.tex(x, y, k));
			}
		}
	}
	void imwrite(const std::string& file_name, const int metric = PHOTOMETRIC_MINISBLACK) const {
		_out.imwrite(file_name, metric);
	}
	const D_mem<T_out>& get_out() const { return _out; }
private:
	D_mem<T_out> _out;
};

// binomial operation of 2 img
template<typename T_in0, typename T_in1, typename T_out> class Op_2img {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_out.open_2d(w, h, ch);
	}
	void do_proc(const D_mem<T_in0>& in0, const D_mem<T_in1>& in1, const unsigned int& ch_in0, const unsigned int& ch_in1, const unsigned int& ch_out, 
		         const std::string mode = "add", const dim3& block = dim3(32, 8)) {
		Proc_unit p(_out.d_data.w, _out.d_data.h, block);
		unsigned int _mode = 0;
		if("add" == mode){
			_mode = 0;
		}
		else if ("sub" == mode) {
			_mode = 1;
		}
		else if ("mul" == mode) {
			_mode = 2;
		}
		else {
			assert(false);
		}
		Stopwatch sw;
		K_func::generic_kernel<Op_2img> << < p.grid, p.block >> > (this, in0.d_data, in1.d_data, _out.d_data, ch_in0, ch_in1, ch_out, _mode);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const D_data<T_in0> in0, const D_data<T_in1> in1, D_data<T_out> out,
		const unsigned int ch_in0, const unsigned int ch_in1, const unsigned int ch_out, unsigned int mode) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (out.w > x && out.h > y) {
			if (0 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(static_cast<float>(in0.tex(x, y, ch_in0)) + static_cast<float>(in1.tex(x, y, ch_in1)));
			}
			else if (1 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(static_cast<float>(in0.tex(x, y, ch_in0)) - static_cast<float>(in1.tex(x, y, ch_in1)));
			}
			else if (2 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(static_cast<float>(in0.tex(x, y, ch_in0)) * static_cast<float>(in1.tex(x, y, ch_in1)));
			}
		}
	}
	void imwrite(const std::string& file_name, const int metric = PHOTOMETRIC_MINISBLACK) const {
		_out.imwrite(file_name, metric);
	}
	const D_mem<T_out>& get_out() const { return _out; }
private:
	D_mem<T_out> _out;
};

// operation of 1 img
template<typename T_in0, typename T_out> class Op_1img {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_out.open_2d(w, h, ch);
	}
	void do_proc(const D_mem<T_in0>& in0, const unsigned int& ch_in0, const unsigned int& ch_out,
		const std::string mode = "add", const float& in1 = 0.0, const float& in2 = 0.0, const dim3& block = dim3(32, 8)) {
		Proc_unit p(in0.d_data.w, in0.d_data.h, block);
		unsigned int _mode = 0;
		if ("add" == mode) {
			_mode = 0;
		}
		else if ("sub" == mode) {
			_mode = 1;
		}
		else if ("mul" == mode) {
			_mode = 2;
		}
		else if ("abs" == mode) {
			_mode = 3;
		}
		else if ("pow" == mode) {
			_mode = 4;
		}
		else if ("th" == mode) {
			_mode = 5;
		}
		else if ("clip" == mode) {
			_mode = 6;
		}
		else if ("div" == mode) {
			_mode = 7;
		}
		else {
			assert(false);
		}
		Stopwatch sw;
		K_func::generic_kernel<Op_1img> << < p.grid, p.block >> > (this, in0.d_data, in1, in2, _out.d_data, ch_in0, ch_out, _mode);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const D_data<T_in0> in0, const float in1, const float in2, D_data<T_out> out,
		const unsigned int ch_in0, const unsigned int ch_out, unsigned int mode) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (in0.w > x && in0.h > y) {
			if (0 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(static_cast<float>(in0.tex(x, y, ch_in0)) + in1);
			}
			else if (1 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(static_cast<float>(in0.tex(x, y, ch_in0)) - in1);
			}
			else if (2 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(static_cast<float>(in0.tex(x, y, ch_in0)) * in1);
			}
			else if (3 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(abs(static_cast<float>(in0.tex(x, y, ch_in0))));
			}
			else if (4 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(pow((static_cast<float>(in0.tex(x, y, ch_in0))), in1));
			}
			else if (5 == mode) {
				if (in1 > static_cast<float>(in0.tex(x, y, ch_in0))) {
					out.val(x, y, ch_out) = static_cast<T_out>(0.0);
				}
				else {
					if (sizeof(uint8) == sizeof(T_out)) {
						out.val(x, y, ch_out) = static_cast<T_out>(255);
					}
					else if (sizeof(uint16) == sizeof(T_out)) {
						out.val(x, y, ch_out) = static_cast<T_out>(65535);
					}
					else {
						out.val(x, y, ch_out) = static_cast<T_out>(1.0);
					}
				}
			}
			else if (6 == mode) {
				float val = static_cast<float>(in0.tex(x, y, ch_in0));
				if (val < in1) {
					val = in1;
				}
				else if (val > in2) {
					val = in2;
				}
				out.val(x, y, ch_out) = static_cast<T_out>(val);
			}
			else if (7 == mode) {
				out.val(x, y, ch_out) = static_cast<T_out>(1.0 / (static_cast<float>(in0.tex(x, y, ch_in0))));
			}
		}
	}
	void imwrite(const std::string& file_name, const int metric = PHOTOMETRIC_MINISBLACK) const {
		_out.imwrite(file_name, metric);
	}
	const D_mem<T_out>& get_out() const { return _out; }
private:
	D_mem<T_out> _out;
};

void unit_test_Basic(int argc, char* argv[]) {
	const std::string in_file = argv[1];
	const std::string out_dir = argv[2];
	// input img
	D_mem<uint16> in_img;
	in_img.imread(in_file);
	// ++++++++++++++++
	// test for Op_1img
	// ++++++++++++++++
	Op_1img<uint16, uint16> op_1img;
	op_1img.alloc_out(in_img.d_data.w, in_img.d_data.h, 5);
	// do process
	op_1img.do_proc(in_img, 2, 0, "mul", 1.0);
	op_1img.do_proc(in_img, 2, 1, "mul", 0.8);
	op_1img.do_proc(in_img, 2, 2, "mul", 0.6);
	op_1img.do_proc(in_img, 2, 3, "mul", 0.4);
	op_1img.do_proc(in_img, 2, 4, "mul", 0.2);
	// output
	op_1img.imwrite(out_dir + "/out_01_op_1img.tif", PHOTOMETRIC_MINISBLACK);
	// ++++++++++++++++
	// test for Op_2img
	// ++++++++++++++++
	Op_2img<uint16, uint16, uint16> op_2img;
	op_2img.alloc_out(in_img.d_data.w, in_img.d_data.h, 2);
	// do process
	op_2img.do_proc(op_1img.get_out(), op_1img.get_out(), 1, 4, 0, "add");
	op_2img.do_proc(op_1img.get_out(), op_1img.get_out(), 2, 3, 1, "add");
	// output
	op_2img.imwrite(out_dir + "/out_01_op_2img.tif", PHOTOMETRIC_MINISBLACK);	
}
