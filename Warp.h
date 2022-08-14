#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Mem.h"
#include "Basic.h"
#include "Util.h"

template<typename T_in, typename T_out> class Warped_img {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_out.open_2d(w, h, ch);
	}
	void alloc_param(unsigned int ch) {
		_param.open_2d(3, 3, ch);
		// set host data
		for (unsigned int k = 0; k < ch; k++) {
			this->set_no_trans(k);
		}
		// trans host to device (trans other channels)
		_param.h2d();
	}
	void set_no_trans(unsigned int ch) {
		for (unsigned int i = 0; i < 3; i++) {
			for (unsigned int j = 0; j < 3; j++) {
				// set host data
				_param.h_data.val(j, i, ch) = 0.0;
			}
		}
		_param.h_data.val(0, 0, ch) = 1.0;
		_param.h_data.val(1, 1, ch) = 1.0;
	}
	void set_resize(const float scale_x, const float scale_y, const unsigned int ch) {
		// set host data
		this->set_no_trans(ch);
		_param.h_data.val(0, 0, ch) = 1.0 / scale_x;
		_param.h_data.val(1, 1, ch) = 1.0 / scale_y;
		// trans host to device (trans other channels)
		_param.h2d();
	}
	void set_rot(const float center_x, const float center_y, const float theta, const unsigned int ch) {
		// set host data
		this->set_no_trans(ch);
		float rad = theta * 0.017453292519943;
		float c = cos(rad);
		float s = sin(rad);
		_param.h_data.val(0, 0, ch) = c;
		_param.h_data.val(1, 0, ch) = s;
		_param.h_data.val(2, 0, ch) = -center_x * c - center_y * s + center_x;
		_param.h_data.val(0, 1, ch) = -s;
		_param.h_data.val(1, 1, ch) = c;
		_param.h_data.val(2, 1, ch) = center_x * s - center_y * c + center_y;
		// trans host to device (trans other channels)
		_param.h2d();
	}
	// enter the coordinates (before trans) of the four corners (clockwise from the top left)
	void set_perspective(const float x0, const float y0, const float x1, const float y1,
		const float x2, const float y2, const float x3, const float y3, const unsigned int ch) {
		// set host data
		float a0 = ((x1 + x3 - x0 - x2) * (y2 - y3) - (y1 + y3 - y0 - y2) * (x2 - x3))
			/ (_out.d_data.w * ((x2 - x1) * (y2 - y3) - (y2 - y1) * (x2 - x3)));
		float b0 = ((x1 + x3 - x0 - x2) * (y2 - y1) - (y1 + y3 - y0 - y2) * (x2 - x1))
			/ (_out.d_data.h * ((x2 - x3) * (y2 - y1) - (y2 - y3) * (x2 - x1)));
		_param.h_data.val(0, 0, ch) = a0 * x1 + (x1 - x0) / _out.d_data.w;
		_param.h_data.val(1, 0, ch) = b0 * x3 + (x3 - x0) / _out.d_data.h;
		_param.h_data.val(2, 0, ch) = x0;
		_param.h_data.val(0, 1, ch) = a0 * y1 + (y1 - y0) / _out.d_data.w;
		_param.h_data.val(1, 1, ch) = b0 * y3 + (y3 - y0) / _out.d_data.h;
		_param.h_data.val(2, 1, ch) = y0;
		_param.h_data.val(0, 2, ch) = a0;
		_param.h_data.val(1, 2, ch) = b0;
		_param.h_data.val(2, 2, ch) = 0;
		// trans host to device (trans other channels)
		_param.h2d();
	}
	// set offset to 0.5 for bilinear scale down
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const unsigned int& ch_out, const unsigned int& ch_param, const float& offset = 0.0, const dim3& block = dim3(32, 8)) {
		Proc_unit p(_out.d_data.w, _out.d_data.h, block);
		Stopwatch sw;
		K_func::generic_kernel<Warped_img> << < p.grid, p.block >> > (this, in.d_data, _out.d_data, _param.d_data, ch_in, ch_out, ch_param, offset);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ const float& bilinear(const D_data<T_in>& in, const float& x_f, const float& y_f, const unsigned int& ch) {
		if ((0.0 <= x_f && x_f <= in.w - 1.0) && (0.0 <= y_f && y_f <= in.h - 1.0)) {
			unsigned int x = static_cast<unsigned int>(x_f);
			unsigned int y = static_cast<unsigned int>(y_f);
			float dx = x_f - static_cast<float>(x);
			float dy = y_f - static_cast<float>(y);
			float v00 = static_cast<float>(in.tex(x, y, ch));
			float v10 = static_cast<float>(in.tex(x + 1, y, ch));
			float v01 = static_cast<float>(in.tex(x, y + 1, ch));
			float v11 = static_cast<float>(in.tex(x + 1, y + 1, ch));
			float vX0 = v00 + (v10 - v00) * dx;
			float vX1 = v01 + (v11 - v01) * dx;
			float vXY = vX0 + (vX1 - vX0) * dy;
			return vXY;
		}
		else {
			return 0.0;
		}
	}
	__device__ __forceinline__ void kernel(const D_data<T_in> in, D_data<T_out> out, D_data<float> param,
		const unsigned int ch_in, const unsigned int ch_out, const unsigned int ch_param, const float offset) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		if (out.w > x && out.h > y) {
			float d = param.tex(0, 2, ch_param) * x + param.tex(1, 2, ch_param) * y + 1.0;
			float in_x = __fdividef(param.tex(0, 0, ch_param) * x + param.tex(1, 0, ch_param) * y + param.tex(2, 0, ch_param), d);
			float in_y = __fdividef(param.tex(0, 1, ch_param) * x + param.tex(1, 1, ch_param) * y + param.tex(2, 1, ch_param), d);
			float val = bilinear(in, in_x + offset, in_y + offset, ch_in);
			if (sizeof(uint16) == sizeof(T_out)) {
				out.val(x, y, ch_out) = K_func::clip_uint16(val);
			}
			else {
				out.val(x, y, ch_out) = static_cast<T_out>(val);
			}
		}
	}
	const D_mem<T_out>& get_out() const { return _out; }
	const D_mem<float>& get_param() const { return _param; }
	void imwrite(const std::string& file_name, const int metric = PHOTOMETRIC_MINISBLACK) const {
		_out.imwrite(file_name, metric);
	}
	void param_to_csv(const std::string& file_name) const {
		_param.write_csv(file_name);
	}
private:
	D_mem<T_out> _out;
	D_mem<float> _param;
};

void unit_test_Warp(int argc, char* argv[]) {
	const std::string in_file = argv[1];
	const std::string out_dir = argv[2];
	// input img
	D_mem<uint16> in_img;
	in_img.imread(in_file);
	// ++++++++++++++++
	// test for warp
	// ++++++++++++++++
	Warped_img<uint16, uint16> img_warp;
	img_warp.alloc_out(in_img.d_data.w * 2, in_img.d_data.h * 2, 10);
	img_warp.alloc_param(10);
	// set param
	img_warp.set_resize(0.4, 0.8, 0);
	img_warp.set_resize(0.8, 0.4, 1);
	img_warp.set_resize(1.0, 1.0, 2);
	img_warp.set_resize(1.4, 1.8, 3);
	img_warp.set_resize(1.8, 1.4, 4);
	img_warp.set_rot(in_img.d_data.w / 2, in_img.d_data.h / 2, 30, 5);
	img_warp.set_rot(in_img.d_data.w / 2, in_img.d_data.h / 2, 60, 6);
	img_warp.set_rot(in_img.d_data.w / 2, in_img.d_data.h / 2, -30, 7);
	img_warp.set_rot(in_img.d_data.w / 2, in_img.d_data.h / 2, -60, 8);
	img_warp.set_perspective(122, 74, 422, 122, 367, 258, 144, 295, 9);
	// do process
	img_warp.do_proc(in_img, 2, 0, 0);
	img_warp.do_proc(in_img, 2, 1, 1);
	img_warp.do_proc(in_img, 2, 2, 2);
	img_warp.do_proc(in_img, 2, 3, 3);
	img_warp.do_proc(in_img, 2, 4, 4);
	img_warp.do_proc(in_img, 2, 5, 5);
	img_warp.do_proc(in_img, 2, 6, 6);
	img_warp.do_proc(in_img, 2, 7, 7);
	img_warp.do_proc(in_img, 2, 8, 8);
	img_warp.do_proc(in_img, 2, 9, 9);
	// output
	img_warp.imwrite(out_dir + "/out_03_warp.tif", PHOTOMETRIC_MINISBLACK);
	img_warp.param_to_csv(out_dir + "/out_03_warp.csv");
}
