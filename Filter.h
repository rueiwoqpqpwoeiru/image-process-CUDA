#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Mem.h"
#include "Basic.h"
#include "Util.h"

// filtering for specified ch (by float calc)
template<typename T_in, typename T_out> class Conv_img {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_out.open_2d(w, h, ch);
	}
	void alloc_flt(unsigned int w, unsigned int h, unsigned int ch) {
		_flt.open_2d(w, h, ch);
	}
	// gaussian (w, h are odd)
	void set_flt_blur(const float sigma, const unsigned int ch) {
		unsigned int w = _flt.h_data.w;
		unsigned int h = _flt.h_data.h;
		// set host data
		float sum = 0.0;
		if (1e-8 < sigma) {
			for (unsigned int i = 0; i < h; i++) {
				for (unsigned int j = 0; j < w; j++) {
					unsigned int d2 = pow(static_cast<int>(i) - static_cast<int>(h / 2), 2) + pow(static_cast<int>(j) - static_cast<int>(w / 2), 2);
					double val = (1.0 / (2.0 * 3.141592653589793 * pow(sigma, 2))) * exp(-(static_cast<float>(d2)) / (sigma * sigma));
					_flt.h_data.val(j, i, ch) = static_cast<float>(val);
					sum = sum + _flt.h_data.val(j, i, ch);
				}
			}
			// normalize
			for (unsigned int i = 0; i < h; i++) {
				for (unsigned int j = 0; j < w; j++) {
					_flt.h_data.val(j, i, ch) = static_cast<float>(_flt.h_data.val(j, i, ch) / sum);
				}
			}
		}
		else {
			for (unsigned int i = 0; i < h; i++) {
				for (unsigned int j = 0; j < w; j++) {
					_flt.h_data.val(j, i, ch) = 0.0;
				}
			}
			_flt.h_data.val(h / 2, w / 2, 0) = 1.0;
		}
		// trans host to device (trans other channels)
		_flt.h2d();
	}
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const unsigned int& ch_out, const unsigned int& ch_flt, const dim3& block = dim3(32, 8)) {
		Proc_unit p(in.d_data.w, in.d_data.h, block);
		Stopwatch sw;
		K_func::generic_kernel<Conv_img> << < p.grid, p.block >> > (this, in.d_data, _out.d_data, _flt.d_data, ch_in, ch_out, ch_flt);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const D_data<T_in> in, D_data<T_out> out, D_data<float> flt,
		const unsigned int ch_in, const unsigned int ch_out, const unsigned int ch_flt) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		unsigned int half_w = flt.w / 2;
		unsigned int half_h = flt.h / 2;
		if ((half_w <= x && x < in.w - half_w) && (half_h <= y && y < in.h - half_h)) {
			float sum = 0.0;
			for (unsigned int i = 0; i < flt.h; i++) {
				for (unsigned int j = 0; j < flt.w; j++) {
					sum = __fmaf_rn(static_cast<float>(in.tex(x - half_w + j, y - half_h + i, ch_in)), flt.tex(j, i, ch_flt), sum);
					// __fmaf_rn(a,b,c) compute (a * b + c) as a single operation, in round-to-nearest-even mode
				}
			}
			if (sizeof(uint16) == sizeof(T_out)) {
				out.val(x, y, ch_out) = K_func::clip_uint16(sum);
			}
			else {
				out.val(x, y, ch_out) = static_cast<T_out>(sum);
			}
		}
	}
	const D_mem<T_out>& get_out() const { return _out; }
	const D_mem<float>& get_flt() const { return _flt; }
private:
	D_mem<T_out> _out;
	D_mem<float> _flt;
};

// morphology_filtered_img_erode_dilate
template<typename T_in, typename T_out> class Morphology_filtered_img_erode_dilate {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_out.open_2d(w, h, ch);
	}
	void alloc_flt(unsigned int w, unsigned int h, unsigned int ch) {
		_flt.open_2d(w, h, ch);
	}
	// binary (w, h are odd)
	void set_flt(unsigned int w, unsigned int h, const unsigned int ch) {
		// set host data ( set zero )
		for (unsigned int i = 0; i < _flt.h_data.h; i++) {
			for (unsigned int j = 0; j < _flt.h_data.w; j++) {
				_flt.h_data.val(j, i, ch) = 0;
			}
		}
		// set host data
		for (unsigned int i = 0; i < h; i++) {
			for (unsigned int j = 0; j < w; j++) {
				_flt.h_data.val(_flt.h_data.w / 2 - w / 2 + j, _flt.h_data.h / 2 - h / 2 + i, ch) = 1;
			}
		}
		// trans host to device (trans other channels)
		_flt.h2d();
	}
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const unsigned int& ch_out, const unsigned int& ch_flt, const std::string& mode, const dim3& block = dim3(32, 8)) {
		Proc_unit p(in.d_data.w, in.d_data.h, block);
		Stopwatch sw;
		bool is_erode;
		if ("erode" == mode) {
			is_erode = true;
		}
		else if ("dilate" == mode) {
			is_erode = false;
		}
		else {
			assert(false);
		}
		K_func::generic_kernel<Morphology_filtered_img_erode_dilate> << < p.grid, p.block >> > (this, in.d_data, _out.d_data, _flt.d_data, ch_in, ch_out, ch_flt, is_erode);
		sw.print_time(__FUNCTION__);
	}
	__device__ __forceinline__ void kernel(const D_data<T_in> in, D_data<T_out> out, D_data<uint8> flt,
		const unsigned int ch_in, const unsigned int ch_out, const unsigned int ch_flt, const bool is_erode) {
		unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
		unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
		unsigned int half_w = flt.w / 2;
		unsigned int half_h = flt.h / 2;
		if ((half_w <= x && x < in.w - half_w) && (half_h <= y && y < in.h - half_h)) {
			T_in tmp = in.tex(x, y, ch_in);
			// erode
			if (true == is_erode) {
				for (unsigned int i = 0; i < flt.h; i++) {
					for (unsigned int j = 0; j < flt.w; j++) {
						if (1 == flt.tex(j, i, ch_flt)) {
							T_in val = in.tex(x - half_w + j, y - half_h + i, ch_in);
							if (tmp > val) {
								tmp = val;
							}
						}
					}
				}
			}
			// dilate
			else {
				for (unsigned int i = 0; i < flt.h; i++) {
					for (unsigned int j = 0; j < flt.w; j++) {
						if (1 == flt.tex(j, i, ch_flt)) {
							T_in val = in.tex(x - half_w + j, y - half_h + i, ch_in);
							if (tmp < val) {
								tmp = val;
							}
						}
					}
				}
			}
			if (sizeof(uint16) == sizeof(T_out)) {
				out.val(x, y, ch_out) = K_func::clip_uint16(tmp);
			}
			else {
				out.val(x, y, ch_out) = static_cast<T_out>(tmp);
			}
		}
	}
	const D_mem<T_out>& get_out() const { return _out; }
	const D_mem<uint8>& get_flt() const { return _flt; }
private:
	D_mem<T_out> _out;
	D_mem<uint8> _flt;
};

// morphology_filtered_img_open_close
template<typename T_in, typename T_out> class Morphology_filtered_img_open_close {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_buf.alloc_out(w, h, ch);
		_out.alloc_out(w, h, ch);
	}
	void alloc_flt(unsigned int w, unsigned int h, unsigned int ch) {
		_buf.alloc_flt(w, h, ch);
		_out.alloc_flt(w, h, ch);
	}
	// binary (w, h are odd)
	void set_flt(unsigned int w, unsigned int h, const unsigned int ch) {
		_buf.set_flt(w, h, ch);
		_out.set_flt(w, h, ch);
	}
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const unsigned int& ch_out, const unsigned int& ch_flt, const std::string& mode, const dim3& block = dim3(32, 8)) {
		if ("open" == mode) {
			_buf.do_proc(in, ch_in, ch_out, ch_flt, "erode");
			_out.do_proc(_buf.get_out(), ch_out, ch_out, ch_flt, "dilate");
		}
		else if ("close" == mode) {
			_buf.do_proc(in, ch_in, ch_out, ch_flt, "dilate");
			_out.do_proc(_buf.get_out(), ch_out, ch_out, ch_flt, "erode");
		}
		else {
			assert(false);
		}
	}
	const D_mem<T_out>& get_buf() const { return _buf.get_out(); }
	const D_mem<T_out>& get_out() const { return _out.get_out(); }
	const D_mem<uint8>& get_flt() const { return _out.get_flt(); }
private:
	Morphology_filtered_img_erode_dilate<T_in, T_out> _buf;  // 1st output
	Morphology_filtered_img_erode_dilate<T_in, T_out> _out;  // 2nd output
};

// Warning : low speed (due to dynamic memory alloc)
template<typename T_in, typename T_out> class Median_filtered_img {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch) {
		_cast_img_0.alloc_out(w, h, ch);
		_out.open_2d(w, h, ch);
		_cast_img_1.alloc_out(w, h, ch);
	}
	void alloc_flt(unsigned int ch) {
		_flt_size.resize(ch);
	}
	// binary (w, h are odd)
	void set_flt(unsigned int w, unsigned int h, const unsigned int ch) {
		_flt_size[ch] = { static_cast<int>(w), static_cast<int>(h) };
	}
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const unsigned int& ch_out, const unsigned int& ch_flt) {
		NppiSize oSizeROI = { in.d_data.w, in.d_data.h };
		NppiPoint oAnchor = { _flt_size[ch_flt].width / 2, _flt_size[ch_flt].height / 2 };
		_cast_img_0.do_proc(in);  // cast all ch
		Stopwatch sw;
		nppiFilterMedian_32f_C1R(&(_cast_img_0.get_out().d_data.val(0, 0, ch_in)), _cast_img_0.get_out().d_data.pitch, &(_out.d_data.val(0, 0, ch_out)), _out.d_data.pitch, oSizeROI, _flt_size[ch_flt], oAnchor, _buf);
		sw.print_time(__FUNCTION__);
		_cast_img_1.do_proc(_out);  // cast all ch
	}
	const D_mem<T_out>& get_out() const { return _cast_img_1.get_out(); }
private:
	Npp8u* _buf;  // It seems that nppiFilterMedianGetBufferSize does not work (buffer is automatically allocated)
	std::vector<NppiSize> _flt_size;
	Cast_img<T_in, float> _cast_img_0;  // in_img (T_in -> float)
	D_mem<float> _out;
	Cast_img<float, T_out> _cast_img_1;  // out_img (float -> T_out)
};

// filtering for specified ch (this class is wrapper)
template<typename T_in, typename T_out> class Filtered_img {
public:
	void alloc_out(unsigned int w, unsigned int h, unsigned int ch, const std::string& mode = "conv") {
		if ("conv" == mode) {
			_mode = _mode_conv;
			_conv.alloc_out(w, h, ch);
		}
		else if ("erode" == mode) {
			_mode = _mode_erode;
			_erode_dilate.alloc_out(w, h, ch);
		}
		else if ("dilate" == mode) {
			_mode = _mode_dilate;
			_erode_dilate.alloc_out(w, h, ch);
		}
		else if ("open" == mode) {
			_mode = _mode_open;
			_open_close.alloc_out(w, h, ch);
		}
		else if ("close" == mode) {
			_mode = _mode_close;
			_open_close.alloc_out(w, h, ch);
		}
		else if ("median" == mode) {
			_mode = _mode_median;
			_median.alloc_out(w, h, ch);
		}
		else {
			assert(false);
		}
	}
	void alloc_flt(unsigned int w, unsigned int h, unsigned int ch) {
		if (_mode_conv == _mode) {
			_conv.alloc_flt(w, h, ch);
		}
		else if (_mode_erode == _mode || _mode_dilate == _mode) {
			_erode_dilate.alloc_flt(w, h, ch);
		}
		else if (_mode_open == _mode || _mode_close == _mode) {
			_open_close.alloc_flt(w, h, ch);
		}
		else {
			assert(false);
		}
	}
	void alloc_flt(unsigned int ch) {
		if (_mode_median == _mode) {
			_median.alloc_flt(ch);
		}
		else {
			assert(false);
		}
	}
	// (w, h are odd)
	void set_flt_blur(const float sigma, const unsigned int ch) {
		if (_mode_conv != _mode) {
			assert(false);
		}
		_conv.set_flt_blur(sigma, ch);
	}
	void set_flt(unsigned int w, unsigned int h, const unsigned int ch) {
		if (_mode_erode == _mode || _mode_dilate == _mode) {
			_erode_dilate.set_flt(w, h, ch);
		}
		else if (_mode_open == _mode || _mode_close == _mode) {
			_open_close.set_flt(w, h, ch);
		}
		else if (_mode_median == _mode) {
			_median.set_flt(w, h, ch);
		}
	}
	void do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const unsigned int& ch_out, const unsigned int& ch_flt, const dim3& block = dim3(32, 8)) {
		if (_mode_conv == _mode) {
			_conv.do_proc(in, ch_in, ch_out, ch_flt, block);
		}
		else if (_mode_erode == _mode) {
			_erode_dilate.do_proc(in, ch_in, ch_out, ch_flt, "erode", block);
		}
		else if (_mode_dilate == _mode) {
			_erode_dilate.do_proc(in, ch_in, ch_out, ch_flt, "dilate", block);
		}
		else if (_mode_open == _mode) {
			_open_close.do_proc(in, ch_in, ch_out, ch_flt, "open", block);
		}
		else if (_mode_close == _mode) {
			_open_close.do_proc(in, ch_in, ch_out, ch_flt, "close", block);
		}
		else if (_mode_median == _mode) {
			_median.do_proc(in, ch_in, ch_out, ch_flt);
		}
	}
	const D_mem<T_out>& get_out() const {
		if (_mode_conv == _mode) {
			return _conv.get_out();
		}
		else if (_mode_erode == _mode || _mode_dilate == _mode) {
			return _erode_dilate.get_out();
		}
		else if (_mode_open == _mode || _mode_close == _mode) {
			return _open_close.get_out();
		}
		else if (_mode_median == _mode) {
			return _median.get_out();
		}
	}
	void imwrite(const std::string& file_name, const int metric = PHOTOMETRIC_MINISBLACK) const {
		if (_mode_conv == _mode) {
			_conv.get_out().imwrite(file_name, metric);
		}
		else if (_mode_erode == _mode || _mode_dilate == _mode) {
			_erode_dilate.get_out().imwrite(file_name, metric);
		}
		else if (_mode_open == _mode || _mode_close == _mode) {
			_open_close.get_out().imwrite(file_name, metric);
		}
		else if (_mode_median == _mode) {
			return _median.get_out().imwrite(file_name, metric);
		}
	}
	void flt_to_csv(const std::string& file_name) const {
		if (_mode_conv == _mode) {
			_conv.get_flt().write_csv(file_name);
		}
		else if (_mode_erode == _mode || _mode_dilate == _mode) {
			_erode_dilate.get_flt().write_csv(file_name);
		}
		else if (_mode_open == _mode || _mode_close == _mode) {
			_open_close.get_flt().write_csv(file_name);
		}
		else if (_mode_median == _mode) {
			assert(false);
		}
	}
private:
	enum Mode {
		_mode_conv, _mode_erode, _mode_dilate, _mode_open, _mode_close, _mode_median
	} _mode;
	Conv_img<T_in, T_out> _conv;
	Morphology_filtered_img_erode_dilate<T_in, T_out> _erode_dilate;
	Morphology_filtered_img_open_close<T_in, T_out> _open_close;
	Median_filtered_img<T_in, T_out> _median;
};

void unit_test_Filter(int argc, char* argv[]) {
	const std::string in_file = argv[1];
	const std::string out_dir = argv[2];
	// input img
	D_mem<uint16> in_img;
	in_img.imread(in_file);
	// ++++++++++++++++
	// test for conv
	// ++++++++++++++++
	Filtered_img<uint16, uint16> img_conv;
	img_conv.alloc_out(in_img.d_data.w, in_img.d_data.h, 5, "conv");
	img_conv.alloc_flt(65, 65, 5);
	// set filter
	img_conv.set_flt_blur(0.0, 0);
	img_conv.set_flt_blur(1.0, 1);
	img_conv.set_flt_blur(3.0, 2);
	img_conv.set_flt_blur(7.0, 3);
	img_conv.set_flt_blur(16.0, 4);
	// do process
	img_conv.do_proc(in_img, 2, 0, 0);
	img_conv.do_proc(in_img, 2, 1, 1);
	img_conv.do_proc(in_img, 2, 2, 2);
	img_conv.do_proc(in_img, 2, 3, 3);
	img_conv.do_proc(in_img, 2, 4, 4);
	// output
	img_conv.imwrite(out_dir + "/out_02_conv.tif", PHOTOMETRIC_MINISBLACK);
	img_conv.flt_to_csv(out_dir + "/out_02_conv.csv");
	// ++++++++++++++++
	// test for median
	// ++++++++++++++++
	Filtered_img<uint16, uint16> img_median;
	img_median.alloc_out(in_img.d_data.w, in_img.d_data.h, 5, "median");
	img_median.alloc_flt(5);
	// set filter
	img_median.set_flt(1, 1, 0);
	img_median.set_flt(3, 1, 1);
	img_median.set_flt(1, 3, 2);
	img_median.set_flt(3, 3, 3);
	img_median.set_flt(5, 5, 4);
	// do process
	img_median.do_proc(in_img, 2, 0, 0);
	img_median.do_proc(in_img, 2, 1, 1);
	img_median.do_proc(in_img, 2, 2, 2);
	img_median.do_proc(in_img, 2, 3, 3);
	img_median.do_proc(in_img, 2, 4, 4);
	// output
	img_median.get_out().imwrite(out_dir + "/out_02_median.tif", PHOTOMETRIC_MINISBLACK);
	// ++++++++++++++++
	// test for morphology filter
	// ++++++++++++++++
	std::string mode[4] = { "erode", "dilate", "open", "close" };
	for (int i = 0; i < 4; i++) {
		Filtered_img<uint16, uint16> img_morphology;
		img_morphology.alloc_out(in_img.d_data.w, in_img.d_data.h, 5, mode[i]);
		img_morphology.alloc_flt(65, 65, 5);
		// set filter
		img_morphology.set_flt(0, 0, 0);
		img_morphology.set_flt(1, 3, 1);
		img_morphology.set_flt(1, 5, 2);
		img_morphology.set_flt(3, 1, 3);
		img_morphology.set_flt(5, 1, 4);
		// do process
		img_morphology.do_proc(in_img, 2, 0, 0);
		img_morphology.do_proc(in_img, 2, 1, 1);
		img_morphology.do_proc(in_img, 2, 2, 2);
		img_morphology.do_proc(in_img, 2, 3, 3);
		img_morphology.do_proc(in_img, 2, 4, 4);
		// output
		img_morphology.imwrite(out_dir + "/out_02_" + mode[i] + ".tif", PHOTOMETRIC_MINISBLACK);
		img_morphology.flt_to_csv(out_dir + "/out_02_" + mode[i] + ".csv");
	}
}
