#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <npp.h>
#include "Mem.h"
#include "Basic.h"

// Warning : low speed (first time)
template<typename T_in> class Op_stat {
public:
	void open(const D_mem<T_in>& in_img) {
		_cast_img_0.alloc_out(in_img.d_data.w, in_img.d_data.h, in_img.d_data.ch);
		int hpBufferSize = _get_max_buffer_size(static_cast<int>(in_img.d_data.w), static_cast<int>(in_img.d_data.h));
		_buf.open_2d(hpBufferSize, 1, 1);
	}
	double do_proc(const D_mem<T_in>& in, const unsigned int& ch_in, const std::string& mode) {
		NppiSize oSizeROI = { static_cast<int>(in.d_data.w), static_cast<int>(in.d_data.h) };
		_cast_img_0.do_proc(in);  // cast all ch
		Stopwatch sw;
		cudaMalloc(&_out_dev, sizeof(Npp64f));
		if ("mean" == mode) {
			nppiMean_32f_C1R(&(_cast_img_0.get_out().d_data.val(0, 0, ch_in)), _cast_img_0.get_out().d_data.pitch, oSizeROI, &(_buf.d_data.val(0, 0, 0)), _out_dev);
		}
		else if ("stdev" == mode) {
			nppiMean_StdDev_32f_C1R(&(_cast_img_0.get_out().d_data.val(0, 0, ch_in)), _cast_img_0.get_out().d_data.pitch, oSizeROI, &(_buf.d_data.val(0, 0, 0)), nullptr, _out_dev);
		}
		else if ("max" == mode) {
			nppiMax_32f_C1R(&(_cast_img_0.get_out().d_data.val(0, 0, ch_in)), _cast_img_0.get_out().d_data.pitch, oSizeROI, &(_buf.d_data.val(0, 0, 0)), (Npp32f*)(_out_dev));
		}
		else if ("min" == mode) {
			nppiMin_32f_C1R(&(_cast_img_0.get_out().d_data.val(0, 0, ch_in)), _cast_img_0.get_out().d_data.pitch, oSizeROI, &(_buf.d_data.val(0, 0, 0)), (Npp32f*)(_out_dev));
		}
		else {
			assert(false);
		}
		cudaMemcpy(&_out_hst, _out_dev, sizeof(Npp64f), cudaMemcpyDeviceToHost);
		cudaFree(_out_dev);
		sw.print_time(__FUNCTION__);
		if ("max" == mode || "min" == mode) {
			_out_hst = (double)(((Npp32f*)(&_out_hst))[0]);
		}
		return _out_hst;
	}
private:
	int _get_max_buffer_size(const int w, const int h) {
		NppiSize oSizeROI = { w, h };
		int hpBufferSize = 0;
		int tmp = 0;
		nppiMaxGetBufferHostSize_32f_C1R(oSizeROI, &tmp);
		if (hpBufferSize < tmp) { hpBufferSize = tmp; }
		nppiMinGetBufferHostSize_32f_C1R(oSizeROI, &tmp);
		if (hpBufferSize < tmp) { hpBufferSize = tmp; }
		nppiMeanGetBufferHostSize_32f_C1R(oSizeROI, &tmp);
		if (hpBufferSize < tmp) { hpBufferSize = tmp; }
		nppiMeanStdDevGetBufferHostSize_32f_C1R(oSizeROI, &tmp);
		if (hpBufferSize < tmp) { hpBufferSize = tmp; }
		return hpBufferSize;
	}
	Cast_img<T_in, float> _cast_img_0;  // in_img (T_in -> float)
	D_mem<Npp8u> _buf;
	Npp64f* _out_dev;
	double _out_hst;
};

void unit_test_Stat(int argc, char* argv[]) {
	const std::string in_file = argv[1];
	const std::string out_dir = argv[2];
	// input img
	D_mem<uint16> in_img;
	in_img.imread(in_file);
	// ++++++++++++++++
	// test for Op_stat
	// ++++++++++++++++
	Op_stat<uint16> op_stat;
	op_stat.open(in_img);
	std::cout << "mean = " << op_stat.do_proc(in_img, 2, "mean") << std::endl;
	std::cout << "stdev = " << op_stat.do_proc(in_img, 2, "stdev") << std::endl;
	std::cout << "max = " << op_stat.do_proc(in_img, 2, "max") << std::endl;
	std::cout << "min = " << op_stat.do_proc(in_img, 2, "min") << std::endl;
}
