//#define _USE_MATH_DEFINES
//#include <cmath>
#pragma once
namespace CudaRasterizer {
	// other data for rasterizer
	struct BasisModelInfo {
		int pixel_count = 0;
		int basis_image_width = 0;
		int basis_image_height = 0;
		int coff_num = 16;
		float basis_min_phi = 0;
		float basis_max_phi = 3.14159265358979323846;
		float basis_min_theta = 0;
		float basis_max_theta = 2 * 3.14159265358979323846;
	};
}