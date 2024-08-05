#pragma once

#include "../../lib.h"

ADD_TESTCASE(Testcase("elemwise_correctness", {"kernel", "correctness"}, "kernel-level elementwise op correctness test", [](Testcase* this_testcase) {
	struct SubcaseConfig {
		string subcase_name;
		vector<int> dims;
		vector<size_t> layout_i0, layout_i1, layout_i2;
	};
	vector<SubcaseConfig> subcases = {
		{
			"1D",
			{180},
			{1}, {1}, {1}
		},
		{
			"2D row major perfect",
			{10, 18},
			{32, 1}, {24, 1}, {48, 1}
		},
		{
			"2D row major non perfect",
			{10, 18},
			{18, 1}, {19, 1}, {123, 1}
		},
		{
			"2D col major perfect",
			{10, 18},
			{1, 16}, {1, 24}, {1, 32}
		},
		{
			"2D col major non perfect",
			{10, 18},
			{1, 10}, {1, 11}, {1, 13}
		},
		{
			"3D row major perfect",
			{2, 5, 18},
			{192, 24, 1}, {192, 24, 1}, {192, 24, 1}
		},
		{
			"3D row major non perfect",
			{2, 5, 18},
			{90, 18, 1}, {92, 18, 1}, {100, 20, 1}
		},
		{
			"3D col major perfect",
			{2, 5, 18},
			{1, 8, 64}, {1, 8, 64}, {1, 8, 64}
		},
		{
			"3D col major non perfect",
			{2, 5, 18},
			{1, 2, 10}, {1, 2, 10}, {1, 2, 10}
		}
	};
	for (const SubcaseConfig &subcase : subcases) {
		int numel = 1;
		for (int dim : subcase.dims) numel *= dim;
		assert(numel == 2*5*18);

		Subcase t({
			1,
			0,
			true,
			{
				80
			}
		}, subcase.subcase_name);
		kn::DTensor i0 = t.new_input(subcase.dims, Gen::ARange(-1.0, 1.0), subcase.layout_i0);
		kn::DTensor i1 = t.new_input(subcase.dims, Gen::ARange(1.0, 2.0), subcase.layout_i1);
		kn::DTensor x0 = t.g.add(i0, i1);
		kn::DTensor i2 = t.new_input(subcase.dims, Gen::ARange(2.0, 3.0), subcase.layout_i2);
		kn::DTensor x1 = t.g.mul(i1, i2);
		kn::DTensor x2 = t.g.div(x0, x1);
		kn::DTensor o0 = t.g.exp(x2);
		kn::DTensor o1 = t.g.exp(o0);
		// o0 = exp((i0 + i1) / (i1 * i2))
		t.mark_output(o0, subcase.dims, {1.000,1.008,1.017,1.025,1.033,1.041,1.049,1.057,1.064,1.072,1.080,1.087,1.095,1.102,1.110,1.117,1.124,1.132,1.139,1.146,1.153,1.160,1.166,1.173,1.180,1.187,1.193,1.200,1.206,1.212,1.219,1.225,1.231,1.237,1.243,1.249,1.255,1.261,1.267,1.273,1.278,1.284,1.289,1.295,1.300,1.306,1.311,1.316,1.321,1.326,1.332,1.337,1.341,1.346,1.351,1.356,1.361,1.365,1.370,1.375,1.379,1.384,1.388,1.392,1.397,1.401,1.405,1.409,1.413,1.417,1.421,1.425,1.429,1.433,1.437,1.441,1.444,1.448,1.452,1.455,1.459,1.462,1.466,1.469,1.473,1.476,1.479,1.482,1.486,1.489,1.492,1.495,1.498,1.501,1.504,1.507,1.510,1.513,1.515,1.518,1.521,1.524,1.526,1.529,1.531,1.534,1.536,1.539,1.541,1.544,1.546,1.549,1.551,1.553,1.555,1.558,1.560,1.562,1.564,1.566,1.568,1.570,1.572,1.574,1.576,1.578,1.580,1.582,1.584,1.586,1.587,1.589,1.591,1.593,1.594,1.596,1.598,1.599,1.601,1.602,1.604,1.606,1.607,1.608,1.610,1.611,1.613,1.614,1.615,1.617,1.618,1.619,1.621,1.622,1.623,1.624,1.626,1.627,1.628,1.629,1.630,1.631,1.632,1.633,1.634,1.635,1.636,1.637,1.638,1.639,1.640,1.641,1.642,1.643,1.644,1.645,1.646,1.646,1.647,1.648});
		t.mark_output(o1, subcase.dims, {2.718,2.741,2.764,2.786,2.809,2.831,2.854,2.877,2.899,2.922,2.944,2.967,2.989,3.011,3.034,3.056,3.078,3.101,3.123,3.145,3.167,3.189,3.211,3.232,3.254,3.276,3.297,3.319,3.340,3.362,3.383,3.404,3.425,3.446,3.467,3.488,3.508,3.529,3.550,3.570,3.590,3.610,3.630,3.650,3.670,3.690,3.710,3.729,3.748,3.768,3.787,3.806,3.825,3.843,3.862,3.881,3.899,3.917,3.935,3.953,3.971,3.989,4.007,4.024,4.042,4.059,4.076,4.093,4.110,4.126,4.143,4.159,4.176,4.192,4.208,4.224,4.239,4.255,4.270,4.286,4.301,4.316,4.331,4.346,4.360,4.375,4.389,4.403,4.417,4.431,4.445,4.459,4.472,4.486,4.499,4.512,4.525,4.538,4.551,4.564,4.576,4.588,4.601,4.613,4.625,4.636,4.648,4.660,4.671,4.682,4.694,4.705,4.716,4.726,4.737,4.748,4.758,4.768,4.779,4.789,4.799,4.808,4.818,4.828,4.837,4.846,4.856,4.865,4.874,4.883,4.891,4.900,4.909,4.917,4.925,4.934,4.942,4.950,4.957,4.965,4.973,4.980,4.988,4.995,5.002,5.010,5.017,5.024,5.030,5.037,5.044,5.050,5.057,5.063,5.069,5.075,5.081,5.087,5.093,5.099,5.105,5.110,5.116,5.121,5.127,5.132,5.137,5.142,5.147,5.152,5.157,5.161,5.166,5.171,5.175,5.180,5.184,5.188,5.192,5.196});
		this_testcase->add_subcase(t);
	}
}));

ADD_TESTCASE(Testcase("elemwise_perf", {"kernel", "perf"}, "kernel-level elementwise op performance test", [](Testcase* this_testcase) {
	const vector<int> dims = {128, 1024, 1024};
	auto epilogue = [dims](const Subcase::RunResult &res) -> string {
		float time_usage = res.avg_time_ms;
		size_t numel = get_cumulative_mul(dims);
		float mem_rw = numel*sizeof(half) * 11;
		static char buf[1000];
		sprintf(buf, "Memory read/write: %.2f GB, Memory BW: %.2f GB/s", (float)mem_rw/(1l<<30), mem_rw / (time_usage/1e3) / 1e9);
		return string(buf);
	};
	Subcase t({
		2,
		10,
		false,
		{
			80
		}
	}, nullopt, epilogue);
	kn::DTensor i0 = t.new_input(dims, Gen::ARange(-1.0, 1.0));
	kn::DTensor i1 = t.new_input(dims, Gen::ARange(1.0, 2.0));
	kn::DTensor x0 = t.g.add(i0, i1);
	kn::DTensor i2 = t.new_input(dims, Gen::ARange(2.0, 3.0));
	kn::DTensor x1 = t.g.mul(i1, i2);
	kn::DTensor x2 = t.g.div(x0, x1);
	kn::DTensor o0 = t.g.exp(x2);
	// o0 = exp((i0 + i1) / (i1 * i2))
	t.mark_output(o0, dims, {});
	this_testcase->add_subcase(t);
}));

ADD_TESTCASE(Testcase("elemwise_bcast_correctness", {"kernel", "correctness"}, "kernel-level elementwise op with bcast correctness test", [](Testcase* this_testcase) {
	Subcase t({
		1,
		0,
		true,
		{
			80
		}
	});
	kn::DTensor i0 = t.new_input({1, 4, 1, 10}, Gen::ARange(-1.0, 1.0));
	kn::DTensor i1 = t.new_input({1, 1, 4, 10}, Gen::ARange(1.0, 2.0));
	kn::DTensor x0 = t.g.add(i0, i1);
	kn::DTensor x1 = t.g.mul(i0, i1);
	kn::DTensor o0 = t.g.add(x0, x1);
	t.mark_output(o0, {1, 4, 4, 10}, {-1.000,-0.899,-0.795,-0.689,-0.580,-0.469,-0.355,-0.239,-0.120,0.001,-1.000,-0.886,-0.770,-0.651,-0.530,-0.406,-0.280,-0.151,-0.020,0.114,-1.000,-0.874,-0.745,-0.614,-0.480,-0.344,-0.205,-0.064,0.080,0.226,-1.000,-0.861,-0.720,-0.576,-0.430,-0.281,-0.130,0.024,0.180,0.339,0.000,0.114,0.230,0.349,0.470,0.594,0.720,0.849,0.980,1.114,0.125,0.251,0.380,0.511,0.645,0.781,0.920,1.061,1.205,1.351,0.250,0.389,0.530,0.674,0.820,0.969,1.120,1.274,1.430,1.589,0.375,0.526,0.680,0.836,0.995,1.156,1.320,1.486,1.655,1.826,1.000,1.126,1.255,1.386,1.520,1.656,1.795,1.936,2.080,2.226,1.250,1.389,1.530,1.674,1.820,1.969,2.120,2.274,2.430,2.589,1.500,1.651,1.805,1.961,2.120,2.281,2.445,2.611,2.780,2.951,1.750,1.914,2.080,2.249,2.420,2.594,2.770,2.949,3.130,3.314,2.000,2.139,2.280,2.424,2.570,2.719,2.870,3.024,3.180,3.339,2.375,2.526,2.680,2.836,2.995,3.156,3.320,3.486,3.655,3.826,2.750,2.914,3.080,3.249,3.420,3.594,3.770,3.949,4.130,4.314,3.125,3.301,3.480,3.661,3.845,4.031,4.220,4.411,4.605,4.801});
	this_testcase->add_subcase(t);
}));
