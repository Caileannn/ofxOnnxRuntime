#pragma once

#include <onnxruntime_cxx_api.h>

namespace ofxOnnxRuntime
{
	enum InferType
	{
		INFER_CPU = 0,
		INFER_CUDA,
		INFER_TENSORRT
	};

	struct BaseSetting
	{
		InferType infer_type;
		int device_id;
	};

	class BaseHandler
	{
	public:
		BaseHandler() {}

		void setup(const std::string& onnx_path, const BaseSetting& base_setting = BaseSetting{ INFER_CPU, 0 });
		void setup2(const std::string& onnx_path, const Ort::SessionOptions& session_options);

		Ort::Value& run();

		// Utilities
		std::string PrintShape(const std::vector<std::int64_t>& v);
		Ort::Value GenerateTensor();
		int CalculateProduct(const std::vector<std::int64_t>& v);
		Ort::Value VectorToTensor(std::vector<float>& data, const std::vector<std::int64_t>& shape);

	protected:
		Ort::Env ort_env;
		std::shared_ptr<Ort::Session> ort_session;

		std::vector<std::string> input_node_names;
		std::vector<int64_t> input_node_dims; // 1 input only.
		std::size_t input_tensor_size = 1;

		Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

		std::vector<std::string> output_node_names;
		std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
		std::vector<Ort::Value> output_values;

		Ort::Value dummy_tensor{ nullptr };
		std::vector<Ort::Value> dummy_output_tensor;

		int num_outputs = 1;
	};
}
