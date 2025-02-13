#include "ofxOnnxRuntime.h"
#include "ofMain.h"

namespace ofxOnnxRuntime
{
#ifdef _MSC_VER
	static std::wstring to_wstring(const std::string &str)
	{
		unsigned len = str.size() * 2;
		setlocale(LC_CTYPE, "");
		wchar_t *p = new wchar_t[len];
		mbstowcs(p, str.c_str(), len);
		std::wstring wstr(p);
		delete[] p;
		return wstr;
	}
#endif

	void BaseHandler::setup(const std::string & onnx_path, const BaseSetting & base_setting, const std::vector<int64_t>& batched_dims, const int & batch_size)
	{
		Ort::SessionOptions session_options;
		session_options.SetIntraOpNumThreads(1);
		session_options.SetIntraOpNumThreads(1);
		session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

		if (base_setting.infer_type == INFER_CUDA) {
			OrtCUDAProviderOptions opts;
			opts.device_id = 0;
			opts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
			opts.do_copy_in_default_stream = 0;
			opts.arena_extend_strategy = 0;
			session_options.AppendExecutionProvider_CUDA(opts);
		}

		// Sets batch size
		this->batch_size = batch_size;
		this->batched_dims = batched_dims;
		this->setup2(onnx_path, session_options);
	}

	void BaseHandler::setup2(const std::string & onnx_path, const Ort::SessionOptions & session_options)
	{
		std::string path = ofToDataPath(onnx_path, true);
#ifdef _MSC_VER
		ort_session = std::make_shared<Ort::Session>(ort_env, to_wstring(path).c_str(), session_options);
#else
		ort_session = std::make_shared<Ort::Session>(ort_env, path.c_str(), session_options);
#endif

		Ort::AllocatorWithDefaultOptions allocator;

		// 1. Gets Input Name/s & Shape ([1, 3, 28, 28]) -- In most cases this is usually just one
		for (std::size_t i = 0; i < ort_session->GetInputCount(); i++) {
			input_node_names.emplace_back(ort_session->GetInputNameAllocated(i, allocator).get());
			input_node_dims = ort_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();

			// Some models might have negative shape values to indicate dynamic shape, e.g., for variable batch size. (?, 3, 28, 28) -> (1, 3, 28, 28)
			for (auto& s : input_node_dims) if (s < 0) s = 1;

			std::cout << input_node_names.at(i) << " : " << PrintShape(input_node_dims) << std::endl;
		}

		// 2. Calculate the product of the dimensions
		for (auto& f : batched_dims) {
			input_node_size *= f;
		}

		// 3. Resize input values array to match input tensor/s
		input_values_handler.resize(batch_size);

		for (auto& tensor : input_values_handler) {
			tensor.resize(input_node_size);
		}

		// 2. Clear up output values
		output_node_dims.clear();
		output_values.clear();
		
		// 3. Gets Output name/s & Shapes
		for (std::size_t i = 0; i < ort_session->GetOutputCount(); i++) {
			output_node_names.emplace_back(ort_session->GetOutputNameAllocated(i, allocator).get());
			auto output_shapes = ort_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
			
			output_values.emplace_back(nullptr);

			std::cout << output_node_names.at(i) << " : " << PrintShape(output_shapes) << std::endl;
		}
	}

	std::vector<Ort::Value>& BaseHandler::run()
	{
		std::vector<Ort::Value> input_tensors;

		// 1. Create 1-D array for all values to create tensor & push all values from input_vals to batch_vals
		std::vector<float> batch_values(input_node_size * batch_size);

		for (const auto& inner_vec : input_values_handler) {
			for (float value : inner_vec) {
				batch_values.push_back(value);
			}
		}

		// 2. Create tensor with batch values { input data, input size, model input dims, model input size}
		input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
			memory_info_handler, batch_values.data(), input_node_size,
			input_node_dims.data(), input_node_dims.size()));

		// transform std::string -> const char*
		std::vector<const char*> input_names_char(input_node_names.size(), nullptr);
		std::transform(std::begin(input_node_names), std::end(input_node_names), std::begin(input_names_char),
			[&](const std::string& str) { return str.c_str(); });

		std::vector<const char*> output_names_char(output_node_names.size(), nullptr);
		std::transform(std::begin(output_node_names), std::end(output_node_names), std::begin(output_names_char),
			[&](const std::string& str) { return str.c_str(); });
		

		try {
			// 3. Run inference, { in names, input data, num of inputs, output names, num of outputs }
			output_values = ort_session->Run(Ort::RunOptions{ nullptr }, 
				input_names_char.data(), input_tensors.data(),
				input_names_char.size(), output_names_char.data(), 
				output_names_char.size());

			return output_values;
		}
		catch (const Ort::Exception& ex) {
			std::cout << "ERROR running model inference: " << ex.what() << std::endl;
			return dummy_output_tensor;
		}
		
	}

	// Prints the shape of the given tensor (ex. input: (1, 1, 512, 512))
	std::string BaseHandler::PrintShape(const std::vector<std::int64_t>& v) {
		std::stringstream ss;
		for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
		ss << v[v.size() - 1];
		return ss.str();
	}

	Ort::Value BaseHandler::GenerateTensor(int batch_size) {
		// Random number generation setup
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(0.0f, 255.0f); // Random values between 0 and 255

		// Calculate the total number of elements for a single tensor (without batch dimension) {?, 8} -> 8
		int tensor_size = CalculateProduct(input_node_dims);

		// Create a vector to hold all the values for the batch (8 * (4)batch_size) -> 32
		std::vector<float> batch_values(batch_size * tensor_size); 

		// Fill the batch with random values
		std::generate(batch_values.begin(), batch_values.end(), [&]() {
			return dis(gen);
		});

		// Fill the batch with random values
		std::generate(batch_values.begin(), batch_values.end(), [&]() {
			return dis(gen);
		});

		// Create the batched dimensions by inserting the batch size at the beginning of the original dimensions
		std::vector<int64_t> batched_dims = {  };  // Start with batch size
		batched_dims.insert(batched_dims.end(), input_node_dims.begin(), input_node_dims.end()); // Add the remaining dimensions
		batched_dims[0] = batch_size;

		return VectorToTensor(batch_values, batched_dims);
	}

	int BaseHandler::CalculateProduct(const std::vector<std::int64_t>& v) {
		int total = 1;
		for (auto& i : v) total *= i;
		return total;
	}

	Ort::Value BaseHandler::VectorToTensor(std::vector<float>& data, const std::vector<std::int64_t>& shape) {
		//// Allocate memory using CPU memory allocator
		//Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

		// Create a tensor from the provided data, shape, and memory info
		auto tensor = Ort::Value::CreateTensor<float>(memory_info_handler, data.data(), data.size(), shape.data(), shape.size());

		// Return the created tensor
		return tensor;
	}
}
