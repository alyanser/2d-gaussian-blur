#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <optional>
#include <algorithm>
#include <chrono>
#include <string_view>
#include <tga.h>

#include "ops.h"

class Tga_image {
public:
	Tga_image() = default;

	Tga_image(const std::uint32_t img_height, const std::uint32_t img_width, const std::uint32_t bytes_per_pixel) noexcept
		: buffer_(img_height * img_width * bytes_per_pixel)
		, img_({buffer_.empty() ? nullptr : &buffer_[0], bytes_per_pixel, img_width * bytes_per_pixel})
	{
	}

	Tga_image(const Tga_image & rhs) noexcept
		: buffer_(rhs.buffer_)
		, img_(rhs.img_)
	{
		img_.pixels = buffer_.empty() ? nullptr : &buffer_[0];
	}

	Tga_image(Tga_image && rhs) noexcept 
		: buffer_(std::move(rhs.buffer_))
		, img_(std::move(rhs.img_))
	{
		img_.pixels = buffer_.empty() ? nullptr : &buffer_[0];
	}

	Tga_image & operator = (Tga_image && rhs) noexcept {
		this->~Tga_image();
		new (this) Tga_image(std::move(rhs));
		return *this;
	}

	Tga_image & operator = (const Tga_image & rhs) noexcept {
		this->~Tga_image();
		new (this) Tga_image(rhs);
		return *this;
	}

	tga::Image * operator -> () noexcept {
		return &img_;
	}

	const tga::Image * operator -> () const noexcept {
		return &img_;
	}

	tga::Image & operator * () noexcept {
		return img_;
	}

	const tga::Image & operator * () const noexcept {
		return img_;
	}

private:
	std::vector<std::uint8_t> buffer_;
	tga::Image img_{};
};

class Timer {
public:
	Timer(){
		start_ = std::chrono::high_resolution_clock::now();
	}

	~Timer(){
		end_ = std::chrono::high_resolution_clock::now();
		std::cout << "-----------\ntime taken:-\nseconds: " << std::chrono::duration_cast<std::chrono::seconds>(end_ - start_).count() << "\n"
			<< "milliseconds: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count() << "\n"
			<< "microseconds: " << std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() << "\n----------\n";
	}

private:
	std::chrono::time_point<std::chrono::high_resolution_clock> start_;
	std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

// basic raii wrapper class over FILE* (unfortunately tga-lib requires FILE* :()
class File {
public:
	File(const std::string_view file_path, const std::string_view open_mode) noexcept
		: file_(std::fopen(file_path.data(), open_mode.data()))
	{}

	File(const File & rhs) = delete;
	File(File && rhs) = delete;
	File & operator = (const File & rhs) = delete;
	File & operator = (File && rhs) = delete;

	~File() noexcept {

		if(file_){
			std::fclose(file_);
		}
	}

	[[nodiscard]]
	FILE * operator * () noexcept {
		return file_;
	}

	[[nodiscard]]
	const FILE * operator * () const noexcept {
		return file_;
	}

	operator bool () const noexcept {
		return file_;
	}
private:
	FILE * file_{};
};

[[nodiscard]]
std::pair<tga::Header, Tga_image> extract_tga_image(const std::string_view tga_img_path){
	File input_img_file(tga_img_path.data(), "rb"); // open the given file in read & binary mode
	
	if(!input_img_file){
		throw std::runtime_error("given file could not be opened for reading");
	}

	tga::StdioFileInterface tga_stdio_interface(*input_img_file);
	tga::Decoder decoder(&tga_stdio_interface);
	tga::Header header;

	if(!decoder.readHeader(header)){
		throw std::runtime_error("given file isn't either a tga image or it's corrupted. tga header could not be extracted");
	}

	Tga_image img(header.height, header.width, header.bytesPerPixel());

	if(!decoder.readImage(header, *img, nullptr)){
		throw std::runtime_error("given tga image seems to be corrupted. pixel blob couldn't be extrracted");
	}

	return std::make_pair(std::move(header), std::move(img));
}

void write_tga_image(const tga::Header & header, const Tga_image & img, const std::string_view output_file_path){
	File output_img_file(output_file_path.data(), "wb");

	if(!output_img_file){
		throw std::runtime_error("output file could not be opened for writing");
	}

	tga::StdioFileInterface tga_stdio_interface(*output_img_file);
	tga::Encoder encoder(&tga_stdio_interface);

	encoder.writeHeader(header);
	encoder.writeImage(header, *img);
}

enum class Process_type {
	CPU,
	GPU
};

template<Process_type PT>
[[nodiscard]]
std::vector<std::vector<double>> get_kernel(const int width, const int height, const int sigma) noexcept {
	std::vector<std::vector<double>> kernel(height, std::vector<double>(width));
	double sum = 0;

	for(int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){

			if constexpr(PT == Process_type::CPU){
				kernel[i][j] = std::exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
				sum += kernel[i][j];
			}else{
				using cuda::Ops;

				kernel[i][j] = Ops(-(Ops(i) * Ops(i) + Ops(j) * Ops(j)) / (Ops(2) * Ops(sigma) * Ops(sigma))).exp()
					/ (Ops(2) * Ops(M_PI) * Ops(sigma) * Ops(sigma));

				sum = Ops(sum) + Ops(kernel[i][j]);
			}
		}
	}

	std::transform(std::begin(kernel), std::end(kernel), std::begin(kernel), [sum](auto & row){

		std::transform(std::begin(row), std::end(row), std::begin(row), [sum](const auto filter_val){
			using cuda::Ops;

			if constexpr (PT == Process_type::CPU){
				return filter_val / sum;
			}else{
				using cuda::Ops;
				return Ops(filter_val) / Ops(sum);
			}
		});

		return row;
	});

	return kernel;
}

template<Process_type PT>
[[nodiscard]]
std::pair<tga::Header, Tga_image> convolve_tga_image(const tga::Header & header, const Tga_image & img, const int sigma) noexcept {
	[[maybe_unused]] Timer timer;
	constexpr auto filter_height = 5;
	constexpr auto filter_width = 5;

	const auto filter = get_kernel<PT>(filter_height, filter_width, sigma);

	tga::Header new_header = header;
	new_header.width -= filter_width + 1;
	new_header.height -= filter_height + 1;

	Tga_image convolved_img(new_header.height, new_header.width, new_header.bytesPerPixel());

	for(int i = 0; i < new_header.height; ++i){
		for(int j = 0; j < new_header.width; ++j){
			for(int k = 0; k < header.bytesPerPixel(); ++k){
				for(int h = i; h < i + filter_height; ++h){
					for(int w = j; w < j + filter_width; ++w){

						if constexpr(PT == Process_type::CPU){
							const auto new_img_pixel_idx = i * convolved_img->rowstride + j * header.bytesPerPixel() + k;
							const auto old_img_pixel_idx = h * img->rowstride + w * header.bytesPerPixel() + k;
							const auto kernel_x = h - i;
							const auto kernel_y = w - j;

							convolved_img->pixels[new_img_pixel_idx] += img->pixels[old_img_pixel_idx] * filter[kernel_x][kernel_y];
						}else{
							using cuda::Ops;

							const auto new_img_pixel_idx = static_cast<std::size_t>(Ops(i) * Ops(convolved_img->rowstride) + Ops(j) 
								* Ops(header.bytesPerPixel()) + Ops(k));

							const auto old_img_pixel_idx = static_cast<std::size_t>(Ops(h) * Ops(img->rowstride) + Ops(w)
								* Ops(header.bytesPerPixel()) + Ops(k));

							const auto kernel_x = static_cast<std::size_t>(Ops(h) - Ops(i));
							const auto kernel_y = static_cast<std::size_t>(Ops(w) - Ops(j));

							convolved_img->pixels[new_img_pixel_idx] = Ops(convolved_img->pixels[new_img_pixel_idx])
								+ Ops(img->pixels[old_img_pixel_idx]) * Ops(filter[kernel_x][kernel_y]);
						}
					}
				}
			}
		}
	}

	return std::make_pair(std::move(new_header), std::move(convolved_img));
}

int main(int argc, char ** argv){

	if(argc < 3 || argc > 7){
		std::cerr << "Usage: " << argv[0] << " path_to_tga_img deviation\n\n"
			"Options:\n"
			"-o output_image_path -> store the convolved image at given path instead of overwriting\n"
			"-g -> use gpu\n"
			"-c -> use cpu\n\n"
			"note: if neither -c or -g is provided, both cpu and gpu will be used\n";

		return 1;
	}

	const auto sigma = [&argv]() -> std::optional<int> {
		try{
			return std::stoi(argv[2]);
		}catch(const std::exception & e){
			return std::nullopt;
		}
	}();

	if(!sigma){
		std::cerr << "deviation must be an integer. exiting...\n";
		return 1;
	}

	if(*sigma <= 0){
		std::cerr << "deviation must be greater than 0. exiting...\n";
		return 1;
	}

	bool use_gpu = false;
	bool use_cpu = false;
	std::string_view output_image_path;

	// todo: improve argparse
	for(int i = 3; i < argc; ++i){
		const std::string_view cur_arg(argv[i]);

		if(cur_arg == "-c"){
			use_cpu = true;
		}else if(cur_arg == "-g"){
			use_gpu = true;
		}else if(cur_arg == "-o"){

			if(i + 1 < argc){
				output_image_path = argv[i + 1];
				++i;
			}else{
				std::cerr << "-o option requires an additional argument.\n";
				return 1;
			}
		}else{
			std::cerr << "unrecognized argument: " << cur_arg << '\n';
			return 1;
		}
	}

	if(output_image_path.empty()){
		output_image_path = argv[1];
	}

	if(!use_cpu && !use_gpu){
		use_cpu = use_gpu = true;
	}

	try{
		std::cout << "extracting tga metadata and pixel blob from the given file...\n";
		const auto [header, tga_img] = extract_tga_image(argv[1]);
		std::cout << "extraction successful.\n";

		if(use_cpu){
			std::cout << "convolving the extracted tga image on CPU using sigma value: " << *sigma << "...\n";
			const auto [convolved_header, convolved_img] = convolve_tga_image<Process_type::CPU>(header, tga_img, *sigma);
			std::cout << "convolution successful.\n";

			if(!use_gpu){
				std::cout << "writing CPU's convoluded image to: " << output_image_path << "...\n";
				write_tga_image(convolved_header, convolved_img, output_image_path.data());
				std::cout << "convolved image written successfully to " << output_image_path << ".\n";
			}
		}

		if(use_gpu){
			std::cout << "convolving the extracted tga image on GPU using sigma value: " << *sigma << "...\n";
			const auto [convolved_header, convolved_img] = convolve_tga_image<Process_type::GPU>(header, tga_img, *sigma);
			std::cout << "convolution successful.\n";

			std::cout << "writing GPU's convoluded image to: " << output_image_path << "...\n";
			write_tga_image(convolved_header, convolved_img, output_image_path.data());
			std::cout << "convolved image written successfully to " << output_image_path << ".\n";
		}

	}catch(const std::exception & e){
		std::cerr << e.what() << '\n';
		return 1;
	}
}
