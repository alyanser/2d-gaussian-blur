#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <optional>
#include <algorithm>
#include <chrono>
#include <tga.h>

#include "ops.h"

// basic raii wrapper class over FILE*
class File {
public:
	File(const char * file_path, const char * open_mode) noexcept
		: file_(std::fopen(file_path, open_mode))
	{}

	File(const File & rhs) = delete;
	File(File && rhs) = delete;
	File & operator = (const File & rhs) = delete;
	File & operator = (File && rhs) = delete;

	~File() noexcept {

		if(*this){ // close the file if it was opened successfully
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

	// returns true if the file was opened sucessfully, false otherwise
	operator bool () const noexcept {
		return file_;
	}
private:
	FILE * file_{};
};

// basic raii wrapper class over tga::Image. associates the pixel buffer with the actual image to avoid dangling buffer
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

// raii based timer class used for measuring elapsed time
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

// extracts the tga image metadata and pixel blob from the given tga image path
[[nodiscard]]
std::pair<tga::Header, Tga_image> extract_tga_image(const char * const tga_img_path){
	File input_img_file(tga_img_path, "rb"); // open the given file in read & binary mode
	
	if(!input_img_file){ // if the file couldn't be opened, don't proceed
		throw std::runtime_error("given file could not be opened for reading");
	}

	tga::StdioFileInterface tga_stdio_interface(*input_img_file);
	tga::Decoder decoder(&tga_stdio_interface);
	tga::Header header;

	// attempt to read the tga header
	if(!decoder.readHeader(header)){ // if the tga header could not extracted, don't proceed
		throw std::runtime_error("given file isn't either a tga image or it's corrupted. tga header could not be extracted");
	}

	Tga_image img(header.height, header.width, header.bytesPerPixel());

	// attempt to read the tga pixel blob
	if(!decoder.readImage(header, *img, nullptr)){ // if the pixel blob from tga image couldn't be extracted, don't proceed
		throw std::runtime_error("given tga image seems to be corrupted. pixel blob couldn't be extrracted");
	}

	// finally, return the successfully extracted tga header and the pixel blob
	return std::make_pair(std::move(header), std::move(img));
}

// write the given tga image to the given output_file_path
void write_tga_image(const tga::Header & header, const Tga_image & img, const char * const output_file_path){
	File output_img_file(output_file_path, "wb"); // open the given file in write & binary mode

	if(!output_img_file){ // if file could not be opened for writing, don't proceed
		throw std::runtime_error("output file could not be opened for writing");
	}

	tga::StdioFileInterface tga_stdio_interface(*output_img_file);
	tga::Encoder encoder(&tga_stdio_interface);

	encoder.writeHeader(header); // write the header part of tga img to given file path
	encoder.writeImage(header, *img); // write the pixel blob part of img to given file path
}

enum class Process_type {
	CPU,
	GPU
};

// populate the kernel to be used for convolution using the given sigma kernel width, height and sigma value
template<Process_type PT>
[[nodiscard]]
std::vector<std::vector<double>> get_kernel(const int width, const int height, const int sigma) noexcept {
	std::vector<std::vector<double>> kernel(height, std::vector<double>(width)); // stores the filter to be applied on tga image
	double sum = 0; // sum of every filter value in the kernel

	for(int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			// calculate the filter value at (i, j) indices

			if constexpr(PT == Process_type::CPU){ // perform the calculation on CPU
				kernel[i][j] = std::exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
				sum += kernel[i][j];
			}else{ // perform the calculation on GPU
				using cuda::Ops;

				kernel[i][j] = Ops(-(Ops(i) * Ops(i) + Ops(j) * Ops(j)) / (Ops(2) * Ops(sigma) * Ops(sigma))).exp()
					/ (Ops(2) * Ops(M_PI) * Ops(sigma) * Ops(sigma));

				sum = Ops(sum) + Ops(kernel[i][j]);
			}
		}
	}

	// downscale each filter value in the kernel using the total sum
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
	Timer _;
	constexpr auto filter_height = 5;
	constexpr auto filter_width = 5;

	// get the filter required for image convolution using the user provided sigma value
	const auto filter = get_kernel<PT>(filter_height, filter_width, sigma);

	// prepare the header for new image
	tga::Header new_header = header;
	// update the width and height of the new image's header
	new_header.width -= filter_width + 1;
	new_header.height -= filter_height + 1;

	Tga_image convolved_img(new_header.height, new_header.width, new_header.bytesPerPixel());

	// apply the filter to given image and store the pixel blob in convolved_img
	for(int i = 0; i < new_header.height; ++i){
		for(int j = 0; j < new_header.width; ++j){
			for(int k = 0; k < header.bytesPerPixel(); ++k){
				for(int h = i; h < i + filter_height; ++h){
					for(int w = j; w < j + filter_width; ++w){

						if constexpr(PT == Process_type::CPU){ // perform the calculation on CPU
							const auto new_img_pixel_idx = i * convolved_img->rowstride + j * header.bytesPerPixel() + k;
							const auto old_img_pixel_idx = h * img->rowstride + w * header.bytesPerPixel() + k;
							const auto kernel_x = h - i;
							const auto kernel_y = w - j;

							// multiply the filter value with current image's pixel and add it to new image's pixel
							convolved_img->pixels[new_img_pixel_idx] += img->pixels[old_img_pixel_idx] * filter[kernel_x][kernel_y];
						}else{ // perform the calculation on GPU
							using cuda::Ops;

							const auto new_img_pixel_idx = static_cast<std::size_t>(Ops(i) * Ops(convolved_img->rowstride) + Ops(j) 
								* Ops(header.bytesPerPixel()) + Ops(k));

							const auto old_img_pixel_idx = static_cast<std::size_t>(Ops(h) * Ops(img->rowstride) + Ops(w)
								* Ops(header.bytesPerPixel()) + Ops(k));

							const auto kernel_x = static_cast<std::size_t>(Ops(h) - Ops(i));
							const auto kernel_y = static_cast<std::size_t>(Ops(w) - Ops(j));

							// multiply the filter value with current image's pixel and add it to new image's pixel
							convolved_img->pixels[new_img_pixel_idx] = Ops(convolved_img->pixels[new_img_pixel_idx])
								+ Ops(img->pixels[old_img_pixel_idx]) * Ops(filter[kernel_x][kernel_y]);
						}
					}
				}
			}
		}
	}

	// finally, return the convolved tga image header and pixel blob
	return std::make_pair(std::move(new_header), std::move(convolved_img));
}

int main(int argc, char ** argv){

	if(argc < 3 || argc > 7){ // if an invalid number of arguments is given, print the usage and stop execution
		std::cerr << "Usage: " << argv[0] << " path_to_image deviation -o output_image_path(optional) -g(run on gpu | optional)"
			"-c(run on cpu | optional)\n";
		return 1;
	}

	// get the sigma value from given arguments
	const auto sigma = [&argv]() -> std::optional<int> {

		try{
			return std::stoi(argv[2]);
		}catch(const std::exception & e){
			return std::nullopt;
		}
	}();

	if(!sigma){ // if the provided signam value wasn't an integer, inform the user and stop execution
		std::cerr << "deviation must be an integer. exiting...\n";
		return 1;
	}

	if(*sigma <= 0){ // if given sigma is negative, inform the user and stop execution
		std::cerr << "sigma must be greater than 0. exiting...\n";
		return 1;
	}

	bool use_gpu = false;
	bool use_cpu = false;
	std::string output_image_path;

	for(int i = 3; i < argc; ++i){
		const auto cur_arg = std::string(argv[i]);

		if(cur_arg == "-c"){ // if arg is -c, enable cpu
			use_cpu = true;
		}else if(cur_arg == "-g"){ // if arg is -g, enable gpu
			use_gpu = true;
		}else if(cur_arg == "-o"){ // if arg is -o, update output file path
			if(i + 1 < argc){
				output_image_path = argv[i + 1];
			}
		}
	}

	if(output_image_path.empty()){ // if -o option wasn't used, default back to overwriting existing image
		output_image_path = argv[1];
	}

	if(!use_cpu && !use_gpu){ // if no argument was provided, enable both cpu and gpu
		use_cpu = use_gpu = true;
	}

	try{
		std::cout << "extracting tga metadata and pixel blob from the given file...\n";
		// extract the tga image from the given file path
		const auto [header, tga_img] = extract_tga_image(argv[1]);
		std::cout << "extraction successful.\n";

		if(use_cpu){
			std::cout << "convolving the extracted tga image on CPU using sigma value: " << *sigma << "...\n";
			// convolve the image using the given sigma value
			const auto [convolved_header, convolved_img] = convolve_tga_image<Process_type::CPU>(header, tga_img, *sigma);
			std::cout << "convolution successful.\n";

			if(!use_gpu){ // if only cpu was enabled, save the convolved image, otherwise gpu's result will be saved
				std::cout << "writing the convoluded image to: " << output_image_path << "...\n";
				// store the convolved image back to the disk
				write_tga_image(convolved_header, convolved_img, output_image_path.c_str());
				std::cout << "covolved image written successfully to " << output_image_path << ".\n";
			}
		}

		if(use_gpu){
			std::cout << "convolving the extracted tga image on GPU using sigma value: " << *sigma << "...\n";
			// convolve the image using the given sigma value
			const auto [convolved_header, convolved_img] = convolve_tga_image<Process_type::GPU>(header, tga_img, *sigma);
			std::cout << "convolution successful.\n";

			std::cout << "writing the convoluded image to: " << output_image_path << "...\n";
			// store the convolved image back to the disk
			write_tga_image(convolved_header, convolved_img, output_image_path.c_str());
			std::cout << "covolved image written successfully to " << output_image_path << ".\n";
		}

	}catch(const std::exception & e){
		std::cerr << e.what() << '\n';
		return 1;
	}
}
