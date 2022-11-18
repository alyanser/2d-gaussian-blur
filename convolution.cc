#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <optional>
#include <algorithm>
#include <tga.h>

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

// populate the kernel to be used for convolution using the given sigma kernel width, height and sigma value
[[nodiscard]]
std::vector<std::vector<double>> get_kernel(const int width, const int height, const int sigma) noexcept {
	std::vector<std::vector<double>> kernel(height, std::vector<double>(width)); // stores the filter to be applied on tga image
	double sum = 0; // sum of every filter value in the kernel

	for(int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			// calculate the filter value at (i, j) indices
			kernel[i][j] = std::exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			sum += kernel[i][j]; // add the current filter value to the total sum
		}
	}

	// downscale each filter value in the kernel using the total sum
	std::transform(std::begin(kernel), std::end(kernel), std::begin(kernel), [sum](auto & row){

		std::transform(std::begin(row), std::end(row), std::begin(row), [sum](const auto filter_val){
			return filter_val / sum;
		});

		return row;
	});

	return kernel;
}

[[nodiscard]]
std::pair<tga::Header, Tga_image> convolve_tga_image(const tga::Header & header, const Tga_image & img, const int sigma) noexcept {
	constexpr auto filter_height = 5;
	constexpr auto filter_width = 5;

	// get the filter required for image convolution using the user provided sigma value
	const auto filter = get_kernel(filter_height, filter_width, sigma);

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
						const auto new_img_pixel_idx = i * convolved_img->rowstride + j * header.bytesPerPixel() + k;
						const auto old_img_pixel_idx = h * img->rowstride + w * header.bytesPerPixel() + k;
						const auto kernel_x = h - i;
						const auto kernel_y = w - j;

						// multiply the filter value with current image's pixel and add it to new image's pixel
						convolved_img->pixels[new_img_pixel_idx] += img->pixels[old_img_pixel_idx] * filter[kernel_x][kernel_y];
					}
				}
			}
		}
	}

	// finally, return the convolved tga image header and pixel blob
	return std::make_pair(std::move(new_header), std::move(convolved_img));
}

int main(int argc, char ** argv){

	if(argc != 3){ // if argument count is not 3, print the usage and stop execution
		std::cerr << "Usage: " << argv[0] << " path_to_image deviation\n";
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

	try{
		// extract the tga image from the given file path
		const auto [header, tga_img] = extract_tga_image(argv[1]);
		// convolve the image using the given sigma value
		const auto [convolved_header, convolved_img] = convolve_tga_image(header, tga_img, *sigma);
		// store the konvoluded image back to the disk
		write_tga_image(convolved_header, convolved_img, "text.tga");
	}catch(const std::exception & e){
		std::cerr << e.what() << '\n';
		return 1;
	}
}
