#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <tga.h>

class File {
public:
	File(const char * file_path, const char * open_mode) noexcept
		: file_(std::fopen(file_path, open_mode)){}

	~File() noexcept {

		if(file_){
			std::fclose(file_);
		}
	}

	[[nodiscard]]
	FILE * operator * () noexcept {
		return file_;
	}

	operator bool () noexcept {
		return file_;
	}
private:
	FILE * file_;
};

class Image {
public:
	Image() = default;

	Image(const std::size_t pixel_size) noexcept
		: buffer_(pixel_size)
	{
		alloc_img_pixels();
	}

	Image(const Image & rhs) noexcept
		: img(rhs.img)
		, buffer_(rhs.buffer_)
	{
		alloc_img_pixels();
	}

	Image(Image && rhs) noexcept 
		: img(std::move(rhs.img))
		, buffer_(std::move(rhs.buffer_))
	{
		alloc_img_pixels();
	}

	Image & operator = (Image && rhs) noexcept {
		this->~Image();
		new (this) Image(std::move(rhs));
		return *this;
	}

	Image & operator = (const Image & rhs) noexcept {
		this->~Image();
		new (this) Image(rhs);
		return *this;
	}

	tga::Image img{};
private:

	void alloc_img_pixels() noexcept {
		img.pixels = buffer_.empty() ? nullptr : &buffer_[0];
	}

	std::vector<std::uint8_t> buffer_;
};

[[nodiscard]]
std::pair<tga::Header, Image> extract_tga_image(const char * const tga_img_path){
	File input_img_file(tga_img_path, "rb");
	
	if(!input_img_file){
		throw std::runtime_error("given image file could not be opened for reading");
	}

	tga::StdioFileInterface tga_file(*input_img_file);
	tga::Decoder decoder(&tga_file);
	tga::Header header;

	if(!decoder.readHeader(header)){
		throw std::runtime_error("given tga image could not be processed");
	}

	Image tga_img(header.width * header.height * header.bytesPerPixel());
	tga_img.img.bytesPerPixel = header.bytesPerPixel();
	tga_img.img.rowstride = header.width * header.bytesPerPixel();

	if(!decoder.readImage(header, tga_img.img, nullptr)){
		throw std::runtime_error("given tga image could not be processed");
	}

	return std::make_pair(std::move(header), std::move(tga_img));
}

void write_tga_image(const tga::Header & header, const Image & img, const char * const output_file_path) noexcept {
	File output_img_file(output_file_path, "wb");
	tga::StdioFileInterface tga_file(*output_img_file);
	tga::Encoder encoder(&tga_file);

	encoder.writeHeader(header);
	encoder.writeImage(header, img.img);
}

[[nodiscard]]
std::vector<std::vector<double>> get_kernel(const int width, const int height, const int sigma) noexcept {
	std::vector<std::vector<double>> kernel(height, std::vector<double>(width));
	double sum = 0;

	for(int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			kernel[i][j] = std::exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
			sum += kernel[i][j];
		}
	}

	for(int i = 0; i < height; i++){
		for(int j = 0; j < width; j++){
			kernel[i][j] /= sum;
		}
	}

	return kernel;
}

[[nodiscard]]
std::pair<tga::Header, Image> convolude_tga_image(const tga::Header & header, const Image & img, const int sigma) noexcept {
	constexpr auto filter_height = 5;
	constexpr auto filter_width = 5;

	const auto kernel = get_kernel(filter_height, filter_width, sigma);

	tga::Header new_header = header;
	new_header.width -= filter_width + 1;
	new_header.height -= filter_height + 1;

	Image convoluded_image(new_header.height * new_header.width * new_header.bytesPerPixel());
	convoluded_image.img.bytesPerPixel = img.img.bytesPerPixel;
	convoluded_image.img.rowstride = new_header.width * new_header.bytesPerPixel();

	for(int i = 0; i < new_header.height; ++i){
		for(int j = 0; j < new_header.width; ++j){
			for(int k = 0; k < header.bytesPerPixel(); ++k){
				for(int h = i; h < i + filter_height; ++h){
					for(int w = j; w < j + filter_width; ++w){
						convoluded_image.img.pixels[i * convoluded_image.img.rowstride + j * header.bytesPerPixel() + k] += 
							img.img.pixels[h * img.img.rowstride + w * header.bytesPerPixel() + k] 
								* kernel[h - i][w - j];
					}
				}
			}
		}
	}

	return std::make_pair(std::move(new_header), std::move(convoluded_image));
}

int main(int argc, char ** argv){

	if(argc != 3){
		std::cerr << "Usage: " << argv[0] << " path_to_image deviation\n";
		return 1;
	}

	int sigma;

	try{
		sigma = std::stoi(argv[2]);
	}catch(const std::exception & e){
		std::cerr << "deviation must be an integer. exiting...\n";
		return 1;
	}

	tga::Header header;
	Image tga_img;

	try{
		std::tie(header, tga_img) = extract_tga_image(argv[1]);
	}catch(const std::exception & e){
		std::cerr << e.what() << '\n';
		return 1;
	}

	auto [convoluded_header, convoluded_img] = convolude_tga_image(header, tga_img, sigma);
	write_tga_image(convoluded_header, convoluded_img, "text.tga");
}
