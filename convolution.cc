#include <string>
#include <iostream>
#include <fstream>
#include <tga.h>

class File {
public:
	File(const char * file_path, const char * open_mode) noexcept
		: file_(std::fopen(file_path, open_mode)){}

	~File() noexcept {
		std::fclose(file_);
	}

	FILE * operator * () noexcept {
		return file_;
	}
private:
	FILE * file_;
};

struct Image {
	tga::Image img;
	std::vector<uint8_t> buffer;
};

std::pair<tga::Header, Image> extract_tga_image(const char * const tga_img_path){
	File input_img_file(tga_img_path, "rb");
	
	if(!*input_img_file){
		throw std::runtime_error("given image file could not be opened for reading");
	}

	tga::StdioFileInterface tga_file(*input_img_file);
	tga::Decoder decoder(&tga_file);
	tga::Header header;

	if(!decoder.readHeader(header)){
		throw std::runtime_error("given tga image could not be processed");
	}

	Image tga_image;
	tga_image.img.bytesPerPixel = header.bytesPerPixel();
	tga_image.img.rowstride = header.width * header.bytesPerPixel();

	tga_image.buffer.resize(tga_image.img.rowstride * header.height);
	tga_image.img.pixels = &tga_image.buffer[0];

	if(!decoder.readImage(header, tga_image.img, nullptr)){
		throw std::runtime_error("given tga image could not be processed");
	}

	return std::make_pair(std::move(header), std::move(tga_image));
}

void write_tga_image(const tga::Header & header, const tga::Image & image, const char * const output_file_path) noexcept {
	File output_img_file(output_file_path, "wb");
	tga::StdioFileInterface tga_file(*output_img_file);
	tga::Encoder encoder(&tga_file);

	encoder.writeHeader(header);
	encoder.writeImage(header, image);
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

	auto [header, tga_image] = extract_tga_image(argv[1]);
	write_tga_image(header, tga_image.img, "text.tga");
}
