#pragma once

#include "file.h"
#include <tga.h>
#include <cstdint>

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


[[nodiscard]]
inline std::pair<tga::Header, Tga_image> extract_tga_image(const std::string_view tga_img_path){
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

inline void write_tga_image(const tga::Header & header, const Tga_image & img, const std::string_view output_file_path){
	File output_img_file(output_file_path.data(), "wb");

	if(!output_img_file){
		throw std::runtime_error("output file could not be opened for writing");
	}

	tga::StdioFileInterface tga_stdio_interface(*output_img_file);
	tga::Encoder encoder(&tga_stdio_interface);

	encoder.writeHeader(header);
	encoder.writeImage(header, *img);
}
