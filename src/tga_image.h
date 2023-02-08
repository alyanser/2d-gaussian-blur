#pragma once

#include "file.h"
#include <tga.h>

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
