#pragma once

#include <string_view>
#include <cstdio>

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
