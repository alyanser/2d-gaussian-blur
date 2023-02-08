#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <tuple>
#include <optional>
#include <algorithm>
#include <unistd.h>

#include "ops.h"
#include "tga_image.h"
#include "timer.h"

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

	auto print_usage = [argv](){
		std::cerr << "Usage: " << argv[0] << " path_to_tga_img deviation\n\n"
			"Options:\n"
			"-o output_image_path -> store the convolved image at given path instead of overwriting\n"
			"-g -> use gpu\n"
			"-c -> use cpu\n\n"
			"note: if neither -c or -g is provided, both cpu and gpu will be used\n";
	};

	if(argc < 3 || argc > 7){
		print_usage();
		return 1;
	}

	bool use_gpu = false;
	bool use_cpu = false;
	std::string_view output_image_path;

	for(int c; (c = getopt(argc, argv, "cgo:")) != -1;){

		switch(c){

			case 'c' : {
				use_cpu = true;
				break;
			}

			case 'g' : {
				use_gpu = true;
				break;
			}

			case 'o' : {
				output_image_path = optarg;
				break;
			}

			default : {
				std::cerr << "invalid argument detected\n";
				print_usage();
				return 1;
			}
		}
	}

	const auto sigma = [&argv]() -> std::optional<int> {
		try{
			return std::stoi(argv[optind + 1]);
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

	if(output_image_path.empty()){
		output_image_path = argv[optind];
	}

	if(!use_cpu && !use_gpu){
		use_cpu = use_gpu = true;
	}

	try{
		std::cout << "extracting tga metadata and pixel blob from the given file...\n";
		const auto [header, tga_img] = extract_tga_image(argv[optind]);
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
