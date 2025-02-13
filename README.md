# 2d-gaussian-blur

performs tga image convolution using 2d gaussian kernel on both gpu and cpu.

**Usage:**

	./convolution path_to_tga_img deviation

	Options:
	-o output_image_path -> store the convolved image at given path instead of overwriting
	-g -> use gpu
	-c -> use cpu

	note: if neither or both -c or -g are provided, both cpu and gpu will be used separately
