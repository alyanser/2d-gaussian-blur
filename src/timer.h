#pragma once

#include <chrono>
#include <iostream>

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

