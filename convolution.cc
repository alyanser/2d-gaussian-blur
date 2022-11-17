#include <string>
#include <iostream>
#include <fstream>

int main(int argc, char ** argv){

	if(argc != 3){
		std::cerr << "Usage: " << argv[0] << " path_to_image deviation\n";
		return 1;
	}
	
	std::ifstream ifs(argv[1]);

	if(!ifs){
		std::cerr << argv[1] << " could not be opened for reading. exiting...\n";
		return 1;
	}

	int sigma;

	try{
		sigma = std::stoi(argv[2]);
	}catch(const std::exception & e){
		std::cerr << "deviation must be an integer. exiting...\n";
		return 1;
	}
}
