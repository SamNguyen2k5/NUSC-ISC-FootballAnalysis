#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <argparse/argparse.hpp>
#include "data/mvfouls/mvfouls.hpp"
namespace fs = std::filesystem;

int main(int argc, char **argv)
{
    argparse::ArgumentParser program("main_tester", "1.0");

    program.add_argument("--testdata")
        .required()
        .help("Test data folder");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err)
    {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }

    std::string test_data_folder_path = program.get<>("testdata");
    MVFoulsDataset mvFouls(test_data_folder_path);
    std::cerr << "Program completed.";
    return 0;
}
