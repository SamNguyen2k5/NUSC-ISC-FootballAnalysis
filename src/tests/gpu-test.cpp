#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// Note: at::hasMPS() is the specific underlying C++ function used
// to verify Apple Silicon GPU support in LibTorch.
int GPUTest()
{
    std::cout << "========================================" << std::endl;
    std::cout << "       LibTorch GPU/MPS Tester          " << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. Check if the Apple Silicon GPU (MPS) backend is available
    if (at::hasMPS())
    {
        std::cout << "[SUCCESS] Apple Silicon GPU (MPS) is AVAILABLE!" << std::endl;

        // Define the target GPU device
        torch::Device device(torch::kMPS); //

        try
        {
            std::cout << "\n[1/3] Creating tensors directly on the GPU..." << std::endl;
            // Create two 3x3 matrices directly on the Mac's GPU memory
            torch::Tensor matrixA = torch::ones({3, 3}, torch::dtype(torch::kFloat32).device(device));
            torch::Tensor matrixB = torch::eye(3, torch::dtype(torch::kFloat32).device(device));

            std::cout << "[2/3] Performing GPU Matrix Multiplication (A x B)..." << std::endl;
            // Execute the multiplication on GPU hardware
            torch::Tensor result = torch::matmul(matrixA, matrixB);

            std::cout << "[3/3] Pulling results back to print..." << std::endl;
            std::cout << "Result Tensor (on " << result.device() << "):\n"
                      << result << std::endl;

            std::cout << "----------------------------------------" << std::endl;
            std::cout << "STATUS: GPU Hardware test PASSED successfully!" << std::endl;
        }
        catch (const c10::Error &e)
        {
            std::cerr << "\n[ERROR] A LibTorch exception occurred during GPU operations:\n"
                      << e.what() << std::endl;
        }
    }
    else
    {
        std::cout << "[WARNING] Apple Silicon GPU (MPS) is NOT available." << std::endl;
        std::cout << "Falling back to CPU mode calculation..." << std::endl;

        torch::Tensor matrixA = torch::ones({3, 3});
        torch::Tensor result = torch::matmul(matrixA, torch::eye(3));
        std::cout << "Result Tensor (on CPU):\n"
                  << result << std::endl;
    }

    std::cout << "========================================" << std::endl;
    return 0;
}
