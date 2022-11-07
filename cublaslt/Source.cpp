#include <cuda_runtime.h>
//#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>
#include <cublasLt.h>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

int main()
{
	const uint32_t inputEntries = 4;
	const uint32_t inputFeatures = 6;
	const uint32_t outputFeatures = 6;

	float* CPUWorkspace = (float*)malloc((inputEntries * inputFeatures + inputFeatures * outputFeatures + outputFeatures + inputEntries * outputFeatures) * sizeof(float));
	float* CPUInputMatrix = (float*)CPUWorkspace;
	float* CPUWeightMatrix = CPUInputMatrix + inputEntries * inputFeatures;
	float* CPUBiasVector = CPUWeightMatrix + inputFeatures * outputFeatures;
	float* CPUOutputMatrix = CPUBiasVector + outputFeatures;

	float* GPUWorkspace = nullptr;
	uint32_t extraWorkspaceSize = 4194304;
	cudaMalloc(&GPUWorkspace, extraWorkspaceSize + (inputEntries * inputFeatures + inputFeatures * outputFeatures + outputFeatures + inputEntries * outputFeatures) * sizeof(float));
	float* GPUInputMatrix = (float*)GPUWorkspace + extraWorkspaceSize;
	float* GPUWeightMatrix = GPUInputMatrix + inputEntries * inputFeatures;
	float* GPUBiasVector = GPUWeightMatrix + inputFeatures * outputFeatures;
	float* GPUOutputMatrix = GPUBiasVector + outputFeatures;

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	curandGenerateNormal(gen, GPUInputMatrix, (inputEntries * inputFeatures + inputFeatures * outputFeatures + outputFeatures), 0.0f, 1.0f);
	cudaMemcpy(CPUInputMatrix, GPUInputMatrix, (inputEntries * inputFeatures + inputFeatures * outputFeatures + outputFeatures) * sizeof(float), cudaMemcpyDeviceToHost);
	
	cout << "Input Matrix" << endl;
	for (uint32_t i = 0; i < inputEntries; i++)
	{
		for (uint32_t j = 0; j < inputFeatures; j++)
		{
			cout << CPUInputMatrix[i * inputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "Weight Matrix" << endl;
	for (uint32_t i = 0; i < inputFeatures; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			cout << CPUWeightMatrix[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	cout << "Bias Vector" << endl;
	for (uint32_t i = 0; i < outputFeatures; i++)
	{
		cout << CPUBiasVector[i] << " ";
	}
	cout << endl << endl;
	
	return 0;
}