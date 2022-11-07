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
	const uint32_t inputEntries = 5;
	const uint32_t inputFeatures = 7;
	const uint32_t outputFeatures = 3;
	
	const uint32_t inputMatrixSize = inputEntries * inputFeatures;
	const uint32_t weightMatrixSize = inputFeatures * outputFeatures;
	const uint32_t biasVectorSize = outputFeatures;
	const uint32_t outputMatrixSize = inputEntries * outputFeatures;

	const uint32_t inputMatrixSizeBytes = inputMatrixSize * sizeof(float);
	const uint32_t weightMatrixSizeBytes = weightMatrixSize * sizeof(float);
	const uint32_t biasVectorSizeBytes = biasVectorSize * sizeof(float);
	const uint32_t outputMatrixSizeBytes = outputMatrixSize * sizeof(float);
	const uint32_t workspaceSizeBytes = 4194304;
	
	float* CPUInputMatrix = (float*)malloc(inputMatrixSizeBytes);
	float* CPUWeightMatrix = (float*)malloc(weightMatrixSizeBytes);
	float* CPUBiasVector = (float*)malloc(biasVectorSizeBytes);
	float* CPUOutputMatrix = (float*)malloc(outputMatrixSizeBytes);
	
	float* GPUInputMatrix;
	float* GPUWeightMatrix;
	float* GPUBiasVector;
	float* GPUOutputMatrix;
	float* GPUWorkspace;
	
	cudaMalloc(&GPUInputMatrix, inputMatrixSizeBytes);
	cudaMalloc(&GPUWeightMatrix, weightMatrixSizeBytes);
	cudaMalloc(&GPUBiasVector, biasVectorSizeBytes);
	cudaMalloc(&GPUOutputMatrix, outputMatrixSizeBytes);
	cudaMalloc(&GPUWorkspace, workspaceSizeBytes);

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	curandGenerateUniform(gen, GPUInputMatrix, inputMatrixSize);
	curandGenerateUniform(gen, GPUWeightMatrix, weightMatrixSize);
	curandGenerateUniform(gen, GPUBiasVector, biasVectorSize);
	
	cudaMemcpy(CPUInputMatrix, GPUInputMatrix, inputMatrixSizeBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(CPUWeightMatrix, GPUWeightMatrix, weightMatrixSizeBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(CPUBiasVector, GPUBiasVector, biasVectorSizeBytes, cudaMemcpyDeviceToHost);
	
	cout << "Input Matrix:" << endl;
	for (uint32_t i = 0; i < inputEntries; i++)
	{
		for (uint32_t j = 0; j < inputFeatures; j++)
		{
			cout << CPUInputMatrix[i * inputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "Weight Matrix:" << endl;
	for (uint32_t i = 0; i < inputFeatures; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			cout << CPUWeightMatrix[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	
	cout << "Bias Vector:" << endl;
	for (uint32_t i = 0; i < outputFeatures; i++)
	{
		cout << CPUBiasVector[i] << " ";
	}
	cout << endl << endl;
	
	cudaMemcpy(CPUInputMatrix, GPUInputMatrix, inputMatrixSizeBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(CPUWeightMatrix, GPUWeightMatrix, weightMatrixSizeBytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(CPUBiasVector, GPUBiasVector, biasVectorSizeBytes, cudaMemcpyDeviceToHost);

	cublasLtHandle_t ltHandle = nullptr;
	cublasLtMatmulDesc_t operationDesc = nullptr;
	cublasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr;
	cublasLtMatmulPreference_t preference = nullptr;

	int returnedResults = 0;
	cublasLtMatmulHeuristicResult_t heuristicResult = {};

	cublasLtCreate(&ltHandle);
	cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
	cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
	cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
	cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
	cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
	cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
	cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &GPUBiasVector, sizeof(GPUBiasVector));
	
	cublasLtMatrixLayoutCreate(&adesc, CUDA_R_32F, inputEntries, inputFeatures, inputEntries);
	cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_32F, inputFeatures, outputFeatures, inputFeatures);
	cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_32F, inputEntries, outputFeatures, inputEntries);

	cublasLtMatmulPreferenceCreate(&preference);
	cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSizeBytes, sizeof(workspaceSizeBytes));
	
	cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, adesc, bdesc, cdesc, cdesc, preference, 1, &heuristicResult, &returnedResults);
	
	float alpha = 1.0f, beta = 0.0f;
	cublasLtMatmul(ltHandle, operationDesc, &alpha, GPUInputMatrix, adesc, GPUWeightMatrix, bdesc, &beta, GPUOutputMatrix, cdesc, GPUOutputMatrix, cdesc, &heuristicResult.algo, GPUWorkspace, workspaceSizeBytes, 0);
	cudaMemcpy(CPUOutputMatrix, GPUOutputMatrix, outputMatrixSizeBytes, cudaMemcpyDeviceToHost);
	
	cout << "Output Matrix:" << endl;
	for (uint32_t i = 0; i < inputEntries; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			cout << CPUOutputMatrix[i * outputFeatures + j] << " ";
		}
		cout << endl;
	}
	cout << endl;

	float avgErr = 0;
	for (uint32_t i = 0; i < inputEntries; i++)
	{
		for (uint32_t j = 0; j < outputFeatures; j++)
		{
			float sum = CPUBiasVector[j];
			for (uint32_t k = 0; k < inputFeatures; k++)
			{
				sum += CPUInputMatrix[i * inputFeatures + k] * CPUWeightMatrix[k * outputFeatures + j];
			}
			cout << sum << " ";
			sum += CPUBiasVector[j];
			if (sum < 0.0f)
			{
				sum = 0.0f;
			}
			avgErr += abs(sum - CPUOutputMatrix[i * outputFeatures + j]);
		}
		cout << endl;
	}
	cout << endl;
	avgErr /= (inputEntries * outputFeatures);
	cout << "Average Error: " << avgErr << endl;
	
	return 0;
}