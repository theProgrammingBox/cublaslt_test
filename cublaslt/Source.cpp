#include <cuda_runtime.h>
#include <curand.h>
#include <cublasLt.h>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

template <typename T>
class Matrix
{
public:
	uint64_t Rows;
	uint64_t Columns;
	uint64_t Size;
	uint64_t Bytes;
	T* Data;

	Matrix() : Rows(0), Columns(0), Size(0), Bytes(0), Data(nullptr) {}
	~Matrix() { free(Data); }
	
	void Allocate(uint64_t rows, uint64_t columns)
	{
		Rows = rows;
		Columns = columns;
		Size = Rows * Columns;
		Bytes = Size * sizeof(T);
		Data = (T*)malloc(Bytes);
	}

	void Randomize(curandGenerator_t& gen)
	{
		T* GPUData;
		cudaMalloc(&GPUData, Bytes);
		curandGenerateNormal(gen, GPUData, Size + (Size & 1), 0, 1);
		cudaMemcpy(Data, GPUData, Bytes, cudaMemcpyDeviceToHost);
		cudaFree(GPUData);
	}

	void Print(bool t = false)
	{
		for (uint64_t i = 0; i < Rows; i++)
		{
			for (uint64_t j = 0; j < Columns; j++)
			{
				cout << (t ? Data[j * Rows + i] : Data[i * Columns + j]) << " ";
			}
			cout << endl;
		}
		cout << endl;
	}

	/*void EqualMatMulMat(Matrix<T>& A, Matrix<T>& B)
	{
		T* GPUDataA;
		T* GPUDataB;
		T* GPUDataC;
		cudaMalloc(&GPUDataA, A.Bytes);
		cudaMalloc(&GPUDataB, B.Bytes);
		cudaMalloc(&GPUDataC, Bytes);
		cudaMemcpy(GPUDataA, A.Data, A.Bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(GPUDataB, B.Data, B.Bytes, cudaMemcpyHostToDevice);
		cublasLtHandle_t handle;
		
	}*/
};

int main()
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());
	
	void* workspace;
	uint64_t workspaceSize = 4194304;
	cudaMalloc(&workspace, workspaceSize);
	
	cublasLtMatmulDesc_t operationDesc = NULL;
	cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
	cublasLtMatmulPreference_t preference = NULL;

	int returnedResults = 0;
	cublasLtMatmulHeuristicResult_t heuristicResult = {};

	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_N;

	cublasLtHandle_t ltHandle;
	cublasLtCreate(&ltHandle);

	cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32I, CUDA_R_32F);
	cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
	cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

	const uint32_t inputEntries = 5;
	const uint32_t inputFeatures = 7;
	const uint32_t outputFeatures = 3;

	Matrix<float> input;
	input.Allocate(inputEntries, inputFeatures);
	input.Randomize(gen);
	input.Print();

	Matrix<float> weights;
	weights.Allocate(inputFeatures, outputFeatures);
	weights.Randomize(gen);
	weights.Print();
	
	Matrix<float> output;
	output.Allocate(inputEntries, outputFeatures);

	cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, inputEntries, inputFeatures, inputEntries);
	cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, inputFeatures, outputFeatures, inputFeatures);
	cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, inputEntries, outputFeatures, inputEntries);

	cublasLtMatmulPreferenceCreate(&preference);
	cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
	
	cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
	
	float* GPUDataA;
	float* GPUDataB;
	float* GPUDataC;
	cudaMalloc(&GPUDataA, input.Bytes);
	cudaMalloc(&GPUDataB, weights.Bytes);
	cudaMalloc(&GPUDataC, output.Bytes);
	cudaMemcpy(GPUDataA, input.Data, input.Bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(GPUDataB, weights.Data, weights.Bytes, cudaMemcpyHostToDevice);
	
	float alpha = 1.0f;
	float beta = 0.0f;
	
	cublasLtMatmul(ltHandle, operationDesc, &alpha, GPUDataA, Adesc, GPUDataB, Bdesc, &beta, GPUDataC, Cdesc, GPUDataC, Cdesc, &heuristicResult.algo, workspace, workspaceSize, 0);
	
	cudaMemcpy(output.Data, GPUDataC, output.Bytes, cudaMemcpyDeviceToHost);
	output.Print();
	
	cudaFree(GPUDataA);
	cudaFree(GPUDataB);
	cudaFree(GPUDataC);
	
	cudaFree(workspace);
	curandDestroyGenerator(gen);
	cublasLtDestroy(ltHandle);
	cublasLtMatrixLayoutDestroy(Adesc);
	cublasLtMatrixLayoutDestroy(Bdesc);
	cublasLtMatrixLayoutDestroy(Cdesc);
	cublasLtMatmulDescDestroy(operationDesc);
	cublasLtMatmulPreferenceDestroy(preference);

	return 0;
}