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

	void Print()
	{
		for (uint64_t i = 0; i < Rows; i++)
		{
			for (uint64_t j = 0; j < Columns; j++)
			{
				cout << Data[i * Columns + j] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
};

int main()
{
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count());

	const uint32_t inputEntries = 5;
	const uint32_t inputFeatures = 7;
	const uint32_t outputFeatures = 3;

	Matrix<float> input;
	input.Allocate(inputEntries, inputFeatures);
	input.Randomize(gen);
	input.Print();

	Matrix<float> weights;
	weights.Allocate(outputFeatures, inputFeatures);
	weights.Randomize(gen);
	weights.Print();
	
	Matrix<float> output;
	output.Allocate(inputEntries, outputFeatures);
	output.Randomize(gen);
	output.Print();
	
	curandDestroyGenerator(gen);

	return 0;
}