#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <chrono>

using std::cout;
using std::endl;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::nanoseconds;

int main()
{
	const int inputEntries = 10;
	const int inputFeatures = 5;
	const int outputFeatures = 5;

	void* workspace = nullptr;
    cublasLtHandle_t ltHandle = nullptr;
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

	void* CPUWorkspace = nullptr;
	malloc(&CPUWorkspace, (inputEntries * inputFeatures + inputFeatures * outputFeatures + outputFeatures + inputEntries * outputFeatures) * sizeof(float));
	
    uint32_t workspaceSize = 4194304;
    cudaMalloc(&workspace, workspaceSize + (inputEntries * inputFeatures + inputFeatures * outputFeatures + outputFeatures + inputEntries * outputFeatures) * sizeof(float));
	float* inputMatrix = (float*)workspace + workspaceSize;
	float* weightMatrix = inputMatrix + inputEntries * inputFeatures;
	float* biasVector = weightMatrix + inputFeatures * outputFeatures;
	float* outputMatrix = biasVector + outputFeatures;
	
	cublasLtMatmulDesc_t operationDesc = NULL;
	cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
	cublasLtMatmulPreference_t preference = NULL;
	
	int returnedResults = 0;
	cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    cublasLtHandle_t ltHandle;
    cublasLtCreate(&ltHandle);
	
	checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
	checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));
	
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc));
	
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));
	
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    checkCublasStatus(cublasLtMatmul(ltHandle,
        operationDesc,
        alpha,
        A,
        Adesc,
        B,
        Bdesc,
        beta,
        C,
        Cdesc,
        C,
        Cdesc,
        &heuristicResult.algo,
        workspace,
        workspaceSize,
        0));

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}