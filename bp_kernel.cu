#include "bp_kernel.h"

__global__ void bp_imaging_kernel(float* d_bp_real,float* d_bp_imag,float Nx,float Ny,float k0,float* x_array,float* y_array,float* R0_xy1,float* d_x,float* d_y,float* d_z,float* d_y_bp_real,float* d_y_bp_imag,int width,int height,int depth,int ybp_w,int ybp_h,int r0w,float rs,float rstart,float rstop,float R0){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	// unique block and thread id
	int blockId = blockIdx.x+blockIdx.y* gridDim.x+ gridDim.x* gridDim.y* blockIdx.z;
	int threadId = blockId * (blockDim.x* blockDim.y* blockDim.z) + (threadIdx.z* (blockDim.x* blockDim.y))+ (threadIdx.y* blockDim.x)+ threadIdx.x;

	float Ri,lr,yi_be4_exp,yi_real,yi_imag;
	int l1,l2,idx1,idx2;
	//printf("%d %d %d %d %d %d\n", blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockIdx.z,threadIdx.z);
	//printf("%d %d %d\n", i,j,k);
	for(int m=0;m<Ny;m++){
		for(int n=0;n<Nx;n++){
			Ri = sqrt(pow(d_x[i]-x_array[n],2) + pow(d_y[j] - y_array[m],2) + pow(d_z[k]+R0,2)) - R0_xy1[m+n*r0w];
			l1 = floor((Ri-rstart)/rs);
			l2 = ceil((Ri-rstart)/rs);
			lr = (Ri-rstart)/rs;
			idx1 = m+n*ybp_w+l1*ybp_w*ybp_h;
			idx2 = m+n*ybp_w+l2*ybp_w*ybp_h;
			yi_real = d_y_bp_real[idx1]+(lr-l1)*(d_y_bp_real[idx2]-d_y_bp_real[idx1]);
			yi_imag = d_y_bp_imag[idx1]+(lr-l1)*(d_y_bp_imag[idx2]-d_y_bp_imag[idx1]);
			yi_be4_exp = k0*2*(Ri+R0_xy1[m+n*r0w]);
			yi_real = yi_real * cos(yi_be4_exp);
			yi_imag = yi_imag * sin(yi_be4_exp);
			//d_bp_real[i+j*width+k*width*height] = d_bp_real[i+j*width+k*width*height]+yi_real;
			//d_bp_imag[i+j*width+k*width*height] = d_bp_imag[i+j*width+k*width*height]+yi_imag;
			d_bp_real[threadId] = d_bp_real[threadId]+yi_real;
			d_bp_imag[threadId] = d_bp_imag[threadId]+yi_imag;
		}
	}
	printf("%f %f %f\n", d_bp_real[0],d_bp_real[5],yi_real);
}

void bp_kernel(float*& bp_real_host,float*& bp_imag_host,float* y_bp_real,float* y_bp_imag,int ybpw,int ybph,int ybpd,fvec x,fvec y,fvec z,fvec x_array,fvec y_array,fmat R0_xy1,int Nx,int Ny,int ixn,int iyn,float k,float rs,float rstart,float rstop,float R0){
	// allocate memory on device
	int width = ixn;
	int height = iyn;
	int depth = iyn;
	int bp_numOfPnts = width*height*depth;
	int ybp_numOfPnts = ybpw*ybph*ybpd;
	float *d_x;
	float *d_y;
	float *d_z;
	float *d_x_array;
	float *d_y_array;
	float *d_R0_xy1;
	float *d_y_bp_real;
	float *d_y_bp_imag;
	float *d_bp_real;
	float *d_bp_imag;

	cout<<y_bp_real[0]<<y_bp_real[5]<<endl;
	cout<<R0_xy1(0)<<endl;
	cudaMalloc((void **)&d_x,ixn*sizeof(float));
	cudaMalloc((void **)&d_y,iyn*sizeof(float));
	cudaMalloc((void **)&d_z,iyn*sizeof(float));
	cudaMalloc((void **)&d_x_array,(x_array.n_elem)*sizeof(float));
	cudaMalloc((void **)&d_y_array,(y_array.n_elem)*sizeof(float));
	cudaMalloc((void **)&d_R0_xy1,(R0_xy1.n_elem)*sizeof(float));
	cudaMalloc(&d_y_bp_real,ybp_numOfPnts*sizeof(float));
	cudaMalloc(&d_y_bp_imag,ybp_numOfPnts*sizeof(float));
	cudaMalloc((void **)&d_bp_real,bp_numOfPnts*sizeof(float));
	cudaMemset(d_bp_real,0,bp_numOfPnts*sizeof(float));
	cudaMalloc((void **)&d_bp_imag,bp_numOfPnts*sizeof(float));
	cudaMemset(d_bp_imag,0,bp_numOfPnts*sizeof(float));

	float* x_mem = x.memptr();
	float* y_mem = y.memptr();
	float* z_mem = z.memptr();
	float* x_array_mem = x_array.memptr();
	float* y_array_mem = y_array.memptr();
	float* R0_xy1_mem = R0_xy1.memptr();

	// copy data from host to device
	cudaMemcpy(d_x,x_mem,ixn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y_mem,iyn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,z_mem,iyn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_array,x_array_mem,(x_array.n_elem)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_array,y_array_mem,(y_array.n_elem)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_R0_xy1,R0_xy1_mem,(R0_xy1.n_elem)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_bp_real,y_bp_real,ybp_numOfPnts*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_bp_imag,y_bp_imag,ybp_numOfPnts*sizeof(float),cudaMemcpyHostToDevice);

	dim3 block_size(8,8,8);
	dim3 grid_size(237/8+1,237/8+1,237/8+1);

	// call cuda function
	bp_imaging_kernel<<<grid_size,block_size>>>(d_bp_real,d_bp_imag,Nx,Ny,k,d_x_array,d_y_array,d_R0_xy1,d_x,d_y,d_z,d_y_bp_real,d_y_bp_imag,width,height,depth,ybpw,ybph,R0_xy1.n_rows,rs,rstart,rstop,R0);

	// copy result back to host
	//cout<<d_bp_real[0]<<" "<<d_bp_real[5]<<endl;
	cudaMemcpy(bp_real_host,d_bp_real,bp_numOfPnts*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(bp_imag_host,d_bp_imag,bp_numOfPnts*sizeof(float),cudaMemcpyDeviceToHost);
	cout<<bp_real_host[0]<<" "<<bp_real_host[5]<<endl;
	cudaFree(d_bp_real);
	cudaFree(d_bp_imag);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	cudaFree(d_x_array);
	cudaFree(d_y_array);
	cudaFree(d_R0_xy1);
	cudaFree(d_y_bp_real);
	cudaFree(d_y_bp_imag);
}
