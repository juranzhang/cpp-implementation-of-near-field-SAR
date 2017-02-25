#include "bp_kernel.h"

__global__ void bp_imaging_kernel(float Nx,float Ny,float k,float* x_array,float* y_array,float* R0_xy1,float* d_x,float* d_y,float* d_z,cuFloatComplex* d_bp,cuFloatComplex* d_y_bp,int width,int height,int depth,int ybp_w,int ybp_h,int r0w,float rs,float rstart,float rstop,float R0){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < width*height*depth){
		//Get ijk indices from each index
		int k = index/(width*height);
		index -= k*width*height;
		int j = index/width;
		index -= j*width;
		int i = index/1;

		float Ri,lr,yi_be4_exp;
		cuFloatComplex yi,yi_cx,l12,tmp;
		int l1,l2;
		for(int m=0;m<Ny;m++){
			for(int n=0;n<Nx;n++){
				Ri = sqrt(pow(d_x[i]-x_array[n],2) + pow(d_y[j] - y_array[m],2) + pow(d_z[k]+R0,2) - R0_xy1[m+n*r0w]);
				l1 = floor((Ri-rstart)/rs)+1;
				l2 = ceil((Ri-rstart)/rs)+1;
				lr = (Ri-rstart)/rs+1;
				l12 = make_cuFloatComplex(lr-l1,0);
				tmp = cuCmulf(l12,(d_y_bp[m+n*ybp_w+l2*ybp_w*ybp_h]));
				yi = cuCaddf(d_y_bp[m+n*ybp_w+l1*ybp_w*ybp_h],tmp);
				yi = cuCsubf(yi,d_y_bp[m+n*ybp_w+l1*ybp_w*ybp_h]);
				yi_be4_exp = k*2*(Ri+R0_xy1[m+n*r0w]);
				yi_cx = make_cuFloatComplex(cos(yi_be4_exp),sin(yi_be4_exp));
				yi = cuCmulf(yi,yi_cx);
				d_bp[i+j*width+k*width*height] = cuCaddf(d_bp[i+j*width+k*width*height],yi);
			}
		}
	}
}

extern "C" cx_fcube bp_kernel(cx_fcube y_bp,fvec x,fvec y,fvec z,fvec x_array,fvec y_array,fmat R0_xy1,int Nx,int Ny,int ixn,int iyn,float k,float rs,float rstart,float rstop,float R0){
	cx_fcube bp_image(ixn,iyn,iyn);

	// allocate memory on device
	int width = ixn;
	int height = iyn;
	int depth = iyn;
	int numOfPnts = width*height*depth;
	int size = numOfPnts*sizeof(cuFloatComplex);
	float *d_x;
	float *d_y;
	float *d_z;
	float *d_x_array;
	float *d_y_array;
	float *d_R0_xy1;
	cuFloatComplex *d_y_bp;
	cuFloatComplex *d_bp;
	cudaMalloc((void **)&d_x,ixn*sizeof(float));
	cudaMalloc((void **)&d_y,iyn*sizeof(float));
	cudaMalloc((void **)&d_z,iyn*sizeof(float));
	cudaMalloc((void **)&d_x_array,(x_array.n_elem)*sizeof(float));
	cudaMalloc((void **)&d_y_array,(y_array.n_elem)*sizeof(float));
	cudaMalloc((void **)&d_R0_xy1,(R0_xy1.n_elem)*sizeof(float));
	cudaMalloc((void **)&d_y_bp,(y_bp.n_elem)*sizeof(cuFloatComplex));
	cudaMalloc((void **)&d_bp,size);
	cout<<"after"<<endl;
	// allocate memory on host
	cuFloatComplex *host_data = (cuFloatComplex*)malloc(size);
	float* x_mem = x.memptr();
	float* y_mem = y.memptr();
	float* z_mem = z.memptr();
	float* x_array_mem = x_array.memptr();
	float* y_array_mem = y_array.memptr();
	float* R0_xy1_mem = R0_xy1.memptr();
	cuFloatComplex *y_bp_host = (cuFloatComplex*)malloc((y_bp.n_elem/2)*sizeof(cuFloatComplex));
	cout<<sizeof(cuFloatComplex)<<endl;
	cout<<y_bp.n_elem<<endl;
	size_t sybp = malloc_usable_size (y_bp_host);
	cout<<sybp<<endl;
	cout<<sizeof(cx_float)<<endl;
	for(int i = 0;i<(y_bp.n_elem/2);i++){
		cout<<i<<endl;
		y_bp_host[i]=make_cuFloatComplex(y_bp(i).real(),y_bp(i).imag());
	}


	cout<<"after"<<endl;
	// copy data from host to device
	cudaMemcpy(d_x,x_mem,ixn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y_mem,iyn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,z_mem,iyn*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_array,x_array_mem,(x_array.n_elem)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_array,y_array_mem,(y_array.n_elem)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_R0_xy1,R0_xy1_mem,(R0_xy1.n_elem)*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_bp,y_bp_host,(y_bp.n_elem)*sizeof(cuFloatComplex),cudaMemcpyHostToDevice);
	cudaMemcpy(d_bp,host_data,size,cudaMemcpyHostToDevice);

	dim3 block_size(32,32);
	dim3 grid_size(235/32+1,235/32+1);

	// call cuda function
	bp_imaging_kernel<<<grid_size,block_size>>>(Nx,Ny,k,d_x_array,d_y_array,d_R0_xy1,d_x,d_y,d_z,d_bp,d_y_bp,width,height,depth,y_bp.n_rows,y_bp.n_cols,R0_xy1.n_rows,rs,rstart,rstop,R0);

	// copy result back to host
	cudaMemcpy(host_data,d_bp,size,cudaMemcpyDeviceToHost);

	for(int i = 0;i<numOfPnts;i++){
		bp_image(i).real() = cuCrealf(host_data[i]);
		bp_image(i).imag() = cuCimagf(host_data[i]);
	}

	return bp_image;
}
