#include "bp_kernel.cuh"

__global__ void bp_imaging_kernel(double Nx,double Ny,double k,double* x_array,double* y_array,double* R0_xy1,double* d_x,double* d_y,double* d_z,cuDoubleComplex* d_bp,cuDoubleComplex* d_y_bp,int width,int height,int depth,int ybp_w,int ybp_h,int r0w,double rs,double rstart,double rstop,double R0){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < width*height*depth){
		//Get ijk indices from each index
		int k = index/(width*height);
		index -= k*width*height;
		int j = index/width;
		index -= j*width;
		int i = index/1;

		double Ri,lr,yi_be4_exp;
		cuDoubleComplex yi,yi_cx,l12,tmp;
		int l1,l2;
		for(int m=0;m<Ny;m++){
			for(int n=0;n<Nx;n++){
				Ri = sqrt(pow(d_x[i]-x_array[n],2) + pow(d_y[j] - y_array[m],2) + pow(d_z[k]+R0,2) - R0_xy1[m+n*r0w]);
				l1 = floor((Ri-rstart)/rs)+1;
				l2 = ceil((Ri-rstart)/rs)+1;
				lr = (Ri-rstart)/rs+1;
				l12 = make_cuDoubleComplex(lr-l1,0);
				tmp = cuCmul(l12,(d_y_bp[m+n*ybp_w+l2*ybp_w*ybp_h]));
				yi = cuCadd(d_y_bp[m+n*ybp_w+l1*ybp_w*ybp_h],tmp);
				yi = cuCsub(yi,d_y_bp[m+n*ybp_w+l1*ybp_w*ybp_h]);
				yi_be4_exp = k*2*(Ri+R0_xy1[m+n*r0w]);
				yi_cx = make_cuDoubleComplex(cos(yi_be4_exp),sin(yi_be4_exp));
				yi = cuCmul(yi,yi_cx);
				d_bp[i+j*width+k*width*height] = cuCadd(d_bp[i+j*width+k*width*height],yi);
			}
		}
	}
}

cx_cube bp_kernel(cx_cube y_bp,vec x,vec y,vec z,vec x_array,vec y_array,mat R0_xy1,int Nx,int Ny,int ixn,int iyn,double k,double rs,double rstart,double rstop,double R0){
	cx_cube bp_image(ixn,iyn,iyn);

	// allocate memory on device
	int width = ixn;
	int height = iyn;
	int depth = iyn;
	int numOfPnts = width*height*depth;
	int size = numOfPnts*sizeof(cuDoubleComplex);
	double *d_x;
	double *d_y;
	double *d_z;
	double *d_x_array;
	double *d_y_array;
	double *d_R0_xy1;
	cuDoubleComplex *d_y_bp;
	cuDoubleComplex *d_bp;
	cudaMalloc((void **)&d_x,ixn*sizeof(double));
	cudaMalloc((void **)&d_y,iyn*sizeof(double));
	cudaMalloc((void **)&d_z,iyn*sizeof(double));
	cudaMalloc((void **)&d_x_array,(x_array.n_elem)*sizeof(double));
	cudaMalloc((void **)&d_y_array,(y_array.n_elem)*sizeof(double));
	cudaMalloc((void **)&d_R0_xy1,(R0_xy1.n_elem)*sizeof(double));
	cudaMalloc((void **)&d_y_bp,(y_bp.n_elem)*sizeof(cuDoubleComplex));
	cudaMalloc((void **)&d_bp,size);

	// allocate memory on host
	cuDoubleComplex *host_data = (cuDoubleComplex*)malloc(size);
	double* x_mem = x.memptr();
	double* y_mem = y.memptr();
	double* z_mem = z.memptr();
	double* x_array_mem = x_array.memptr();
	double* y_array_mem = y_array.memptr();
	double* R0_xy1_mem = R0_xy1.memptr();
	cuDoubleComplex *y_bp_host = (cuDoubleComplex*)malloc((y_bp.n_elem)*sizeof(cuDoubleComplex));
	for(int i = 0;i<(y_bp.n_elem);i++){
		y_bp_host[i]=make_cuDoubleComplex(y_bp(i).real(),y_bp(i).imag());
	}

	// copy data from host to device
	cudaMemcpy(d_x,x_mem,ixn*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,y_mem,iyn*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_z,z_mem,iyn*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_array,x_array_mem,(x_array.n_elem)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_array,y_array_mem,(y_array.n_elem)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_R0_xy1,R0_xy1_mem,(R0_xy1.n_elem)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_bp,y_bp_host,(y_bp.n_elem)*sizeof(cuDoubleComplex),cudaMemcpyHostToDevice);
	cudaMemcpy(d_bp,host_data,size,cudaMemcpyHostToDevice);

	dim3 block_size(32,32);
	dim3 grid_size(235/32+1,235/32+1);

	// call cuda function
	bp_imaging_kernel<<<grid_size,block_size>>>(Nx,Ny,k,d_x_array,d_y_array,d_R0_xy1,d_x,d_y,d_z,d_bp,d_y_bp,width,height,depth,y_bp.n_rows,y_bp.n_cols,R0_xy1.n_rows,rs,rstart,rstop,R0);

	// copy result back to host
	cudaMemcpy(host_data,d_bp,size,cudaMemcpyDeviceToHost);

	for(int i = 0;i<numOfPnts;i++){
		bp_image(i).real() = cuCreal(host_data[i]);
		bp_image(i).imag() = cuCimag(host_data[i]);
	}

	return bp_image;
}
