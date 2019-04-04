#include "LetNet.h"

float expf(float x) {
 x = 1.0 + x / 1024;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x; x *= x; x *= x;
 x *= x; x *= x;
 return x;
}

float Conv_5x5(float input[25], float kernel[25]){
	int x,y;
	float result = 0;
	for(y = 0; y < 5; y++){
		for(x = 0; x < 5; x++){
			result += input[x+y*5] * kernel[x+y*5];
		}
	}
	return result;
}

//kernel 5x5x6 = 25x6 = 150
void ConvLayer_1(float input[1024],float * C1_value,float * weights){
	int i_y,i_x,matrix_y,matrix_x;
	int k_num,mat_i = 0;
	top_loop:for(int k_num = 0; k_num < 6; k_num+=1){
		//TODO 内存kernel
		float matrix_2[25];
		for(mat_i = 0;mat_i<25;mat_i++){
			matrix_2[mat_i] = weights[mat_i + k_num*25];
		}
		i_y_loop:for(i_y = 0; i_y < 28; i_y++){
			for(i_x = 0; i_x < 28; i_x++){
				float matrix[25];
				int pic_value_index = i_x + i_y * 32;
				matrix_loop:for(matrix_y = 0; matrix_y <5; matrix_y++){
					caculate:for(matrix_x = 0; matrix_x <5; matrix_x++){
//						图片索引  0 ~ 24
						int matrix_index = matrix_x + matrix_y * 5;
//						图片像素索引 0 ~ 1024,与matrix_x,matrix_y相关,x、y=32
						int input_value_index = pic_value_index + matrix_x + matrix_y * 32;
						matrix[matrix_index] = input[input_value_index];
					}
				}
				int out_pic_index = i_x + i_y * 28 + k_num * 784;
				C1_value[out_pic_index] = Conv_5x5(matrix,matrix_2);
			}
		}
	}
}

float AvgPool_2x2(float input[4]){
	float res = 0;
	int i;
	for(i = 0; i < 4 ; i++){
		res += input[i];
	}
	res /= 4;
	return res;
}

float sigmoid(float x)
{
    return (1 / (1 + expf(-x)));
}

void AvgpoolLayer_2(float input[4704],float *A2_value){
	int k_num,i_y,i_x,matrix_x,matrix_y;
	int count = 0;
	for(k_num = 0; k_num < 6; k_num++){
		for(i_y = 0; i_y < 27; i_y+=2){
			for(i_x = 0;  i_x < 27; i_x+=2){
				float matrix[4];
				int index_now = i_x + i_y * 28 + k_num * 784;
				for(matrix_y = 0; matrix_y < 2; matrix_y++){
					for(matrix_x = 0; matrix_x < 2; matrix_x++){
						int input_index = index_now + matrix_x + matrix_y * 28 ;
						matrix[matrix_x + matrix_y*2] = input[input_index];
					}
				}
				A2_value[count] = sigmoid(AvgPool_2x2(matrix));
				count++;
			}
		}
	}
}

//kernel 5x5x6x16 = 25x6x16 =2400
void ConvLayer_3(float input[1176],float *C3_value,float * weights){
	int k_num,nk_num,i_y,i_x,matrix_x,matrix_y;
	int mat_i;
    for(nk_num = 0; nk_num < 16; nk_num++){
		for(i_y = 0; i_y < 10; i_y++){
			for(i_x = 0; i_x < 10; i_x++){
				float res = 0;
				float res_total_6 = 0;
				float matrix[25];
				int index_now = i_x + i_y * 10 + nk_num * 100;
				for(k_num = 0; k_num < 6; k_num++){
					float matrix_2[25];
					for(mat_i = 0;mat_i<25;mat_i++){
						int weights_index = mat_i + k_num*25 + (nk_num+1)*150;
						matrix_2[mat_i] = weights[weights_index];
					}
					for(matrix_y = 0; matrix_y <5; matrix_y++){
						for(matrix_x = 0; matrix_x <5; matrix_x++){
							int matrix_index = matrix_x + matrix_y * 5;
							int input_value_index = index_now + matrix_x + matrix_y * 14;
							matrix[matrix_index] = input[input_value_index];
						}
					}
					res_total_6 += Conv_5x5(matrix,matrix_2);
				}
				C3_value[index_now] = res_total_6;
			}
		}
	}
}

void AvgpoolLayer_4(float input[1600],float *A4_value){
	int k_num,i_y,i_x,matrix_x,matrix_y;
	int count = 0;
	for(k_num = 0; k_num < 16; k_num++){
		for(i_y = 0; i_y < 10; i_y+=2){
			for(i_x = 0;  i_x < 10; i_x+=2){
				float matrix[4];
				int index_now = i_x + i_y * 10 + k_num * 100;
				for(matrix_y = 0; matrix_y < 2; matrix_y++){
					for(matrix_x = 0; matrix_x < 2; matrix_x++){
						int input_index = index_now + matrix_x + matrix_y * 10 ;
						matrix[matrix_x + matrix_y*2] = input[input_index];
					}
				}
				A4_value[count] = sigmoid(AvgPool_2x2(matrix));
				count++;
			}
		}
	}
}

//kernel 400x120 = 48000
void FullyConnLayer_5(float input[400],float *F5_value,float * weights){
	int i_y,i_x;
	for(i_y = 0; i_y < 120; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 400; i_x++){
			int index = i_x + i_y * 400;
			res += input[i_x] * weights[index + 2550];
		}
		F5_value[i_y] = res;
	}
}
//kernel 84x120 = 10080
void FullyConnLayer_6(float input[120],float *F6_value,float * weights){
	int i_y,i_x;
	for(i_y = 0; i_y < 84; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 120; i_x++){
			int index = i_x + i_y * 120;
			res += input[i_x] * weights[index + 50550];
		}
		F6_value[i_y] = res;
	}
}

//kernel 10x120 = 1200
void FullyConnLayer_7(float input[84],float *F6_value,float * weights){
	int i_y,i_x;
	for(i_y = 0; i_y < 10; i_y++){
		float res = 0;
		for(i_x = 0;  i_x < 84; i_x++){
			int index = i_x + i_y * 84;
			res += input[i_x] * weights[index + 60630];
		}
		F6_value[i_y] = res;
	}
}

int Softmax_1_8(float input[10],float *probability,float *res){
	int index;
	float sum = 0;
	for(index = 0; index < 10; index++ ){
		probability[index] = expf(input[index]/1000);
		sum += probability[index];
	}
	int max_index = 0;
	for(index = 0; index < 10; index++ ){
			res[index] = probability[index]/sum;
			float res1 = res[index];
			float res2 = res[max_index];
			if(res1 > res2){
				max_index = index;
			}
	}
	return max_index;
}


void LetNet(volatile float *addrMaster,int* r){
#pragma HLS INTERFACE m_axi depth=62855 port=addrMaster offset=slave bundle=MASTER_BUS
#pragma HLS INTERFACE s_axilite port=r bundle=CRTL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CRTL_BUS

	// 32x32 iamge
	float photo[1024];
	//layer1 weights  5x5x6 = 25x6 = 150
	//layer3 weights  5x5x6x16 = 25x6x16 =2400
	//layer5 weights 400x120 = 48000
	//layer6 weights 84x120 = 10080
	//layer7 weights 10x120 = 1200
	float data[62855];
	//The output of each layer
	float C1_value[4704];
	float A2_value[1176];
	float C3_value[1600];
	float A4_value[400];
	float F5_value[120];
	float F6_value[84];
	float F7_value[10];

	float probability[10];
	float res[10];
	int loop1_i;
	//memory copy from BRAM to FPGA's RAM
	memcpy(data,(const float*)addrMaster,62855*sizeof(float));
	//get the image data
	for(loop1_i = 0; loop1_i<1024; loop1_i++){
		photo[loop1_i] = data[loop1_i+61830];
	}
	//calulation of each layer
	ConvLayer_1(photo,C1_value,data);
	AvgpoolLayer_2(C1_value,A2_value);
	ConvLayer_3(A2_value,C3_value,data);
	AvgpoolLayer_4(C3_value,A4_value);
	FullyConnLayer_5(A4_value,F5_value,data);
	FullyConnLayer_6(F5_value,F6_value,data);
	FullyConnLayer_7(F6_value,F7_value,data);
	*r = Softmax_1_8(F7_value,probability,res);
}
