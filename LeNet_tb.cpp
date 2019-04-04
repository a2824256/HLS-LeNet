#include "LeNet.h"
#include "stdlib.h"
#include <time.h>
using namespace std;
int main(){
	srand((unsigned)time(NULL));
	float weights[62855];
	int r = 0;
//	float test[1024]={0},weights[51750]={0};
	for(long i_2 = 0; i_2<61830; i_2++){
		int r = rand()%1000;
		float r2 = (float)r/10000;
		weights[i_2] = r2;
	}
	for(int i_1 = 0; i_1<1024; i_1++){
		int r = rand()%255;
		weights[i_1+61830] = (float)r;
	}
	LetNet(weights,&r);
	std::cout<<"res:"<<r<<"\n";
	return 0;
}
