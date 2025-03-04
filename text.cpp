#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "genann.h"
int main(void) {
	char* filename = "/home/love/Desktop/wonderland.txt";
	FILE* file = fopen(filename, "rb"); // Open the file in binary mode
	fseek(file, 0, SEEK_END);
   	long size = ftell(file);
    	rewind(file); // Go back to the beginning of the file
    	printf("File is %i bytes long\n", size);
    	 char* buffer = (char*)malloc(size + 1); // +1 for null terminator
	fread(buffer, 1, size, file);
    	 buffer[size] = '\0';

    	int len = 128;
    	double *x, *y;
	x = (double*)malloc(sizeof(double)*len);
	y = malloc(sizeof(double));
	/* New network with 2 inputs,
	 * 1 hidden layer of 3 neurons each,
	 * and 2 outputs. */
	genann *ann = genann_init(len, 3, 1024*2, 1);
	char* str = "hello";
	double* gen = (double*)malloc(sizeof(double)*128);
	for (int i = 0; i < sizeof(str); i++)
	gen[len - sizeof(str)+i] = str[i];
	for (int l = 0; l < size; l++) {
	    y[0] = x[l+1];
	    printf("Training...\n");
	    genann_train(ann, x, y, 0.0001);
	    
	    for (int i = 0; i < len - 1; i++) {
		x[i] = x[i + 1];
	    }
	    x[len - 1] = buffer[l];
	    
	   
		for (int i = 0; i < 128; i++) {
			gen[i] = genann_run(ann, gen)[0];
			for (int v = 0; v < len - 1; v++) {
				x[l] = x[v + 1];
		    	}
		    	x[len - 1] = buffer[i];
		    	
		    	printf("%c", (int)gen[i]);
		}

		
	}
	genann_train(ann, x, y, 0.1);
	FILE* f = fopen("gen", "w");
	genann_write(ann, f);
 	fclose(f);
	

	genann_train(ann, x, y, 0.1);
	double* generated = genann_run(ann, x);
	/* Learn on the training set. */
}
