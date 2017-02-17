//This is the file which works on GPU config info. 

/*
important info: 
	- only ONE cuda device can work on this code
	= MUST set the num-of-body is 2048 nodes (less than 3072 node, i.e. max 3071). block size can be small. 
*/

#define SSM_SIZE 49152
#define SSM_NUM_STREAM_MULTIPROCESSOR 16
#define SSM_CLEAN_BLOCK_SIZE 1024

/* ##This config will change if last 3 definition change## */
//if data type is float
#define SSM_CLEAN_EXPAND 12 // 48KB/(1K*sizeof(float)) = 12
//if data type is double
//#define SSM_CLEAN_EXPAND 6 // 48KB/(1K*sizeof(double)) = 6


