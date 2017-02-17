#ifndef OPT_KERNEL
#define OPT_KERNEL

void static_hist() ;

/* Include below the function headers of any other functions that you implement */

void AllocateOnDevice(uint8_t **input ); // uint8_t histoBins[HISTO_HEIGHT*HISTO_WIDTH]);
void CopyFromDevice(uint32_t *kernel_bins);
void FreeOnDevice();

#endif
