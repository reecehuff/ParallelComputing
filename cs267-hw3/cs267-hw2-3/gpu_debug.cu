#include "common.h"
#include <cuda.h>
#include <cmath>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define NUM_THREADS 256

// Number of blocks
int num_blocks; 

// Bin width, the number of bins across the domain, and the total number of bins
double bin_width;
int num_bins_across;
int num_bins; 

// Arrays for storing information about paticle and bin ids
int* particle_ids;   // array of particle ids sorted by bin id
int* bin_ids;        // array of bin ids
int* bin_counts;     // array of the number of particles in each bin 

__device__ void apply_force_gpu(particle_t& particle, particle_t& neighbor) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    // r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r * min_r) ? r2 : min_r * min_r;
    double r = sqrt(r2);

    //
    //  very simple short-range repulsive force
    //
    double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, double bin_width, int num_bins_across, int* particle_ids, int* bin_ids, int* bin_counts) {
    
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // Zero acceleration
    particles[tid].ax = particles[tid].ay = 0;
    // Calculate the particle's bin
    int bin_x = floor(particles[tid].x/bin_width);
    int bin_y = floor(particles[tid].y/bin_width);
    // Loop through the neighboring bins 
    for (int j = -1; j <= 1; j++){
        for (int k = -1; k <= 1; k++){
            if ((bin_x+j) >= 0 and (bin_x+j) < num_bins_across and (bin_y+k) >= 0 and (bin_y+k) < num_bins_across){
                int neighbor_bin = (bin_x+j) + (bin_y+k)*num_bins_across; 
                for (int offset = 0; offset < bin_counts[neighbor_bin]; ++offset){
                    // Calculate force
                    apply_force_gpu(particles[tid], particles[ particle_ids[bin_ids[neighbor_bin] + offset] ]);
                }
            }
        }
    }
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    particle_t* p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    //
    //  bounce from walls
    //
    while (p->x < 0 || p->x > size) {
        p->x = p->x < 0 ? -(p->x) : 2 * size - p->x;
        p->vx = -(p->vx);
    }
    while (p->y < 0 || p->y > size) {
        p->y = p->y < 0 ? -(p->y) : 2 * size - p->y;
        p->vy = -(p->vy);
    }
}

__global__ void num_parts_in_each_bin(particle_t* parts, int num_parts, double bin_width, int num_bins_across, int* bin_ids){

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // Calculate the particle's bin
    int bin_x = floor(parts[tid].x/bin_width);
    int bin_y = floor(parts[tid].y/bin_width);
    int bin   = bin_x + bin_y * num_bins_across;

    // Add 1 to bin_ids array at the bin where the particle is located
    atomicAdd(&bin_ids[bin], 1);

}

__global__ void set_particle_ids_and_bin_counts(particle_t* parts, int num_parts, double bin_width, int num_bins_across, int num_bins, int* particle_ids, int* bin_ids, int* bin_counts){

     // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_parts)
        return;

    // Calculate the particle's bin
    int bin_x = floor(parts[tid].x/bin_width);
    int bin_y = floor(parts[tid].y/bin_width);
    int bin   = bin_x + bin_y * num_bins_across;

    // Add 1 to bin_counts array at the bin where the particle is located and
    // save its output as an offset for storing the particle in particle_ids 
    int offset = atomicAdd(&bin_counts[bin], 1);
    // Save the particle to the bin it is in plus an offset
    particle_ids[bin_ids[bin] + offset] = tid;

}

__global__ void debug_bin_counts(int* bin_counts, int num_bins){

    printf("bin_counts: \n");
    for (int i = 0; i < num_bins; ++i){
        printf("bin %d has %d particles \n", i, bin_counts[i]);
    }
    printf("\n");

}

__global__ void debug_bin_ids(int* bin_ids, int num_bins){

    printf("bin_ids: \n");
    for (int i = 0; i < num_bins; ++i){
        printf("bin %d has a starting place of %d \n", i, bin_ids[i]);
    }
    printf("\n");

}

__global__ void debug_particle_ids(int* particle_ids, int* bin_ids, int* counts, int num_parts, int num_bins){

    printf("particle_ids: \n");
    for (int bin = 0; bin < num_bins; ++bin){
        printf("bin %d has particles: ", bin);
        for (int j = 0; j < counts[bin]; ++j){
            printf(" %d ", particle_ids[bin_ids[bin] + j]);
        }
        printf("\n");
    }
    printf("\n");

}

void init_simulation(particle_t* parts, int num_parts, double size) {
    // You can use this space to initialize data objects that you may need
    // This function will be called once before the algorithm begins
    // parts live in GPU memory

    // Define the bin width, the number of bins across, and the total number of bins
    bin_width       = 0.03; // cutoff; 
    num_bins_across = ceil(size/bin_width);
    num_bins        = num_bins_across * num_bins_across;

    // Allocate memory for particle and bin data structures 
    cudaMalloc((void**)&particle_ids, num_parts * sizeof(int));
    cudaMalloc((void**)&bin_ids     , num_bins  * sizeof(int));
    cudaMalloc((void**)&bin_counts  , num_bins  * sizeof(int));

    // Define the number of blocks
    num_blocks = ( num_parts + NUM_THREADS - 1 ) / NUM_THREADS;
    
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {

    // Zero out the arrays
    cudaMemset(particle_ids, 0, num_parts * sizeof(int));
    cudaMemset(bin_ids     , 0, num_bins  * sizeof(int));
    cudaMemset(bin_counts  , 0, num_bins  * sizeof(int));
    
    // Calculate how many particles are in each bin and store the counts in bin_ids 
    num_parts_in_each_bin<<<num_blocks, NUM_THREADS>>>(parts, num_parts, bin_width, num_bins_across, bin_ids);

    // Next, perform a prefix sum on bin_ids using thrust
    // For example, if num_bins = 5 and num_parts = 15 then...
    // before prefix sum, bin_ids = {4  2  1  5  3}
    // after  prefix sum, bin_ids = {0  4  6  7 12}
    thrust::exclusive_scan(thrust::device, bin_ids, bin_ids+num_bins, bin_ids); 
    // see https://docs.nvidia.com/cuda/thrust/ for an example of exclusive_scan
    // and https://thrust.github.io/doc/group__prefixsums_ga0f1b7e1931f6ccd83c67c8cfde7c8144.html#ga0f1b7e1931f6ccd83c67c8cfde7c8144

    // Set particle_ids and bin_counts
    set_particle_ids_and_bin_counts<<<num_blocks, NUM_THREADS>>>(parts, num_parts, bin_width, num_bins_across, num_bins, particle_ids, bin_ids, bin_counts);
    
    // Compute forces
    compute_forces_gpu<<<num_blocks, NUM_THREADS>>>(parts, num_parts, bin_width, num_bins_across, particle_ids, bin_ids, bin_counts);

    // Move particles
    move_gpu<<<num_blocks, NUM_THREADS>>>(parts, num_parts, size);

    /* DEBUG data structures */
    debug_bin_counts<<<1, 1>>>(bin_counts, num_bins);
    debug_bin_ids<<<1, 1>>>(bin_ids, num_bins);
    debug_particle_ids<<<1, 1>>>(particle_ids, bin_ids, bin_counts, num_parts, num_bins); 

}
