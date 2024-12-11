#include "kernels.h"
#include "aie_api/utils.hpp"

using namespace adf;

void antiKt(input_stream<float> * __restrict in, output_stream<float> * __restrict out)
{
    // data variables
    alignas(aie::vector_decl_align) float pts_arr[NUM_PARTICLES] = { 0.0 };
    alignas(aie::vector_decl_align) float etas_arr[NUM_PARTICLES] = { 0.0 };
    alignas(aie::vector_decl_align) float phis_arr[NUM_PARTICLES] = { 0.0 };

    // auxiliary variables
    alignas(aie::vector_decl_align) int16 index_arr[V_SIZE] = {0, 1, 2, 3, 4, 5, 6, 7};
    alignas(aie::vector_decl_align) float pi_arr[V_SIZE] = {PI_FLOAT, PI_FLOAT, PI_FLOAT, PI_FLOAT, PI_FLOAT, PI_FLOAT, PI_FLOAT, PI_FLOAT};
    alignas(aie::vector_decl_align) float mpi_arr[V_SIZE] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_FLOAT};
    alignas(aie::vector_decl_align) float twopi_arr[V_SIZE] = {TWOPI_FLOAT, TWOPI_FLOAT, TWOPI_FLOAT, TWOPI_FLOAT, TWOPI_FLOAT, TWOPI_FLOAT, TWOPI_FLOAT, TWOPI_FLOAT};
    alignas(aie::vector_decl_align) float mtwopi_arr[V_SIZE] = {MTWOPI_FLOAT, MTWOPI_FLOAT, MTWOPI_FLOAT, MTWOPI_FLOAT, MTWOPI_FLOAT, MTWOPI_FLOAT, MTWOPI_FLOAT, MTWOPI_FLOAT};

    aie::vector<int16, V_SIZE> *index_vec_pt = (aie::vector<int16, V_SIZE> *) index_arr;
    aie::vector<float, V_SIZE> *pi_vec_pt = (aie::vector<float, V_SIZE> *) pi_arr;
    aie::vector<float, V_SIZE> *mpi_vec_pt = (aie::vector<float, V_SIZE> *) mpi_arr;
    aie::vector<float, V_SIZE> *twopi_vec_pt = (aie::vector<float, V_SIZE> *) twopi_arr;
    aie::vector<float, V_SIZE> *mtwopi_vec_pt = (aie::vector<float, V_SIZE> *) mtwopi_arr;

    // algorithm variables
    float min_dist = 1000;
    alignas(aie::vector_decl_align) float min_dist_arr[V_SIZE / 2] = { 1000.0 };

    // read input data
    aie::vector<float, V_SIZE> *pts_pt = (aie::vector<float, V_SIZE> *) pts_arr;
    aie::vector<float, V_SIZE> *etas_pt = (aie::vector<float, V_SIZE> *) etas_arr;
    aie::vector<float, V_SIZE> *phis_pt = (aie::vector<float, V_SIZE> *) phis_arr;

    for (int i0=0; i0<P_BUNCHES; i0++)
    chess_loop_count(P_BUNCHES)
    chess_flatten_loop
    {
        *pts_pt = readincr_v<V_SIZE>(in);
        *etas_pt = readincr_v<V_SIZE>(in);
        *phis_pt = readincr_v<V_SIZE>(in);
        pts_pt++;
        etas_pt++;
        phis_pt++;
    }
    
    pts_pt -= P_BUNCHES;
    etas_pt -= P_BUNCHES;
    phis_pt -= P_BUNCHES;

    // select bunch to process
    int16 selected_bunch = 0;
    int16 idx_i0_arr[V_SIZE / 2] = { selected_bunch };
    int16 idx_j0_arr[V_SIZE / 2] = { -1 };
    int16 idx_i1_arr[V_SIZE / 2] = { selected_bunch };
    int16 idx_j1_arr[V_SIZE / 2] = { -1 };
    int16 idx_i0 = selected_bunch;
    int16 idx_i1 = selected_bunch;
    int16 idx_j0 = -1;
    int16 idx_j1 = -1;

    for (int j0=0; j0<(V_SIZE/2); j0++)
    chess_prepare_for_pipelining
    {   
        int16 idx_shift = j0 + 1;
        aie::vector<int16, V_SIZE> index_vec_shifted = aie::shuffle_up_fill(*index_vec_pt, *index_vec_pt, idx_shift);
        aie::vector<float, V_SIZE> pts_cur_shifted = aie::shuffle_up_fill(*pts_pt, *pts_pt, idx_shift);
        aie::vector<float, V_SIZE> etas_cur_shifted = aie::shuffle_up_fill(*etas_pt, *etas_pt, idx_shift);
        aie::vector<float, V_SIZE> phis_cur_shifted = aie::shuffle_up_fill(*phis_pt, *phis_pt, idx_shift);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDATA__)
        aie::print(*pts_pt, true, "\t\tpts_cur :");
        aie::print(*etas_pt, true, "\t\tetas_cur :");
        aie::print(*phis_pt, true, "\t\tphis_cur :");
        printf("\t\t--------------------\n");
        aie::print(pts_cur_shifted, true, "\t\tpts_cur_shifted :");
        aie::print(etas_cur_shifted, true, "\t\tetas_cur_shifted:");
        aie::print(phis_cur_shifted, true, "\t\tphis_cur_shifted:");
        printf("\t\t--------------------\n");
        #endif

        // COMPUTE PT FACTOR
        aie::mask<V_SIZE> pt_mask = aie::gt(*pts_pt, pts_cur_shifted);
        aie::vector<float, V_SIZE> pt_factor = aie::select(pts_cur_shifted, *pts_pt, pt_mask);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        aie::print(pt_factor, true, "\t\tSelected max pt (pt_factor)");
        printf("\t\t--------------------\n");
        #endif

        // once max {p_t1, p_tj} has been found, calculate the inverse and square it
        pt_factor = aie::inv(pt_factor); // pt_factor = 1 / pt_factor
        aie::accum<accfloat, V_SIZE> acc_float = aie::mul_square(pt_factor); // acc_float = (1 / pt_max) ^ 2
        pt_factor = acc_float.to_vector<float>(0);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        aie::print(pt_factor, true, "\t\tpt_factor: ");
        printf("\t\t--------------------\n");
        #endif

        // COMPUTE DR2 FACTOR
        // d_eta
        aie::vector<float, V_SIZE> d_eta = aie::sub(*etas_pt, etas_cur_shifted);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        aie::print(d_eta, true, "\t\td_eta: ");
        printf("\t\t--------------------\n");
        #endif

        // d_phi by taking into account -pi +pi boundaries
        aie::vector<float, V_SIZE> d_phi = aie::sub(*phis_pt, phis_cur_shifted);
        aie::vector<float, V_SIZE> d_phi_ptwopi = aie::add(d_phi, *twopi_vec_pt); // d_eta + 2 * pi
        aie::vector<float, V_SIZE> d_phi_mtwopi = aie::add(d_phi, *mtwopi_vec_pt); // d_eta - 2 * pi
        aie::mask<V_SIZE> is_gt_pi = aie::gt(d_phi, *pi_vec_pt);
        aie::mask<V_SIZE> is_lt_mpi = aie::lt(d_phi, *mpi_vec_pt);
        d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
        d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwop    

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        aie::print(d_phi, true, "\t\td_phi: ");
        printf("\t\t--------------------\n");
        #endif

        // multiply & accumulate to get dr2 / R0
        acc_float = aie::mul_square(d_eta);
        acc_float = aie::mac_square(acc_float, d_phi);
        aie::vector<float, V_SIZE> dr2_factor = acc_float.to_vector<float>(0);
        acc_float = aie::mul(dr2_factor, invR02); // acc_float = dr2 * (1 / R0) ^ 2
        dr2_factor = acc_float.to_vector<float>(0);   
        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        aie::print(dr2_factor, true, "\t\tdr2_factor: ");
        printf("\t\t--------------------\n");
        #endif

        // COMPUTE DISTANCE
        acc_float = aie::mul(pt_factor, dr2_factor);
        aie::vector<float, V_SIZE> dist = acc_float.to_vector<float>(0);

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        aie::print(dist, true, "\t\tdist: ");
        printf("\t\t--------------------\n");
        #endif

        // FIND MINIMUM DISTANCE IN THE CURRENT BUNCH
        min_dist_arr[j0] = aie::reduce_min(dist); 

        aie::mask<V_SIZE> min_idx_mask = aie::gt(dist, min_dist_arr[j0]); // has ones where elements are gt the min distance found
        min_idx_mask = ~min_idx_mask;
        int16 min_idx = V_SIZE - min_idx_mask.clz() - 1; 
        idx_j0_arr[j0] = index_arr[min_idx];
        idx_j1_arr[j0] = index_vec_shifted[min_idx];

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        printf("\t\tmin_dist_arr[idx_shif] = %f between (%d, %d) and (%d, %d)\n", min_dist_arr[idx_shift-1], idx_i0_arr[idx_shift-1], idx_j0_arr[idx_shift-1], idx_i1_arr[idx_shift-1], idx_j1_arr[idx_shift-1]);
        printf("\t\t--------------------\n");
        printf("\t\t--------------------\n\n\n");
        #endif
    }

    aie::vector<float, V_SIZE / 2> min_dist_vec = aie::load_v<V_SIZE / 2>(min_dist_arr);

    #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
    aie::print(min_dist_vec, true, "min_dist_vec: ");
    printf("--------------------\n");
    #endif

    min_dist = aie::reduce_min(min_dist_vec);
    aie::mask<V_SIZE / 2> min_idx_mask = aie::gt(min_dist_vec, min_dist); // has ones where elements are gt the min distance found
    min_idx_mask = ~min_idx_mask;
    int16 min_idx = (V_SIZE / 2) - min_idx_mask.clz() - 1; 
    idx_j0 = idx_j0_arr[min_idx];
    idx_j1 = idx_j1_arr[min_idx];

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("Min dist found between (%d, %d) and (%d, %d) = %f\n", idx_i0, idx_j0, idx_i1, idx_j1, min_dist);
    printf("\n");
    #endif

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("\n\n\n");
    #endif

    // flush output data
    writeincr(out, idx_i0);
    writeincr(out, idx_j0);
    writeincr(out, idx_i1);
    writeincr(out, idx_j1);
    writeincr(out, min_dist);
}