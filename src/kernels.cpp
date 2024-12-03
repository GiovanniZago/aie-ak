#include "kernels.h"
#include "aie_api/utils.hpp"

using namespace adf;

void antiKt(input_stream<int16> * __restrict in, output_stream<int16> * __restrict out)
{
    // data variables
    aie::vector<int16, V_SIZE> pts[P_BUNCHES], etas[P_BUNCHES], phis[P_BUNCHES];

    // auxiliary variables
    alignas(aie::vector_decl_align) static int16 idx_array[V_SIZE] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    const aie::vector<int16, V_SIZE> idx_vector = aie::load_v<V_SIZE>(idx_array);
    const aie::vector<int16, V_SIZE> zeros_vector = aie::zeros<int16, V_SIZE>();
    const aie::vector<int16, V_SIZE> pi_vector = aie::broadcast<int16, V_SIZE>(PI);
    const aie::vector<int16, V_SIZE> mpi_vector = aie::broadcast<int16, V_SIZE>(MPI);
    const aie::vector<int16, V_SIZE> twopi_vector = aie::broadcast<int16, V_SIZE>(TWOPI);
    const aie::vector<int16, V_SIZE> mtwopi_vector = aie::broadcast<int16, V_SIZE>(MTWOPI);

    // algorithm variables
    float min_dist = 1000;
    float min_dist_bunch = 0;
    int16 idx_i = 1000, idx_j = 1000;

    // read input data
    pts[0] = readincr_v<V_SIZE>(in);
    pts[1] = readincr_v<V_SIZE>(in);
    etas[0] = readincr_v<V_SIZE>(in);
    etas[1] = readincr_v<V_SIZE>(in);
    phis[0] = readincr_v<V_SIZE>(in);
    phis[1] = readincr_v<V_SIZE>(in);

    // Algorithm implementation
    for (int i0=0; i0<P_BUNCHES; i0++)
    {
        for (int j0=0; j0<V_SIZE; j0++)
        {
            min_dist = 1000;

            #if defined(__X86SIM__) && defined(__X86DEBUG__)
            printf("Candindate (%d, %d)\n", i0, j0);
            #endif

            for (int i1=0; i1<P_BUNCHES; i1++)
            {
                // COMPUTE PT FACTOR
                aie::mask<V_SIZE> pt_mask = aie::gt(pts[i1], pts[i0][j0]);
                aie::vector<int16, V_SIZE> pt_pair_max = aie::select(pts[i0][j0], pts[i1], pt_mask);

                // once max {p_t1, p_tj} has been found, calculate the inverse and square it
                aie::vector<float, V_SIZE> pt_pair_max_float = aie::to_float(pt_pair_max, 0);

                #if defined(__X86SIM__) && defined(__X86DEBUG__)
                aie::print(pt_pair_max, true, "Selected minimum pt ");
                printf("--------------------\n");
                #endif

                aie::vector<float, V_SIZE> pt_factor = aie::inv(pt_pair_max_float); // pt_factor = 1 / pt_min
                aie::accum<accfloat, V_SIZE> acc_float = aie::mul_square(pt_factor); // acc_float = (1 / pt_min) ^ 2
                pt_factor = acc_float.to_vector<float>(0); // pt_factor = (1 / pt_min) ^ 2
                acc_float = aie::mul(pt_factor, INVPT_CONV2); // acc_float = ((1 / pt_min) ^ 2) * ((1 / 0.25) ^ 2)
                pt_factor = acc_float.to_vector<float>(0); // pt_factor = ((1 / pt_min) ^ 2) * ((1 / 0.25) ^ 2)

                #if defined(__X86SIM__) && defined(__X86DEBUG__)
                aie::print(pt_factor, true, "Pt factor ");
                printf("--------------------\n");
                #endif

                // COMPUTE DR2 FACTOR
                // d_eta
                aie::vector<int16, V_SIZE> d_eta = aie::sub(etas[i0][j0], etas[i1]);

                // d_phi by taking into account -pi +pi boundaries
                aie::vector<int16, V_SIZE> d_phi = aie::sub(phis[i0][j0], phis[i1]);
                aie::vector<int16, V_SIZE> d_phi_ptwopi = aie::add(d_phi, twopi_vector); // d_eta + 2 * pi
                aie::vector<int16, V_SIZE> d_phi_mtwopi = aie::add(d_phi, mtwopi_vector); // d_eta - 2 * pi
                aie::mask<V_SIZE> is_gt_pi = aie::gt(d_phi, pi_vector);
                aie::mask<V_SIZE> is_lt_mpi = aie::lt(d_phi, mpi_vector);
                d_phi = aie::select(d_phi, d_phi_ptwopi, is_lt_mpi); // select element from d_phi if element is geq of -pi, otherwise from d_phi_ptwopi
                d_phi = aie::select(d_phi, d_phi_mtwopi, is_gt_pi); // select element from d_phi if element is leq of pi, otherwise from d_phi_mtwopi

                // multiply & accumulate to get dr2 / R0
                aie::accum<acc48, V_SIZE> acc = aie::mul_square(d_eta);
                acc = aie::mac_square(acc, d_phi);
                aie::vector<int32, V_SIZE> dr2 = acc.to_vector<int32>(0);
                aie::vector<float, V_SIZE> dr2_factor = aie::to_float(dr2, 0); // dr2_factor = dr2
                acc_float = aie::mul(dr2_factor, ANG_CONV2); // acc_float = dr2_int * ((pi / 720) ^ 2)
                dr2_factor = acc_float.to_vector<float>(0); // dr2_factor = dr2_int * ((pi / 720) ^ 2)
                acc_float = aie::mul(dr2_factor, invR02); // acc_float = dr2_factor * (1 / R0) ^ 2
                dr2_factor = acc_float.to_vector<float>(0); // dr2_factor = dr2_factor * (1 / R0) ^ 2

                #if defined(__X86SIM__) && defined(__X86DEBUG__)
                aie::print(dr2_factor, true, "dr2 factor ");
                printf("--------------------\n");
                #endif

                // COMPUTE DISTANCE
                acc_float = aie::mul(pt_factor, dr2_factor);
                aie::vector<float, V_SIZE> dist = acc_float.to_vector<float>(0);
                aie::mask<V_SIZE> valid_mask = aie::eq(dr2, (int32) 0); // spot the dr2=0 due to considering the same particle
                dist = aie::select(dist, (float) 10000, valid_mask);
                valid_mask = aie::eq(pts[i1], (int16) 0); // do not consider particles that are zeros (due to the padding)
                dist = aie::select(dist, (float) 10000, valid_mask);

                #if defined(__X86SIM__) && defined(__X86DEBUG__)
                aie::print(dist, true, "Distances ");
                printf("--------------------\n");
                #endif

                // FIND MINIMUM DISTANCE IN THE CURRENT BUNCH
                min_dist_bunch = aie::reduce_min(dist);

                #if defined(__X86SIM__) && defined(__X86DEBUG__)
                printf("min_dist_bunch = %f\n", min_dist_bunch);
                printf("--------------------\n");
                #endif

                if (min_dist_bunch < min_dist)
                {
                    min_dist = min_dist_bunch;
                    aie::mask<V_SIZE> min_idx_mask = aie::gt(dist, min_dist_bunch); // has ones where elements are gt the min distance found
                    min_idx_mask = ~min_idx_mask; // flip 0s and 1s in the mask
                    aie::vector<int16, V_SIZE> min_idx_vec = aie::select(zeros_vector, idx_vector, min_idx_mask); // contains the vector index of the minimum
                    idx_j = aie::reduce_add(min_idx_vec);
                    idx_i = i1;
                }
            }

            #if defined(__X86SIM__) && defined(__X86DEBUG__)
            printf("Min dist %f (%d, %d)\n", min_dist, idx_i, idx_j);
            printf("\n\n");
            #endif
        }
    }

    // flush output data
    writeincr(out, pts[0]);
    writeincr(out, pts[1]);
    writeincr(out, etas[0]);
    writeincr(out, etas[1]);
    writeincr(out, phis[0]);
    writeincr(out, phis[1]);
}
