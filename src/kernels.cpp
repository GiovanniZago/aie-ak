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
    float min_dist_beam = 1000;
    int16 idx_i0 = -1, idx_j0 = -1;
    int16 idx_i1 = -1, idx_j1 = -1;
    int16 idx_i_beam = -1, idx_j_beam = -1;
    aie::vector<int16, N_JETS> pts_jets = aie::zeros<int16, N_JETS>();
    aie::vector<int16, N_JETS> etas_jets = aie::zeros<int16, N_JETS>();
    aie::vector<int16, N_JETS> phis_jets = aie::zeros<int16, N_JETS>();
    aie::mask<N_JETS> jets_mask;
    jets_mask.set(0);

    // read input data
    pts[0] = readincr_v<V_SIZE>(in);
    pts[1] = readincr_v<V_SIZE>(in);
    etas[0] = readincr_v<V_SIZE>(in);
    etas[1] = readincr_v<V_SIZE>(in);
    phis[0] = readincr_v<V_SIZE>(in);
    phis[1] = readincr_v<V_SIZE>(in);

    // Algorithm implementation
    for (int i_ep=0; i_ep<N_EPOCH; i_ep++)
    {
        #if defined(__X86SIM__) && defined(__X86DEBUG__)
        printf("*************************************************************************************************\n");
        printf("Iteration # %d\n\n", i_ep);
        #endif

        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDATA__)
        aie::print(pts[0], true, "pts[0]: ");
        aie::print(pts[1], true, "pts[1]: ");
        aie::print(etas[0], true, "etas[0]: ");
        aie::print(etas[1], true, "etas[1]: ");
        aie::print(phis[0], true, "phis[0]: ");
        aie::print(phis[1], true, "phis[1]: ");
        #endif

        int16 max_dist_beam_int = 0;

        for (int i0=0; i0<P_BUNCHES; i0++)
        {
            int16 dist_beam_int = aie::reduce_max(pts[i0]);

            if (dist_beam_int >= max_dist_beam_int)
            {
                max_dist_beam_int = dist_beam_int;
                float max_dist_beam_float = aie::to_float(max_dist_beam_int, 0);
                min_dist_beam = aie::inv(max_dist_beam_float);
                min_dist_beam = min_dist_beam * min_dist_beam;
                min_dist_beam = min_dist_beam * INVPT_CONV2;

                aie::mask<V_SIZE> max_idx_mask = aie::lt(pts[i0], max_dist_beam_int); // has ones where elements are lt the max pt found
                max_idx_mask = ~max_idx_mask;
                int16 first_one_index = max_idx_mask.clz();
                idx_i_beam = i0;
                idx_j_beam = idx_vector[first_one_index];

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDISTBEAM__)
                printf("\t\tMin beam dist found on (%d, %d) = %f (pt int = %d)\n", idx_i_beam, idx_j_beam, min_dist_beam, max_dist_beam_int);
                printf("\n");
                #endif  
            }
        }

        #if defined(__X86SIM__) && defined(__X86DEBUG__)
        printf("Min beam dist found on (%d, %d) = %f\n", idx_i_beam, idx_j_beam, min_dist_beam);
        printf("\n");
        #endif
        
        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
        printf("\t\tDEBUG DIST\n\n");
        #endif

        for (int i0=0; i0<P_BUNCHES; i0++)
        {
            for (int j0=0; j0<V_SIZE; j0++)
            {
                if (!pts[i0][j0]) continue;

                min_dist = 1000;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
                printf("\t\tCandindate (%d, %d)\n", i0, j0);
                #endif

                for (int i1=0; i1<P_BUNCHES; i1++)
                {
                    // COMPUTE PT FACTOR
                    aie::mask<V_SIZE> pt_mask = aie::gt(pts[i1], pts[i0][j0]);
                    aie::vector<int16, V_SIZE> pt_pair_max = aie::select(pts[i0][j0], pts[i1], pt_mask);

                    #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
                    aie::print(pt_pair_max, true, "\t\tSelected max pt ");
                    printf("\t\t--------------------\n");
                    #endif

                    // once max {p_t1, p_tj} has been found, calculate the inverse and square it
                    aie::vector<float, V_SIZE> pt_pair_max_float = aie::to_float(pt_pair_max, 0);
                    aie::vector<float, V_SIZE> pt_factor = aie::inv(pt_pair_max_float); // pt_factor = 1 / pt_min
                    aie::accum<accfloat, V_SIZE> acc_float = aie::mul_square(pt_factor); // acc_float = (1 / pt_min) ^ 2
                    pt_factor = acc_float.to_vector<float>(0); // pt_factor = (1 / pt_min) ^ 2
                    acc_float = aie::mul(pt_factor, INVPT_CONV2); // acc_float = ((1 / pt_min) ^ 2) * ((1 / 0.25) ^ 2)
                    pt_factor = acc_float.to_vector<float>(0); // pt_factor = ((1 / pt_min) ^ 2) * ((1 / 0.25) ^ 2)

                    #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
                    aie::print(pt_factor, true, "\t\tPt factor ");
                    printf("\t\t--------------------\n");
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

                    #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
                    aie::print(dr2_factor, true, "\t\tdr2 factor ");
                    printf("\t\t--------------------\n");
                    #endif

                    // COMPUTE DISTANCE
                    acc_float = aie::mul(pt_factor, dr2_factor);
                    aie::vector<float, V_SIZE> dist = acc_float.to_vector<float>(0);
                    aie::mask<V_SIZE> valid_mask = aie::eq(dr2, (int32) 0); // spot the dr2=0 due to considering the same particle
                    dist = aie::select(dist, (float) 10000, valid_mask);
                    valid_mask = aie::eq(pts[i1], (int16) 0); // do not consider particles that are zeros (due to the padding)
                    dist = aie::select(dist, (float) 10000, valid_mask);

                    #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
                    aie::print(dist, true, "\t\tDistances ");
                    printf("\t\t--------------------\n");
                    #endif

                    // FIND MINIMUM DISTANCE IN THE CURRENT BUNCH
                    min_dist_bunch = aie::reduce_min(dist);

                    #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__)
                    printf("\t\tmin_dist_bunch = %f\n", min_dist_bunch);
                    printf("\t\t--------------------\n\n");
                    #endif

                    if (min_dist_bunch < min_dist)
                    {
                        min_dist = min_dist_bunch;
                        aie::mask<V_SIZE> min_idx_mask = aie::gt(dist, min_dist); // has ones where elements are gt the min distance found
                        min_idx_mask = ~min_idx_mask; // flip 0s and 1s in the mask
                        int16 first_one_index = V_SIZE - min_idx_mask.clz() - 1; // find the index of the first 1 element of the mask

                        #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGDIST__) && defined(__X86DEBUGDIST1__)
                        aie::print(min_idx_mask, true, "\t\t\t\tmin_idx_mask ");
                        printf("\t\t\t\t--------------------\n");
                        printf("\t\t\t\tfirst_one_index = %d\n", first_one_index);
                        printf("\t\t\t\t--------------------\n\n");
                        #endif

                        idx_i0 = i0; 
                        idx_j0 = j0;
                        idx_i1 = i1;
                        idx_j1 = idx_vector[first_one_index];
                    }
                }
            }
        }

        #if defined(__X86SIM__) && defined(__X86DEBUG__)
        printf("Min dist found between (%d, %d) and (%d, %d) = %f\n", idx_i0, idx_j0, idx_i1, idx_j1, min_dist);
        printf("\n");
        #endif

        if (min_dist_beam < min_dist)
        {
            bool updated = false;

            #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGUPDATE__)
            printf("\t\tDEBUG UPDATE\n\n");
            #endif

            for (int i1=0; i1<P_BUNCHES; i1++)
            {

                // find the mask that spots the location of the pseudo-jet close to the beam, if there's any
                aie::mask<V_SIZE> match_pt = aie::eq(pts[i1], pts[idx_i_beam][idx_j_beam]);
                aie::mask<V_SIZE> match_eta = aie::eq(etas[i1], etas[idx_i_beam][idx_j_beam]);
                aie::mask<V_SIZE> match_phi = aie::eq(phis[i1], phis[idx_i_beam][idx_j_beam]);
                aie::mask<V_SIZE> match_mask = match_pt & match_eta & match_phi;
                bool updated_cur = updated ? false : !match_mask.empty();
                updated = updated_cur;
                aie::mask<V_SIZE> updated_mask(updated);
                match_mask &= updated_mask;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGUPDATE__)
                aie::print(match_mask, true, "\t\tmatch_mask bunch ");
                printf("\t\t--------------------\n");
                #endif

                // once selected, extract its pt, eta and phi
                aie::vector<int16, V_SIZE> pt_selected_vec = aie::select(zeros_vector, pts[i1], match_mask);
                aie::vector<int16, V_SIZE> eta_selected_vec = aie::select(zeros_vector, etas[i1], match_mask);
                aie::vector<int16, V_SIZE> phi_selected_vec = aie::select(zeros_vector, phis[i1], match_mask);
                int16 pt_selected = aie::reduce_add(pt_selected_vec);
                int16 eta_selected = aie::reduce_add(eta_selected_vec);
                int16 phi_selected = aie::reduce_add(phi_selected_vec);

                aie::mask<N_JETS> updated_mask_shift(updated);
                aie::mask<N_JETS> jets_mask_cur = jets_mask & updated_mask_shift;

                // mark the pseudo-jet as jet by inserting the selected pt, eta and phi inside the jets vectors
                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGUPDATE__)
                aie::print(jets_mask, true, "\t\tjets_mask ");
                aie::print(jets_mask_cur, true, "\t\tjets_mask_cur ");
                printf("\t\t--------------------\n");
                #endif

                pts_jets = aie::select(pts_jets, pt_selected, jets_mask);
                etas_jets = aie::select(pts_jets, eta_selected, jets_mask);
                phis_jets = aie::select(pts_jets, phi_selected, jets_mask);

                // set to zero the pseudo-jet in the pts, etas, phis vectors
                pts[i1] = aie::select(pts[i1], zeros_vector, match_mask);
                etas[i1] = aie::select(etas[i1], zeros_vector, match_mask);
                phis[i1] = aie::select(phis[i1], zeros_vector, match_mask);

                // shift right jets_mask in order to insert a new jet in the following entry next time it occurs
                jets_mask = jets_mask >> 1;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGUPDATE__)
                aie::print(jets_mask, true, "\t\tjets_mask after (potential) shift bunch ");
                printf("\t\t--------------------\n");
                #endif
            }
        } else
        {
            int32 pt_sum = 0;
            int32 eta_wsum = 0, phi_wsum = 0; 

            for (int i1=0; i1<P_BUNCHES; i1++)
            {
                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGUPDATE__)
                printf("\t\tDEBUG RECOMBINATION - 1\n");
                #endif

                aie::mask<V_SIZE> match_pt = aie::eq(pts[i1], pts[idx_i0][idx_j0]);
                aie::mask<V_SIZE> match_eta = aie::eq(etas[i1], etas[idx_i0][idx_j0]);
                aie::mask<V_SIZE> match_phi = aie::eq(phis[i1], phis[idx_i0][idx_j0]);
                aie::mask<V_SIZE> match_mask0 = match_pt & match_eta & match_phi;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGRECOMB__)
                aie::print(match_mask0, true, "\t\tmatch_mask0 bunch ");
                printf("\t\t--------------------\n");
                #endif

                match_pt = aie::eq(pts[i1], pts[idx_i1][idx_j1]);
                match_eta = aie::eq(etas[i1], etas[idx_i1][idx_j1]);
                match_phi = aie::eq(phis[i1], phis[idx_i1][idx_j1]);
                aie::mask<V_SIZE> match_mask1 = match_pt & match_eta & match_phi;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGRECOMB__)
                aie::print(match_mask1, true, "\t\tmatch_mask1 bunch ");
                printf("\t\t--------------------\n");
                #endif

                aie::mask<V_SIZE> match_mask_tot = match_mask0 | match_mask1;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGRECOMB__)
                aie::print(match_mask_tot, true, "\t\tmatch_mask_tot bunch ");
                printf("\t\t--------------------\n\n");
                #endif

                aie::vector<int16, V_SIZE> pts_to_sum = aie::select(zeros_vector, pts[i1], match_mask_tot);
                pt_sum += aie::reduce_add(pts_to_sum);

                aie::vector<int16, V_SIZE> etas_to_sum = aie::select(zeros_vector, etas[i1], match_mask_tot);
                aie::accum<acc48, V_SIZE> acc = aie::mul(etas_to_sum, pts_to_sum);
                aie::vector<int32, V_SIZE> eta_pt_prod = acc.to_vector<int32>(0);
                eta_wsum += aie::reduce_add(eta_pt_prod);

                aie::vector<int16, V_SIZE> phis_to_sum = aie::select(zeros_vector, phis[i1], match_mask_tot);
                acc = aie::mul(phis_to_sum, pts_to_sum);
                aie::vector<int32, V_SIZE> phi_pt_prod = acc.to_vector<int32>(0);
                phi_wsum += aie::reduce_add(phi_pt_prod);
            }

            float pt_sum_float = aie::to_float(pt_sum, 0);
            float invpt_sum_float = aie::inv(pt_sum_float);

            float eta_wsum_float = aie::to_float(eta_wsum, 0);
            float eta_updated_float = eta_wsum_float * invpt_sum_float;
            int32 eta_updated = aie::to_fixed(eta_updated_float, 0);

            float phi_wsum_float = aie::to_float(phi_wsum, 0);
            float phi_updated_float = phi_wsum_float * invpt_sum_float;
            int32 phi_updated = aie::to_fixed(phi_updated_float, 0);

            bool recomb0 = false, recomb1 = false;

            for (int i1=0; i1<P_BUNCHES; i1++)
            {
                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGUPDATE__)
                printf("\t\tDEBUG RECOMBINATION - 2\n");
                #endif

                aie::mask<V_SIZE> match_pt = aie::eq(pts[i1], pts[idx_i0][idx_j0]);
                aie::mask<V_SIZE> match_eta = aie::eq(etas[i1], etas[idx_i0][idx_j0]);
                aie::mask<V_SIZE> match_phi = aie::eq(phis[i1], phis[idx_i0][idx_j0]);
                aie::mask<V_SIZE> match_mask0 = match_pt & match_eta & match_phi;
                bool recomb0_cur = recomb0 ? false : !match_mask0.empty();
                recomb0 = recomb0_cur;
                aie::mask<V_SIZE> recomb0_mask(recomb0);
                match_mask0 &= recomb0_mask;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGRECOMB__)
                aie::print(match_mask0, true, "\t\tmatch_mask0 bunch ");
                printf("\t\t--------------------\n");
                #endif

                match_pt = aie::eq(pts[i1], pts[idx_i1][idx_j1]);
                match_eta = aie::eq(etas[i1], etas[idx_i1][idx_j1]);
                match_phi = aie::eq(phis[i1], phis[idx_i1][idx_j1]);
                aie::mask<V_SIZE> match_mask1 = match_pt & match_eta & match_phi;
                bool recomb1_cur = recomb1 ? false : !match_mask1.empty();
                recomb1 = recomb1_cur;
                aie::mask<V_SIZE> recomb1_mask(recomb1);
                match_mask1 &= recomb1_mask;

                #if defined(__X86SIM__) && defined(__X86DEBUG__) && defined(__X86DEBUGRECOMB__)
                aie::print(match_mask1, true, "\t\tmatch_mask1 bunch ");
                printf("\t\t--------------------\n\n");
                #endif

                pts[i1] = aie::select(pts[i1], (int16) pt_sum, match_mask0);
                etas[i1] = aie::select(etas[i1], (int16) eta_updated, match_mask0);
                phis[i1] = aie::select(phis[i1], (int16) phi_updated, match_mask0);

                pts[i1] = aie::select(pts[i1], zeros_vector, match_mask1);
                etas[i1] = aie::select(etas[i1], zeros_vector, match_mask1);
                phis[i1] = aie::select(phis[i1], zeros_vector, match_mask1);
            }
        }
    }

    #if defined(__X86SIM__) && defined(__X86DEBUG__)
    printf("\n\n\n");
    #endif

    // flush output data
    writeincr(out, pts[0]);
    writeincr(out, pts[1]);
    writeincr(out, etas[0]);
    writeincr(out, etas[1]);
    writeincr(out, phis[0]);
    writeincr(out, phis[1]);
}
