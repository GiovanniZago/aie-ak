#include "adf.h"
#include "kernels.h"

using namespace adf;

class AntiKtGraph : public graph {
    private:
        kernel antiKt_k;

    public:
        input_plio in;
        output_plio out;

        AntiKtGraph() {
            antiKt_k = kernel::create(antiKt);

            in = input_plio::create(plio_128_bits, "data/event_data_TTbar.csv", 360);
            out = output_plio::create(plio_32_bits, "data/event_out_TTbar.csv", 360);

            // PL inputs
            connect<stream>(in.out[0], antiKt_k.in[0]);

            // PL outputs
            connect<stream>(antiKt_k.out[0], out.in[0]);

            // sources and runtime ratios
            source(antiKt_k) = "kernels.cpp";
            runtime<ratio>(antiKt_k) = 1;
        }
};

