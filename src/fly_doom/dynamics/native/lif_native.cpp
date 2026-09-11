#include "lif_native.h"

#include <cmath>
#include <vector>
#include <cstring>
#include <algorithm>

constexpr uint32_t MAX_DELAY_STEPS = 128;

struct LIFNativeEngine {
    uint32_t num_neurons;
    uint32_t num_edges;
    NativeLIFParams params;

    double alpha_m;
    std::vector<double> alpha_syn; // per-neuron decay factor
    int32_t refractory_steps;
    uint64_t step_count;

    // Graph CSR storage
    std::vector<uint32_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<double> weights;
    std::vector<uint32_t> edge_delays;
    bool has_delays;

    // Dynamic state
    std::vector<double> v;
    std::vector<double> i_syn;
    std::vector<int32_t> refractory_timer;

    // Delay ring buffer: [slot][neuron]
    std::vector<std::vector<double>> delay_buffer;
    uint32_t ring_idx;

    LIFNativeEngine(
        uint32_t n_neurons,
        uint32_t n_edges,
        const uint32_t* r_ptr,
        const uint32_t* c_idx,
        const double* w,
        const NativeLIFParams* p,
        const double* tau_syn_per_neuron = nullptr,
        const uint32_t* delays = nullptr
    ) : num_neurons(n_neurons),
        num_edges(n_edges),
        params(*p),
        step_count(0),
        row_ptr(r_ptr, r_ptr + n_neurons + 1),
        col_idx(c_idx, c_idx + n_edges),
        weights(w, w + n_edges),
        has_delays(delays != nullptr),
        v(n_neurons, p->v_rest),
        i_syn(n_neurons, 0.0),
        refractory_timer(n_neurons, 0),
        delay_buffer(MAX_DELAY_STEPS, std::vector<double>(n_neurons, 0.0)),
        ring_idx(0)
    {
        alpha_m = std::exp(-params.dt / params.tau_m);
        refractory_steps = static_cast<int32_t>(std::round(params.t_ref / params.dt));

        alpha_syn.resize(n_neurons);
        for (uint32_t i = 0; i < n_neurons; ++i) {
            double tau = tau_syn_per_neuron ? tau_syn_per_neuron[i] : params.tau_syn;
            if (tau <= 0.0) tau = params.tau_syn;
            alpha_syn[i] = std::exp(-params.dt / tau);
        }

        if (delays) {
            edge_delays.assign(delays, delays + n_edges);
        }
    }
};

extern "C" {

LIFNativeEngine* lif_create_engine(
    uint32_t num_neurons,
    uint32_t num_edges,
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const double* weights,
    const NativeLIFParams* params
) {
    if (!row_ptr || !col_idx || !weights || !params) {
        return nullptr;
    }
    return new LIFNativeEngine(num_neurons, num_edges, row_ptr, col_idx, weights, params);
}

LIFNativeEngine* lif_create_engine_extended(
    uint32_t num_neurons,
    uint32_t num_edges,
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const double* weights,
    const NativeLIFParams* params,
    const double* tau_syn_per_neuron,
    const uint32_t* edge_delays
) {
    if (!row_ptr || !col_idx || !weights || !params) {
        return nullptr;
    }
    return new LIFNativeEngine(
        num_neurons,
        num_edges,
        row_ptr,
        col_idx,
        weights,
        params,
        tau_syn_per_neuron,
        edge_delays
    );
}

void lif_destroy_engine(LIFNativeEngine* engine) {
    delete engine;
}

uint32_t lif_step(
    LIFNativeEngine* engine,
    const double* external_current,
    uint32_t* out_spikes,
    uint32_t max_spikes
) {
    if (!engine) return 0;

    const uint32_t N = engine->num_neurons;
    const double* alpha_syn = engine->alpha_syn.data();
    const double alpha_m = engine->alpha_m;
    const double v_rest = engine->params.v_rest;
    const double v_reset = engine->params.v_reset;
    const double v_thresh = engine->params.v_thresh;
    const double dt_rm = engine->params.dt * engine->params.r_m;
    const int32_t ref_steps = engine->refractory_steps;

    double* v = engine->v.data();
    double* i_syn = engine->i_syn.data();
    int32_t* ref = engine->refractory_timer.data();

    // 0. Drain delayed current inputs scheduled for this current step
    const uint32_t curr_slot = engine->ring_idx;
    double* scheduled_inputs = engine->delay_buffer[curr_slot].data();
    for (uint32_t i = 0; i < N; ++i) {
        i_syn[i] += scheduled_inputs[i];
        scheduled_inputs[i] = 0.0; // Clear after delivery
    }

    // 1. Decay synaptic currents & add external stimulus
    for (uint32_t i = 0; i < N; ++i) {
        i_syn[i] *= alpha_syn[i];
        if (external_current) {
            i_syn[i] += external_current[i];
        }
    }

    uint32_t spike_count = 0;

    // 2. Membrane potential update & refractory handling
    for (uint32_t i = 0; i < N; ++i) {
        if (ref[i] <= 0) {
            // Non-refractory integration
            double dv = (1.0 - alpha_m) * v_rest + dt_rm * i_syn[i];
            v[i] = alpha_m * v[i] + dv;

            // Spike detection
            if (v[i] >= v_thresh) {
                if (out_spikes && spike_count < max_spikes) {
                    out_spikes[spike_count++] = i;
                }
                v[i] = v_reset;
                ref[i] = ref_steps;
            }
        } else {
            // Refractory period clamp & decrement
            v[i] = v_reset;
            ref[i] -= 1;
        }
    }

    // 3. Synaptic transmission from spiking neurons
    if (out_spikes && spike_count > 0) {
        const uint32_t* row_ptr = engine->row_ptr.data();
        const uint32_t* col_idx = engine->col_idx.data();
        const double* weights = engine->weights.data();
        const uint32_t* delays = engine->has_delays ? engine->edge_delays.data() : nullptr;

        for (uint32_t s = 0; s < spike_count; ++s) {
            uint32_t pre = out_spikes[s];
            uint32_t start = row_ptr[pre];
            uint32_t end = row_ptr[pre + 1];
            for (uint32_t e = start; e < end; ++e) {
                uint32_t post = col_idx[e];
                uint32_t d = delays ? delays[e] : 0;
                if (d == 0) {
                    // Immediate delivery into next step
                    i_syn[post] += weights[e];
                } else {
                    // Schedule into future ring buffer slot
                    uint32_t target_slot = (curr_slot + d) % MAX_DELAY_STEPS;
                    engine->delay_buffer[target_slot][post] += weights[e];
                }
            }
        }
    }

    // Advance ring buffer slot
    engine->ring_idx = (curr_slot + 1) % MAX_DELAY_STEPS;
    engine->step_count++;
    return spike_count;
}

const double* lif_get_v(const LIFNativeEngine* engine) {
    return engine ? engine->v.data() : nullptr;
}

const double* lif_get_i_syn(const LIFNativeEngine* engine) {
    return engine ? engine->i_syn.data() : nullptr;
}

const int32_t* lif_get_refractory(const LIFNativeEngine* engine) {
    return engine ? engine->refractory_timer.data() : nullptr;
}

uint64_t lif_get_step_count(const LIFNativeEngine* engine) {
    return engine ? engine->step_count : 0;
}

void lif_set_state(
    LIFNativeEngine* engine,
    const double* v,
    const double* i_syn,
    const int32_t* refractory_timer
) {
    if (!engine) return;
    const size_t n = engine->num_neurons;
    if (v) std::memcpy(engine->v.data(), v, n * sizeof(double));
    if (i_syn) std::memcpy(engine->i_syn.data(), i_syn, n * sizeof(double));
    if (refractory_timer) std::memcpy(engine->refractory_timer.data(), refractory_timer, n * sizeof(int32_t));
}

} // extern "C"
