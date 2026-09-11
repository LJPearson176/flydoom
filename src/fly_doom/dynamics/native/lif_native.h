#pragma once

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque engine handle
typedef struct LIFNativeEngine LIFNativeEngine;

// Parameter structure matching Python LIFParameters
typedef struct {
    double v_rest;
    double v_reset;
    double v_thresh;
    double tau_m;
    double tau_syn;
    double t_ref;
    double dt;
    double r_m;
} NativeLIFParams;

// Create engine instance
LIFNativeEngine* lif_create_engine(
    uint32_t num_neurons,
    uint32_t num_edges,
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const double* weights,
    const NativeLIFParams* params
);

// Destroy engine instance
void lif_destroy_engine(LIFNativeEngine* engine);

// Single simulation step
// Returns number of spikes written to out_spikes buffer
uint32_t lif_step(
    LIFNativeEngine* engine,
    const double* external_current,
    uint32_t* out_spikes,
    uint32_t max_spikes
);

// Read-only buffer accessors
const double* lif_get_v(const LIFNativeEngine* engine);
const double* lif_get_i_syn(const LIFNativeEngine* engine);
const int32_t* lif_get_refractory(const LIFNativeEngine* engine);
uint64_t lif_get_step_count(const LIFNativeEngine* engine);

// Direct state setters (for deterministic test synchronization)
void lif_set_state(
    LIFNativeEngine* engine,
    const double* v,
    const double* i_syn,
    const int32_t* refractory_timer
);

#ifdef __cplusplus
}
#endif
