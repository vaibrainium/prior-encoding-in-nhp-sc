
class PoissonGLMConfig:
    """Configuration class for NeuroGLM model parameters."""

    # TRIAL TIMING PARAMETERS
    # ========================================================================
    BIN_SIZE_MS = 1.0           # Temporal resolution
    PRE_TARGET_MS = 50          # Time before target onset
    # BASIS FUNCTION TYPES
    BASIS_RAISED_COSINE = "raised_cosine"
    BASIS_BOXCAR = "boxcar"
    # EFFECT TYPES FOR TEMPORAL KERNELS
    EFFECT_CAUSAL = "causal"           # Forward in time
    EFFECT_ANTI_CAUSAL = "anti-causal" # Backward in time
    # Target onset component
    TARGET_DURATION_MS = 200
    TARGET_N_BASES = int(TARGET_DURATION_MS/50 + 1)
    TARGET_EFFECT = EFFECT_CAUSAL
    TARGET_BASIS = BASIS_RAISED_COSINE
    # Stimulus coherence component
    STIMULUS_DURATION_MS = 500
    STIMULUS_N_BASES = int(STIMULUS_DURATION_MS/50 + 1)
    N_COHERENCE_LEVELS = 7  # Based on COH_LEVELS
    STIMULUS_EFFECT = EFFECT_CAUSAL
    STIMULUS_BASIS = BASIS_RAISED_COSINE
    # Saccade/Choice component
    SACCADE_DURATION_MS = 1500
    SACCADE_N_BASES = int(SACCADE_DURATION_MS/50 + 1)
    N_CHOICE_OPTIONS = 2  # Binary choice task
    SACCADE_EFFECT = EFFECT_ANTI_CAUSAL
    SACCADE_BASIS = BASIS_RAISED_COSINE
    # Bias state component
    BIAS_DURATION_MS = 2000
    BIAS_N_BASES = int(BIAS_DURATION_MS/50 + 1)
    N_BIAS_STATES = 2  # Biased vs Unbiased from GLM-HMM
    BIAS_EFFECT = EFFECT_ANTI_CAUSAL
    BIAS_BASIS = BASIS_RAISED_COSINE
    # Post-spike history component
    HISTORY_UNIFORM_MS = 11      # Fast refractory period
    HISTORY_NONLINEAR_MS = 265   # Slower dynamics
    HISTORY_N_UNIFORM = 10       # Uniform bases for fast dynamics
    HISTORY_N_NONLINEAR = 10     # Raised cosine for slower dynamics
    # Direct feature counts
    FEATURES_TARGET = TARGET_N_BASES
    FEATURES_STIMULUS = STIMULUS_N_BASES * N_COHERENCE_LEVELS
    FEATURES_SACCADE = SACCADE_N_BASES * N_CHOICE_OPTIONS
    FEATURES_BIAS = BIAS_N_BASES * N_BIAS_STATES
    FEATURES_HISTORY = HISTORY_N_UNIFORM + HISTORY_N_NONLINEAR
    FEATURES_INTERCEPT = 1

    @classmethod
    def get_total_features(cls):
        """Calculate total number of features."""
        return (cls.FEATURES_TARGET + cls.FEATURES_STIMULUS + cls.FEATURES_SACCADE +
                cls.FEATURES_BIAS + cls.FEATURES_HISTORY + cls.FEATURES_INTERCEPT)



class StateBasedPoissonGLMConfig:
    """Configuration class for NeuroGLM model parameters."""

    # TRIAL TIMING PARAMETERS
    BIN_SIZE_MS = 1.0           # Temporal resolution
    PRE_TARGET_MS = 50          # Time before target onset
    # BASIS FUNCTION TYPES
    BASIS_RAISED_COSINE = "raised_cosine"
    BASIS_BOXCAR = "boxcar"
    # EFFECT TYPES FOR TEMPORAL KERNELS
    EFFECT_CAUSAL = "causal"           # Forward in time
    EFFECT_ANTI_CAUSAL = "anti-causal" # Backward in time

    # Target onset component
    TARGET_DURATION_MS = 200
    TARGET_SPACING_MS = 10
    TARGET_N_BASES = int(TARGET_DURATION_MS/TARGET_SPACING_MS + 1)
    TARGET_EFFECT = EFFECT_CAUSAL
    TARGET_BASIS = BASIS_RAISED_COSINE

    # Stimulus coherence component
    STIMULUS_DURATION_MS = 300
    STIMULUS_SPACING_MS = 10
    STIMULUS_N_BASES = int(STIMULUS_DURATION_MS/STIMULUS_SPACING_MS + 1)
    N_COHERENCE_LEVELS = 1#7  # Based on COH_LEVELS
    STIMULUS_EFFECT = EFFECT_CAUSAL
    STIMULUS_BASIS = BASIS_RAISED_COSINE

    # Stimulus offset component
    STIMULUS_OFFSET_DURATION_MS = 50
    STIMULUS_OFFSET_SPACING_MS = 10
    STIMULUS_OFFSET_N_BASES = int(STIMULUS_OFFSET_DURATION_MS/STIMULUS_OFFSET_SPACING_MS + 1)
    STIMULUS_OFFSET_EFFECT = EFFECT_CAUSAL
    STIMULUS_OFFSET_BASIS = BASIS_RAISED_COSINE

    # Saccade/Choice component
    SACCADE_DURATION_MS = 1500
    SACCADE_SPACING_MS = 10
    SACCADE_N_BASES = int(SACCADE_DURATION_MS/SACCADE_SPACING_MS + 1)
    N_CHOICE_OPTIONS = 2  # Binary choice task
    SACCADE_EFFECT = EFFECT_ANTI_CAUSAL
    SACCADE_BASIS = BASIS_RAISED_COSINE
    # N_BIAS_STATES = 2  # Biased vs Unbiased from GLM-HMM
    N_BIAS_STATES = 1  # Only use equal block

    # Post-spike history component
    HISTORY_UNIFORM_MS = 11      # Fast refractory period
    HISTORY_NONLINEAR_MS = 265   # Slower dynamics
    HISTORY_N_UNIFORM = 10       # Uniform bases for fast dynamics
    HISTORY_N_NONLINEAR = 10     # Raised cosine for slower dynamics

    # Direct feature counts
    FEATURES_TARGET = TARGET_N_BASES * N_BIAS_STATES
    FEATURES_STIMULUS = STIMULUS_N_BASES * N_COHERENCE_LEVELS * N_BIAS_STATES
    FEATURE_STIMULUS_OFFSET = STIMULUS_OFFSET_N_BASES * N_COHERENCE_LEVELS * N_BIAS_STATES
    FEATURES_SACCADE = SACCADE_N_BASES * N_CHOICE_OPTIONS * N_BIAS_STATES
    FEATURES_HISTORY = HISTORY_N_UNIFORM + HISTORY_N_NONLINEAR
    FEATURES_INTERCEPT = 1

    @classmethod
    def get_total_features(cls):
        """Calculate total number of features."""
        return (cls.FEATURES_TARGET + cls.FEATURES_STIMULUS + cls.FEATURE_STIMULUS_OFFSET + cls.FEATURES_SACCADE +
                cls.FEATURES_HISTORY + cls.FEATURES_INTERCEPT)
