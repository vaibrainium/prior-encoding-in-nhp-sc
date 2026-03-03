
import json
from dataclasses import dataclass, asdict

@dataclass
class PoissonGLMConfig:
    """Instance-based configuration for NeuroGLM model parameters."""

    # TRIAL TIMING PARAMETERS
    BIN_SIZE_MS: float = 1.0
    PRE_TARGET_MS: int = 50

    # Basis types
    BASIS_RAISED_COSINE: str = "raised_cosine"
    BASIS_BOXCAR: str = "boxcar"

    # Effects
    EFFECT_CAUSAL: str = "causal"
    EFFECT_ANTI_CAUSAL: str = "anti-causal"

    # Target
    TARGET_DURATION_MS: int = 200
    TARGET_EFFECT: str = "causal"
    TARGET_BASIS: str = "raised_cosine"

    # Stimulus
    STIMULUS_DURATION_MS: int = 500
    N_COHERENCE_LEVELS: int = 7
    STIMULUS_EFFECT: str = "causal"
    STIMULUS_BASIS: str = "raised_cosine"

    # Saccade
    SACCADE_DURATION_MS: int = 1500
    N_CHOICE_OPTIONS: int = 2
    SACCADE_EFFECT: str = "anti-causal"
    SACCADE_BASIS: str = "raised_cosine"

    # Bias state
    BIAS_DURATION_MS: int = 2000
    N_BIAS_STATES: int = 2
    BIAS_EFFECT: str = "anti-causal"
    BIAS_BASIS: str = "raised_cosine"

    # Spike history
    HISTORY_UNIFORM_MS: int = 11
    HISTORY_NONLINEAR_MS: int = 265
    HISTORY_N_UNIFORM: int = 10
    HISTORY_N_NONLINEAR: int = 10

    FEATURES_INTERCEPT: int = 1

    # ================================
    # COMPUTED PROPERTIES
    # ================================

    @property
    def TARGET_N_BASES(self):
        return int(self.TARGET_DURATION_MS/50 + 1)

    @property
    def STIMULUS_N_BASES(self):
        return int(self.STIMULUS_DURATION_MS/50 + 1)

    @property
    def SACCADE_N_BASES(self):
        return int(self.SACCADE_DURATION_MS/50 + 1)

    @property
    def BIAS_N_BASES(self):
        return int(self.BIAS_DURATION_MS/50 + 1)

    @property
    def FEATURES_TARGET(self):
        return self.TARGET_N_BASES

    @property
    def FEATURES_STIMULUS(self):
        return self.STIMULUS_N_BASES * self.N_COHERENCE_LEVELS

    @property
    def FEATURES_SACCADE(self):
        return self.SACCADE_N_BASES * self.N_CHOICE_OPTIONS

    @property
    def FEATURES_BIAS(self):
        return self.BIAS_N_BASES * self.N_BIAS_STATES

    @property
    def FEATURES_HISTORY(self):
        return self.HISTORY_N_UNIFORM + self.HISTORY_N_NONLINEAR

    def get_total_features(self):
        return (self.FEATURES_TARGET + self.FEATURES_STIMULUS +
                self.FEATURES_SACCADE + self.FEATURES_BIAS +
                self.FEATURES_HISTORY + self.FEATURES_INTERCEPT)

    # ================================
    # SAVE/LOAD SUPPORT
    # ================================
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
@dataclass
class StateBasedPoissonGLMConfig:
    BIN_SIZE_MS: float = 1.0
    PRE_TARGET_MS: int = 50

    BASIS_RAISED_COSINE: str = "raised_cosine"
    BASIS_BOXCAR: str = "boxcar"

    EFFECT_CAUSAL: str = "causal"
    EFFECT_ANTI_CAUSAL: str = "anti-causal"

    TARGET_DURATION_MS: int = 200
    TARGET_SPACING_MS: int = 10
    TARGET_EFFECT: str = "causal"
    TARGET_BASIS: str = "raised_cosine"

    STIMULUS_DURATION_MS: int = 300
    STIMULUS_SPACING_MS: int = 10
    N_COHERENCE_LEVELS: int = 1
    STIMULUS_EFFECT: str = "causal"
    STIMULUS_BASIS: str = "raised_cosine"

    SACCADE_DURATION_MS: int = 50
    SACCADE_SPACING_MS: int = 10
    N_CHOICE_OPTIONS: int = 2
    SACCADE_EFFECT: str = "anti-causal"
    SACCADE_BASIS: str = "raised_cosine"

    N_BIAS_STATES: int = 1

    HISTORY_UNIFORM_MS: int = 11
    HISTORY_NONLINEAR_MS: int = 265
    HISTORY_N_UNIFORM: int = 10
    HISTORY_N_NONLINEAR: int = 10

    FEATURES_INTERCEPT: int = 1

    # computed properties
    @property
    def TARGET_N_BASES(self):
        return int(self.TARGET_DURATION_MS / self.TARGET_SPACING_MS + 1)

    @property
    def STIMULUS_N_BASES(self):
        return int(self.STIMULUS_DURATION_MS / self.STIMULUS_SPACING_MS + 1)

    @property
    def SACCADE_N_BASES(self):
        return int(self.SACCADE_DURATION_MS / self.SACCADE_SPACING_MS + 1)

    # feature counts
    @property
    def FEATURES_TARGET(self):
        return self.TARGET_N_BASES * self.N_BIAS_STATES

    @property
    def FEATURES_STIMULUS(self):
        return self.STIMULUS_N_BASES * self.N_COHERENCE_LEVELS * self.N_BIAS_STATES

    @property
    def FEATURES_SACCADE(self):
        return self.SACCADE_N_BASES * self.N_CHOICE_OPTIONS * self.N_BIAS_STATES

    @property
    def FEATURES_HISTORY(self):
        return self.HISTORY_N_UNIFORM + self.HISTORY_N_NONLINEAR

    def get_total_features(self):
        return (
            self.FEATURES_TARGET + 
            self.FEATURES_STIMULUS + 
            self.FEATURES_SACCADE + 
            self.FEATURES_HISTORY + 
            self.FEATURES_INTERCEPT
        )

    # saving/loading
    def to_dict(self): return asdict(self)
    @classmethod
    def from_dict(cls, d): return cls(**d)
    def save(self, path): json.dump(self.to_dict(), open(path, "w"), indent=4)
    @classmethod
    def load(cls, path): return cls.from_dict(json.load(open(path)))