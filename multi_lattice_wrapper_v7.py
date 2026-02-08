"""
MULTI-LATTICE WRAPPER V7 - UNIFIED PHYSICS ENGINE
==================================================

Python wrapper for lattice_v7.dll with runtime profile switching.

Profiles:
    - FAST: Bulk ingestion, signature ops (minimal physics)
    - BALANCED: Interactive demo, good recall (default)
    - QUALITY: Best correlation, full V4 physics

Usage:
    from multi_lattice_wrapper_v7 import MultiLatticeCUDAv7

    ml = MultiLatticeCUDAv7()
    ml.set_profile("fast")        # For bulk ingestion
    ml.imprint_vector(embedding)
    ml.settle(frames=2, learn=True)

    ml.set_profile("quality")     # For recall
    ml.imprint_vector(query)
    ml.settle(frames=30, learn=False)
    result = ml.recall()

Training Distinct Patterns (for highly-similar inputs):
    # When patterns share 95%+ content (boilerplate), use reset between each:
    ml.train_distinct_patterns(embeddings, epochs=10, settle_frames=5)

    # This prevents cross-contamination where residual activation from
    # pattern A bleeds into pattern B's learning, causing merged basins.
"""

import ctypes
import numpy as np
import os
import sys
import time


# ============================================================
# PHYSICS CONFIG STRUCT (matches lattice_v7.h)
# ============================================================


class PhysicsConfig(ctypes.Structure):
    _fields_ = [
        # Core dynamics
        ("energy_persist", ctypes.c_float),
        ("fatigue_rate", ctypes.c_float),
        ("fatigue_recovery", ctypes.c_float),
        ("temperature", ctypes.c_float),
        # Mexican hat
        ("facilitation", ctypes.c_float),
        ("inhibition", ctypes.c_float),
        ("anti_facilitation_mult", ctypes.c_float),
        # Learning
        ("hebbian_rate", ctypes.c_float),
        # Hierarchy gains - bottom-up
        ("alpha_l4", ctypes.c_float),
        ("alpha_l3", ctypes.c_float),
        ("alpha_l2", ctypes.c_float),
        ("alpha_l1", ctypes.c_float),
        # Hierarchy gains - top-down
        ("beta_l4", ctypes.c_float),
        ("beta_l3", ctypes.c_float),
        ("beta_l2", ctypes.c_float),
        ("beta_l1", ctypes.c_float),
        # Layer persistence
        ("l3_persist", ctypes.c_float),
        ("l2_persist", ctypes.c_float),
        ("l1_persist", ctypes.c_float),
        # V7 Profile controls
        ("kwta_threshold", ctypes.c_float),
        ("hierarchy_depth", ctypes.c_int),
        ("temp_annealing", ctypes.c_int),
        ("hebbian_in_settle", ctypes.c_int),
        ("hybrid_topdown", ctypes.c_int),
        # Reserved
        ("reserved", ctypes.c_int * 2),
    ]

    @classmethod
    def default(cls):
        """Return BALANCED profile defaults."""
        return cls(
            energy_persist=0.96,
            fatigue_rate=0.2,
            fatigue_recovery=0.94,
            temperature=14.0,
            facilitation=0.27,
            inhibition=0.055,
            anti_facilitation_mult=0.5,
            hebbian_rate=0.25,
            alpha_l4=0.86,
            alpha_l3=0.5,
            alpha_l2=0.5,
            alpha_l1=0.5,
            beta_l4=1.0,
            beta_l3=0.7,
            beta_l2=0.6,
            beta_l1=0.5,
            l3_persist=0.1,
            l2_persist=0.2,
            l1_persist=0.2,
            kwta_threshold=0.35,
            hierarchy_depth=2,
            temp_annealing=1,
            hebbian_in_settle=1,
            hybrid_topdown=1,
        )


# ============================================================
# PHYSICS PROFILE ENUM
# ============================================================


class PhysicsProfile:
    FAST = 0
    BALANCED = 1
    QUALITY = 2


# ============================================================
# MAIN WRAPPER CLASS
# ============================================================


class MultiLatticeCUDAv7:
    """
    V7 Unified Lattice Engine with runtime profile switching.

    Key improvements over V6:
    - Restored V4 physics: anti-facilitation, hybrid top-down, kWTA, full hierarchy
    - Runtime profile switching: fast/balanced/quality
    - All V6 features: GPU imprint, hippocampus, signatures, brain dump
    """

    def __init__(
        self,
        lattice_size: int = 4096,
        max_layers: int = 4,
        verbose: int = 1,
        cooldown_sec: float = 0.0,
        dll_path: str = None,
    ):
        """
        Initialize V7/V8 engine.

        Args:
            lattice_size: Width/height of L4 lattice (default 4096)
            max_layers: Reserved for future use (default 4)
            verbose: Print status messages (0=quiet, 1=normal)
            cooldown_sec: Sleep after heavy GPU ops (default 0 = disabled)
                          Recommended: 0.125 for bulk ingestion to prevent thermal issues
            dll_path: Path to DLL (default: lattice_v7.dll next to this file).
                      Pass int8/lattice_v8_int8.dll to use V8 engine.
        """
        self.size = lattice_size
        self.verbose = verbose
        self.cooldown_sec = cooldown_sec

        # Load DLL / shared library
        if dll_path is None:
            base_dir = os.path.dirname(__file__)
            if sys.platform == "win32":
                dll_path = os.path.join(base_dir, "lattice_v7.dll")
            else:
                dll_path = os.path.join(base_dir, "lattice_cuda_v7.so")

        if not os.path.exists(dll_path):
            raise FileNotFoundError(
                f"Cannot find {dll_path}. "
                f"Compile the appropriate engine (.dll on Windows, .so on Linux)."
            )

        self.lib = ctypes.CDLL(dll_path)
        self._setup_function_signatures()

        # Create engine
        if self.verbose:
            print(f"Initializing V7 Unified Engine ({lattice_size}x{lattice_size})...")

        self.engine = self.lib.CreateEngine(lattice_size, max_layers)

        if not self.engine:
            raise RuntimeError("Failed to create V7 engine")

        self.current_profile = "balanced"

        # Contrastive imprinting: subtract global mean to suppress common features
        self.contrastive_mode = False
        self.contrastive_alpha = 0.3  # How much of mean to subtract
        self.global_mean_pattern = None
        self.pattern_count = 0

    def _setup_function_signatures(self):
        """Define ctypes argument/return types for all API functions."""

        # Lifecycle
        self.lib.CreateEngine.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.CreateEngine.restype = ctypes.c_void_p

        self.lib.DestroyEngine.argtypes = [ctypes.c_void_p]
        self.lib.DestroyEngine.restype = None

        # Configuration
        self.lib.SetPhysics.argtypes = [ctypes.c_void_p, PhysicsConfig]
        self.lib.SetPhysics.restype = None

        self.lib.SetProfile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.SetProfile.restype = None

        self.lib.GetProfileDefaults.argtypes = [ctypes.c_int]
        self.lib.GetProfileDefaults.restype = PhysicsConfig

        # Operations
        self.lib.ImprintPattern.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.ImprintPattern.restype = None

        self.lib.ImprintNomic.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
        ]
        self.lib.ImprintNomic.restype = None

        self.lib.ResetLattice.argtypes = [ctypes.c_void_p]
        self.lib.ResetLattice.restype = None

        self.lib.LearnImmediate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.LearnImmediate.restype = None

        self.lib.Settle.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
        self.lib.Settle.restype = None

        self.lib.Recall.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.Recall.restype = None

        # V7.1: Hierarchy layer recall (L3=256x256, L2=64x64, L1=16x16)
        self.lib.RecallL3.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.RecallL3.restype = None

        self.lib.RecallL2.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.RecallL2.restype = None

        self.lib.RecallL1.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.RecallL1.restype = None

        self.lib.EncodeHippocampus.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_bool,
        ]
        self.lib.EncodeHippocampus.restype = None

        # Search
        self.lib.GenerateSignature.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ]
        self.lib.GenerateSignature.restype = None

        self.lib.ScanSignatures.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
            ctypes.c_int,
        ]
        self.lib.ScanSignatures.restype = None

        # Persistence
        self.lib.SaveCore.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        ]
        self.lib.SaveCore.restype = None

        self.lib.LoadCore.argtypes = [
            ctypes.c_void_p,
            np.ctypeslib.ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS"),
        ]
        self.lib.LoadCore.restype = None

        # Diagnostic
        self.lib.GetActiveCount.argtypes = [ctypes.c_void_p]
        self.lib.GetActiveCount.restype = ctypes.c_int

        # V7.2: Protected rows (for hippocampus metadata)
        self.lib.SetProtectedRows.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.SetProtectedRows.restype = None

        self.lib.GetProtectedRows.argtypes = [ctypes.c_void_p]
        self.lib.GetProtectedRows.restype = ctypes.c_uint64

        # V7.3: Per-row physics control
        self.lib.SetRowPhysics.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint8]
        self.lib.SetRowPhysics.restype = None

        self.lib.GetRowPhysics.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.GetRowPhysics.restype = ctypes.c_uint8

        self.lib.SetAllRowPhysics.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.SetAllRowPhysics.restype = None

        self.lib.GetAllRowPhysics.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.GetAllRowPhysics.restype = None

        # V8/V9 extensions (graceful - only available when using V8 DLL)
        self._setup_v8_functions()

    def _setup_v8_functions(self):
        """Bind V8/V9-specific functions if available. Sets has_v8 / has_bitpacked."""
        self.has_v8 = False
        self.has_bitpacked = False

        try:
            # Version / capabilities
            self.lib.GetVersionString.argtypes = [ctypes.c_void_p]
            self.lib.GetVersionString.restype = ctypes.c_char_p

            self.lib.GetCapabilities.argtypes = [ctypes.c_void_p]
            self.lib.GetCapabilities.restype = ctypes.c_int

            # U8 pattern imprint
            self.lib.ImprintPatternU8.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
            self.lib.ImprintPatternU8.restype = None

            # L4-only settle
            self.lib.SettleL4Only.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.SettleL4Only.restype = None

            # Batch operations
            self.lib.AllocateBatch.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.AllocateBatch.restype = None

            self.lib.FreeBatch.argtypes = [ctypes.c_void_p]
            self.lib.FreeBatch.restype = None

            self.lib.BatchImprintU8.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
            self.lib.BatchImprintU8.restype = None

            self.lib.BatchSettle.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self.lib.BatchSettle.restype = None

            self.lib.BatchSettleL4Only.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            self.lib.BatchSettleL4Only.restype = None

            self.lib.BatchRecallL2.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            self.lib.BatchRecallL2.restype = None

            # Pipeline (u8 patterns)
            self.lib.PipelineProcessU8.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
            self.lib.PipelineProcessU8.restype = None

            # Pipeline (Nomic embeddings)
            self.lib.PipelineIngestNomic.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)]
            self.lib.PipelineIngestNomic.restype = None

            # Compact brain save/load
            self.lib.SaveBrainCompact.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
            self.lib.SaveBrainCompact.restype = ctypes.c_int

            self.lib.LoadBrainCompact.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int]
            self.lib.LoadBrainCompact.restype = None

            self.lib.GetCompactBrainSize.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.GetCompactBrainSize.restype = ctypes.c_int

            self.has_v8 = True
        except (AttributeError, OSError):
            pass

        if not self.has_v8:
            return

        try:
            # V9: Bitpacked binary physics
            self.lib.PrepareBitpacked.argtypes = [ctypes.c_void_p]
            self.lib.PrepareBitpacked.restype = None

            self.lib.FreeBitpacked.argtypes = [ctypes.c_void_p]
            self.lib.FreeBitpacked.restype = None

            self.lib.SetBitpackedParams.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int]
            self.lib.SetBitpackedParams.restype = None

            self.lib.SettleBitpacked.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.lib.SettleBitpacked.restype = None

            self.lib.PipelineSearchBitpacked.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float)]
            self.lib.PipelineSearchBitpacked.restype = None

            self.has_bitpacked = True
        except (AttributeError, OSError):
            pass

    def __del__(self):
        """Clean up GPU resources."""
        if hasattr(self, "engine") and self.engine:
            self.lib.DestroyEngine(self.engine)

    # ============================================================
    # CONFIGURATION
    # ============================================================

    def set_physics(self, config: PhysicsConfig):
        """Set physics parameters directly."""
        self.lib.SetPhysics(self.engine, config)

    def set_profile(self, profile: str):
        """
        Switch physics profile at runtime.

        Args:
            profile: "fast", "balanced", or "quality"

        Profiles:
            - fast: Minimal physics for bulk ops (500+ patterns/sec)
            - balanced: Good recall quality (default)
            - quality: Full V4 physics, best correlation
        """
        profiles = {
            "fast": PhysicsProfile.FAST,
            "balanced": PhysicsProfile.BALANCED,
            "quality": PhysicsProfile.QUALITY,
        }

        profile_lower = profile.lower()
        if profile_lower not in profiles:
            raise ValueError(
                f"Unknown profile: {profile}. Use: fast, balanced, quality"
            )

        self.lib.SetProfile(self.engine, profiles[profile_lower])
        self.current_profile = profile_lower

    def get_profile_defaults(self, profile: str) -> PhysicsConfig:
        """
        Get the default PhysicsConfig for a profile.

        Useful for inspection or as a starting point for customization.
        """
        profiles = {
            "fast": PhysicsProfile.FAST,
            "balanced": PhysicsProfile.BALANCED,
            "quality": PhysicsProfile.QUALITY,
        }
        return self.lib.GetProfileDefaults(profiles[profile.lower()])

    def set_contrastive(self, enabled: bool, alpha: float = 0.3):
        """
        Enable/disable contrastive imprinting.

        When enabled, subtracts a running mean of all patterns before imprinting.
        This suppresses common features (boilerplate) and emphasizes unique content.

        Args:
            enabled: True to enable contrastive mode
            alpha: How much of the mean to subtract (0.0-1.0, default 0.3)
        """
        self.contrastive_mode = enabled
        self.contrastive_alpha = alpha
        if enabled and self.verbose:
            print(f"Contrastive imprinting ENABLED (alpha={alpha})")
        elif not enabled and self.verbose:
            print("Contrastive imprinting DISABLED")

    def reset_contrastive_mean(self):
        """Reset the running mean for contrastive imprinting."""
        self.global_mean_pattern = None
        self.pattern_count = 0

    # ============================================================
    # CORE OPERATIONS
    # ============================================================

    def imprint_pattern(self, pattern: np.ndarray):
        """
        Imprint a raw float pattern (16M values for 4096x4096).

        Args:
            pattern: 2D or 1D array, values > 0.5 become ON
        """
        pat_c = np.ascontiguousarray(pattern.flatten(), dtype=np.float32)
        self.lib.ImprintPattern(self.engine, pat_c)

    def imprint_pattern_contrastive(self, pattern: np.ndarray):
        """
        Imprint with contrastive subtraction to suppress common features.

        Maintains a running mean of all imprinted patterns and subtracts
        a fraction of it before imprinting. This emphasizes unique content
        and suppresses boilerplate that appears across many patterns.

        Must call set_contrastive(True) first, or this behaves like imprint_pattern.

        Args:
            pattern: 2D or 1D array, values > 0.5 become ON
        """
        pat_flat = pattern.flatten().astype(np.float32)

        if self.contrastive_mode:
            # Update running mean
            if self.global_mean_pattern is None:
                self.global_mean_pattern = pat_flat.copy()
                self.pattern_count = 1
            else:
                # Exponential moving average
                decay = 0.95
                self.global_mean_pattern = (
                    decay * self.global_mean_pattern + (1 - decay) * pat_flat
                )
                self.pattern_count += 1

            # Subtract mean to emphasize unique features
            pat_contrast = pat_flat - self.contrastive_alpha * self.global_mean_pattern
            pat_contrast = np.clip(pat_contrast, 0.0, 1.0)
            pat_c = np.ascontiguousarray(pat_contrast, dtype=np.float32)
        else:
            pat_c = np.ascontiguousarray(pat_flat, dtype=np.float32)

        self.lib.ImprintPattern(self.engine, pat_c)

    def imprint_vector(self, embedding: np.ndarray, normalize: str = "auto"):
        """
        Imprint a Nomic embedding directly (768 or similar dims).

        Uses GPU-native thermometer expansion. The thermometer encoder
        expects values in [0, 1] range. This method auto-detects and
        normalizes embeddings that fall outside this range.

        Args:
            embedding: 1D array of embedding values (typically 768-dim)
            normalize: "auto" (default) - detect and normalize if needed
                       "always" - force normalization
                       "never" - pass through raw (use if pre-normalized)
        """
        emb = embedding.flatten().astype(np.float32)
        val_min, val_max = emb.min(), emb.max()

        needs_norm = False
        if normalize == "always":
            needs_norm = True
        elif normalize == "auto":
            # Heuristic: if values significantly outside [0,1], normalize
            # Nomic embeddings typically range [-4, +4]
            if val_min < -0.1 or val_max > 1.1:
                needs_norm = True

        if needs_norm:
            # Linear rescale to [0, 1]
            val_range = val_max - val_min
            if val_range > 1e-6:
                emb = (emb - val_min) / val_range
            else:
                emb = np.full_like(emb, 0.5)  # Degenerate case: constant embedding

            if self.verbose > 1:
                print(
                    f"  Normalized embedding [{val_min:.2f}, {val_max:.2f}] -> [0, 1]"
                )

        emb_c = np.ascontiguousarray(emb, dtype=np.float32)
        self.lib.ImprintNomic(self.engine, emb_c, len(emb_c))

    def reset(self):
        """
        Clear all neuron states to zero (fresh slate for new pattern).

        Call this before imprint_vector when you need independent patterns
        (e.g., signature building, query processing).
        """
        self.lib.ResetLattice(self.engine)

    def settle(self, frames: int = 30, learn: bool = True):
        """
        Run physics simulation for N frames.

        Args:
            frames: Number of physics frames to run
            learn: If True and hebbian_in_settle enabled, update weights
        """
        self.lib.Settle(self.engine, frames, learn)

        # Optional cooldown to prevent GPU thermal issues during bulk ops
        if self.cooldown_sec > 0:
            time.sleep(self.cooldown_sec)

    def learn_immediate(self, iterations: int = 5):
        """
        Apply Hebbian learning directly without running physics.

        This is V4's approach: learn the clean imprinted pattern before
        any degradation from physics. Call after imprint, before settle.

        Args:
            iterations: Number of Hebbian update passes (default: 5)
        """
        self.lib.LearnImmediate(self.engine, iterations)

    def train_distinct_patterns(
        self,
        patterns: list,
        epochs: int = 10,
        settle_frames: int = 5,
        shuffle: bool = True,
        is_embedding: bool = True,
        progress_callback=None,
    ):
        """
        Train multiple patterns with reset between each to prevent cross-contamination.

        This is essential for highly-similar patterns (e.g., 95% boilerplate with
        small differences). Resetting between patterns forces each to carve its
        own distinct attractor basin rather than merging into one muddy basin.

        Args:
            patterns: List of patterns (embeddings or raw lattice arrays)
            epochs: Number of training passes through all patterns (default: 10)
            settle_frames: Frames per pattern per epoch (default: 5)
            shuffle: Randomize order each epoch (default: True)
            is_embedding: If True, use imprint_vector; if False, use imprint_pattern
            progress_callback: Optional fn(epoch, total_epochs) called after each epoch

        Returns:
            dict with training stats
        """
        indices = list(range(len(patterns)))

        for epoch in range(epochs):
            if shuffle:
                np.random.shuffle(indices)

            for idx in indices:
                self.reset()  # Clean slate - prevents cross-contamination
                if is_embedding:
                    self.imprint_vector(patterns[idx])
                else:
                    self.imprint_pattern(patterns[idx])
                self.settle(frames=settle_frames, learn=True)

            if progress_callback:
                progress_callback(epoch + 1, epochs)

        # Final consolidation
        self.settle(frames=settle_frames * 3, learn=True)

        return {
            "patterns": len(patterns),
            "epochs": epochs,
            "total_imprints": len(patterns) * epochs,
        }

    def recall(self) -> np.ndarray:
        """
        Extract current L4 state as float array.

        Returns:
            2D array (4096x4096) with 1.0 = ON, 0.0 = OFF
        """
        out = np.zeros(self.size * self.size, dtype=np.float32)
        self.lib.Recall(self.engine, out)
        return out.reshape(self.size, self.size)

    def recall_l3(self) -> np.ndarray:
        """
        Extract current L3 state (256x256 abstraction layer).

        L3 captures mid-level abstractions from L4 via 16x16 pooling.

        Returns:
            2D array (256x256) of float activations
        """
        out = np.zeros(256 * 256, dtype=np.float32)
        self.lib.RecallL3(self.engine, out)
        return out.reshape(256, 256)

    def recall_l2(self) -> np.ndarray:
        """
        Extract current L2 state (64x64 abstraction layer).

        L2 captures high-level abstractions from L3 via 4x4 pooling.
        Same dimensions as generate_signature() - can be used as drop-in replacement.

        Returns:
            2D array (64x64) of float activations, or flattened 4096-float vector
        """
        out = np.zeros(64 * 64, dtype=np.float32)
        self.lib.RecallL2(self.engine, out)
        return out.reshape(64, 64)

    def recall_l1(self) -> np.ndarray:
        """
        Extract current L1 state (16x16 ultra-compact abstraction).

        L1 is the highest abstraction level, capturing global patterns.
        256 floats total - useful for fast approximate matching or hashing.

        Returns:
            2D array (16x16) of float activations
        """
        out = np.zeros(16 * 16, dtype=np.float32)
        self.lib.RecallL1(self.engine, out)
        return out.reshape(16, 16)

    def encode_hippocampus(self, pattern_id: int, is_deleted: bool = False):
        """
        Encode pattern ID into hippocampus region (row 63).

        Args:
            pattern_id: Integer ID to encode (up to 60 bits)
            is_deleted: If True, sets tombstone flag
        """
        self.lib.EncodeHippocampus(self.engine, pattern_id, is_deleted)

    # ============================================================
    # SEARCH OPERATIONS
    # ============================================================

    def generate_signature(self) -> np.ndarray:
        """
        Generate 4096-float signature from current L4 state.

        Each value is mean activation of a 64x64 region.

        Returns:
            1D array of 4096 floats
        """
        sig = np.zeros(4096, dtype=np.float32)
        self.lib.GenerateSignature(self.engine, sig)
        return sig

    def scan_signatures(
        self, query_sig: np.ndarray, signature_db: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and database signatures.

        Args:
            query_sig: 4096-float query signature
            signature_db: 2D array (N x 4096) of database signatures

        Returns:
            1D array of N similarity scores
        """
        num_items = signature_db.shape[0]
        query = np.ascontiguousarray(query_sig, dtype=np.float32)
        db = np.ascontiguousarray(signature_db.flatten(), dtype=np.float32)
        indices = np.zeros(num_items, dtype=np.int32)
        scores = np.zeros(num_items, dtype=np.float32)

        self.lib.ScanSignatures(self.engine, query, db, num_items, indices, scores, 0)
        return scores

    # ============================================================
    # PERSISTENCE
    # ============================================================

    def save_brain(self, filename: str):
        """
        Save L4 state + weights to file (legacy 128 MB format).

        For 4096x4096: 2 * 16M * 4 bytes = 134 MB

        Args:
            filename: Output .npy file path
        """
        total_ints = self.size * self.size * 2
        buffer = np.zeros(total_ints, dtype=np.uint32)
        self.lib.SaveCore(self.engine, buffer)
        np.save(filename, buffer)

        if self.verbose:
            print(f"Brain saved to {filename}")

    def load_brain(self, filename: str) -> bool:
        """
        Load L4 state + weights from file (supports legacy and compact formats).

        Args:
            filename: Input .npy file path

        Returns:
            True if successful, False if file not found
        """
        if not os.path.exists(filename):
            return False

        buffer = np.load(filename)

        N = self.size * self.size

        # Detect format by size
        if buffer.dtype == np.uint32 and len(buffer) == N * 2:
            # Legacy format: state + weights (128 MB)
            buffer = np.ascontiguousarray(buffer, dtype=np.uint32)
            self.lib.LoadCore(self.engine, buffer)
            if self.verbose:
                print(f"Brain restored from {filename} (legacy 128 MB)")

        elif buffer.dtype == np.uint32 and len(buffer) == N:
            # Compact format: weights only (64 MB)
            full = np.zeros(N * 2, dtype=np.uint32)
            full[N:] = buffer  # Weights go in second half
            full = np.ascontiguousarray(full)
            self.lib.LoadCore(self.engine, full)
            if self.verbose:
                print(f"Brain restored from {filename} (compact 64 MB, weights-only)")

        elif buffer.dtype == np.uint16:
            # Compact uint4-quantized format (32 MB)
            weights_u8 = self._dequantize_weights_u4(buffer)
            full = np.zeros(N * 2, dtype=np.uint32)
            full[N:] = weights_u8
            full = np.ascontiguousarray(full)
            self.lib.LoadCore(self.engine, full)
            if self.verbose:
                print(f"Brain restored from {filename} (quantized 32 MB)")

        else:
            # Unknown format, try legacy
            buffer = np.ascontiguousarray(buffer, dtype=np.uint32)
            self.lib.LoadCore(self.engine, buffer)
            if self.verbose:
                print(f"Brain restored from {filename} (unknown format, tried legacy)")

        return True

    def save_brain_compact(self, filename: str, quantize: str = "none"):
        """
        Save only Hebbian weights (no transient state). Much smaller files.

        Formats:
            "none"  - uint8 weights, 64 MB (lossless)
            "uint4" - 4-bit quantized, 32 MB (~1% fidelity loss)

        Args:
            filename: Output .npy file path (will add .npy if not present)
            quantize: "none" or "uint4"
        """
        N = self.size * self.size
        full_buffer = np.zeros(N * 2, dtype=np.uint32)
        self.lib.SaveCore(self.engine, full_buffer)

        # Extract weights (second half)
        weights = full_buffer[N:].copy()

        if quantize == "uint4":
            packed = self._quantize_weights_u4(weights)
            np.save(filename, packed)
            size_mb = packed.nbytes / (1024 * 1024)
        else:
            np.save(filename, weights)
            size_mb = weights.nbytes / (1024 * 1024)

        if self.verbose:
            print(f"Brain saved to {filename} (compact {quantize}, {size_mb:.1f} MB)")

    def _quantize_weights_u4(self, weights_u32: np.ndarray) -> np.ndarray:
        """
        Quantize uint8 weights to uint4 (packed into uint16).

        Each uint32 holds 4 × uint8 weights (N/E/S/W).
        Quantize each uint8 (0-127) → uint4 (0-15) by >> 3.
        Pack all 4 as uint4 into one uint16 (4 × 4 bits = 16 bits).

        Result: 16M uint32 → 16M uint16 = 32 MB (was 64 MB).
        """
        # Unpack 4 uint8 weights from each uint32
        w0 = (weights_u32 & 0xFF).astype(np.uint8)
        w1 = ((weights_u32 >> 8) & 0xFF).astype(np.uint8)
        w2 = ((weights_u32 >> 16) & 0xFF).astype(np.uint8)
        w3 = ((weights_u32 >> 24) & 0xFF).astype(np.uint8)

        # Quantize to 4-bit (0-15)
        q0 = (w0 >> 3).astype(np.uint16)
        q1 = (w1 >> 3).astype(np.uint16)
        q2 = (w2 >> 3).astype(np.uint16)
        q3 = (w3 >> 3).astype(np.uint16)

        # Pack into uint16: q0 in bits 0-3, q1 in 4-7, q2 in 8-11, q3 in 12-15
        packed = q0 | (q1 << 4) | (q2 << 8) | (q3 << 12)
        return packed

    def _dequantize_weights_u4(self, packed_u16: np.ndarray) -> np.ndarray:
        """
        Dequantize uint4 weights back to uint32 (4 × uint8 packed).

        Reverses _quantize_weights_u4: uint4 × 8 → uint8, repack into uint32.
        """
        # Unpack 4-bit values
        q0 = (packed_u16 & 0xF).astype(np.uint32)
        q1 = ((packed_u16 >> 4) & 0xF).astype(np.uint32)
        q2 = ((packed_u16 >> 8) & 0xF).astype(np.uint32)
        q3 = ((packed_u16 >> 12) & 0xF).astype(np.uint32)

        # Dequantize back to uint8 range (× 8)
        w0 = q0 * 8
        w1 = q1 * 8
        w2 = q2 * 8
        w3 = q3 * 8

        # Repack into uint32
        return w0 | (w1 << 8) | (w2 << 16) | (w3 << 24)

    # ============================================================
    # V7.2: PROTECTED ROWS (for hippocampus metadata)
    # ============================================================

    def set_protected_rows(self, rows: list):
        """
        Set region rows that skip lateral inhibition and kWTA enforcement.

        Protected rows preserve sparse metadata patterns that would otherwise
        be cleaned up by the physics (lateral inhibition + kWTA). This is
        essential for storing pattern IDs or other metadata in hippocampus
        regions.

        Args:
            rows: List of region row indices (0-63) to protect.
                  Row 63 = bottom row = hippocampus region.
                  Example: [63] protects only the hippocampus row.
                  Example: [62, 63] protects last two rows.
                  Example: [] clears all protection (normal physics everywhere).

        Example:
            ml.set_protected_rows([63])  # Protect hippocampus row
            ml.encode_hippocampus(pattern_id)  # Now metadata survives settle
            ml.settle(frames=20, learn=True)  # Row 63 preserved
        """
        mask = 0
        for row in rows:
            if 0 <= row <= 63:
                mask |= (1 << row)
            else:
                raise ValueError(f"Row index must be 0-63, got {row}")
        self.lib.SetProtectedRows(self.engine, mask)

    def get_protected_rows(self) -> list:
        """
        Get list of currently protected region rows.

        Returns:
            List of row indices (0-63) that are currently protected.
        """
        mask = self.lib.GetProtectedRows(self.engine)
        rows = []
        for i in range(64):
            if mask & (1 << i):
                rows.append(i)
        return rows

    # ============================================================
    # V7.3: PER-ROW PHYSICS CONTROL
    # ============================================================

    # Flag constants matching CUDA defines
    ROW_SKIP_DECAY      = 0x01
    ROW_SKIP_FATIGUE    = 0x02
    ROW_SKIP_INHIBITION = 0x04
    ROW_SKIP_WEIGHTS    = 0x08
    ROW_SKIP_BOLTZMANN  = 0x10
    ROW_SKIP_KWTA       = 0x20
    ROW_SKIP_LEARNING   = 0x40
    ROW_SKIP_TOPDOWN    = 0x80

    ROW_FULLY_PROTECTED = 0xFF  # All physics skipped
    ROW_LEARN_ONLY      = 0xBF  # All skipped EXCEPT Hebbian learning

    def set_row_physics(self, row: int, flags: int):
        """
        Set physics flags for a single region row.

        Args:
            row: Region row index (0-63)
            flags: Bitmask of ROW_SKIP_* flags.
                   0x00 = full physics (normal)
                   0xFF = fully protected (V7.2 equivalent)
                   0xBF = learn-only (Hebbian learning active, everything else off)

        Example:
            # ROM hippocampus: fully frozen
            ml.set_row_physics(63, ml.ROW_FULLY_PROTECTED)

            # RAM hippocampus: learns associations but no competitive dynamics
            ml.set_row_physics(62, ml.ROW_LEARN_ONLY)

            # Custom: skip only kWTA and inhibition
            ml.set_row_physics(60, ml.ROW_SKIP_KWTA | ml.ROW_SKIP_INHIBITION)
        """
        if not 0 <= row <= 63:
            raise ValueError(f"Row index must be 0-63, got {row}")
        self.lib.SetRowPhysics(self.engine, row, flags)

    def get_row_physics(self, row: int) -> int:
        """
        Get physics flags for a single region row.

        Returns:
            Bitmask of ROW_SKIP_* flags for that row.
        """
        if not 0 <= row <= 63:
            raise ValueError(f"Row index must be 0-63, got {row}")
        return self.lib.GetRowPhysics(self.engine, row)

    def set_all_row_physics(self, flags_list: list):
        """
        Set physics flags for all 64 rows at once.

        Args:
            flags_list: List of 64 integers (0-255), one per row.
        """
        if len(flags_list) != 64:
            raise ValueError(f"Expected 64 flags, got {len(flags_list)}")
        arr = (ctypes.c_uint8 * 64)(*flags_list)
        self.lib.SetAllRowPhysics(self.engine, arr)

    def get_all_row_physics(self) -> list:
        """
        Get physics flags for all 64 rows.

        Returns:
            List of 64 integers (0-255), one per row.
        """
        arr = (ctypes.c_uint8 * 64)()
        self.lib.GetAllRowPhysics(self.engine, arr)
        return list(arr)

    def describe_row_physics(self, row: int) -> dict:
        """
        Get a human-readable description of physics flags for a row.

        Returns:
            Dict mapping component names to True (active) / False (skipped).
        """
        flags = self.get_row_physics(row)
        return {
            'energy_decay':   not bool(flags & self.ROW_SKIP_DECAY),
            'fatigue':        not bool(flags & self.ROW_SKIP_FATIGUE),
            'inhibition':     not bool(flags & self.ROW_SKIP_INHIBITION),
            'weight_readout': not bool(flags & self.ROW_SKIP_WEIGHTS),
            'boltzmann':      not bool(flags & self.ROW_SKIP_BOLTZMANN),
            'kwta':           not bool(flags & self.ROW_SKIP_KWTA),
            'learning':       not bool(flags & self.ROW_SKIP_LEARNING),
            'topdown':        not bool(flags & self.ROW_SKIP_TOPDOWN),
        }

    # ============================================================
    # DIAGNOSTICS
    # ============================================================

    def get_active_count(self) -> int:
        """Get current number of active (ON) neurons."""
        return self.lib.GetActiveCount(self.engine)

    def get_on_ratio(self, expected_density: float = 0.048) -> float:
        """
        Get ratio of actual active neurons to expected.

        Args:
            expected_density: Expected sparsity (default 4.8%)

        Returns:
            Ratio (1.0 = exactly as expected, >1.0 = bloom, <1.0 = death)
        """
        active = self.get_active_count()
        expected = self.size * self.size * expected_density
        return active / (expected + 1e-8)

    # ============================================================
    # V8: VERSION / CAPABILITIES
    # ============================================================

    def get_version_string(self) -> str:
        """Get engine version string (V8+ only)."""
        if not self.has_v8:
            return "V7 (no version API)"
        return self.lib.GetVersionString(self.engine).decode()

    def get_capabilities(self) -> dict:
        """Get engine capabilities as dict (V8+ only)."""
        if not self.has_v8:
            return {"dp4a": False, "tensor": False}
        caps = self.lib.GetCapabilities(self.engine)
        return {"dp4a": bool(caps & 1), "tensor": bool(caps & 2)}

    # ============================================================
    # V8: L4-ONLY SETTLE
    # ============================================================

    def settle_l4_only(self, frames: int):
        """
        Run L4-only settle (skip hierarchy during settle, compute once at end).

        23% faster than full settle. 0.9999 fidelity vs full hierarchy.
        Requires V8 engine.
        """
        if not self.has_v8:
            raise RuntimeError("settle_l4_only requires V8 engine")
        self.lib.SettleL4Only(self.engine, frames)

        if self.cooldown_sec > 0:
            time.sleep(self.cooldown_sec)

    # ============================================================
    # V8: PIPELINE OPERATIONS
    # ============================================================

    def pipeline_ingest_nomic(
        self,
        embeddings: np.ndarray,
        settle_frames: int = 2,
        learn: bool = True,
        pattern_ids: np.ndarray = None,
    ) -> np.ndarray:
        """
        Ingest N Nomic embeddings in a single C call.

        For each pattern: Reset -> ImprintNomic -> Hippocampus -> Settle -> L2.
        Eliminates Python round-trip overhead.

        Args:
            embeddings: 2D array (N x dims), raw Nomic embeddings (auto-normalized)
            settle_frames: Physics frames per pattern (default: 2)
            learn: Enable Hebbian learning (default: True)
            pattern_ids: Optional int32 array of N pattern IDs for hippocampus

        Returns:
            2D array (N x 4096) of L2 signatures
        """
        if not self.has_v8:
            raise RuntimeError("pipeline_ingest_nomic requires V8 engine")

        n_patterns = len(embeddings)
        n_dims = embeddings.shape[1]

        # Normalize each embedding to [0, 1]
        normed = np.zeros_like(embeddings, dtype=np.float32)
        for i in range(n_patterns):
            emb = embeddings[i].astype(np.float32)
            v_min, v_max = emb.min(), emb.max()
            if v_max - v_min > 1e-6:
                normed[i] = (emb - v_min) / (v_max - v_min)
            else:
                normed[i] = 0.5

        all_emb = np.ascontiguousarray(normed.flatten(), dtype=np.float32)
        all_emb_ptr = all_emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ids_ptr = None
        if pattern_ids is not None:
            ids_arr = np.ascontiguousarray(pattern_ids, dtype=np.int32)
            ids_ptr = ids_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        l2_buf = np.zeros(n_patterns * 4096, dtype=np.float32)
        l2_ptr = l2_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.PipelineIngestNomic(
            self.engine, all_emb_ptr, n_dims, n_patterns,
            settle_frames, learn, ids_ptr, l2_ptr)

        return l2_buf.reshape(n_patterns, 4096)

    def pipeline_search_nomic(
        self,
        embeddings: np.ndarray,
        settle_frames: int = 2,
    ) -> np.ndarray:
        """
        Search N Nomic embeddings via full-physics pipeline (no learning).

        Args:
            embeddings: 2D array (N x dims), raw Nomic embeddings
            settle_frames: Physics frames per query (default: 2)

        Returns:
            2D array (N x 4096) of L2 signatures
        """
        if not self.has_v8:
            raise RuntimeError("pipeline_search_nomic requires V8 engine")

        n_patterns = len(embeddings)
        n_dims = embeddings.shape[1]

        normed = np.zeros_like(embeddings, dtype=np.float32)
        for i in range(n_patterns):
            emb = embeddings[i].astype(np.float32)
            v_min, v_max = emb.min(), emb.max()
            if v_max - v_min > 1e-6:
                normed[i] = (emb - v_min) / (v_max - v_min)
            else:
                normed[i] = 0.5

        all_emb = np.ascontiguousarray(normed.flatten(), dtype=np.float32)
        all_emb_ptr = all_emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        l2_buf = np.zeros(n_patterns * 4096, dtype=np.float32)
        l2_ptr = l2_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.PipelineIngestNomic(
            self.engine, all_emb_ptr, n_dims, n_patterns,
            settle_frames, False, None, l2_ptr)

        return l2_buf.reshape(n_patterns, 4096)

    # ============================================================
    # V8: BATCH OPERATIONS
    # ============================================================

    def allocate_batch(self, batch_size: int):
        """Allocate GPU memory for batch processing."""
        if not self.has_v8:
            raise RuntimeError("Batch operations require V8 engine")
        self.lib.AllocateBatch(self.engine, batch_size)

    def free_batch(self):
        """Free batch GPU memory."""
        if not self.has_v8:
            return
        self.lib.FreeBatch(self.engine)

    # ============================================================
    # V8: COMPACT BRAIN SAVE/LOAD
    # ============================================================

    def save_brain_gpu_compact(self, filename: str, fmt: int = 0):
        """
        Save brain using GPU-side quantization (V8 only).

        Formats:
            0 = uint8 weights-only (64 MB)
            1 = uint4 quantized (32 MB)
            2 = uint2 aggressive (16 MB)

        Args:
            filename: Output file path
            fmt: Quantization format (0, 1, or 2)
        """
        if not self.has_v8:
            raise RuntimeError("GPU compact save requires V8 engine")

        buf_size = self.lib.GetCompactBrainSize(self.engine, fmt)
        buf = np.zeros(buf_size, dtype=np.uint8)
        buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        bytes_written = self.lib.SaveBrainCompact(self.engine, buf_ptr, fmt)
        np.save(filename, buf[:bytes_written])

        if self.verbose:
            size_mb = bytes_written / (1024 * 1024)
            print(f"Brain saved to {filename} (GPU compact fmt={fmt}, {size_mb:.1f} MB)")

    def load_brain_gpu_compact(self, filename: str, fmt: int = 0) -> bool:
        """
        Load brain from GPU-compact format (V8 only).

        Args:
            filename: Input .npy file path
            fmt: Quantization format that was used to save
        """
        if not self.has_v8:
            raise RuntimeError("GPU compact load requires V8 engine")
        if not os.path.exists(filename):
            return False

        buf = np.load(filename)
        buf = np.ascontiguousarray(buf, dtype=np.uint8)
        buf_ptr = buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        self.lib.LoadBrainCompact(self.engine, buf_ptr, fmt, len(buf))

        if self.verbose:
            print(f"Brain restored from {filename} (GPU compact fmt={fmt})")
        return True

    # ============================================================
    # V9: BITPACKED BINARY PHYSICS
    # ============================================================

    def prepare_bitpacked(self):
        """
        Prepare bit-packed mode: allocate 12 MB buffers, extract 1-bit weights.

        Must be called after learning patterns with full physics.
        After this, use settle_bitpacked() or pipeline_search_bitpacked()
        for 1.86x faster search with 0.915 fidelity.
        """
        if not self.has_bitpacked:
            raise RuntimeError("Bitpacked requires V8 engine with V9 extensions")
        self.lib.PrepareBitpacked(self.engine)

    def free_bitpacked(self):
        """Free bit-packed GPU buffers."""
        if not self.has_bitpacked:
            return
        self.lib.FreeBitpacked(self.engine)

    def set_bitpacked_params(
        self, facil: int = 8, anti_facil: int = 2, inhib: int = 1,
        thresh_on: int = 3, thresh_off: int = -1
    ):
        """
        Set bitpacked physics parameters.

        Args:
            facil: Facilitation weight for dist-1 ON neighbors (default: 8)
            anti_facil: Anti-facilitation penalty for dist-1 OFF neighbors (default: 2)
            inhib: Inhibition weight for dist>=2 ON neighbors (default: 1)
            thresh_on: Score threshold for OFF->ON transition (default: 3)
            thresh_off: Score threshold for ON->OFF transition (default: -1)
        """
        if not self.has_bitpacked:
            raise RuntimeError("Bitpacked requires V8 engine with V9 extensions")
        self.lib.SetBitpackedParams(
            self.engine, facil, anti_facil, inhib, thresh_on, thresh_off)

    def settle_bitpacked(self, frames: int = 2):
        """
        Run binary physics settle (1-bit neurons, popcount Mexican hat).

        1.86x faster than full physics, 0.915 fidelity.
        Must call prepare_bitpacked() first.
        """
        if not self.has_bitpacked:
            raise RuntimeError("Bitpacked requires V8 engine with V9 extensions")
        self.lib.SettleBitpacked(self.engine, frames)

    def pipeline_search_bitpacked(
        self,
        embeddings: np.ndarray,
        settle_frames: int = 2,
    ) -> np.ndarray:
        """
        Search N Nomic embeddings using bitpacked binary physics.

        Single C call: Reset -> ImprintNomic -> bits -> binary settle -> L2.
        1.86x faster than full-physics pipeline.

        Args:
            embeddings: 2D array (N x dims), raw Nomic embeddings
            settle_frames: Binary physics frames per query (default: 2)

        Returns:
            2D array (N x 4096) of L2 signatures
        """
        if not self.has_bitpacked:
            raise RuntimeError("Bitpacked search requires V8 engine with V9 extensions")

        n_patterns = len(embeddings)
        n_dims = embeddings.shape[1]

        normed = np.zeros_like(embeddings, dtype=np.float32)
        for i in range(n_patterns):
            emb = embeddings[i].astype(np.float32)
            v_min, v_max = emb.min(), emb.max()
            if v_max - v_min > 1e-6:
                normed[i] = (emb - v_min) / (v_max - v_min)
            else:
                normed[i] = 0.5

        all_emb = np.ascontiguousarray(normed.flatten(), dtype=np.float32)
        all_emb_ptr = all_emb.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        l2_buf = np.zeros(n_patterns * 4096, dtype=np.float32)
        l2_ptr = l2_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self.lib.PipelineSearchBitpacked(
            self.engine, all_emb_ptr, n_dims, n_patterns,
            settle_frames, l2_ptr)

        return l2_buf.reshape(n_patterns, 4096)


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    import pickle

    print("=" * 60)
    print("V7 WRAPPER QUICK TEST - WITH REAL EMBEDDINGS")
    print("=" * 60)

    # Create engine
    ml = MultiLatticeCUDAv7(lattice_size=4096, verbose=1)

    # Test profile switching
    print("\nTesting profile switching...")
    ml.set_profile("fast")
    ml.set_profile("balanced")
    ml.set_profile("quality")

    # Load REAL embeddings from pkl file
    print("\nLoading real Nomic embeddings from pkl file...")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pkl_path = os.path.join(data_dir, "wiki_nomic_10k.pkl")

    embeddings = None  # Will hold the full array if loaded
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Handle different pkl formats
        if isinstance(data, dict) and "embeddings" in data:
            embeddings = np.array(data["embeddings"])
        elif isinstance(data, dict) and "data" in data:
            embeddings = np.array(data["data"])
        elif isinstance(data, list):
            embeddings = np.array(data)
        else:
            embeddings = np.array(data)

        print(f"Loaded {len(embeddings)} embeddings, shape: {embeddings.shape}")
        test_emb = embeddings[0].astype(np.float32)
        print(
            f"Test embedding dims: {len(test_emb)}, range: [{test_emb.min():.4f}, {test_emb.max():.4f}]"
        )
        if test_emb.min() < -0.1 or test_emb.max() > 1.1:
            print("  -> Will be auto-normalized to [0, 1] for thermometer encoding")
    else:
        print(f"WARNING: Could not find {pkl_path}")
        print("Falling back to synthetic wave pattern...")
        test_emb = np.zeros(768, dtype=np.float32)
        for i in range(768):
            test_emb[i] = 0.5 + 0.4 * np.sin(i * 0.05) + 0.1 * np.cos(i * 0.13)
        test_emb = np.clip(test_emb, 0.0, 1.0)

    # Test with BALANCED profile
    print("\nTesting imprint + settle + recall with BALANCED profile...")
    ml.set_profile("balanced")
    ml.imprint_vector(test_emb)
    ml.settle(frames=10, learn=True)

    result = ml.recall()
    active = (result > 0.5).sum()
    ratio = ml.get_on_ratio()

    print(f"Active neurons: {active:,}")
    print(f"On-ratio: {ratio:.3f}")

    # Check if in acceptable range
    if 0.9 <= ratio <= 1.3:
        print("On-ratio: GOOD (within 0.9-1.3 range)")
    elif ratio < 0.9:
        print("On-ratio: WARNING - possible pattern death")
    else:
        print("On-ratio: WARNING - mild bloom (may need tuning)")

    # Test signature
    print("\nTesting signature generation...")
    sig = ml.generate_signature()
    print(f"Signature shape: {sig.shape}, range: [{sig.min():.4f}, {sig.max():.4f}]")

    # Test with QUALITY profile to compare bloom control
    print("\nTesting same embedding with QUALITY profile...")
    ml.set_profile("quality")
    ml.imprint_vector(test_emb)
    ml.settle(frames=10, learn=True)
    ratio_quality = ml.get_on_ratio()
    print(f"QUALITY profile on-ratio: {ratio_quality:.3f}")

    # Test with FAST profile (no anti-facilitation)
    print("\nTesting same embedding with FAST profile...")
    ml.set_profile("fast")
    ml.imprint_vector(test_emb)
    ml.settle(frames=10, learn=True)
    ratio_fast = ml.get_on_ratio()
    print(f"FAST profile on-ratio: {ratio_fast:.3f}")

    # Summary
    print("\n" + "-" * 40)
    print("PROFILE COMPARISON:")
    print(f"  FAST:     {ratio_fast:.3f}  (no anti-facil)")
    print(f"  BALANCED: {ratio:.3f}  (0.5x anti-facil)")
    print(f"  QUALITY:  {ratio_quality:.3f}  (1.0x anti-facil)")
    print("-" * 40)

    if 0.9 <= ratio_quality <= 1.1:
        print("QUALITY profile is well-tuned!")
    elif ratio_quality > 1.1:
        print("Suggestion: Increase kWTA aggressiveness or anti-facil")
    else:
        print("Suggestion: Reduce anti-facil or kWTA threshold")

    print("\n" + "=" * 60)
    print("V7 WRAPPER TEST COMPLETE")
    print("=" * 60)
