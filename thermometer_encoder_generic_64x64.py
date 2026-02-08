"""
64×64 THERMOMETER ENCODER FOR NOMIC EMBEDDINGS
===============================================

Encoder for Nomic Embed (768-dim) → 4096×4096 lattice patterns
Using 64×64 regions with thermometer encoding (4,096 bits per region).

NOMIC EMBED SPECS:
- 768 dimensions (2× larger than SBERT's 384-dim)
- State-of-the-art semantic understanding
- Long context (8K tokens)

ENCODING STRATEGY:
- 768 dimensions → 768 regions of 64×64 bits
- Each dimension value [0, 1] → byte value [0, 255]
- Byte value → thermometer pattern (progressive activation)
  - 0 → all zeros
  - 128 → half filled (2048 bits)
  - 255 → fully filled (4096 bits)

LAYOUT:
- 24 rows × 32 columns of 64×64 regions
- Total: 768 regions covering 4096×4096 lattice (16M neurons!)
- Activation: ~18.7% (optimal for Hopfield network)

Based on: thermometer_encoder_32x32.py (proven at 10K patterns)
"""

import numpy as np


class ThermometerEncoderNomic64x64:
    """64×64 region thermometer encoder for Nomic embeddings (768-dim)."""

    def __init__(
        self,
        n_dims: int = 768,
        lattice_size: int = 4096,
        region_size: int = 64,
        encoding_type: str = "linear",
        layout: str = "spread",
        start_region_row: int = 0,
        seed: int = 42,
    ):
        """
        Initialize thermometer encoder for Nomic.

        Args:
            n_dims: Number of embedding dimensions (768 for Nomic)
            lattice_size: Lattice dimension (4096)
            region_size: Size of each region (64 for 64×64)
            encoding_type: "linear" (progressive fill)
            layout: "spread" (scattered with gaps) or "contiguous" (packed row-major)
            start_region_row: For contiguous layout, offset embedding to this row.
                              Use >0 to leave room for RAM/hippocampus rows ABOVE
                              the embedding data (needed for Hebbian adjacency since
                              thermometer fills from top of each region).
            seed: Random seed for reproducibility
        """
        self.n_dims = n_dims
        self.lattice_size = lattice_size
        self.region_size = region_size
        self.encoding_type = encoding_type
        self.layout = layout
        self.start_region_row = start_region_row
        self.seed = seed

        np.random.seed(seed)

        print(f"Creating {encoding_type} thermometer lookup (64x64)...")
        self.lookup_table = self._create_linear_thermometer()

        print(f"Creating layout for {n_dims} regions (64x64, layout={layout})...")
        if layout == "contiguous":
            self.region_positions = self._create_contiguous_layout()
        else:
            self.region_positions = self._create_region_layout()

        total_bits = self._estimate_average_bits()
        total_lattice_bits = lattice_size * lattice_size
        activation_pct = (total_bits / total_lattice_bits) * 100

        print(f"OK 64x64 Thermometer Encoder initialized:")
        print(f"  Regions: {n_dims} x {region_size}x{region_size}")
        print(
            f"  Total lattice: {lattice_size}x{lattice_size} = {total_lattice_bits:,} neurons"
        )
        print(
            f"  Total activation: {total_bits}/{total_lattice_bits} ({activation_pct:.2f}%)"
        )

    def _create_linear_thermometer(self):
        """
        Create lookup table for thermometer encoding.

        For each byte value [0, 255], create a 64×64 pattern
        with progressive activation (fill from top-left).

        Returns:
            np.ndarray: Shape (256, 64, 64) with int8 values
        """
        lookup = np.zeros((256, self.region_size, self.region_size), dtype=np.int8)

        for byte_val in range(256):
            # Calculate number of active bits (0-4096)
            n_active = int((byte_val / 255.0) * 4096)

            # Create pattern with progressive fill
            pattern = np.zeros((self.region_size, self.region_size), dtype=np.int8)

            for i in range(n_active):
                row = i // self.region_size
                col = i % self.region_size
                pattern[row, col] = 1

            lookup[byte_val] = pattern

        return lookup

    def _create_region_layout(self):
        """
        Create positions for 768 regions in 4096×4096 lattice.

        Layout: 24 rows × 32 columns of 64×64 regions
        Total: 768 regions (perfect fit!)

        Returns:
            List of (start_row, start_col) tuples
        """
        positions = []

        # Grid layout
        # rows_of_regions = 24  # 24 × 64 = 1536... wait that's not 4096
        # cols_of_regions = 32  # 32 × 64 = 2048... that's also not 4096

        # Actually: 4096 / 64 = 64 regions per side
        # So we can do: 64 × 64 = 4,096 regions total!
        # But we only need 768 regions for Nomic

        # # Layout options:
        # # Option 1: 24 rows × 32 cols = 768 regions (leaving space unused)
        # # Option 2: Spread 768 regions across the 4096×4096 space

        # # Let's do Option 1 for simplicity
        # rows_of_regions = 24
        # cols_of_regions = 32

        # row_spacing = self.lattice_size // rows_of_regions  # 4096/24 = 170.67
        # col_spacing = self.lattice_size // cols_of_regions  # 4096/32 = 128

        # for r in range(rows_of_regions):
        #     for c in range(cols_of_regions):
        #         start_row = r * row_spacing
        #         start_col = c * col_spacing

        #         # Boundary check
        #         if start_row + self.region_size > self.lattice_size:
        #             start_row = self.lattice_size - self.region_size
        #         if start_col + self.region_size > self.lattice_size:
        #             start_col = self.lattice_size - self.region_size

        #         positions.append((start_row, start_col))

        # return positions[: self.n_dims]

        # Compute grid shape
        n = self.n_dims
        regions_per_row = int(np.floor(np.sqrt(n)))
        regions_per_col = int(np.ceil(n / regions_per_row))

        # Compute spacing
        row_spacing = self.lattice_size // regions_per_row
        col_spacing = self.lattice_size // regions_per_col

        for r in range(regions_per_row):
            for c in range(regions_per_col):
                if len(positions) >= n:
                    break

                start_row = r * row_spacing
                start_col = c * col_spacing

                # Clamp to lattice boundary
                start_row = min(start_row, self.lattice_size - self.region_size)
                start_col = min(start_col, self.lattice_size - self.region_size)

                positions.append((start_row, start_col))

        return positions

    def _create_contiguous_layout(self):
        """
        Create contiguous (packed) layout: regions fill row-major with no gaps.

        768 dims at 64px regions = 12 rows x 64 columns.
        Each region is exactly 64x64 pixels, packed edge-to-edge.

        start_region_row offsets placement so RAM/hippocampus rows can sit
        ABOVE the embedding data. This matters because thermometer encoding
        fills from TOP of each region - so the top edge of the first embedding
        row has active neurons, providing Hebbian contact with RAM rows above.

        Returns:
            List of (start_row, start_col) tuples
        """
        positions = []
        regions_per_row = self.lattice_size // self.region_size  # 64

        for dim_idx in range(self.n_dims):
            region_row = self.start_region_row + dim_idx // regions_per_row
            region_col = dim_idx % regions_per_row

            start_row = region_row * self.region_size
            start_col = region_col * self.region_size

            positions.append((start_row, start_col))

        return positions

    def used_region_rows(self):
        """
        Return sorted list of region row indices that contain embedding data.

        For contiguous layout with 768 dims: rows 0-11 (12 rows x 64 cols = 768).
        For spread layout: depends on spacing.

        Useful for determining which rows are free for hippocampus/RAM/text.
        """
        used = set()
        for start_row, start_col in self.region_positions:
            region_row = start_row // self.region_size
            used.add(region_row)
        return sorted(used)

    def free_region_rows(self):
        """
        Return sorted list of region row indices NOT used by embedding data.

        These rows are available for hippocampus, RAM, text storage, etc.
        """
        all_rows = set(range(self.lattice_size // self.region_size))
        used = set(self.used_region_rows())
        return sorted(all_rows - used)

    #
    # NEW HIPPOCAMPUS ENCODING METHODS
    # This is an experimental addition to store pattern IDs
    # in a reserved area of the lattice for quick identification.
    # This allows for instant pattern ID retrieval.
    # Normal usage does not require this.
    # This region can hold simple indexing information.
    # Other metadata can be stored here as well.
    # The total_regions method is a helper function to get the total number of neurons in the lattice.
    #
    # FUTURE USE: Store metadata, timestamps, checksums, etc.
    # But, more importantly, the hippocampus flagging system
    #
    # THE MAIN LIST SO FAR:
    # Temporal: past / present / future / indeterminate
    # Spatial: internal / external / both / indeterminate
    # Valence: negative / neutral / positive / mixed
    # Certainty: low / medium / high / unknown
    # Agency: self / other / joint / unknown
    # Modality: language / vision / audio / proprio/other
    # Self-reference: me / not-me / group / unknown
    # Novelty: new / familiar / repeated / unknown

    @property
    def total_neurons(self):
        return self.lattice_size * self.lattice_size

    @property
    def total_regions(self):
        return (
            self.lattice_size
            // self.region_size
            * self.lattice_size
            // self.region_size
        )

    def encode_hippocampus(self, pattern_id):
        """
        Encodes the Pattern ID into the LAST 64 REGIONS of the lattice.
        Instead of lighting up 1 pixel, we light up an entire 64x64 block per bit.
        This makes the ID 'Physical' and robust to noise.
        """
        # 1. Create a blank canvas of the FULL 16M neurons
        header_layer = np.zeros(self.total_neurons, dtype=np.float32)

        # 2. Reshape to manipulate it as (RegionRow, RegionCol, NeuronRow, NeuronCol)
        # This allows us to address "Region X" easily
        grid_view = header_layer.reshape(64, 64, 64, 64)

        # 3. We use the very last row of Regions (Row 63)
        # It has 64 regions (Cols 0-63). We can store 64 bits here.
        region_row_idx = 63

        # 4. Convert ID to Binary (up to 64 bits)
        bin_str = format(pattern_id, "064b")  # e.g., '00...0101'

        for i, bit in enumerate(bin_str):
            if bit == "1":
                # TURN ON THE WHOLE REGION
                # Set all 4096 neurons in Region[63, i] to 1.0
                grid_view[region_row_idx, i, :, :] = 1.0

        # 5. Flatten back to 1D array for the engine
        return header_layer

    def decode_hippocampus(self, lattice_vector):
        """
        Reads the 'Bottom Row' of Regions to decode the ID.
        """
        # 1. Reshape
        grid_view = lattice_vector.reshape(64, 64, 64, 64)
        region_row_idx = 63

        readout_bits = []

        # 2. Read the 64 Regions
        for i in range(64):
            # Check the average energy of the region
            region_energy = np.mean(grid_view[region_row_idx, i, :, :])

            # Threshold: If the region is mostly active (>0.5), it's a 1
            readout_bits.append("1" if region_energy > 0.5 else "0")

        try:
            return int("".join(readout_bits), 2)
        except:
            return -1

    def _estimate_average_bits(self):
        """
        Estimate average number of active bits.

        Assumes uniform distribution of embedding values.
        Average byte value = 128 → ~2048 bits per region
        Total: 768 regions × 2048 bits = 1,572,864 bits (~9.4% of 16M)

        Actually, let me recalculate:
        768 regions × 2048 avg bits = 1,572,864 / 16,777,216 = 9.4%

        Wait that seems low. Let me check...
        Oh! The issue is I'm using 24×32 layout which wastes space.
        But the activation % is still based on the actual bits we write.

        Returns:
            int: Estimated active bits
        """
        avg_byte = 128
        avg_pattern = self.lookup_table[avg_byte]
        avg_bits_per_region = np.sum(avg_pattern)
        return int(avg_bits_per_region * self.n_dims)

    def quantize_embedding(self, embedding):
        """
        Quantize embedding to byte values [0, 255].

        Normalizes embedding to [0, 1] then scales to [0, 255].

        Args:
            embedding: np.ndarray of shape (768,)

        Returns:
            np.ndarray: uint8 array of shape (768,)
        """
        # Normalize to [0, 1]
        embedding_min = embedding.min()
        embedding_max = embedding.max()

        if embedding_max - embedding_min < 1e-8:
            # Flat embedding (unlikely but handle gracefully)
            return np.zeros(self.n_dims, dtype=np.uint8)

        normalized = (embedding - embedding_min) / (embedding_max - embedding_min)

        # Quantize to [0, 255]
        quantized = (normalized * 255).astype(np.uint8)

        return quantized

    def encode(self, embedding):
        """
        Encode Nomic embedding to 4096×4096 lattice pattern.

        Args:
            embedding: np.ndarray of shape (768,) with float values

        Returns:
            np.ndarray: Shape (4096, 4096) with int8 values (0 or 1)
        """
        # Quantize to bytes
        quantized = self.quantize_embedding(embedding)

        # Create empty lattice
        lattice = np.zeros((self.lattice_size, self.lattice_size), dtype=np.int8)

        # Fill regions with thermometer patterns
        for dim_idx in range(self.n_dims):
            byte_val = quantized[dim_idx]
            pattern_64x64 = self.lookup_table[byte_val]

            start_row, start_col = self.region_positions[dim_idx]

            lattice[
                start_row : start_row + self.region_size,
                start_col : start_col + self.region_size,
            ] = pattern_64x64

        return lattice

    def compute_sparsity(self, lattice):
        """
        Compute activation sparsity of lattice pattern.

        Args:
            lattice: np.ndarray of shape (4096, 4096)

        Returns:
            float: Percentage of active neurons (0-100)
        """
        active = np.sum(lattice > 0)
        total = lattice.size
        return (active / total) * 100

    def decode(self, lattice):
        """
        Decode lattice pattern back to normalized embedding [0, 1].

        Reverses thermometer encoding by counting active bits in each region.
        NOTE: Returns normalized values - original scale (min/max) is lost.

        Args:
            lattice: np.ndarray of shape (4096, 4096) or flattened

        Returns:
            np.ndarray: Shape (768,) with float values in [0, 1]
        """
        # Handle flattened input
        if lattice.ndim == 1:
            lattice = lattice.reshape(self.lattice_size, self.lattice_size)

        # Binarize if needed (handle float patterns from settle)
        binary_lattice = (lattice > 0.5).astype(np.float32)

        embedding = np.zeros(self.n_dims, dtype=np.float32)

        for dim_idx in range(self.n_dims):
            start_row, start_col = self.region_positions[dim_idx]

            # Extract region
            region = binary_lattice[
                start_row : start_row + self.region_size,
                start_col : start_col + self.region_size,
            ]

            # Count active bits (0 to 4096)
            active_bits = np.sum(region)

            # Reverse thermometer: active_bits / 4096 → [0, 1]
            normalized_value = active_bits / (self.region_size * self.region_size)
            embedding[dim_idx] = normalized_value

        return embedding


# =================================================================
# SIMPLE TEST
# =================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("NOMIC 64×64 THERMOMETER ENCODER TEST")
    print("=" * 80)

    # Create encoder
    encoder = ThermometerEncoderNomic64x64()

    # Test embedding (random Nomic-like embedding)
    test_embedding = np.random.randn(768)

    # Encode
    print("\nEncoding test embedding...")
    lattice_pattern = encoder.encode(test_embedding)

    # Stats
    sparsity = encoder.compute_sparsity(lattice_pattern)

    print(f"\nResults:")
    print(f"  Embedding shape: {test_embedding.shape}")
    print(f"  Lattice shape: {lattice_pattern.shape}")
    print(
        f"  Active neurons: {np.sum(lattice_pattern > 0):,} / {lattice_pattern.size:,}"
    )
    print(f"  Sparsity: {sparsity:.2f}%")

    # Verify quantization
    quantized = encoder.quantize_embedding(test_embedding)
    print(f"\nQuantization:")
    print(f"  Min byte: {quantized.min()}")
    print(f"  Max byte: {quantized.max()}")
    print(f"  Mean byte: {quantized.mean():.1f}")

    print("\nOK: Nomic 64×64 thermometer encoder test complete!")
    print("=" * 80)
