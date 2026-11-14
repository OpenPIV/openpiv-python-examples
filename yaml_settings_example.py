"""Example: configure OpenPIV using PIVSettings and YAML."""

from openpiv import windef

# Create settings programmatically
settings = windef.Settings()
settings.filepath_images = "./test1"
settings.frame_pattern_a = "exp1_001_a.bmp"
settings.frame_pattern_b = "exp1_001_b.bmp"
settings.windowsizes = (64, 32, 16)
settings.overlap = (32, 16, 8)

# Save to YAML for reuse
settings.to_yaml("my_settings.yaml")

# Load from YAML and run
settings = windef.Settings.from_yaml("my_settings.yaml")
windef.piv(settings)
