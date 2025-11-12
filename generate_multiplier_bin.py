#!/usr/bin/env python3
"""
Generate binary lookup table from EvoApproxLib C multiplier
Usage: python generate_multiplier_bin.py path/to/mul8u_XXX.c output.bin
"""
import sys
import subprocess
import struct
import tempfile
import os

def generate_bin_from_c(c_file, output_bin):
    """Generate .bin lookup table from .c multiplier file"""

    # Read the C file
    with open(c_file, 'r') as f:
        c_code = f.read()

    # Extract function name
    func_name = os.path.basename(c_file).replace('.c', '')

    # Create a wrapper program
    wrapper_code = f"""
{c_code}

#include <stdio.h>

int main() {{
    FILE *f = fopen("{output_bin}", "wb");
    if (!f) {{
        fprintf(stderr, "Cannot open output file\\n");
        return 1;
    }}

    for (unsigned int a = 0; a < 256; a++) {{
        for (unsigned int b = 0; b < 256; b++) {{
            uint64_t result = {func_name}(b, a);
            uint16_t val = (uint16_t)(result & 0xFFFF);
            fwrite(&val, sizeof(uint16_t), 1, f);
        }}
    }}

    fclose(f);
    return 0;
}}
"""

    # Compile and run
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as tmp:
        tmp.write(wrapper_code)
        tmp_c = tmp.name

    try:
        tmp_exe = tmp_c.replace('.c', '.exe')

        # Compile
        compile_cmd = ['gcc', tmp_c, '-o', tmp_exe, '-O2']
        subprocess.run(compile_cmd, check=True, capture_output=True)

        # Run
        subprocess.run([tmp_exe], check=True)

        print(f"Generated {output_bin} from {c_file}")

        # Verify size
        size = os.path.getsize(output_bin)
        expected = 256 * 256 * 2  # 131072 bytes
        if size == expected:
            print(f"✓ File size correct: {size} bytes")
        else:
            print(f"✗ File size incorrect: {size} bytes (expected {expected})")

    finally:
        # Cleanup
        if os.path.exists(tmp_c):
            os.remove(tmp_c)
        if os.path.exists(tmp_exe):
            os.remove(tmp_exe)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python generate_multiplier_bin.py input.c output.bin")
        sys.exit(1)

    c_file = sys.argv[1]
    output_bin = sys.argv[2]

    generate_bin_from_c(c_file, output_bin)
