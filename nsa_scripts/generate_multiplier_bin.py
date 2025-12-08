import sys
import subprocess
import struct
import tempfile
import os

def generate_bin_from_c(c_file, output_bin):

    with open(c_file, 'r') as f:
        c_code = f.read()

    func_name = os.path.basename(c_file).replace('.c', '')

    wrapper_code = f"""
{c_code}


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
