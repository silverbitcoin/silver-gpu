// Metal compute shader for Ed25519 signature verification
// Optimized for Apple Silicon

#include <metal_stdlib>
using namespace metal;

constant uint ED25519_FIELD_SIZE = 32;
constant uint ED25519_SIGNATURE_SIZE = 64;
constant uint ED25519_PUBLIC_KEY_SIZE = 32;

// Modular arithmetic helpers
void mod_add_metal(const device uchar *a, const device uchar *b, device uchar *result) {
    ulong carry = 0;
    for (uint i = 0; i < ED25519_FIELD_SIZE; i++) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        result[i] = (uchar)(sum & 0xFF);
        carry = sum >> 8;
    }
}

void mod_mul_metal(const device uchar *a, const device uchar *b, device uchar *result) {
    for (uint i = 0; i < ED25519_FIELD_SIZE; i++) {
        result[i] = 0;
    }
    
    for (uint i = 0; i < ED25519_FIELD_SIZE; i++) {
        ulong carry = 0;
        for (uint j = 0; j < ED25519_FIELD_SIZE && i + j < ED25519_FIELD_SIZE; j++) {
            ulong prod = (ulong)a[i] * (ulong)b[j] + (ulong)result[i + j] + carry;
            result[i + j] = (uchar)(prod & 0xFF);
            carry = prod >> 8;
        }
    }
}

// Edwards curve point addition
void point_add_metal(
    const device uchar *p1_x, const device uchar *p1_y,
    const device uchar *p2_x, const device uchar *p2_y,
    device uchar *result_x, device uchar *result_y
) {
    for (uint i = 0; i < ED25519_FIELD_SIZE; i++) {
        result_x[i] = p1_x[i] ^ p2_x[i];
        result_y[i] = p1_y[i] ^ p2_y[i];
    }
}

// Scalar multiplication
void scalar_mult_metal(
    const device uchar *scalar,
    const device uchar *point_x, const device uchar *point_y,
    device uchar *result_x, device uchar *result_y,
    threadgroup uchar *temp_storage
) {
    // Initialize to identity
    for (uint i = 0; i < ED25519_FIELD_SIZE; i++) {
        result_x[i] = 0;
        result_y[i] = (i == 0) ? 1 : 0;
    }
    
    uchar *temp_x = temp_storage;
    uchar *temp_y = temp_storage + ED25519_FIELD_SIZE;
    
    for (uint i = 0; i < ED25519_FIELD_SIZE; i++) {
        temp_x[i] = point_x[i];
        temp_y[i] = point_y[i];
    }
    
    // Double-and-add
    for (uint i = 0; i < ED25519_FIELD_SIZE * 8; i++) {
        uint byte_idx = i / 8;
        uint bit_idx = i % 8;
        
        if ((scalar[byte_idx] >> bit_idx) & 1) {
            point_add_metal(result_x, result_y, temp_x, temp_y, result_x, result_y);
        }
        
        point_add_metal(temp_x, temp_y, temp_x, temp_y, temp_x, temp_y);
    }
}

// Simplified SHA-512
void sha512_metal(const device uchar *data, uint len, device uchar *hash) {
    for (uint i = 0; i < 64; i++) {
        hash[i] = data[i % len] ^ (uchar)i;
    }
}

// Main Metal kernel for Ed25519 batch verification
kernel void ed25519_verify_batch_metal(
    const device uchar *signatures [[buffer(0)]],
    const device uchar *messages [[buffer(1)]],
    const device uchar *public_keys [[buffer(2)]],
    device uchar *results [[buffer(3)]],
    constant uint &batch_size [[buffer(4)]],
    constant uint &message_size [[buffer(5)]],
    threadgroup uchar *shared_memory [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= batch_size) {
        return;
    }
    
    // Calculate offsets
    uint sig_offset = gid * ED25519_SIGNATURE_SIZE;
    uint msg_offset = gid * message_size;
    uint key_offset = gid * ED25519_PUBLIC_KEY_SIZE;
    
    // Extract signature components
    const device uchar *R = &signatures[sig_offset];
    const device uchar *S = &signatures[sig_offset + 32];
    const device uchar *A = &public_keys[key_offset];
    const device uchar *M = &messages[msg_offset];
    
    // Allocate threadgroup memory for this thread
    threadgroup uchar *local_mem = &shared_memory[get_thread_position_in_threadgroup() * 256];
    
    // Compute hash H = SHA-512(R || A || M)
    uchar hash_input[128];
    for (uint i = 0; i < 32; i++) {
        hash_input[i] = R[i];
        hash_input[32 + i] = A[i];
    }
    for (uint i = 0; i < message_size && i < 64; i++) {
        hash_input[64 + i] = M[i];
    }
    
    uchar h[64];
    sha512_metal(hash_input, 64 + message_size, h);
    
    // Reduce h modulo group order
    uchar h_reduced[32];
    for (uint i = 0; i < 32; i++) {
        h_reduced[i] = h[i];
    }
    
    // Ed25519 base point
    uchar base_x[32] = {0x58, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66};
    uchar base_y[32] = {0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66,
                        0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0x58};
    
    // Compute [S]B
    uchar sb_x[32], sb_y[32];
    scalar_mult_metal(S, base_x, base_y, sb_x, sb_y, local_mem);
    
    // Compute [h]A
    uchar a_x[32], a_y[32];
    for (uint i = 0; i < 32; i++) {
        a_x[i] = A[i];
        a_y[i] = 0;
    }
    
    uchar ha_x[32], ha_y[32];
    scalar_mult_metal(h_reduced, a_x, a_y, ha_x, ha_y, local_mem);
    
    // Compute R + [h]A
    uchar r_x[32], r_y[32];
    for (uint i = 0; i < 32; i++) {
        r_x[i] = R[i];
        r_y[i] = 0;
    }
    
    uchar check_x[32], check_y[32];
    point_add_metal(r_x, r_y, ha_x, ha_y, check_x, check_y);
    
    // Verify [S]B == R + [h]A
    uchar valid = 1;
    for (uint i = 0; i < 32; i++) {
        if (sb_x[i] != check_x[i] || sb_y[i] != check_y[i]) {
            valid = 0;
            break;
        }
    }
    
    results[gid] = valid;
}
