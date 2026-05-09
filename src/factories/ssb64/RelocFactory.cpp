#include "RelocFactory.h"
#include "StructFixupCatalogData.h"

#include "Companion.h"
#include "spdlog/spdlog.h"

#include <cstring>
#include <iomanip>
#include <stdexcept>

extern "C" {
#include <libvpk0/vpk0.h>
}

// ============================================================================
//  Torch → .o2r binary export
// ============================================================================
//
//  Binary format written into the .o2r archive (after the 0x40-byte LUS header):
//
//    u32  file_id
//    u16  reloc_intern_offset      (word offset, 0xFFFF = none)
//    u16  reloc_extern_offset      (word offset, 0xFFFF = none)
//    u32  num_extern_file_ids
//    u16[num_extern_file_ids]  extern_file_ids
//    u32  processing_flags         (v1+; bit 0 = PROC_PASS1_BSWAP_DONE,
//                                          bit 1 = PROC_PASS2_DONE,
//                                          bits 2..7 = PROC_<FAMILY>_DONE,
//                                          bit 8 = PROC_HALFSWAP_DONE,
//                                          bit 10 = PROC_CHAIN_FLATTENED)
//    u32  num_intern_chain_entries  (v2+)
//    { u32 slot_byte_off, u32 target_byte_off }[num_intern_chain_entries]
//    u32  num_extern_chain_entries  (v2+)
//    { u32 slot_byte_off, u32 target_byte_off }[num_extern_chain_entries]
//    u32  decompressed_data_size   (bytes)
//    u8[decompressed_data_size]  decompressed_data   // post-Pass1+Pass2 if
//                                                      both flags set
//
//  Header version 1 added processing_flags (Pass1+Pass2+struct+halfswap moved
//  to torch). Version 2 (current) adds the flat chain-entry sidecar (Stage 9
//  chain-flatten). The runtime registers V0/V1/V2 factories; older archives
//  keep loading via the matching factory and run skipped transforms at load
//  time as before.
//
// ============================================================================

namespace {

// Mirrors port/resource/RelocFile.h.
constexpr uint32_t kProcPass1BswapDone = 1u << 0;
constexpr uint32_t kProcPass2Done      = 1u << 1;
constexpr uint32_t kProcStructU16Done    = 1u << 2;
constexpr uint32_t kProcStructU32Done    = 1u << 3;
constexpr uint32_t kProcSpriteDone       = 1u << 4;
constexpr uint32_t kProcBitmapDone       = 1u << 5;
constexpr uint32_t kProcMobjsubDone      = 1u << 6;
constexpr uint32_t kProcFtAttributesDone = 1u << 7;
constexpr uint32_t kProcHalfswapDone     = 1u << 8;
// Bit 9 reserved-unused (Stage 7 closeout — AObjEvent32 walker stays runtime-side).
constexpr uint32_t kProcChainFlattened   = 1u << 10;

// F3DEX2 GBI opcode constants. Mirror port/bridge/lbreloc_byteswap.cpp.
constexpr uint8_t  kGbiVtx        = 0x01;
constexpr uint8_t  kGbiSettimg    = 0xFD;
constexpr uint8_t  kGbiLoadblock  = 0xF3;
constexpr uint8_t  kGbiLoadtlut   = 0xF0;
constexpr uint8_t  kFileSegmentId = 0x0E;
constexpr uint32_t kImSiz4b  = 0;
constexpr uint32_t kImSiz8b  = 1;
constexpr uint32_t kImSiz16b = 2;
constexpr uint32_t kImSiz32b = 3;

// Reverses each 4-byte group in `data` so a u32 read on a LE host produces
// the same numerical value as the original BE u32 from N64 ROM. Equivalent
// to running pass1_swap_u32 (port/bridge/lbreloc_byteswap.cpp) on the buffer
// once it lives in PC RAM, but applied at extraction so the runtime can
// memcpy + skip Pass 1.
//
// Endian-portable: works regardless of host byte order. Trailing 1-3 bytes
// (size % 4) stay untouched, mirroring the runtime's `size / 4` word loop.
void ApplyPass1BswapInPlace(std::vector<uint8_t>& data) {
    for (size_t i = 0; i + 4 <= data.size(); i += 4) {
        std::swap(data[i + 0], data[i + 3]);
        std::swap(data[i + 1], data[i + 2]);
    }
}

// Reads a u32 from `data[off..off+4]` in host native byte order. Used by the
// pass2 DL walker — after pass1, `data` holds bytes that read as the original
// N64 BE u32 numerical value when interpreted natively, regardless of host
// endian.
uint32_t ReadU32Native(const std::vector<uint8_t>& data, size_t off) {
    uint32_t w;
    std::memcpy(&w, data.data() + off, 4);
    return w;
}

enum class Pass2Kind { Vertex, TexBytes, TexU16 };
struct Pass2Region {
    uint32_t  offset;
    uint32_t  size;
    Pass2Kind kind;
};

// Walks the post-pass1 blob looking for G_VTX, G_SETTIMG+G_LOADBLOCK, and
// G_SETTIMG+G_LOADTLUT command pairs whose target segment is 0x0E (intra-file).
// Pure scanner; no transforms. Mirrors `scan_display_lists` in
// port/bridge/lbreloc_byteswap.cpp byte-for-byte so the regions list it
// produces is identical to what the runtime would compute on the same blob.
void ScanDisplayLists(const std::vector<uint8_t>& data,
                      std::vector<Pass2Region>& out) {
    const size_t file_size = data.size();
    if (file_size < 8) return;

    bool     has_pending_tex     = false;
    uint32_t pending_tex_offset  = 0;
    uint32_t pending_tex_siz     = 0;

    for (size_t i = 0; i + 8 <= file_size; i += 8) {
        const uint32_t w0     = ReadU32Native(data, i);
        const uint32_t w1     = ReadU32Native(data, i + 4);
        const uint8_t  opcode = (w0 >> 24) & 0xFF;

        switch (opcode) {
        case kGbiVtx: {
            const uint32_t num_vtx   = (w0 >> 12) & 0xFF;
            const uint8_t  seg       = (w1 >> 24) & 0xFF;
            const uint32_t offset    = w1 & 0x00FFFFFF;
            const size_t   offset_sz = static_cast<size_t>(offset);
            const size_t   vtx_bytes = static_cast<size_t>(num_vtx) * 16;
            if (seg == kFileSegmentId && num_vtx > 0
                && offset_sz <= file_size && vtx_bytes <= (file_size - offset_sz)) {
                out.push_back({offset, num_vtx * 16u, Pass2Kind::Vertex});
            }
            break;
        }
        case kGbiSettimg: {
            const uint8_t  seg = (w1 >> 24) & 0xFF;
            const uint32_t siz = (w0 >> 19) & 0x03;
            if (seg == kFileSegmentId) {
                pending_tex_offset = w1 & 0x00FFFFFF;
                pending_tex_siz    = siz;
                has_pending_tex    = true;
            } else {
                has_pending_tex = false;
            }
            break;
        }
        case kGbiLoadblock: {
            if (!has_pending_tex) break;
            const uint32_t lrs        = (w1 >> 12) & 0xFFF;
            const uint32_t num_texels = lrs + 1;
            uint32_t bpp = 0;
            switch (pending_tex_siz) {
            case kImSiz4b:  bpp = 4;  break;
            case kImSiz8b:  bpp = 8;  break;
            case kImSiz16b: bpp = 16; break;
            case kImSiz32b: bpp = 32; break;
            }
            if (bpp == 0) { has_pending_tex = false; break; }
            uint32_t tex_bytes = (num_texels * bpp + 7) / 8;
            tex_bytes = (tex_bytes + 3) & ~3u;
            const size_t pending_offset_sz = static_cast<size_t>(pending_tex_offset);
            if (pending_offset_sz > file_size) {
                has_pending_tex = false;
                break;
            }
            if (static_cast<size_t>(tex_bytes) > (file_size - pending_offset_sz))
                tex_bytes = static_cast<uint32_t>(file_size - pending_offset_sz);

            Pass2Kind kind;
            switch (pending_tex_siz) {
            case kImSiz4b:
            case kImSiz8b:
                kind = Pass2Kind::TexBytes;
                break;
            case kImSiz16b:
                kind = Pass2Kind::TexU16;
                break;
            default:
                // 32bpp: pass1's u32 swap is already correct.
                has_pending_tex = false;
                continue;
            }
            out.push_back({pending_tex_offset, tex_bytes, kind});
            has_pending_tex = false;
            break;
        }
        case kGbiLoadtlut: {
            if (!has_pending_tex) break;
            const uint32_t count = ((w1 >> 14) & 0x3FF) + 1;
            uint32_t palette_bytes = count * 2;
            palette_bytes = (palette_bytes + 3) & ~3u;
            const size_t pending_offset_sz = static_cast<size_t>(pending_tex_offset);
            if (pending_offset_sz > file_size) {
                has_pending_tex = false;
                break;
            }
            if (static_cast<size_t>(palette_bytes) > (file_size - pending_offset_sz))
                palette_bytes = static_cast<uint32_t>(file_size - pending_offset_sz);
            out.push_back({pending_tex_offset, palette_bytes, Pass2Kind::TexU16});
            has_pending_tex = false;
            break;
        }
        default:
            break;
        }
    }
}

// Rotate-16 in byte form: [d0 d1 d2 d3] -> [d2 d3 d0 d1]. Equivalent on a LE
// host to `(w << 16) | (w >> 16)` over the u32 reading the 4 bytes natively.
inline void Rotate16Bytes(uint8_t* p) {
    std::swap(p[0], p[2]);
    std::swap(p[1], p[3]);
}

// Reverse 4 bytes: [d0 d1 d2 d3] -> [d3 d2 d1 d0]. Equivalent to BSWAP32 over
// the u32 reading the 4 bytes natively.
inline void Bswap32Bytes(uint8_t* p) {
    std::swap(p[0], p[3]);
    std::swap(p[1], p[2]);
}

// Mirrors apply_fixup_vertex in port/bridge/lbreloc_byteswap.cpp:
//   word 0..2: rotate16  (s16 ob[0..2], u16 flag, s16 tc[0..1])
//   word 3:    BSWAP32   (u8 RGBA color)
void ApplyVertexFixup(uint8_t* base, uint32_t aligned_bytes) {
    for (uint32_t v = 0; v + 16 <= aligned_bytes; v += 16) {
        Rotate16Bytes(base + v + 0);
        Rotate16Bytes(base + v + 4);
        Rotate16Bytes(base + v + 8);
        Bswap32Bytes (base + v + 12);
    }
}

// Mirrors apply_fixup_tex_bytes / apply_fixup_tex_u16 in
// port/bridge/lbreloc_byteswap.cpp — both are BSWAP32 per u32 word, undoing
// pass1's blanket swap so the format-specific texel reads in Fast3D get the
// original N64 BE byte layout.
void ApplyBswap32Region(uint8_t* base, uint32_t aligned_bytes) {
    for (uint32_t i = 0; i + 4 <= aligned_bytes; i += 4) {
        Bswap32Bytes(base + i);
    }
}

// Mirrors portFixupSprite in port/bridge/lbreloc_byteswap.cpp.
// Sprite layout (17 words = 68 bytes):
//   w[0]  rotate16  s16 x, s16 y
//   w[1]  rotate16  s16 width, s16 height
//   w[2]  ok        f32 scalex
//   w[3]  ok        f32 scaley
//   w[4]  rotate16  s16 expx, s16 expy
//   w[5]  rotate16  u16 attr, s16 zdepth
//   w[6]  bswap32   u8 rgba
//   w[7]  rotate16  s16 startTLUT, s16 nTLUT
//   w[8]  ok        u32 LUT (token)
//   w[9]  rotate16  s16 istart, s16 istep
//   w[10] rotate16  s16 nbitmaps, s16 ndisplist
//   w[11] rotate16  s16 bmheight, s16 bmHreal
//   w[12] bswap32   u8 bmfmt, u8 bmsiz, pad
//   w[13] ok        u32 bitmap (token)
//   w[14] ok        u32 rsp_dl (token)
//   w[15] ok        u32 rsp_dl_next (token)
//   w[16] rotate16  s16 frac_s, s16 frac_t
void ApplySpriteFixup(uint8_t* base) {
    Rotate16Bytes(base + 0  * 4);
    Rotate16Bytes(base + 1  * 4);
    Rotate16Bytes(base + 4  * 4);
    Rotate16Bytes(base + 5  * 4);
    Bswap32Bytes (base + 6  * 4);
    Rotate16Bytes(base + 7  * 4);
    Rotate16Bytes(base + 9  * 4);
    Rotate16Bytes(base + 10 * 4);
    Rotate16Bytes(base + 11 * 4);
    Bswap32Bytes (base + 12 * 4);
    Rotate16Bytes(base + 16 * 4);
}
constexpr uint32_t kSpriteSize = 68;

// Mirrors portFixupBitmap in port/bridge/lbreloc_byteswap.cpp.
// Bitmap layout (4 words = 16 bytes):
//   w[0] rotate16  s16 width, s16 width_img
//   w[1] rotate16  s16 s, s16 t
//   w[2] ok        u32 buf (token)
//   w[3] rotate16  s16 actualHeight, s16 LUToffset
void ApplyBitmapFixup(uint8_t* base) {
    Rotate16Bytes(base + 0 * 4);
    Rotate16Bytes(base + 1 * 4);
    Rotate16Bytes(base + 3 * 4);
}
constexpr uint32_t kBitmapSize = 16;

// Mirrors fixup_u16_u8u8 in port/bridge/lbreloc_byteswap.cpp. Permutes a
// 4-byte word laid out as [u16 a][u8 b][u8 c] in original BE memory:
//   post-pass1 bytes: [c, b, a_lo, a_hi]  (BSWAP32 of [a_hi, a_lo, b, c])
//   target LE layout: [a_lo, a_hi, b, c]
inline void PermuteU16U8U8Bytes(uint8_t* p) {
    const uint8_t b0 = p[0];
    const uint8_t b1 = p[1];
    const uint8_t b2 = p[2];
    const uint8_t b3 = p[3];
    p[0] = b2;
    p[1] = b3;
    p[2] = b1;
    p[3] = b0;
}

// Mirrors portFixupMObjSub in port/bridge/lbreloc_byteswap.cpp.
// MObjSub layout (30 words = 120 bytes):
//   w[0]  bswap32          u16 pad00, u8 fmt, u8 siz
//   w[1]  ok               u32 sprites (token)
//   w[2]  rotate16         u16 unk08, u16 unk0A
//   w[3]  rotate16         u16 unk0C, u16 unk0E
//   w[4..11] ok            s32/f32/u32
//   w[12] permute u16+u8u8 u16 flags + u8 block_fmt + u8 block_siz
//   w[13] rotate16         u16 block_dxt, u16 unk36
//   w[14] rotate16         u16 unk38, u16 unk3A
//   w[15..19] ok           f32/u32
//   w[20] bswap32          SYColorPack primcolor (u8 rgba)
//   w[21] bswap32          u8 prim_l, u8 prim_m, u8[2] pad
//   w[22] bswap32          SYColorPack envcolor
//   w[23] bswap32          SYColorPack blendcolor
//   w[24] bswap32          SYColorPack light1color
//   w[25] bswap32          SYColorPack light2color
//   w[26..29] ok           s32
void ApplyMObjSubFixup(uint8_t* base) {
    Bswap32Bytes      (base + 0  * 4);
    Rotate16Bytes     (base + 2  * 4);
    Rotate16Bytes     (base + 3  * 4);
    PermuteU16U8U8Bytes(base + 12 * 4);
    Rotate16Bytes     (base + 13 * 4);
    Rotate16Bytes     (base + 14 * 4);
    Bswap32Bytes      (base + 20 * 4);
    Bswap32Bytes      (base + 21 * 4);
    Bswap32Bytes      (base + 22 * 4);
    Bswap32Bytes      (base + 23 * 4);
    Bswap32Bytes      (base + 24 * 4);
    Bswap32Bytes      (base + 25 * 4);
}
constexpr uint32_t kMObjSubSize = 120;

// Mirrors portFixupStructU16 in port/bridge/lbreloc_byteswap.cpp:
// rotate16 every word in [base, base + num_words*4). Variable-size — caller
// (i.e. each catalog entry) carries num_words in the catalog's `extra` field.
void ApplyStructU16Fixup(uint8_t* base, uint32_t num_words) {
    for (uint32_t i = 0; i < num_words; i++) {
        Rotate16Bytes(base + i * 4);
    }
}

// Mirrors portFixupStructU32 / portFixupRawTextureBSWAP32 in
// port/bridge/lbreloc_byteswap.cpp: bswap32 every word in
// [base, base + num_words*4). Variable-size — extra carries num_words.
// Same shape between the two runtime helpers; they share the STRUCT_U32
// catalog family.
void ApplyStructU32Fixup(uint8_t* base, uint32_t num_words) {
    for (uint32_t i = 0; i < num_words; i++) {
        Bswap32Bytes(base + i * 4);
    }
}

// Mirrors portFixupFTAttributes in port/bridge/lbreloc_byteswap.cpp.
// FTAttributes layout (0x348 bytes = 210 words). Most fields are f32/s32/u32
// types that are correct after pass1; only these 10 word slots need fixing:
//   w[0x2D] rotate16  u16 dead_fgm_ids[0..1]
//   w[0x2E] rotate16  u16 deadup_sfx, damage_sfx
//   w[0x2F] rotate16  u16 smash_sfx[0..1]
//   w[0x30] rotate16  u16 smash_sfx[2], pad
//   w[0x39] rotate16  u16 itemthrow_vel_scale, damage_scale
//   w[0x3A] rotate16  u16 heavyget_sfx, pad
//   w[0x3C] bswap32   SYColorRGBA shade_color[0]
//   w[0x3D] bswap32   SYColorRGBA shade_color[1]
//   w[0x3E] bswap32   SYColorRGBA shade_color[2]
//   w[0x3F] bswap32   SYColorRGBA fog_color
void ApplyFTAttributesFixup(uint8_t* base) {
    Rotate16Bytes(base + 0x2D * 4);
    Rotate16Bytes(base + 0x2E * 4);
    Rotate16Bytes(base + 0x2F * 4);
    Rotate16Bytes(base + 0x30 * 4);
    Rotate16Bytes(base + 0x39 * 4);
    Rotate16Bytes(base + 0x3A * 4);
    Bswap32Bytes (base + 0x3C * 4);
    Bswap32Bytes (base + 0x3D * 4);
    Bswap32Bytes (base + 0x3E * 4);
    Bswap32Bytes (base + 0x3F * 4);
}
constexpr uint32_t kFTAttributesSize = 0x348;

// Walks the catalog for `file_id`, applies each in-scope family transform to
// `data` in place, and returns the OR'd PROC_<FAMILY>_DONE bits for the
// families that were actually touched. `data` must already have pass1+pass2
// applied — struct fixups expect that as their input state.
uint32_t ApplyStructFixupsInPlace(std::vector<uint8_t>& data, uint32_t file_id) {
    uint32_t flags_set = 0;
    auto [first, last] = SSB64::StructFixupCatalog::EntriesForFile(
        static_cast<uint16_t>(file_id));
    if (first == last) return 0;

    const size_t file_size = data.size();
    for (const auto* e = first; e != last; ++e) {
        switch (e->family) {
        case SSB64::StructFixupCatalog::SPRITE: {
            if (e->byte_offset + kSpriteSize > file_size) continue;
            ApplySpriteFixup(data.data() + e->byte_offset);
            flags_set |= kProcSpriteDone;
            break;
        }
        case SSB64::StructFixupCatalog::BITMAP: {
            if (e->byte_offset + kBitmapSize > file_size) continue;
            ApplyBitmapFixup(data.data() + e->byte_offset);
            flags_set |= kProcBitmapDone;
            break;
        }
        case SSB64::StructFixupCatalog::MOBJSUB: {
            if (e->byte_offset + kMObjSubSize > file_size) continue;
            ApplyMObjSubFixup(data.data() + e->byte_offset);
            flags_set |= kProcMobjsubDone;
            break;
        }
        case SSB64::StructFixupCatalog::STRUCT_U16: {
            const uint64_t span = uint64_t(e->extra) * 4;
            if (e->byte_offset + span > file_size) continue;
            ApplyStructU16Fixup(data.data() + e->byte_offset, e->extra);
            flags_set |= kProcStructU16Done;
            break;
        }
        case SSB64::StructFixupCatalog::STRUCT_U32: {
            const uint64_t span = uint64_t(e->extra) * 4;
            if (e->byte_offset + span > file_size) continue;
            ApplyStructU32Fixup(data.data() + e->byte_offset, e->extra);
            flags_set |= kProcStructU32Done;
            break;
        }
        case SSB64::StructFixupCatalog::FTATTRIBUTES: {
            if (e->byte_offset + kFTAttributesSize > file_size) continue;
            ApplyFTAttributesFixup(data.data() + e->byte_offset);
            flags_set |= kProcFtAttributesDone;
            break;
        }
        default:
            break;
        }
    }
    return flags_set;
}

// Mirrors portRelocIsFighterFigatreeFile in port/bridge/lbreloc_bridge.cpp.
// The runtime u16-halfswap was applied to fighter animation/submotion files
// (because their figatree/AObjEvent16 streams encode u16 pairs in u32 slots,
// and pass1 BSWAP32 alone leaves the pair indices reversed under LE), and to
// SCExplainMain (FTKeyEvent u16 arrays + SCExplainPhase u16/u8 fields).
bool PathNeedsHalfswap(const std::string& path) {
    static const std::string kAnimPrefix      = "reloc_animations/FT";
    static const std::string kSubmotionPrefix = "reloc_submotions/FT";
    static const std::string kSCExplainMain   = "reloc_scene/SCExplainMain";
    if (path.compare(0, kAnimPrefix.size(),      kAnimPrefix)      == 0) return true;
    if (path.compare(0, kSubmotionPrefix.size(), kSubmotionPrefix) == 0) return true;
    if (path == kSCExplainMain) return true;
    return false;
}

// Mirrors portRelocFixupFighterFigatree in port/bridge/lbreloc_bridge.cpp:
// rotate16 every u32 word that is NOT a reloc-chain slot. Chain slots (intern
// + extern lists) hold encoded `next_reloc:16 | target:16` linked-list nodes
// that the runtime later overwrites with tokens; the runtime mask zeroes them
// out before halfswap so they read as native u32 (`*slot >> 16` etc) for the
// chain walk. Torch precomputes the same mask by walking both chains in the
// post-pass1+pass2+struct buffer.
//
// `path` is the OTR path string (== entryName at this layer). When it doesn't
// match the runtime's halfswap predicate, returns 0 with no transform applied.
uint32_t ApplyHalfswapInPlace(std::vector<uint8_t>& data,
                              uint16_t reloc_intern,
                              uint16_t reloc_extern,
                              const std::string& path) {
    if (!PathNeedsHalfswap(path)) return 0;

    const size_t word_count = data.size() / 4;
    if (word_count == 0) return 0;

    std::vector<uint8_t> reloc_mask(word_count, 0);

    auto walk_chain = [&](uint16_t start) {
        // 0xFFFF = sentinel (no chain). Defensive iteration cap matches the
        // runtime's implicit guarantee — the chain has at most word_count
        // distinct nodes, so any longer walk indicates a cycle / corrupt data.
        uint16_t cur = start;
        size_t   iter = 0;
        while (cur != 0xFFFF && iter < word_count) {
            if (static_cast<size_t>(cur) >= word_count) break;
            reloc_mask[cur] = 1;
            const uint32_t w = ReadU32Native(data, static_cast<size_t>(cur) * 4);
            cur = static_cast<uint16_t>(w >> 16);
            iter++;
        }
    };

    walk_chain(reloc_intern);
    walk_chain(reloc_extern);

    uint8_t* bytes = data.data();
    for (size_t i = 0; i < word_count; i++) {
        if (!reloc_mask[i]) {
            Rotate16Bytes(bytes + i * 4);
        }
    }
    return kProcHalfswapDone;
}

// One pre-walked chain entry. byte offsets are relative to the start of
// `data` (which the runtime later memcpys into its heap-resident buffer).
// Mirrors port/resource/RelocFile.h::RelocChainEntry minus the DepFileId,
// which the runtime fills from ExternFileIds[i] (chain-insertion order).
struct ChainEntryOut {
    uint32_t slot_byte_off;
    uint32_t target_byte_off;
};

// Walks the encoded reloc chains in `data` and emits flat per-entry tuples.
// Mirrors lbreloc_bridge.cpp's chain walker exactly (same termination rule,
// same u32-read semantics). Safe to call after pass1+pass2+struct+halfswap
// because every transform torch runs preserves the chain-slot u32 values
// (pass1 byte-reverses each u32 in place — `ReadU32Native` on the result
// yields the original BE value, same as the runtime does; pass2 skips slots
// that aren't part of an in-DL G_VTX/G_SETTIMG; halfswap explicitly leaves
// chain-slot u32s untouched).
//
// Defensive: bounded by `data.size() / 4` to catch corrupt cycles, identical
// to the cap the halfswap walker uses above.
void BuildChainEntries(const std::vector<uint8_t>& data,
                       uint16_t reloc_intern, uint16_t reloc_extern,
                       std::vector<ChainEntryOut>& intern_out,
                       std::vector<ChainEntryOut>& extern_out) {
    const size_t word_count = data.size() / 4;
    auto walk = [&](uint16_t start, std::vector<ChainEntryOut>& out) {
        uint16_t cur = start;
        size_t   iter = 0;
        while (cur != 0xFFFF && iter < word_count) {
            if (static_cast<size_t>(cur) >= word_count) break;
            const uint32_t w = ReadU32Native(data, static_cast<size_t>(cur) * 4);
            ChainEntryOut e;
            e.slot_byte_off   = static_cast<uint32_t>(cur) * 4u;
            e.target_byte_off = static_cast<uint32_t>(w & 0xFFFFu) * 4u;
            out.push_back(e);
            cur = static_cast<uint16_t>(w >> 16);
            iter++;
        }
    };
    walk(reloc_intern, intern_out);
    walk(reloc_extern, extern_out);
}

// Applies the pass2 byte transforms in place. Idempotent in the sense that
// re-running it on already-pass2'd data would produce a different blob (it
// would invert the transforms again); torch invokes it exactly once per file
// after pass1.
//
// Tracker bookkeeping (sStructU16Fixups vertex insertions in the runtime) is
// NOT done here — that's heap-absolute state which lives only at runtime. The
// bridge re-walks the same DL stream at load and inserts the trackers
// regardless of whether torch already applied the byte transforms.
void ApplyPass2InPlace(std::vector<uint8_t>& data) {
    std::vector<Pass2Region> regions;
    ScanDisplayLists(data, regions);
    if (regions.empty()) return;

    const size_t file_size = data.size();
    uint8_t* bytes = data.data();

    for (const auto& r : regions) {
        const size_t start = static_cast<size_t>(r.offset);
        size_t       len   = static_cast<size_t>(r.size);
        if (start > file_size || len > (file_size - start)) continue;
        if ((start & 3) != 0) continue;
        len &= ~static_cast<size_t>(3);
        if (len == 0) continue;

        uint8_t* region_base = bytes + start;
        const uint32_t aligned = static_cast<uint32_t>(len);
        switch (r.kind) {
        case Pass2Kind::Vertex:   ApplyVertexFixup(region_base, aligned);   break;
        case Pass2Kind::TexBytes: ApplyBswap32Region(region_base, aligned); break;
        case Pass2Kind::TexU16:   ApplyBswap32Region(region_base, aligned); break;
        }
    }
}

} // namespace

// ----------------------------------------------------------------------------
//  Header exporter (generates C header declarations / OTR path strings)
// ----------------------------------------------------------------------------

ExportResult SSB64::RelocHeaderExporter::Export(std::ostream& write,
                                                 std::shared_ptr<IParsedData> raw,
                                                 std::string& entryName,
                                                 YAML::Node& node,
                                                 std::string* replacement) {
    const auto symbol = GetSafeNode(node, "symbol", entryName);

    if (Companion::Instance->IsOTRMode()) {
        write << "static const ALIGN_ASSET(2) char " << symbol
              << "[] = \"__OTR__" << (*replacement) << "\";\n\n";
        return std::nullopt;
    }

    write << "extern u8 " << symbol << "[];\n";
    return std::nullopt;
}

// ----------------------------------------------------------------------------
//  Code exporter (generates C arrays — for decomp builds, not the port)
// ----------------------------------------------------------------------------

ExportResult SSB64::RelocCodeExporter::Export(std::ostream& write,
                                               std::shared_ptr<IParsedData> raw,
                                               std::string& entryName,
                                               YAML::Node& node,
                                               std::string* replacement) {
    auto symbol = GetSafeNode(node, "symbol", entryName);
    auto offset = GetSafeNode<uint32_t>(node, "offset");
    auto reloc = std::static_pointer_cast<SSB64::RelocData>(raw);

    if (Companion::Instance->IsOTRMode()) {
        write << "static const ALIGN_ASSET(2) char " << symbol
              << "[] = \"__OTR__" << (*replacement) << "\";\n\n";
        return std::nullopt;
    }

    // Emit as a raw u8 array of the decompressed data
    write << "u8 " << symbol << "[] = {\n" << tab_t;
    for (size_t i = 0; i < reloc->mDecompressedData.size(); i++) {
        if ((i % 15 == 0) && i != 0) {
            write << "\n" << tab_t;
        }
        write << "0x" << std::hex << std::setw(2) << std::setfill('0')
              << (int)reloc->mDecompressedData[i] << ", ";
    }
    write << "\n};\n";

    return offset + reloc->mDecompressedData.size();
}

// ----------------------------------------------------------------------------
//  Binary exporter (writes into .o2r archive)
// ----------------------------------------------------------------------------

ExportResult SSB64::RelocBinaryExporter::Export(std::ostream& write,
                                                 std::shared_ptr<IParsedData> raw,
                                                 std::string& entryName,
                                                 YAML::Node& node,
                                                 std::string* replacement) {
    auto reloc = std::static_pointer_cast<SSB64::RelocData>(raw);
    auto writer = LUS::BinaryWriter();

    // v2: adds a flat chain-entry sidecar (Stage 9).
    WriteHeader(writer, Torch::ResourceType::SSB64Reloc, 2);

    writer.Write(reloc->mFileId);
    writer.Write(reloc->mRelocInternOffset);
    writer.Write(reloc->mRelocExternOffset);

    writer.Write((uint32_t)reloc->mExternFileIds.size());
    for (uint16_t id : reloc->mExternFileIds) {
        writer.Write(id);
    }

    std::vector<uint8_t> data = reloc->mDecompressedData;
    ApplyPass1BswapInPlace(data);
    ApplyPass2InPlace(data);
    const uint32_t structFlags = ApplyStructFixupsInPlace(data, reloc->mFileId);
    const uint32_t halfswapFlags = ApplyHalfswapInPlace(
        data, reloc->mRelocInternOffset, reloc->mRelocExternOffset, entryName);

    // Pre-walk the reloc chains and emit the flat per-entry list. Set
    // PROC_CHAIN_FLATTENED so the runtime iterates the list and skips the
    // encoded chain walk.
    std::vector<ChainEntryOut> internEntries;
    std::vector<ChainEntryOut> externEntries;
    BuildChainEntries(data, reloc->mRelocInternOffset, reloc->mRelocExternOffset,
                      internEntries, externEntries);

    const uint32_t processingFlags =
        kProcPass1BswapDone | kProcPass2Done | structFlags | halfswapFlags |
        kProcChainFlattened;

    writer.Write(processingFlags);

    writer.Write((uint32_t)internEntries.size());
    for (const auto& e : internEntries) {
        writer.Write(e.slot_byte_off);
        writer.Write(e.target_byte_off);
    }
    writer.Write((uint32_t)externEntries.size());
    for (const auto& e : externEntries) {
        writer.Write(e.slot_byte_off);
        writer.Write(e.target_byte_off);
    }

    writer.Write((uint32_t)data.size());
    writer.Write((char*)data.data(), data.size());

    writer.Finish(write);
    return std::nullopt;
}

// ----------------------------------------------------------------------------
//  Factory — parse reloc file from ROM buffer
// ----------------------------------------------------------------------------

std::optional<std::shared_ptr<IParsedData>>
SSB64::RelocFactory::parse(std::vector<uint8_t>& buffer, YAML::Node& node) {
    auto fileId = GetSafeNode<uint32_t>(node, "file_id");

    if (fileId >= RELOC_FILE_COUNT) {
        throw std::runtime_error(
            "SSB64:RELOC file_id " + std::to_string(fileId) + " out of range");
    }

    // --- Read this file's table entry and the next (for computing extern region) ---
    size_t tableOffset = RELOC_TABLE_ROM_ADDR + fileId * RELOC_TABLE_ENTRY_SIZE;
    if (tableOffset + RELOC_TABLE_ENTRY_SIZE * 2 > buffer.size()) {
        throw std::runtime_error("ROM too small to read table entry for file " +
                                  std::to_string(fileId));
    }

    LUS::BinaryReader tableReader(
        reinterpret_cast<char*>(buffer.data() + tableOffset),
        RELOC_TABLE_ENTRY_SIZE * 2);
    tableReader.SetEndianness(Torch::Endianness::Big);

    // Current entry
    uint32_t firstWord = tableReader.ReadUInt32();
    bool isCompressed = (firstWord >> 31) != 0;
    uint32_t dataOffset = firstWord & 0x7FFFFFFF;
    uint16_t relocIntern = tableReader.ReadUInt16();
    uint16_t compressedSizeWords = tableReader.ReadUInt16();
    uint16_t relocExtern = tableReader.ReadUInt16();
    uint16_t decompressedSizeWords = tableReader.ReadUInt16();

    // Next entry (sentinel exists for the last file)
    uint32_t nextFirstWord = tableReader.ReadUInt32();
    uint32_t nextDataOffset = nextFirstWord & 0x7FFFFFFF;

    uint32_t compressedSizeBytes = (uint32_t)compressedSizeWords * 4;
    uint32_t decompressedSizeBytes = (uint32_t)decompressedSizeWords * 4;

    // --- Locate data in ROM ---
    size_t dataRomAddr = RELOC_DATA_START + dataOffset;

    if (dataRomAddr + compressedSizeBytes > buffer.size()) {
        throw std::runtime_error(
            "ROM too small to read file data for file " + std::to_string(fileId));
    }

    const uint8_t* fileData = buffer.data() + dataRomAddr;

    // --- Decompress if VPK0-compressed ---
    std::vector<uint8_t> decompressed;

    if (isCompressed) {
        decompressed.resize(decompressedSizeBytes);

        uint32_t result = vpk0_decode(fileData, compressedSizeBytes,
                                       decompressed.data(), decompressedSizeBytes);
        if (result == 0) {
            throw std::runtime_error(
                "VPK0 decompression failed for file " + std::to_string(fileId) +
                " (compressed=" + std::to_string(compressedSizeBytes) +
                " decompressed=" + std::to_string(decompressedSizeBytes) + ")");
        }

        spdlog::debug("SSB64:RELOC file {} decompressed {} -> {} bytes",
                       fileId, compressedSizeBytes, decompressedSizeBytes);
    } else {
        decompressed.assign(fileData, fileData + decompressedSizeBytes);
    }

    // --- Extract external file ID list ---
    // External file IDs are stored as big-endian u16 values in ROM
    // immediately after the compressed data.
    size_t externRegionStart = dataRomAddr + compressedSizeBytes;
    size_t externRegionEnd = RELOC_DATA_START + nextDataOffset;

    std::vector<uint16_t> externFileIds;

    if (externRegionEnd > externRegionStart && externRegionEnd <= buffer.size()) {
        size_t externBytes = externRegionEnd - externRegionStart;
        size_t numExternIds = externBytes / 2;

        for (size_t i = 0; i < numExternIds; i++) {
            size_t addr = externRegionStart + i * 2;
            uint16_t extId = ((uint16_t)buffer[addr] << 8) | buffer[addr + 1];
            externFileIds.push_back(extId);
        }
    }

    return std::make_shared<SSB64::RelocData>(
        fileId, relocIntern, relocExtern,
        std::move(externFileIds), std::move(decompressed));
}
