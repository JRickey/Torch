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
//                                          bit 1 = PROC_PASS2_DONE)
//    u32  decompressed_data_size   (bytes)
//    u8[decompressed_data_size]  decompressed_data   // post-Pass1+Pass2 if
//                                                      both flags set
//
//  Header version is bumped to 1 because the runtime (port/resource) registers
//  a separate V1 factory; v0 archives without `processing_flags` keep loading
//  via the V0 factory and run every transform at runtime as before.
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
        // Other families land in subsequent Stage 6d steps.
        default:
            break;
        }
    }
    return flags_set;
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

    // v1: post-Pass1 bytes + processing_flags advertise the transform.
    WriteHeader(writer, Torch::ResourceType::SSB64Reloc, 1);

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
    const uint32_t processingFlags =
        kProcPass1BswapDone | kProcPass2Done | structFlags;

    writer.Write(processingFlags);
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
