#include "Decompressor.h"
#include "TorchUtils.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include "spdlog/spdlog.h"
#include <Companion.h>
#include "factories/ssb64/RelocFactory.h"

extern "C" {
#include <libmio0/mio0.h>
#include <libyay0/yay0.h>
#include <libyay0/yay1.h>
#include <libmio0/tkmk00.h>
}

std::unordered_map<uint32_t, DataChunk*> gCachedChunks;
std::unordered_map<std::string, DataChunk*> gRelocatedRelocChunks;

static uint32_t ReadBE32(const uint8_t* data) {
    return (static_cast<uint32_t>(data[0]) << 24) | (static_cast<uint32_t>(data[1]) << 16) |
           (static_cast<uint32_t>(data[2]) << 8) | static_cast<uint32_t>(data[3]);
}

static void WriteBE32(uint8_t* data, uint32_t value) {
    data[0] = static_cast<uint8_t>((value >> 24) & 0xFF);
    data[1] = static_cast<uint8_t>((value >> 16) & 0xFF);
    data[2] = static_cast<uint8_t>((value >> 8) & 0xFF);
    data[3] = static_cast<uint8_t>(value & 0xFF);
}

static DataChunk* GetSSB64RelocatedChunk() {
    const auto parentSymbol = Companion::Instance->GetCurrentSSB64RelocParent();
    if (!parentSymbol.has_value()) {
        return nullptr;
    }

    const auto parentParse = Companion::Instance->GetParseDataBySymbol(parentSymbol.value());
    if (!parentParse.has_value() || !parentParse->data.has_value()) {
        throw std::runtime_error("Could not find parsed SSB64 reloc parent '" + parentSymbol.value() + "'");
    }

    if (Torch::contains(gRelocatedRelocChunks, parentParse->name)) {
        return gRelocatedRelocChunks[parentParse->name];
    }

    const auto reloc = std::static_pointer_cast<SSB64::RelocData>(parentParse->data.value());
    if (!reloc) {
        throw std::runtime_error("Parsed reloc parent '" + parentSymbol.value() + "' is not SSB64::RelocData");
    }

    auto chunk = new DataChunk{ new uint8_t[reloc->mDecompressedData.size()], reloc->mDecompressedData.size() };
    std::memcpy(chunk->data, reloc->mDecompressedData.data(), reloc->mDecompressedData.size());

    const auto wordCount = chunk->size / sizeof(uint32_t);
    auto current = reloc->mRelocInternOffset;
    while (current != 0xFFFF && current < wordCount) {
        const auto wordOffset = static_cast<size_t>(current) * sizeof(uint32_t);
        const auto descriptor = ReadBE32(chunk->data + wordOffset);
        const auto next = static_cast<uint16_t>((descriptor >> 16) & 0xFFFF);
        const auto targetWord = static_cast<uint16_t>(descriptor & 0xFFFF);
        WriteBE32(chunk->data + wordOffset, static_cast<uint32_t>(targetWord) * sizeof(uint32_t));
        current = next;
    }

    gRelocatedRelocChunks[parentParse->name] = chunk;
    return chunk;
}

DataChunk* Decompressor::Decode(const std::vector<uint8_t>& buffer, const uint32_t offset, const CompressionType type,
                                bool ignoreCache) {

    if (!ignoreCache && Torch::contains(gCachedChunks, offset)) {
        return gCachedChunks[offset];
    }

    const unsigned char* in_buf = buffer.data() + offset;

    switch (type) {
        case CompressionType::MIO0: {
            mio0_header_t head;
            if (!mio0_decode_header(in_buf, &head)) {
                throw std::runtime_error("Failed to decode MIO0 header");
            }

            const auto decompressed = new uint8_t[head.dest_size];
            mio0_decode(in_buf, decompressed, nullptr);
            gCachedChunks[offset] = new DataChunk{ decompressed, head.dest_size };
            return gCachedChunks[offset];
        }
        case CompressionType::YAY0: {
            uint32_t size = 0;
            uint8_t* decompressed = yay0_decode(in_buf, &size);

            if (!decompressed) {
                throw std::runtime_error("Failed to decode YAY0");
            }

            gCachedChunks[offset] = new DataChunk{ decompressed, size };
            return gCachedChunks[offset];
        }
        case CompressionType::YAY1: {
            uint32_t size = 0;
            uint8_t* decompressed = yay1_decode(in_buf, &size);

            if (!decompressed) {
                throw std::runtime_error("Failed to decode YAY1");
            }

            gCachedChunks[offset] = new DataChunk{ decompressed, size };
            return gCachedChunks[offset];
        }
        default:
            throw std::runtime_error("Unknown compression type");
    }
}

DataChunk* Decompressor::DecodeTKMK00(const std::vector<uint8_t>& buffer, const uint32_t offset, const uint32_t size,
                                      const uint32_t alpha) {
    if (Torch::contains(gCachedChunks, offset)) {
        return gCachedChunks[offset];
    }

    const uint8_t* in_buf = buffer.data() + offset;

    const auto decompressed = new uint8_t[size];
    const auto rgba = new uint8_t[size];
    tkmk00_decode(in_buf, decompressed, rgba, alpha);
    gCachedChunks[offset] = new DataChunk{ rgba, size };
    return gCachedChunks[offset];
}

DecompressedData Decompressor::AutoDecode(YAML::Node& node, std::vector<uint8_t>& buffer,
                                          std::optional<size_t> manualSize) {
    auto offset = GetSafeNode<uint32_t>(node, "offset");

    if (auto* relocated = GetSSB64RelocatedChunk(); relocated != nullptr) {
        if (offset > relocated->size) {
            throw std::runtime_error("SSB64 reloc child offset is past relocated parent buffer");
        }

        auto availableSize = relocated->size - offset;
        size_t size;

        if (node["size"]) {
            size = node["size"].as<size_t>();
        } else if (manualSize.has_value()) {
            size = manualSize.value();
        } else {
            size = availableSize;
        }

        if (size > availableSize) {
            SPDLOG_WARN("Requested size 0x{:X} exceeds relocated SSB64 asset size 0x{:X} at offset 0x{:X}. Reducing to "
                        "available size.",
                        size, availableSize, offset);
            size = availableSize;
        }

        return { .root = relocated, .segment = { relocated->data + offset, size } };
    }

    CompressionType type = Companion::Instance->GetCurrCompressionType();

    auto fileOffset = TranslateAddr(offset, true);

    // Check if an asset in a yaml file is mio0 compressed and extract.
    if (node["mio0"]) {
        auto assetPtr = ASSET_PTR(offset);
        auto gameSize = Companion::Instance->GetRomData().size();

        auto fileOffset = TranslateAddr(offset, true);
        offset = ASSET_PTR(offset);

        auto decoded = Decode(buffer, fileOffset + offset, CompressionType::MIO0);
        size_t decodedSize = decoded->size - offset;
        size_t size;

        if (node["size"]) {
            size = node["size"].as<size_t>();
        } else if (manualSize.has_value()) {
            size = manualSize.value();
        } else {
            size = decodedSize;
        }

        if (size > decodedSize) {
            SPDLOG_WARN("Requested size 0x{:X} exceeds decoded MIO0 asset size 0x{:X} at offset 0x{:X}. Reducing to "
                        "available size.",
                        size, decodedSize, assetPtr);
            size = decodedSize;
        }

        return { .root = decoded, .segment = { decoded->data, size } };
    }

    // Check if an asset in a yaml file is tkmk00 compressed and extract (mk64).
    if (node["tkmk00"]) {
        const auto alpha = GetSafeNode<uint32_t>(node, "alpha");
        const auto width = GetSafeNode<uint32_t>(node, "width");
        const auto height = GetSafeNode<uint32_t>(node, "height");
        const auto textureSize = width * height * 2;

        auto fileOffset = TranslateAddr(offset, true);
        offset = ASSET_PTR(offset);

        auto assetPtr = fileOffset + offset;
        auto decoded = DecodeTKMK00(buffer, assetPtr, textureSize, alpha);
        size_t decodedSize = decoded->size - offset;
        size_t size;

        if (node["size"]) {
            size = node["size"].as<size_t>();
        } else if (manualSize.has_value()) {
            size = manualSize.value();
        } else {
            size = decodedSize;
        }

        if (size > decodedSize) {
            SPDLOG_WARN("Requested size 0x{:X} exceeds decoded TKMK00 asset size 0x{:X} at offset 0x{:X}. Reducing to "
                        "available size.",
                        size, decodedSize, assetPtr);
            size = decodedSize;
        }

        return { .root = decoded, .segment = { decoded->data, size } };
    }

    // Extract a compressed file which contains many assets.
    switch (type) {
        case CompressionType::YAY0:
        case CompressionType::YAY1:
        case CompressionType::MIO0: {
            offset = ASSET_PTR(offset);

            auto decoded = Decode(buffer, fileOffset, type);
            auto availableSize = decoded->size - offset;
            size_t size;

            if (node["size"]) {
                size = node["size"].as<size_t>();
            } else if (manualSize.has_value()) {
                size = manualSize.value();
            } else {
                size = availableSize;
            }

            if (size > availableSize) {
                SPDLOG_WARN("Requested size 0x{:X} exceeds decoded asset size 0x{:X} at offset 0x{:X}. Reducing to "
                            "available size.",
                            size, availableSize, fileOffset);
                size = availableSize;
            }

            return { .root = decoded, .segment = { decoded->data + offset, size } };
        }
        case CompressionType::YAZ0:
            throw std::runtime_error(
                "Found compressed yaz0 segment.\nDecompression of yaz0 has not been implemented yet.");
        case CompressionType::None: // The data does not have compression
        {
            fileOffset = TranslateAddr(offset, false);

            auto availableSize = buffer.size() - fileOffset;
            size_t size;

            if (node["size"]) {
                size = node["size"].as<size_t>();
            } else if (manualSize.has_value()) {
                size = manualSize.value();
            } else {
                size = availableSize;
            }

            if (size > availableSize) {
                SPDLOG_WARN("Requested size 0x{:X} exceeds available asset size 0x{:X} at offset 0x{:X}. Reducing to "
                            "available size.",
                            size, availableSize, fileOffset);
                size = availableSize;
            }

            return { .root = nullptr, .segment = { buffer.data() + fileOffset, size } };
        }
    }

    throw std::runtime_error("Auto decode could not find a compression type nor uncompressed segment.\nThis is one of "
                             "those issues that should never really happen.");
}

DecompressedData Decompressor::AutoDecode(uint32_t offset, std::optional<size_t> size, std::vector<uint8_t>& buffer) {
    YAML::Node node;
    node["offset"] = offset;

    return AutoDecode(node, buffer, size);
}

uint32_t Decompressor::TranslateAddr(uint32_t addr, bool baseAddress) {
    if (IS_SEGMENTED(addr)) {
        const auto segment = Companion::Instance->GetFileOffsetFromSegmentedAddr(SEGMENT_NUMBER(addr));
        if (!segment.has_value()) {
            SPDLOG_ERROR("Segment data missing from game config\nPlease add an entry for segment {}",
                         SEGMENT_NUMBER(addr));
            return 0;
        }

        return segment.value() + (!baseAddress ? SEGMENT_OFFSET(addr) : 0);
    }

    const auto vramEntry = Companion::Instance->GetCurrentVRAM();

    if (vramEntry.has_value()) {
        const auto vram = vramEntry.value();

        if (addr >= vram.addr) {
            return vram.offset + (addr - vram.addr);
        }
    }

    return addr;
}

CompressionType Decompressor::GetCompressionType(std::vector<uint8_t>& buffer, const uint32_t offset) {
    if (offset) {
        LUS::BinaryReader reader((char*)buffer.data() + offset, sizeof(uint32_t));
        reader.SetEndianness(Torch::Endianness::Big);

        const std::string header = reader.ReadCString();

        // Check if a compressed header exists
        if (header == "MIO0") {
            return CompressionType::MIO0;
        }

        if (header == "Yay0" || header == "PERS") {
            return CompressionType::YAY0;
        }

        if (header == "Yay1") {
            return CompressionType::YAY1;
        }

        if (header == "Yaz0") {
            return CompressionType::YAZ0;
        }
    }
    return CompressionType::None;
}

bool Decompressor::IsSegmented(uint32_t addr) {
    if (IS_SEGMENTED(addr)) {
        const auto segment = Companion::Instance->GetFileOffsetFromSegmentedAddr(SEGMENT_NUMBER(addr));

        if (!segment.has_value()) {
            SPDLOG_ERROR("Segment data missing from game config\nPlease add an entry for segment {}",
                         SEGMENT_NUMBER(addr));
            return false;
        }

        return true;
    }

    return false;
}

void Decompressor::ClearCache() {
    for (auto& [key, value] : gCachedChunks) {
        delete[] value->data;
    }
    gCachedChunks.clear();

    for (auto& [key, value] : gRelocatedRelocChunks) {
        delete[] value->data;
    }
    gRelocatedRelocChunks.clear();
}