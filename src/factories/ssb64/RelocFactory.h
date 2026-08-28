#pragma once

#include <factories/BaseFactory.h>
#include <vector>
#include <cstdint>

namespace SSB64 {

// ROM layout constants for the US NTSC v1.0 ROM
static constexpr uint32_t RELOC_TABLE_ROM_ADDR   = 0x001AC870;
static constexpr uint32_t RELOC_FILE_COUNT        = 2132;
static constexpr uint32_t RELOC_TABLE_ENTRY_SIZE  = 12;
static constexpr uint32_t RELOC_TABLE_SIZE        = (RELOC_FILE_COUNT + 1) * RELOC_TABLE_ENTRY_SIZE;
static constexpr uint32_t RELOC_DATA_START        = RELOC_TABLE_ROM_ADDR + RELOC_TABLE_SIZE;

/**
 * Parsed reloc file data — holds decompressed file contents and relocation metadata.
 */
class RelocData : public IParsedData {
public:
    uint32_t mFileId;
    uint16_t mRelocInternOffset;    // internal reloc chain start (in u32 words), 0xFFFF = none
    uint16_t mRelocExternOffset;    // external reloc chain start (in u32 words), 0xFFFF = none
    std::vector<uint16_t> mExternFileIds;  // IDs of files referenced by external relocations
    std::vector<uint8_t> mDecompressedData;

    RelocData(uint32_t fileId, uint16_t relocIntern, uint16_t relocExtern,
              std::vector<uint16_t> externIds, std::vector<uint8_t> data)
        : mFileId(fileId)
        , mRelocInternOffset(relocIntern)
        , mRelocExternOffset(relocExtern)
        , mExternFileIds(std::move(externIds))
        , mDecompressedData(std::move(data))
    {}
};

class RelocHeaderExporter : public BaseExporter {
    ExportResult Export(std::ostream& write, std::shared_ptr<IParsedData> data,
                        std::string& entryName, YAML::Node& node,
                        std::string* replacement) override;
};

class RelocBinaryExporter : public BaseExporter {
    ExportResult Export(std::ostream& write, std::shared_ptr<IParsedData> data,
                        std::string& entryName, YAML::Node& node,
                        std::string* replacement) override;
};

class RelocCodeExporter : public BaseExporter {
    ExportResult Export(std::ostream& write, std::shared_ptr<IParsedData> data,
                        std::string& entryName, YAML::Node& node,
                        std::string* replacement) override;
};

class RelocFactory : public BaseFactory {
public:
    std::optional<std::shared_ptr<IParsedData>> parse(std::vector<uint8_t>& buffer,
                                                       YAML::Node& data) override;
    inline std::unordered_map<ExportType, std::shared_ptr<BaseExporter>> GetExporters() override {
        return {
            REGISTER(Header, RelocHeaderExporter)
            REGISTER(Binary, RelocBinaryExporter)
            REGISTER(Code, RelocCodeExporter)
        };
    }
};

} // namespace SSB64
