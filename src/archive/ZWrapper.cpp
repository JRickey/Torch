#include "ZWrapper.h"
#include <filesystem>
#include <iostream>
#include <fstream>

#include "spdlog/spdlog.h"
#include <Companion.h>
#include <miniz/zip_file.hpp>

namespace fs = std::filesystem;

ZWrapper::ZWrapper(const std::string& path) {
    this->mPath = path;
    this->mZip = new miniz_cpp::zip_file;
}

int32_t ZWrapper::CreateArchive() {
    SPDLOG_INFO("Loaded ZIP (O2R) archive: {}", mPath.c_str());
    return 0;
}

bool ZWrapper::AddFile(const std::string& path, std::vector<char> data) {
    char* fileData = data.data();
    size_t fileSize = data.size();

    if (Companion::Instance != nullptr && Companion::Instance->IsDebug()) {
        SPDLOG_INFO("Creating debug file: debug/{}", path);
        std::string dpath = "debug/" + path;
        if (!fs::exists(fs::path(dpath).parent_path())) {
            fs::create_directories(fs::path(dpath).parent_path());
        }
        std::ofstream stream(dpath, std::ios::binary);
        stream.write(fileData, fileSize);
        stream.close();
    }

    this->mZip->writebytes(path, data);
    return true;
}

int32_t ZWrapper::Close(void) {
    // miniz_cpp::zip_file::save(path) writes through a bare ofstream and
    // never checks the stream state, so a failed open or a short write
    // (disk full, permissions) succeeded silently. Own the stream and
    // check it after the write.
    std::ofstream stream(this->mPath, std::ios::binary);
    if (!stream) {
        SPDLOG_ERROR("Failed to open {} for writing", this->mPath);
        return 1;
    }
    this->mZip->save(stream);
    stream.flush();
    if (!stream) {
        SPDLOG_ERROR("Short write saving {} (disk full?)", this->mPath);
        return 1;
    }
    return 0;
}
