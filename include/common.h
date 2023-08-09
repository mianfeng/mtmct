#ifndef COMMON_H
#define COMMON_H

#include <unistd.h>

#include <opencv2/core/core.hpp>
#include <variant>

// #include "WebSocketClient.h"

struct Position {
    int32_t centerX;
    int32_t centerY;
    int32_t width;
    int32_t height;
};

struct MatchMessage {
    std::string cameraId;
    std::string personId;
    Position postion;
    std::vector<Position> postion2;
    std::int32_t timesTamp;
};

enum class MatSource { SOURCE_1, SOURCE_2, SOURCE_3 };

struct MatWrapper {
    MatSource source;
    cv::Mat mat;
};

using MyVariant = std::variant<MatWrapper, MatchMessage>;
/**
 * Returns the directory name from the given file path.
 *
 * @param filepath the file path from which to extract the directory name.
 *
 * @return the directory name extracted from the file path.
 *
 * @throws std::string::npos if the directory name cannot be found.
 */
std::string DirName(const std::string& filepath);

/**
 * Checks if a given path exists.
 *
 * @param path the path to check
 *
 * @return true if the path exists, false otherwise
 *
 * @throws None
 */
bool PathExists(const std::string& path);

/**
 * Creates a new directory at the specified path if it does not already exist.
 *
 * @param path The path where the directory should be created.
 *
 * @throws std::runtime_error If the directory creation fails.
 */
void MkDir(const std::string& path);

/**
 * Create all directories in the given path.
 *
 * @param path the path where the directories will be created
 *
 * @throws ErrorType if there is an error creating the directories
 */
void MkDirs(const std::string& path);
#endif