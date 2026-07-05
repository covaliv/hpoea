#pragma once

// shared CLI-driving helpers for cli_tests and cli_pagmo_tests
// each including TU is compiled with a CLI_EXECUTABLE definition.

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace hpoea::tests_v2 {

struct CommandResult {
    int exit_code{0};
    std::string stdout_text;
    std::string stderr_text;
};

inline std::filesystem::path unique_test_dir(const std::string &name) {
    return std::filesystem::temp_directory_path()
        / (name + "_" + std::to_string(::getpid()));
}

inline std::string shell_quote(const std::string &value) {
    std::string quoted = "'";
    for (const auto ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted += ch;
        }
    }
    quoted += "'";
    return quoted;
}

inline std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream{path};
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

inline void write_file(const std::filesystem::path &path,
                       const std::string &text) {
    std::ofstream stream{path};
    stream << text;
}

inline CommandResult run_cli(const std::vector<std::string> &args,
                             const std::filesystem::path &work_dir) {
    static int command_index = 0;
    std::filesystem::create_directories(work_dir);
    const auto stdout_path = work_dir / ("stdout_" + std::to_string(command_index) + ".txt");
    const auto stderr_path = work_dir / ("stderr_" + std::to_string(command_index) + ".txt");
    ++command_index;

    std::string command = shell_quote(CLI_EXECUTABLE);
    for (const auto &arg : args) {
        command += ' ';
        command += shell_quote(arg);
    }
    command += " > ";
    command += shell_quote(stdout_path.string());
    command += " 2> ";
    command += shell_quote(stderr_path.string());

    const int raw_status = std::system(command.c_str());
    CommandResult result;
    if (raw_status == -1) {
        result.exit_code = errno == 0 ? 1 : errno;
    } else if (WIFEXITED(raw_status)) {
        result.exit_code = WEXITSTATUS(raw_status);
    } else {
        result.exit_code = 1;
    }
    result.stdout_text = read_file(stdout_path);
    result.stderr_text = read_file(stderr_path);
    return result;
}

inline bool contains(const std::string &text,
                     const std::string &needle) {
    return text.find(needle) != std::string::npos;
}

} // namespace hpoea::tests_v2
