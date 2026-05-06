#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr int exit_usage_error = 2;

void print_help(std::ostream &out) {
    out << "usage: hpoea <command> [args]\n"
        << "\n"
        << "commands:\n"
        << "  validate <config.toml>  validate a config for this build\n"
        << "  plan <config.toml>      preview expanded runs without executing\n"
        << "  run <config.toml>       execute supported config runs\n"
        << "\n"
        << "options:\n"
        << "  --help                  show this help\n"
        << "  --version               show version\n";
}

void print_version(std::ostream &out) {
    out << "hpoea " << HPOEA_PROJECT_VERSION << '\n';
}

int usage_error(std::string_view message) {
    std::cerr << "error: " << message << '\n'
              << "Try 'hpoea --help'.\n";
    return exit_usage_error;
}

int command_not_ready(std::string_view command) {
    std::cerr << "error: command is not implemented yet: " << command << '\n';
    return 1;
}

} // namespace

int main(int argc, char **argv) {
    if (argc == 1) {
        return usage_error("missing command");
    }

    const std::string_view command{argv[1]};
    if (command == "--help") {
        print_help(std::cout);
        return 0;
    }
    if (command == "--version") {
        print_version(std::cout);
        return 0;
    }
    if (command == "validate" || command == "plan" || command == "run") {
        return command_not_ready(command);
    }

    return usage_error("unknown command: " + std::string{command});
}
