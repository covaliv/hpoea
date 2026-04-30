#include "test_harness.hpp"

#include "hpoea/core/logging.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <thread>
#include <vector>
#include <unistd.h>

namespace {
std::filesystem::path unique_test_path(const std::string &base_name) {
    auto dir = std::filesystem::temp_directory_path();
    auto name = base_name + "_" + std::to_string(::getpid()) + ".jsonl";
    return dir / name;
}

std::vector<std::string> read_lines(const std::filesystem::path &path) {
    std::ifstream in(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        lines.push_back(line);
    }
    return lines;
}

std::size_t count_occurrences(const std::string &text, const std::string &needle) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = text.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

bool has_numeric_field_close(const std::string &text,
                             const std::string &field,
                             double expected,
                             double tolerance) {
    const auto needle = "\"" + field + "\":";
    const auto pos = text.find(needle);
    if (pos == std::string::npos) {
        return false;
    }

    try {
        std::size_t consumed = 0;
        const double value = std::stod(text.substr(pos + needle.size()), &consumed);
        return consumed > 0 && std::fabs(value - expected) <= tolerance;
    } catch (const std::exception &) {
        return false;
    }
}

bool has_raw_control_character(const std::string &text) {
    for (const auto ch : text) {
        if (static_cast<unsigned char>(ch) < 0x20) {
            return true;
        }
    }
    return false;
}
}

int main() {
    hpoea::tests_v2::TestRunner runner;
    using namespace hpoea::core;

    RunRecord record;
    record.experiment_id = "exp_v2";
    record.problem_id = "sphere";
    record.evolutionary_algorithm = {"DE", "pagmo::de", "2.x"};
    record.hyper_optimizer = AlgorithmIdentity{"CMAES", "pagmo::cmaes", "2.x"};
    record.algorithm_parameters = {
        {"population_size", std::int64_t{25}},
        {"scaling_factor", 0.8},
        {"memory", true},
        {"variant", std::string{"best"}},
    };
    record.optimizer_parameters = {
        {"generations", std::int64_t{10}},
    };
    record.status = RunStatus::Success;
    record.objective_value = 1.25;
    record.requested_budget = Budget{100u, 5u, std::chrono::milliseconds{250}};
    record.effective_budget = EffectiveBudget{100u, 5u, std::chrono::milliseconds{250}};
    record.algorithm_usage = AlgorithmRunUsage{80u, 5u, std::chrono::milliseconds{200}};
    record.algorithm_seed = 7u;
    record.optimizer_seed = 11u;
    record.message = "ok";

    const auto serialized = serialize_run_record(record);
    HPOEA_V2_CHECK(runner, serialized.find("\"schema_version\":3") != std::string::npos,
                   "schema_version serialized");
    HPOEA_V2_CHECK(runner, serialized.find("\"experiment_id\":\"exp_v2\"") != std::string::npos,
                   "experiment_id serialized");
    HPOEA_V2_CHECK(runner, serialized.find("\"objective_value\":1.25") != std::string::npos,
                   "objective_value serialized");
    HPOEA_V2_CHECK(runner, serialized.find("\"optimizer_seed\":11") != std::string::npos,
                   "optimizer_seed serialized");

    record.objective_value = std::numeric_limits<double>::infinity();
    const auto serialized_inf = serialize_run_record(record);
    HPOEA_V2_CHECK(runner, serialized_inf.find("\"objective_value\":null") != std::string::npos,
                   "non-finite objective serialized as null");

    record.objective_value = 2.5;
    record.experiment_id = "exp_\"control\n";
    record.problem_id = "prob\\name\t";
    record.message = "line1\nline2\t\"quoted\"";
    record.algorithm_parameters = {
        {"param\"name", std::string{"value\\data\n"}},
        {"control", std::string{"\b\f\r"}},
    };
    const auto serialized_escape = serialize_run_record(record);
    HPOEA_V2_CHECK(runner, serialized_escape.find("exp_\\\"control\\n") != std::string::npos,
                   "escape_json handles quotes/newlines in ids");
    HPOEA_V2_CHECK(runner, serialized_escape.find("prob\\\\name\\t") != std::string::npos,
                   "escape_json handles backslashes/tabs in ids");
    HPOEA_V2_CHECK(runner, serialized_escape.find("line1\\nline2\\t\\\"quoted\\\"") != std::string::npos,
                   "escape_json handles message control characters");
    HPOEA_V2_CHECK(runner, serialized_escape.find("\\b") != std::string::npos,
                   "escape_json emits backspace escape");
    HPOEA_V2_CHECK(runner, serialized_escape.find("\\f") != std::string::npos,
                   "escape_json emits formfeed escape");
    HPOEA_V2_CHECK(runner, serialized_escape.find("\\r") != std::string::npos,
                   "escape_json emits carriage return escape");

    record.objective_value = 1.25;
    record.error_info = ErrorInfo{"validation", "bad_input", "detail"};
    const auto serialized_error = serialize_run_record(record);
    HPOEA_V2_CHECK(runner,
                   serialized_error.find("\"error_info\":{\"category\":\"validation\"") != std::string::npos,
                   "error_info object serialized");

    const auto log_path = unique_test_path("logging_v2_test");
    if (std::filesystem::exists(log_path)) {
        std::filesystem::remove(log_path);
    }

    try {
        JsonlLogger logger(log_path);
        record.error_info.reset();
        logger.log(record);
        logger.flush();

        HPOEA_V2_CHECK(runner, logger.records_written() == 1u,
                       "logger increments records_written");
        HPOEA_V2_CHECK(runner, std::filesystem::exists(log_path),
                       "logger creates file");

        std::ifstream in(log_path);
        std::string line;
        std::getline(in, line);
        HPOEA_V2_CHECK(runner, line == serialize_run_record(record),
                       "log file content matches serialize_run_record");
    } catch (const std::exception &ex) {
        HPOEA_V2_CHECK(runner, false, std::string("unexpected logger exception: ") + ex.what());
    }

    if (std::filesystem::exists(log_path)) {
        std::filesystem::remove(log_path);
    }


    {
        RunRecord rt;
        rt.experiment_id = "round_trip_test_42";
        rt.problem_id = "ackley_10d";
        rt.evolutionary_algorithm = {"PSO", "pagmo::pso", "3.0"};
        rt.hyper_optimizer = AlgorithmIdentity{"SA", "pagmo::sa", "3.0"};
        rt.algorithm_parameters = {
            {"weight",     2.718},
            {"max_iter",   std::int64_t{7}},
            {"adaptive",   true},
            {"variant",    std::string{"exponential"}},
        };
        rt.optimizer_parameters = {
            {"cooling_rate", 0.95},
        };
        rt.status = RunStatus::Success;
        rt.objective_value = 3.14159;
        rt.requested_budget = Budget{5000u, 100u, std::chrono::milliseconds{3000}};
        rt.effective_budget = EffectiveBudget{5000u, 100u, std::chrono::milliseconds{3000}};
        rt.algorithm_usage = AlgorithmRunUsage{1234u, 56u, std::chrono::milliseconds{789}};
        rt.error_info = ErrorInfo{"config_error", "E001", "value \"out\" of\trange\n"};
        rt.algorithm_seed = 42;
        rt.optimizer_seed = 99u;
        rt.message = "round-trip verification";

        const auto rt_json = serialize_run_record(rt);
        HPOEA_V2_CHECK(runner, count_occurrences(rt_json, "\"schema_version\":3") == 1u,
                        "rt: schema_version present exactly once");
        HPOEA_V2_CHECK(runner, count_occurrences(rt_json, "\"experiment_id\":\"round_trip_test_42\"") == 1u,
                        "rt: experiment_id present exactly once");
        HPOEA_V2_CHECK(runner, count_occurrences(rt_json, "\"problem_id\":\"ackley_10d\"") == 1u,
                        "rt: problem_id present exactly once");
        HPOEA_V2_CHECK(runner, has_numeric_field_close(rt_json, "objective_value", 3.14159, 1e-12),
                        "rt: objective_value numeric value");
        HPOEA_V2_CHECK(runner, rt_json.find("\"algorithm_seed\":42") != std::string::npos,
                        "rt: algorithm_seed value");
        HPOEA_V2_CHECK(runner, rt_json.find("\"optimizer_seed\":99") != std::string::npos,
                        "rt: optimizer_seed value");


        HPOEA_V2_CHECK(runner, rt_json.find("\"family\":\"PSO\"") != std::string::npos,
                        "rt: ea family");
        HPOEA_V2_CHECK(runner, rt_json.find("\"implementation\":\"pagmo::pso\"") != std::string::npos,
                        "rt: ea implementation");
        HPOEA_V2_CHECK(runner, rt_json.find("\"family\":\"SA\"") != std::string::npos,
                        "rt: hyper family");


        HPOEA_V2_CHECK(runner, rt_json.find("\"adaptive\":true") != std::string::npos,
                        "rt: bool parameter true");
        HPOEA_V2_CHECK(runner, rt_json.find("\"max_iter\":7") != std::string::npos,
                        "rt: int64 parameter");
        HPOEA_V2_CHECK(runner, rt_json.find("\"variant\":\"exponential\"") != std::string::npos,
                        "rt: string parameter");
        HPOEA_V2_CHECK(runner, has_numeric_field_close(rt_json, "weight", 2.718, 1e-12),
                        "rt: double parameter");
        HPOEA_V2_CHECK(runner,
                        has_numeric_field_close(rt_json, "cooling_rate", 0.95, 1e-12),
                        "rt: optimizer parameter numeric value");


        HPOEA_V2_CHECK(runner, rt_json.find("\"function_evaluations\":1234") != std::string::npos,
                        "rt: algorithm_usage function_evaluations");
        HPOEA_V2_CHECK(runner, rt_json.find("\"generations\":56") != std::string::npos,
                        "rt: algorithm_usage generations");
        HPOEA_V2_CHECK(runner, rt_json.find("\"wall_time_ms\":789") != std::string::npos,
                        "rt: algorithm_usage wall_time_ms");


        HPOEA_V2_CHECK(runner, rt_json.find("\"category\":\"config_error\"") != std::string::npos,
                        "rt: error_info category");
        HPOEA_V2_CHECK(runner, rt_json.find("\"code\":\"E001\"") != std::string::npos,
                        "rt: error_info code");
        HPOEA_V2_CHECK(runner, rt_json.find("\\\"out\\\"") != std::string::npos,
                        "rt: error_info detail escapes quotes");
        HPOEA_V2_CHECK(runner, rt_json.find("of\\trange\\n") != std::string::npos,
                        "rt: error_info detail escapes tab/newline");


        HPOEA_V2_CHECK(runner, rt_json.find("\"status\":\"success\"") != std::string::npos,
                        "rt: status field");


        HPOEA_V2_CHECK(runner, rt_json.find("round-trip verification") != std::string::npos,
                        "rt: message field");


        HPOEA_V2_CHECK(runner, !rt_json.empty() && rt_json.front() == '{',
                        "rt: json starts with {");
        HPOEA_V2_CHECK(runner, !rt_json.empty() && rt_json.back() == '}',
                        "rt: json ends with }");


        HPOEA_V2_CHECK(runner, !has_raw_control_character(rt_json),
                        "rt: no unescaped control characters in output");


        int brace_depth = 0;
        int bracket_depth = 0;
        bool in_string = false;
        bool escaped = false;
        for (const auto ch : rt_json) {
            if (escaped) {
                escaped = false;
                continue;
            }
            if (ch == '\\' && in_string) {
                escaped = true;
                continue;
            }
            if (ch == '"') {
                in_string = !in_string;
                continue;
            }
            if (!in_string) {
                if (ch == '{') ++brace_depth;
                else if (ch == '}') --brace_depth;
                else if (ch == '[') ++bracket_depth;
                else if (ch == ']') --bracket_depth;
            }
        }
        HPOEA_V2_CHECK(runner, brace_depth == 0,
                        "rt: balanced braces");
        HPOEA_V2_CHECK(runner, bracket_depth == 0,
                        "rt: balanced brackets");


        const auto rt_log_path = unique_test_path("rt_serialization_test");
        if (std::filesystem::exists(rt_log_path)) {
            std::filesystem::remove(rt_log_path);
        }

        try {
            JsonlLogger rt_logger(rt_log_path);
            rt_logger.log(rt);
            rt_logger.flush();

            HPOEA_V2_CHECK(runner, rt_logger.records_written() == 1u,
                            "rt: logger wrote one record");

            const auto rt_lines = read_lines(rt_log_path);
            HPOEA_V2_CHECK(runner, rt_lines.size() == 1u,
                            "rt: file contains exactly one JSONL line");
            HPOEA_V2_CHECK(runner, !rt_lines.empty() && rt_lines.front() == rt_json,
                            "rt: file line matches serialize_run_record exactly");
        } catch (const std::exception &ex) {
            HPOEA_V2_CHECK(runner, false,
                            std::string("rt: unexpected logger exception: ") + ex.what());
        }

        if (std::filesystem::exists(rt_log_path)) {
            std::filesystem::remove(rt_log_path);
        }
    }


    {
        RunRecord record;
        record.experiment_id = "null_test";
        record.problem_id = "sphere";
        record.evolutionary_algorithm = {"DE", "pagmo::de", "2.x"};
        record.hyper_optimizer = std::nullopt;
        record.status = RunStatus::Success;
        record.objective_value = 1.0;
        record.optimizer_seed = std::nullopt;
        record.message = "test";

        const auto json = serialize_run_record(record);
        HPOEA_V2_CHECK(runner, json.find("\"hyper_optimizer\":null") != std::string::npos,
                       "nullopt hyper_optimizer serialized as null");
        HPOEA_V2_CHECK(runner, json.find("\"optimizer_seed\":null") != std::string::npos,
                       "nullopt optimizer_seed serialized as null");
    }


    {
        auto check_status = [&](RunStatus status, const std::string &expected) {
            RunRecord record;
            record.experiment_id = "status_test";
            record.problem_id = "sphere";
            record.evolutionary_algorithm = {"DE", "pagmo::de", "2.x"};
            record.status = status;
            record.objective_value = 0.0;
            record.message = "test";
            const auto json = serialize_run_record(record);
            const auto needle = "\"status\":\"" + expected + "\"";
            HPOEA_V2_CHECK(runner, json.find(needle) != std::string::npos,
                           "status serialization: " + expected);
        };
        check_status(RunStatus::Success, "success");
        check_status(RunStatus::BudgetExceeded, "budget_exceeded");
        check_status(RunStatus::FailedEvaluation, "failed_evaluation");
        check_status(RunStatus::InvalidConfiguration, "invalid_configuration");
        check_status(RunStatus::InternalError, "internal_error");
    }


    {
        const auto path = unique_test_path("logger_api_test");
        if (std::filesystem::exists(path)) std::filesystem::remove(path);

        try {
            JsonlLogger logger(path);
            HPOEA_V2_CHECK(runner, logger.good(), "logger good() returns true after open");
            HPOEA_V2_CHECK(runner, logger.path() == path, "logger path() returns expected path");
        } catch (const std::exception &ex) {
            HPOEA_V2_CHECK(runner, false, std::string("logger api test failed: ") + ex.what());
        }

        if (std::filesystem::exists(path)) std::filesystem::remove(path);
    }


    {
        const auto path = unique_test_path("logger_multi_test");
        if (std::filesystem::exists(path)) std::filesystem::remove(path);

        try {
            JsonlLogger logger(path);

            RunRecord r;
            r.experiment_id = "multi";
            r.problem_id = "sphere";
            r.evolutionary_algorithm = {"DE", "pagmo::de", "2.x"};
            r.status = RunStatus::Success;
            r.objective_value = 1.0;
            r.message = "record";

            logger.log(r);
            r.objective_value = 2.0;
            logger.log(r);
            r.objective_value = 3.0;
            logger.log(r);
            logger.flush();

            HPOEA_V2_CHECK(runner, logger.records_written() == 3u,
                           "logger multi: records_written is 3");


            const auto lines = read_lines(path);
            HPOEA_V2_CHECK(runner, lines.size() == 3u,
                           "logger multi: file has 3 lines");
            r.objective_value = 1.0;
            HPOEA_V2_CHECK(runner, !lines.empty() && lines[0] == serialize_run_record(r),
                           "logger multi: first line matches first record exactly");
            r.objective_value = 2.0;
            HPOEA_V2_CHECK(runner, lines.size() > 1u && lines[1] == serialize_run_record(r),
                           "logger multi: second line matches second record exactly");
            r.objective_value = 3.0;
            HPOEA_V2_CHECK(runner, lines.size() > 2u && lines[2] == serialize_run_record(r),
                           "logger multi: third line matches third record exactly");
        } catch (const std::exception &ex) {
            HPOEA_V2_CHECK(runner, false, std::string("logger multi test failed: ") + ex.what());
        }

        if (std::filesystem::exists(path)) std::filesystem::remove(path);
    }


    {
        const auto path = unique_test_path("logger_noflush_test");
        if (std::filesystem::exists(path)) std::filesystem::remove(path);

        try {
            std::string expected_line;
            {
                JsonlLogger logger(path, false);

                RunRecord r;
                r.experiment_id = "noflush";
                r.problem_id = "sphere";
                r.evolutionary_algorithm = {"DE", "pagmo::de", "2.x"};
                r.status = RunStatus::Success;
                r.objective_value = 1.0;
                r.message = "test";
                expected_line = serialize_run_record(r);

                logger.log(r);
                HPOEA_V2_CHECK(runner, logger.records_written() == 1u,
                               "noflush: records_written incremented even without flush");


                logger.flush();
            }


            const auto lines = read_lines(path);
            HPOEA_V2_CHECK(runner, lines.size() == 1u, "noflush: exactly one line present after flush and close");
            HPOEA_V2_CHECK(runner, !lines.empty() && lines.front() == expected_line,
                           "noflush: line matches logged record exactly");
        } catch (const std::exception &ex) {
            HPOEA_V2_CHECK(runner, false, std::string("noflush test failed: ") + ex.what());
        }

        if (std::filesystem::exists(path)) std::filesystem::remove(path);
    }


    {
        RunRecord record;
        record.experiment_id = "nan_test";
        record.problem_id = "sphere";
        record.evolutionary_algorithm = {"DE", "pagmo::de", "2.x"};
        record.status = RunStatus::Success;
        record.objective_value = std::numeric_limits<double>::quiet_NaN();
        record.message = "test";
        const auto json = serialize_run_record(record);
        HPOEA_V2_CHECK(runner, json.find("\"objective_value\":null") != std::string::npos,
                       "NaN objective serialized as null");
    }


    {
        auto path = unique_test_path("concurrent_log");
        constexpr int num_threads = 4;
        constexpr int records_per_thread = 50;
        std::set<std::string> expected_lines;
        auto make_record = [](int thread_id, int index) {
            RunRecord r;
            r.experiment_id = "concurrent_" + std::to_string(thread_id);
            r.problem_id = "sphere";
            r.evolutionary_algorithm = {"DE", "pagmo::de", "2.x"};
            r.status = RunStatus::Success;
            r.objective_value = static_cast<double>(thread_id * 100 + index);
            r.message = "thread " + std::to_string(thread_id) + " record " + std::to_string(index);
            return r;
        };
        for (int t = 0; t < num_threads; ++t) {
            for (int i = 0; i < records_per_thread; ++i) {
                expected_lines.insert(serialize_run_record(make_record(t, i)));
            }
        }

        {
            JsonlLogger logger(path, true);

            auto writer = [&](int thread_id) {
                for (int i = 0; i < records_per_thread; ++i) {
                    logger.log(make_record(thread_id, i));
                }
            };

            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; ++t) {
                threads.emplace_back(writer, t);
            }
            for (auto &t : threads) {
                t.join();
            }

            HPOEA_V2_CHECK(runner,
                           logger.records_written() == num_threads * records_per_thread,
                           "concurrent logger records_written matches expected count");
            HPOEA_V2_CHECK(runner, logger.good(),
                           "concurrent logger stream still good after parallel writes");
        }


        const auto lines = read_lines(path);
        std::set<std::string> actual_lines(lines.begin(), lines.end());
        bool all_valid_json = true;
        bool all_schema_v3 = true;
        bool no_raw_controls = true;
        for (const auto &line : lines) {
            if (line.empty() || line.front() != '{' || line.back() != '}') {
                all_valid_json = false;
            }
            if (line.find("\"schema_version\":3") == std::string::npos) {
                all_schema_v3 = false;
            }
            if (has_raw_control_character(line)) {
                no_raw_controls = false;
            }
        }
        HPOEA_V2_CHECK(runner, lines.size() == expected_lines.size(),
                       "concurrent log file has exactly 200 lines");
        HPOEA_V2_CHECK(runner, all_valid_json,
                       "all concurrent log lines are JSON object-shaped");
        HPOEA_V2_CHECK(runner, all_schema_v3,
                       "all concurrent log lines include schema_version 3");
        HPOEA_V2_CHECK(runner, no_raw_controls,
                       "all concurrent log lines avoid raw control characters");
        HPOEA_V2_CHECK(runner, actual_lines.size() == expected_lines.size(),
                       "concurrent log file has no duplicate records");
        HPOEA_V2_CHECK(runner, actual_lines == expected_lines,
                       "concurrent log file contains every expected record exactly once");
        std::filesystem::remove(path);
    }

    return runner.summarize("logging_tests");
}
