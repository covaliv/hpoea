#include "test_harness.hpp"
#include "test_utils.hpp"

#include "hpoea/core/parameters.hpp"
#include "hpoea/core/search_space.hpp"
#include "hpoea/wrappers/pagmo/de_algorithm.hpp"
#include "hpoea/wrappers/pagmo/sade_algorithm.hpp"
#include "hpoea/wrappers/problems/benchmark_problems.hpp"

#include "hyper_tuning_udp.hpp"

#include <cmath>

int main() {
    hpoea::tests_v2::TestRunner runner;

    hpoea::wrappers::problems::SphereProblem problem(3);
    hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory factory;


    auto search = std::make_shared<hpoea::core::SearchSpace>();
    search->fix("population_size", std::int64_t{20});
    search->optimize("scaling_factor", hpoea::core::ContinuousRange{0.4, 0.6});
    search->optimize("crossover_rate", hpoea::core::ContinuousRange{0.8, 0.9});

    auto ctx = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
    ctx->factory = &factory;
    ctx->problem = &problem;
    ctx->algorithm_budget.generations = 2;
    ctx->base_seed = 42UL;
    ctx->trials = std::make_shared<std::vector<hpoea::core::HyperparameterTrialRecord>>();
    ctx->search_space = search;

    hpoea::pagmo_wrappers::HyperparameterTuningProblem udp(ctx);
    auto bounds = udp.get_bounds();
    const std::size_t expected_dim = bounds.first.size();

    HPOEA_V2_CHECK(runner, expected_dim == 6u,
                   "bounds dimension excludes fixed parameters but includes all tunable descriptors");


    pagmo::vector_double candidate(expected_dim, 0.0);
    candidate[0] = 0.5;
    candidate[1] = 0.85;
    candidate[2] = 2.0;
    candidate[3] = 5.0;
    candidate[4] = 1e-6;
    candidate[5] = 1e-6;
    const auto fitness = udp.fitness(candidate);
    HPOEA_V2_CHECK(runner, fitness.size() == 1u, "fitness returns single objective");

    HPOEA_V2_CHECK(runner, !ctx->trials->empty(), "trial recorded in context");
    if (!ctx->trials->empty()) {
        const auto &trial = ctx->trials->front();
        HPOEA_V2_CHECK(runner, std::get<std::int64_t>(trial.parameters.at("population_size")) == 20,
                       "fixed parameter propagated");
        const auto sf = std::get<double>(trial.parameters.at("scaling_factor"));
        const auto cr = std::get<double>(trial.parameters.at("crossover_rate"));
        HPOEA_V2_CHECK(runner, sf >= 0.4 && sf <= 0.6, "scaling_factor clamped to bounds");
        HPOEA_V2_CHECK(runner, cr >= 0.8 && cr <= 0.9, "crossover_rate clamped to bounds");
    }


    bool threw = false;
    try {
        (void)udp.fitness({0.1});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    HPOEA_V2_CHECK(runner, threw, "candidate dimension mismatch throws");
















    {
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        auto ss = std::make_shared<hpoea::core::SearchSpace>();
        ss->fix("population_size", std::int64_t{10});
        ss->optimize("scaling_factor", hpoea::core::ContinuousRange{0.4, 0.6});
        ss->optimize("crossover_rate", hpoea::core::ContinuousRange{0.3, 0.9});

        auto c = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
        c->factory = &de_factory;
        c->problem = &problem;
        c->algorithm_budget.generations = 2;
        c->base_seed = 1UL;
        c->trials = std::make_shared<std::vector<hpoea::core::HyperparameterTrialRecord>>();
        c->search_space = ss;

        hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_bv(c);
        auto bv = udp_bv.get_bounds();

        HPOEA_V2_CHECK(runner, bv.first.size() == 6u && bv.second.size() == 6u,
                       "bounds_verification: dimension is 6 (7 descriptors minus 1 fixed)");

        using hpoea::tests_v2::nearly_equal;

        HPOEA_V2_CHECK(runner, nearly_equal(bv.first[0], 0.3) && nearly_equal(bv.second[0], 0.9),
                       "bounds_verification: crossover_rate bounds [0.3, 0.9]");

        HPOEA_V2_CHECK(runner, nearly_equal(bv.first[1], 0.4) && nearly_equal(bv.second[1], 0.6),
                       "bounds_verification: scaling_factor bounds [0.4, 0.6]");

        HPOEA_V2_CHECK(runner, nearly_equal(bv.first[2], 1.0) && nearly_equal(bv.second[2], 10.0),
                       "bounds_verification: variant bounds [1, 10]");

        HPOEA_V2_CHECK(runner, nearly_equal(bv.first[3], 1.0) && nearly_equal(bv.second[3], 1000.0),
                       "bounds_verification: generations bounds [1, 1000]");

        HPOEA_V2_CHECK(runner, nearly_equal(bv.first[4], 0.0) && nearly_equal(bv.second[4], 1.0),
                       "bounds_verification: ftol bounds [0, 1]");

        HPOEA_V2_CHECK(runner, nearly_equal(bv.first[5], 0.0) && nearly_equal(bv.second[5], 1.0),
                       "bounds_verification: xtol bounds [0, 1]");
    }







    {
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        hpoea::wrappers::problems::SphereProblem sphere2(2);


        auto ss = std::make_shared<hpoea::core::SearchSpace>();
        ss->fix("population_size", std::int64_t{10});
        ss->fix("crossover_rate", 0.9);
        ss->fix("scaling_factor", 0.5);
        ss->fix("generations", std::int64_t{2});
        ss->fix("ftol", 1e-6);
        ss->fix("xtol", 1e-6);


        auto c = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
        c->factory = &de_factory;
        c->problem = &sphere2;
        c->algorithm_budget.generations = 2;
        c->base_seed = 7UL;
        c->trials = std::make_shared<std::vector<hpoea::core::HyperparameterTrialRecord>>();
        c->search_space = ss;

        hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_int(c);
        auto int_bounds = udp_int.get_bounds();
        HPOEA_V2_CHECK(runner, int_bounds.first.size() == 1u,
                       "integer_snapping: single dimension for variant");


        c->trials->clear();
        (void)udp_int.fitness({2.3});
        HPOEA_V2_CHECK(runner, !c->trials->empty(), "integer_snapping: trial recorded for 2.3");
        if (!c->trials->empty()) {
            auto val = std::get<std::int64_t>(c->trials->back().parameters.at("variant"));
            HPOEA_V2_CHECK(runner, val == 2, "integer_snapping: 2.3 rounds to 2");
        }


        c->trials->clear();
        (void)udp_int.fitness({2.5});
        HPOEA_V2_CHECK(runner, !c->trials->empty(), "integer_snapping: trial recorded for 2.5");
        if (!c->trials->empty()) {
            auto val = std::get<std::int64_t>(c->trials->back().parameters.at("variant"));
            HPOEA_V2_CHECK(runner, val == 3, "integer_snapping: 2.5 rounds to 3 (half away from zero)");
        }


        c->trials->clear();
        (void)udp_int.fitness({2.7});
        HPOEA_V2_CHECK(runner, !c->trials->empty(), "integer_snapping: trial recorded for 2.7");
        if (!c->trials->empty()) {
            auto val = std::get<std::int64_t>(c->trials->back().parameters.at("variant"));
            HPOEA_V2_CHECK(runner, val == 3, "integer_snapping: 2.7 rounds to 3");
        }
    }






    {
        hpoea::pagmo_wrappers::PagmoSelfAdaptiveDEFactory sade_factory;
        hpoea::wrappers::problems::SphereProblem sphere2(2);




        auto ss = std::make_shared<hpoea::core::SearchSpace>();
        ss->fix("population_size", std::int64_t{10});
        ss->fix("generations", std::int64_t{2});
        ss->fix("variant", std::int64_t{2});
        ss->fix("variant_adptv", std::int64_t{1});
        ss->fix("ftol", 1e-6);
        ss->fix("xtol", 1e-6);


        auto c = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
        c->factory = &sade_factory;
        c->problem = &sphere2;
        c->algorithm_budget.generations = 2;
        c->base_seed = 13UL;
        c->trials = std::make_shared<std::vector<hpoea::core::HyperparameterTrialRecord>>();
        c->search_space = ss;

        hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_bool(c);
        auto bool_bounds = udp_bool.get_bounds();
        HPOEA_V2_CHECK(runner, bool_bounds.first.size() == 1u,
                       "boolean_threshold: single dimension for memory");
        HPOEA_V2_CHECK(runner,
                       hpoea::tests_v2::nearly_equal(bool_bounds.first[0], 0.0) &&
                       hpoea::tests_v2::nearly_equal(bool_bounds.second[0], 1.0),
                       "boolean_threshold: memory bounds [0, 1]");


        c->trials->clear();
        (void)udp_bool.fitness({0.49});
        HPOEA_V2_CHECK(runner, !c->trials->empty(), "boolean_threshold: trial recorded for 0.49");
        if (!c->trials->empty()) {
            auto val = std::get<bool>(c->trials->back().parameters.at("memory"));
            HPOEA_V2_CHECK(runner, val == false, "boolean_threshold: 0.49 decodes to false");
        }


        c->trials->clear();
        (void)udp_bool.fitness({0.5});
        HPOEA_V2_CHECK(runner, !c->trials->empty(), "boolean_threshold: trial recorded for 0.5");
        if (!c->trials->empty()) {
            auto val = std::get<bool>(c->trials->back().parameters.at("memory"));
            HPOEA_V2_CHECK(runner, val == false, "boolean_threshold: 0.5 decodes to false (strictly greater)");
        }


        c->trials->clear();
        (void)udp_bool.fitness({0.51});
        HPOEA_V2_CHECK(runner, !c->trials->empty(), "boolean_threshold: trial recorded for 0.51");
        if (!c->trials->empty()) {
            auto val = std::get<bool>(c->trials->back().parameters.at("memory"));
            HPOEA_V2_CHECK(runner, val == true, "boolean_threshold: 0.51 decodes to true");
        }
    }






    {
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        hpoea::wrappers::problems::SphereProblem sphere2(2);


        auto ctx_all = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
        ctx_all->factory = &de_factory;
        ctx_all->problem = &sphere2;
        ctx_all->algorithm_budget.generations = 2;
        ctx_all->base_seed = 99UL;
        ctx_all->trials = std::make_shared<std::vector<hpoea::core::HyperparameterTrialRecord>>();


        hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_all(ctx_all);
        auto bounds_all = udp_all.get_bounds();
        const auto dim_all = bounds_all.first.size();
        HPOEA_V2_CHECK(runner, dim_all == 7u,
                       "fixed_excluded: baseline dimension is 7 (all DE descriptors)");


        auto ss_fix = std::make_shared<hpoea::core::SearchSpace>();
        ss_fix->fix("variant", std::int64_t{5});

        auto ctx_fix = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
        ctx_fix->factory = &de_factory;
        ctx_fix->problem = &sphere2;
        ctx_fix->algorithm_budget.generations = 2;
        ctx_fix->base_seed = 99UL;
        ctx_fix->trials = std::make_shared<std::vector<hpoea::core::HyperparameterTrialRecord>>();
        ctx_fix->search_space = ss_fix;

        hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_fix(ctx_fix);
        auto bounds_fix = udp_fix.get_bounds();
        const auto dim_fix = bounds_fix.first.size();
        HPOEA_V2_CHECK(runner, dim_fix == dim_all - 1,
                       "fixed_excluded: fixing one param reduces dimension by 1");








        pagmo::vector_double cand_fix(dim_fix);
        cand_fix[0] = 10.0;
        cand_fix[1] = 0.9;
        cand_fix[2] = 0.5;
        cand_fix[3] = 5.0;
        cand_fix[4] = 1e-6;
        cand_fix[5] = 1e-6;

        (void)udp_fix.fitness(cand_fix);
        HPOEA_V2_CHECK(runner, !ctx_fix->trials->empty(),
                       "fixed_excluded: trial recorded after fitness call");
        if (!ctx_fix->trials->empty()) {
            const auto &params = ctx_fix->trials->back().parameters;
            HPOEA_V2_CHECK(runner, params.count("variant") == 1,
                           "fixed_excluded: fixed param 'variant' present in decoded ParameterSet");
            auto variant_val = std::get<std::int64_t>(params.at("variant"));
            HPOEA_V2_CHECK(runner, variant_val == 5,
                           "fixed_excluded: fixed variant value is 5");
        }
    }






    {
        hpoea::pagmo_wrappers::PagmoDifferentialEvolutionFactory de_factory;
        hpoea::wrappers::problems::SphereProblem sphere2(2);

        auto ss = std::make_shared<hpoea::core::SearchSpace>();
        ss->fix("population_size", std::int64_t{10});
        ss->fix("crossover_rate", 0.9);
        ss->fix("variant", std::int64_t{2});
        ss->fix("generations", std::int64_t{2});
        ss->fix("ftol", 1e-6);
        ss->fix("xtol", 1e-6);

        ss->optimize("scaling_factor",
                     hpoea::core::ContinuousRange{0.01, 1.0},
                     hpoea::core::Transform::log);

        auto c = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
        c->factory = &de_factory;
        c->problem = &sphere2;
        c->algorithm_budget.generations = 2;
        c->base_seed = 77UL;
        c->trials = std::make_shared<std::vector<hpoea::core::HyperparameterTrialRecord>>();
        c->search_space = ss;

        hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_log(c);
        auto log_bounds = udp_log.get_bounds();
        HPOEA_V2_CHECK(runner, log_bounds.first.size() == 1u,
                       "log_transform: single dimension for scaling_factor");

        using hpoea::tests_v2::nearly_equal;

        HPOEA_V2_CHECK(runner,
                       nearly_equal(log_bounds.first[0], -2.0, 1e-9),
                       "log_transform: lower bound is log10(0.01) = -2.0");
        HPOEA_V2_CHECK(runner,
                       nearly_equal(log_bounds.second[0], 0.0, 1e-9),
                       "log_transform: upper bound is log10(1.0) = 0.0");


        c->trials->clear();
        (void)udp_log.fitness({-1.0});
        HPOEA_V2_CHECK(runner, !c->trials->empty(),
                       "log_transform: trial recorded for candidate -1.0");
        if (!c->trials->empty()) {
            auto sf = std::get<double>(c->trials->back().parameters.at("scaling_factor"));
            HPOEA_V2_CHECK(runner, nearly_equal(sf, 0.1, 1e-9),
                           "log_transform: candidate -1.0 decodes to scaling_factor=0.1");
        }
    }






    {

        struct CatFactory final : public hpoea::core::IEvolutionaryAlgorithmFactory {
            hpoea::core::ParameterSpace space_;
            hpoea::core::AlgorithmIdentity id_{"CatStub", "tests", "1.0"};

            explicit CatFactory(std::vector<std::string> choices) {
                hpoea::core::ParameterDescriptor desc;
                desc.name = "strategy";
                desc.type = hpoea::core::ParameterType::Categorical;
                desc.categorical_choices = std::move(choices);
                space_.add_descriptor(desc);
            }

            [[nodiscard]] hpoea::core::EvolutionaryAlgorithmPtr create() const override {
                return nullptr;
            }
            [[nodiscard]] const hpoea::core::ParameterSpace &parameter_space() const noexcept override {
                return space_;
            }
            [[nodiscard]] const hpoea::core::AlgorithmIdentity &identity() const noexcept override {
                return id_;
            }
        };


        {
            CatFactory fac({"a"});
            hpoea::wrappers::problems::SphereProblem sphere(2);

            auto c = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
            c->factory = &fac;
            c->problem = &sphere;
            c->algorithm_budget.generations = 1;
            c->base_seed = 0UL;

            hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_cat(c);
            auto cat_bounds = udp_cat.get_bounds();

            HPOEA_V2_CHECK(runner, cat_bounds.first.size() == 1u,
                           "categorical_bounds: single-choice gives dimension 1");
            HPOEA_V2_CHECK(runner,
                           hpoea::tests_v2::nearly_equal(cat_bounds.first[0], 0.0) &&
                           hpoea::tests_v2::nearly_equal(cat_bounds.second[0], 0.0),
                           "categorical_bounds: single choice => bounds [0, 0]");
        }


        {
            CatFactory fac({"a", "b", "c"});
            hpoea::wrappers::problems::SphereProblem sphere(2);

            auto c = std::make_shared<hpoea::pagmo_wrappers::HyperparameterTuningProblem::Context>();
            c->factory = &fac;
            c->problem = &sphere;
            c->algorithm_budget.generations = 1;
            c->base_seed = 0UL;

            hpoea::pagmo_wrappers::HyperparameterTuningProblem udp_cat(c);
            auto cat_bounds = udp_cat.get_bounds();

            HPOEA_V2_CHECK(runner, cat_bounds.first.size() == 1u,
                           "categorical_bounds: three-choice gives dimension 1");
            HPOEA_V2_CHECK(runner,
                           hpoea::tests_v2::nearly_equal(cat_bounds.first[0], 0.0) &&
                           hpoea::tests_v2::nearly_equal(cat_bounds.second[0], 2.0),
                           "categorical_bounds: three choices => bounds [0, 2]");
        }

    }

    return runner.summarize("hyper_tuning_udp_tests");
}
