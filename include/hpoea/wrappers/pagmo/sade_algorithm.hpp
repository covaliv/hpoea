#pragma once

#include "hpoea/wrappers/pagmo/algorithm_base.hpp"

namespace hpoea::pagmo_wrappers {

class PagmoSelfAdaptiveDE final : public PagmoAlgorithmBase {
public:
    PagmoSelfAdaptiveDE();

    [[nodiscard]] core::OptimizationResult run(const core::IProblem &problem,
                                               const core::Budget &budget,
                                               unsigned long seed) override;

    [[nodiscard]] std::unique_ptr<core::IEvolutionaryAlgorithm> clone() const override;
};

class PagmoSelfAdaptiveDEFactory final : public PagmoAlgorithmFactoryBase {
public:
    PagmoSelfAdaptiveDEFactory();

    [[nodiscard]] core::EvolutionaryAlgorithmPtr create() const override;
};

} // namespace hpoea::pagmo_wrappers
