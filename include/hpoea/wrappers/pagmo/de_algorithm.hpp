#pragma once

#include "hpoea/wrappers/pagmo/algorithm_base.hpp"

namespace hpoea::pagmo_wrappers {

class PagmoDifferentialEvolution final : public PagmoAlgorithmBase {
public:
    PagmoDifferentialEvolution();

    [[nodiscard]] core::OptimizationResult run(const core::IProblem &problem,
                                               const core::Budget &budget,
                                               unsigned long seed) override;

    [[nodiscard]] std::unique_ptr<core::IEvolutionaryAlgorithm> clone() const override;
};

class PagmoDifferentialEvolutionFactory final : public PagmoAlgorithmFactoryBase {
public:
    PagmoDifferentialEvolutionFactory();

    [[nodiscard]] core::EvolutionaryAlgorithmPtr create() const override;
};

} // namespace hpoea::pagmo_wrappers
