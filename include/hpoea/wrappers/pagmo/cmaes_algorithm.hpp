#pragma once

#include "hpoea/wrappers/pagmo/algorithm_base.hpp"

namespace hpoea::pagmo_wrappers {

class PagmoCmaes final : public PagmoAlgorithmBase {
public:
    PagmoCmaes();

    [[nodiscard]] core::OptimizationResult run(const core::IProblem &problem,
                                               const core::Budget &budget,
                                               unsigned long seed) override;

    [[nodiscard]] std::unique_ptr<core::IEvolutionaryAlgorithm> clone() const override;
};

class PagmoCmaesFactory final : public PagmoAlgorithmFactoryBase {
public:
    PagmoCmaesFactory();

    [[nodiscard]] core::EvolutionaryAlgorithmPtr create() const override;
};

} // namespace hpoea::pagmo_wrappers
