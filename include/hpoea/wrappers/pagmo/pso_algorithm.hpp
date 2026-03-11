#pragma once

#include "hpoea/wrappers/pagmo/algorithm_base.hpp"

namespace hpoea::pagmo_wrappers {

class PagmoParticleSwarmOptimization final : public PagmoAlgorithmBase {
public:
    PagmoParticleSwarmOptimization();

    [[nodiscard]] core::OptimizationResult run(const core::IProblem &problem,
                                               const core::Budget &budget,
                                               unsigned long seed) override;

    [[nodiscard]] std::unique_ptr<core::IEvolutionaryAlgorithm> clone() const override;
};

class PagmoParticleSwarmOptimizationFactory final : public PagmoAlgorithmFactoryBase {
public:
    PagmoParticleSwarmOptimizationFactory();

    [[nodiscard]] core::EvolutionaryAlgorithmPtr create() const override;
};

} // namespace hpoea::pagmo_wrappers
