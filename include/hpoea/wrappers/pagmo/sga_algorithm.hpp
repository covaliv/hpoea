#pragma once

#include "hpoea/wrappers/pagmo/algorithm_base.hpp"

namespace hpoea::pagmo_wrappers {

class PagmoSga final : public PagmoAlgorithmBase {
public:
    PagmoSga();

    [[nodiscard]] core::OptimizationResult run(const core::IProblem &problem,
                                               const core::Budget &budget,
                                               unsigned long seed) override;

    [[nodiscard]] std::unique_ptr<core::IEvolutionaryAlgorithm> clone() const override;
};

class PagmoSgaFactory final : public PagmoAlgorithmFactoryBase {
public:
    PagmoSgaFactory();

    [[nodiscard]] core::EvolutionaryAlgorithmPtr create() const override;
};

} // namespace hpoea::pagmo_wrappers
