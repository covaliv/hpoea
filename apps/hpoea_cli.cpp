#include "hpoea/core/parameters.hpp"

#include <iostream>
#include <variant>

int main() {
    using namespace hpoea::core;

    ParameterSpace space;

    ParameterDescriptor population_size;
    population_size.name = "population_size";
    population_size.type = ParameterType::Integer;
    population_size.integer_range = IntegerRange{10, 1000};
    population_size.default_value = static_cast<std::int64_t>(100);
    population_size.required = true;
    space.add_descriptor(population_size);

    ParameterDescriptor crossover_rate;
    crossover_rate.name = "crossover_rate";
    crossover_rate.type = ParameterType::Continuous;
    crossover_rate.continuous_range = ContinuousRange{0.0, 1.0};
    crossover_rate.default_value = 0.9;
    space.add_descriptor(crossover_rate);

    const auto defaults = space.apply_defaults({});

    std::cout << "hpoea_cli placeholder" << std::endl;
    std::cout << "population_size = " << std::get<std::int64_t>(defaults.at("population_size")) << std::endl;
    std::cout << "crossover_rate = " << std::get<double>(defaults.at("crossover_rate")) << std::endl;

    return 0;
}

