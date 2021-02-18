
/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <iostream>
#include <map>
#include <string>

std::map<std::string, size_t> fastest_of_ids;

template <int64_t N>
struct cthulhu;

template <>
struct cthulhu<-1> {
  template <class... Args>
  static void execute(Args... args) {}
};

template <int64_t N>
struct cthulhu {
  static auto execute(size_t index) {}
  template <class Arg, class... Args>
  static auto execute(size_t index, Arg arg, Args... args) {
    if (index == 0) {
      return arg();
    } else {
      return cthulhu<N - 1>::execute(index - 1, args...);
    }
  }
};

size_t get_name_id(){
  using namespace Kokkos::Tools::Experimental;
  static bool initialized;
  static size_t id;
  if(!initialized) {
    VariableInfo info;
    info.type = ValueType::kokkos_value_string;
    info.category = StatisticalCategory::kokkos_value_categorical;
    info.valueQuantity = CandidateValueType::kokkos_value_unbounded;
    id = Kokkos::Tools::Experimental::declare_input_type("cthulhu_name", info);
    initialized = true;
  }
  return id;
}
template <class... Kernels>
auto fastest_of(std::string name, Kernels... kernels) {
  constexpr size_t num_kernels = sizeof...(kernels);
  auto id_iter                 = [&]() {
    auto my_tuner = fastest_of_ids.find(name);
    if (my_tuner == fastest_of_ids.end()) {
      Kokkos::Tools::Experimental::VariableInfo info;
      info.type = Kokkos::Tools::Experimental::ValueType::kokkos_value_int64;
      std::array<int64_t, num_kernels> choices;
      for (int x = 0; x < num_kernels; ++x) {
        choices[x] = x;
      }
      info.candidates = Kokkos::Tools::Experimental::make_candidate_set(
          choices.size(), choices.data());
      info.valueQuantity =
          Kokkos::Tools::Experimental::CandidateValueType::kokkos_value_set;
      info.category = Kokkos::Tools::Experimental::StatisticalCategory::
          kokkos_value_categorical;
      auto id = Kokkos::Tools::Experimental::declare_output_type(name, info);
      return (fastest_of_ids.emplace(name, id).first);
    }
    return my_tuner;
  }();
  auto context = Kokkos::Tools::Experimental::get_new_context_id();
  size_t name_id = get_name_id();
  Kokkos::Tools::Experimental::VariableValue name_value = Kokkos::Tools::Experimental::make_variable_value(name_id, name);
  Kokkos::Tools::Experimental::begin_context(context);
  Kokkos::Tools::Experimental::VariableValue default_value =
      Kokkos::Tools::Experimental::make_variable_value(id_iter->second,
                                                       int64_t(0));
  Kokkos::Tools::Experimental::set_input_values(context, 1, &name_value);
  Kokkos::Tools::Experimental::request_output_values(context, 1,
                                                     &default_value);
  cthulhu<num_kernels>::execute(default_value.value.int_value, kernels...);
  Kokkos::Tools::Experimental::end_context(context);
}

int main(int argc, char* argv[]) {
  using namespace Kokkos::Tools::Experimental;
  Kokkos::initialize(argc, argv);
  {
    int num_iters                 = 3500000;
    constexpr const int dimension = 100;
    std::cout << argc << std::endl;
    if (argc > 1) {
      num_iters = atoi(argv[2]);
    }
#if 0 && !defined(KOKKOS_ENABLE_CUDA)
    Kokkos::View<float***, Kokkos::LayoutLeft,
                 Kokkos::DefaultExecutionSpace::memory_space>
        left("left", dimension, dimension, dimension);
    Kokkos::View<float***, Kokkos::LayoutRight,
                 Kokkos::DefaultExecutionSpace::memory_space>
        right("right", dimension, dimension, dimension);
    for (int x = 0; x < num_iters; ++x) {
      if ((x % (num_iters / 100)) == 0) {
        std::cout << x << std::endl;
      }
      Kokkos::deep_copy(right, left);
    }
#else

    Kokkos::View<float***, Kokkos::HostSpace> host("host", dimension, dimension,
                                                   dimension);
    Kokkos::View<float***, Kokkos::CudaSpace> device("device", dimension,
                                                     dimension, dimension);
    for (int x = 0; x < num_iters; ++x) {
      fastest_of(
          "copy_to_gpu",
          [&]() {
            // Transpose on Host
            auto mirror = Kokkos::create_mirror_view(device);
            Kokkos::deep_copy(Kokkos::OpenMP{}, mirror, host);  // transposition kernel
            Kokkos::deep_copy(device, mirror);
          },
          [&]() {
            // Transpose on Device
            auto mirror = Kokkos::create_mirror_view(
                Kokkos::CudaSpace{}, host,
                Kokkos::Impl::WithoutInitializing_t{});
            Kokkos::deep_copy(mirror, host);
            Kokkos::deep_copy(device, mirror);  // transposition kernel
          });
    }
#endif
  }
  Kokkos::finalize();
}
