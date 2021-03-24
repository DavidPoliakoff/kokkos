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

#ifndef KOKKOS_KOKKOS_TRAITS_FWD_HPP
#define KOKKOS_KOKKOS_TRAITS_FWD_HPP

namespace Kokkos {
namespace Impl {

template <class Enable, class... TraitsList>
struct AnalyzeExecPolicy;

template <class AnalysisResults>
struct ExecPolicyTraitsWithDefaults;

template <class TraitSpec, template <class...> class PolicyTemplate,
          class AlreadyProcessedList, class ToProcessList, class NewTrait,
          class Enable = void>
struct PolicyTraitAdaptorImpl;

template <class TraitSpec, class Policy, class NewTrait>
struct PolicyTraitAdaptor;

// A tag class for dependent defaults that must be handled by the
// ExecPolicyTraitsWithDefaults wrapper, since their defaults depend on other
// traits
struct dependent_policy_trait_default;

template <class Enable, class... TraitsList>
struct AnalyzeView;

template <class AnalysisResults>
struct ViewTraitsWithDefaults;


}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_KOKKOS_TRAITS_FWD_HPP
