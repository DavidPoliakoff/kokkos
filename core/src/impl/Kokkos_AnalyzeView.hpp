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

#ifndef KOKKOS_KOKKOS_ANALYZEVIEW_HPP
#define KOKKOS_KOKKOS_ANALYZEVIEW_HPP
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>  // IndexType
#include <traits/Kokkos_Traits_fwd.hpp>
#include <traits/Kokkos_Traits_ViewHookMechanismTrait.hpp>

namespace Kokkos {
namespace Impl {

//------------------------------------------------------------------------------

using view_trait_specifications =
type_list<ViewHookMechanismTrait>;

//------------------------------------------------------------------------------
// Ignore void for backwards compatibility purposes, though hopefully no one is
// using this in application code
template <class... Traits>
struct AnalyzeView<void, void, Traits...>
    : AnalyzeView<void, Traits...> {
  using base_t = AnalyzeView<void, Traits...>;
  using base_t::base_t;
};

//------------------------------------------------------------------------------
// Mix in the defaults (base_traits) for the traits that aren't yet handled

// MSVC workaround: inheriting from more than one base_traits causes EBO to no
// longer work, so we need to linearize the inheritance hierarchy
template <class>
struct msvc_workaround_get_next_base_traits;
template <class T>
struct msvc_workaround_get_next_base_traits {
  template <class... Ts>
  using apply =
  typename T::template base_traits<msvc_workaround_get_next_base_traits,
      Ts...>;
};

template <class TraitSpecList>
struct AnalyzeViewBaseTraits;
template <class... TraitSpecifications>
struct AnalyzeViewBaseTraits<type_list<TraitSpecifications...>>
: linearize_bases<msvc_workaround_get_next_base_traits,
    TraitSpecifications...> {};

template <>
struct AnalyzeView<void>
    : AnalyzeViewBaseTraits<view_trait_specifications> {
  // Ensure default constructibility since a converting constructor causes it to
  // be deleted.
  AnalyzeExecPolicy() = default;

  // Base converting constructor and assignment operator: unless an individual
  // policy analysis deletes a constructor, assume it's convertible
  template <class Other>
  AnalyzeExecPolicy(ViewTraitsWithDefaults<Other> const&) {}

  template <class Other>
  AnalyzeExecPolicy& operator=(ViewTraitsWithDefaults<Other> const&) {
    return *this;
  }
};

//------------------------------------------------------------------------------
// Used for defaults that depend on other analysis results
template <class AnalysisResults>
struct ViewTraitsWithDefaults : AnalysisResults {
  using base_t = AnalysisResults;
  using base_t::base_t;
  // DZP TODO: figure out what this actually means
  //   instead of the wrapped IndexType<T> for backwards compatibility
  using view_hook_mechanism = typename std::conditional_t<
      base_t::view_hook_mechanism_is_defaulted,
      DefaultViewHookMechanism,
      typename base_t::view_hook_mechanism>::type;
};

//------------------------------------------------------------------------------
template <typename... Traits>
struct PolicyTraits
    : ExecPolicyTraitsWithDefaults<AnalyzeView<void, Traits...>> {
  using base_t =
  ExecPolicyTraitsWithDefaults<AnalyzeView<void, Traits...>>;
  using base_t::base_t;
};

}  // namespace Impl
}  // namespace Kokkos


#endif  // KOKKOS_KOKKOS_ANALYZEVIEW_HPP
