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

// This file calls most of the basic Kokkos primitives. When combined with a
// testing library this tests that our shared-library loading based profiling
// mechanisms work

#include <iostream>
#include <Kokkos_Core.hpp>
#include <Kokkos_LogicalSpaces.hpp>
#include <Kokkos_DualView.hpp>
struct MPrologue {
  static void exec(){};
};
struct MEpilogue {
  static void exec(){};
};

int main() {
  Kokkos::initialize();
  {
    Kokkos::Tools::Experimental::set_allocate_data_callback([](const Kokkos::Tools::SpaceHandle hand, const char* name, const void* const ptr, const size_t size){
      std::cout << hand.name << ", ["<<name<<","<<ptr<<"]"<< std::endl;
		    });
    using fake_exec_space = Kokkos::LogicalExecutionSpace<Kokkos::Serial, void>;
    using multi_fake_exec_space = Kokkos::LogicalExecutionSpace<fake_exec_space>;
    using fake_memory_space =
        Kokkos::LogicalMemorySpace<Kokkos::HostSpace, fake_exec_space,
                                   Kokkos::DefaultMemorySpaceNamer, true>;
    using multi_fake_memory_space = Kokkos::LogicalMemorySpace<fake_memory_space>;
    using ofs = Kokkos::LogicalMemorySpace<Kokkos::HostSpace, Kokkos::Serial, Kokkos::DefaultMemorySpaceNamer, false>;
    Kokkos::View<double*, fake_memory_space> pup_view("pup_view", 1000);
    Kokkos::View<double*, multi_fake_memory_space> oopup_view("oopup_view",1000);
    Kokkos::View<double*, Kokkos::HostSpace> opup_view("opup_view",1000);
std::cout << "================================="<<std::endl;
    Kokkos::DualView<double*,ofs> pup_dual_vew("pup_dv",1000);
std::cout << "================================="<<std::endl;
    std::cout << pup_dual_vew.h_view.data()<<std::endl;
    std::cout << pup_dual_vew.d_view.data()<<std::endl;
    deep_copy(pup_view, opup_view);
    deep_copy(opup_view, pup_view);
    deep_copy(pup_view, pup_view);
    Kokkos::parallel_for(
        "pup_kernel", Kokkos::RangePolicy<multi_fake_exec_space>(0, 1000),
        KOKKOS_LAMBDA(const int i) { pup_view(i) = i; });
  }
  Kokkos::finalize();
}
