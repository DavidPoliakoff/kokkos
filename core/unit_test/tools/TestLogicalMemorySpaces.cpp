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

void debug_print(const Kokkos_Profiling_SpaceHandle hand, const char* name,
   const void* ptr, const size_t size) {
  std::cout << "Alloc: " << hand.name << ", [" << name << "," << ptr << "] "
            << size << std::endl;
}
void debug_dealloc(const Kokkos_Profiling_SpaceHandle hand, const char* name,
                   const void* ptr, const size_t size) {
  std::cout << "Dealloc: " << hand.name << ", [" << name << "," << ptr << "] "
            << size << std::endl;
}

void fail_on_event(const Kokkos::Profiling::SpaceHandle hand, const char* name,
                   const void* ptr, const size_t size) {
  std::cout << ":(\n";
  assert(false);
}

void expect_no_events() {
  Kokkos::Tools::Experimental::set_allocate_data_callback(fail_on_event);
  Kokkos::Tools::Experimental::set_deallocate_data_callback(fail_on_event);
}

void expect_allocation_event(const std::string& evn, const std::string& esn) {
  static std::string expected_view_name  = evn;
  static std::string expected_space_name = esn;
  Kokkos::Tools::Experimental::set_allocate_data_callback(
      [](const Kokkos_Profiling_SpaceHandle hand, const char* name,
         const void* ptr, const size_t size) {
        assert(std::string(hand.name) == expected_space_name);
        assert(std::string(name) == expected_view_name);
        expect_no_events();
      });
}
void expect_deallocation_event(const std::string& evn, const std::string& esn) {
  static std::string expected_view_name  = evn;
  static std::string expected_space_name = esn;
  Kokkos::Tools::Experimental::set_deallocate_data_callback(
      [](const Kokkos_Profiling_SpaceHandle hand, const char* name,
         const void* ptr, const size_t size) {
        assert(std::string(hand.name) == expected_space_name);
        assert(std::string(name) == expected_view_name);
        expect_no_events();
      });
}

class TestSpaceNamer {
  static constexpr const char* get_name() { return "TestSpace"; }
};

int main() {
  Kokkos::initialize();
  {
    using fake_memory_space =
        Kokkos::LogicalMemorySpace<Kokkos::HostSpace, Kokkos::Serial, Kokkos::DefaultMemorySpaceNamer, true
                                   >;
    expect_allocation_event("puppy_view", "TestSpace");
    Kokkos::View<double*, fake_memory_space> pup_view("puppy_view", 1000);
    expect_allocation_event("does_malloc_work", "TestSpace");
    auto* temp =
        Kokkos::kokkos_malloc<fake_memory_space>("does_malloc_work", 1000);
    expect_deallocation_event("allocation_from_space", "TestSpace");
    Kokkos::kokkos_free(temp);
    fake_memory_space debug_space;
    expect_allocation_event("allocation_from_space", "TestSpace");
    temp = debug_space.allocate("allocation_from_space", 1000);
    expect_deallocation_event("allocation_from_space", "TestSpace");
    debug_space.deallocate("allocation_from_space", temp, 1000);
    expect_deallocation_event("puppy_view", "TestSpace");
  }
  Kokkos::finalize();
}
