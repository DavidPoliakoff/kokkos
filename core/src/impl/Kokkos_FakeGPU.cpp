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

#include <Kokkos_Core.hpp>
#if defined(KOKKOS_ENABLE_FAKEGPU)

#include <cstdlib>
#include <sstream>
#include <Kokkos_FakeGPU.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>

#include <impl/Kokkos_SharedAlloc.hpp>
#include <sstream>

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
bool FakeGPU::s_on_gpu;
namespace Impl {

using Kokkos::Impl::HostThreadTeamData;
using Kokkos::Impl::SharedAllocationRecord;
using  Kokkos::Impl::initialize_space_factory;
using Kokkos::Impl::init_lock_array_host_space;
namespace {

HostThreadTeamData g_fake_gpu_thread_team_data;

bool g_fake_gpu_is_initialized = false;

}  // namespace

// Resize thread team data scratch memory
void fake_gpu_resize_thread_team_data(size_t pool_reduce_bytes,
                                    size_t team_reduce_bytes,
                                    size_t team_shared_bytes,
                                    size_t thread_local_bytes) {
  if (pool_reduce_bytes < 512) pool_reduce_bytes = 512;
  if (team_reduce_bytes < 512) team_reduce_bytes = 512;

  const size_t old_pool_reduce = g_fake_gpu_thread_team_data.pool_reduce_bytes();
  const size_t old_team_reduce = g_fake_gpu_thread_team_data.team_reduce_bytes();
  const size_t old_team_shared = g_fake_gpu_thread_team_data.team_shared_bytes();
  const size_t old_thread_local =
      g_fake_gpu_thread_team_data.thread_local_bytes();
  const size_t old_alloc_bytes = g_fake_gpu_thread_team_data.scratch_bytes();

  // Allocate if any of the old allocation is tool small:

  const bool allocate = (old_pool_reduce < pool_reduce_bytes) ||
                        (old_team_reduce < team_reduce_bytes) ||
                        (old_team_shared < team_shared_bytes) ||
                        (old_thread_local < thread_local_bytes);

  if (allocate) {
    FakeGPU::memory_space space;

    if (old_alloc_bytes) {
      g_fake_gpu_thread_team_data.disband_team();
      g_fake_gpu_thread_team_data.disband_pool();

      space.deallocate("Kokkos::FakeGPU::scratch_mem",
                       g_fake_gpu_thread_team_data.scratch_buffer(),
                       g_fake_gpu_thread_team_data.scratch_bytes());
    }

    if (pool_reduce_bytes < old_pool_reduce) {
      pool_reduce_bytes = old_pool_reduce;
    }
    if (team_reduce_bytes < old_team_reduce) {
      team_reduce_bytes = old_team_reduce;
    }
    if (team_shared_bytes < old_team_shared) {
      team_shared_bytes = old_team_shared;
    }
    if (thread_local_bytes < old_thread_local) {
      thread_local_bytes = old_thread_local;
    }

    const size_t alloc_bytes =
        HostThreadTeamData::scratch_size(pool_reduce_bytes, team_reduce_bytes,
                                         team_shared_bytes, thread_local_bytes);

    void* ptr = nullptr;
    try {
      ptr = space.allocate("Kokkos::FakeGPU::scratch_mem", alloc_bytes);
    } catch (Kokkos::Experimental::RawMemoryAllocationFailure const& failure) {
      // For now, just rethrow the error message the existing way
      Kokkos::Impl::throw_runtime_exception(failure.get_error_message());
    }

    g_fake_gpu_thread_team_data.scratch_assign(
        ((char*)ptr), alloc_bytes, pool_reduce_bytes, team_reduce_bytes,
        team_shared_bytes, thread_local_bytes);

    HostThreadTeamData* pool[1] = {&g_fake_gpu_thread_team_data};

    g_fake_gpu_thread_team_data.organize_pool(pool, 1);
    g_fake_gpu_thread_team_data.organize_team(1);
  }
}

HostThreadTeamData* fake_gpu_get_thread_team_data() {
  return &g_fake_gpu_thread_team_data;
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {
bool FakeGPU::impl_is_initialized() { return Impl::g_fake_gpu_is_initialized; }

void FakeGPU::impl_initialize() {
  Impl::SharedAllocationRecord<void, void>::tracking_enable();

  // Init the array of locks used for arbitrarily sized atomics
  Impl::init_lock_array_host_space();

  Impl::g_fake_gpu_is_initialized = true;
}

void FakeGPU::impl_finalize() {
  if (Impl::g_fake_gpu_thread_team_data.scratch_buffer()) {
    Impl::g_fake_gpu_thread_team_data.disband_team();
    Impl::g_fake_gpu_thread_team_data.disband_pool();

    FakeGPU::memory_space space;

    space.deallocate(Impl::g_fake_gpu_thread_team_data.scratch_buffer(),
                     Impl::g_fake_gpu_thread_team_data.scratch_bytes());

    Impl::g_fake_gpu_thread_team_data.scratch_assign(nullptr, 0, 0, 0, 0, 0);
  }

  Kokkos::Profiling::finalize();

  Impl::g_fake_gpu_is_initialized = false;
}

const char* FakeGPU::name() { return "FakeGPU"; }

namespace Impl {

int g_fake_gpu_space_factory_initialized =
    initialize_space_factory<FakeGPUSpaceInitializer>("110_FakeGPU");

void FakeGPUSpaceInitializer::initialize(const InitArguments& args) {
  // Prevent "unused variable" warning for 'args' input struct.  If
  // FakeGPU::initialize() ever needs to take arguments from the input
  // struct, you may remove this line of code.
  (void)args;

  // Always initialize FakeGPU if it is configure time enabled
  Kokkos::Experimental::FakeGPU::impl_initialize();
}

void FakeGPUSpaceInitializer::finalize(const bool) {
  if (Kokkos::Experimental::FakeGPU::impl_is_initialized()) Kokkos::Experimental::FakeGPU::impl_finalize();
}

void FakeGPUSpaceInitializer::fence() { Kokkos::Experimental::FakeGPU::impl_static_fence(); }

void FakeGPUSpaceInitializer::print_configuration(std::ostream& msg,
                                                 const bool detail) {
  msg << "Host FakeGPU Execution Space:" << std::endl;
  msg << "  KOKKOS_ENABLE_FAKEGPU: ";
  msg << "yes" << std::endl;

  msg << "FakeGPU Atomics:" << std::endl;
  msg << "  KOKKOS_ENABLE_FAKEGPU_ATOMICS: ";
#ifdef KOKKOS_ENABLE_FAKEGPU_ATOMICS
  msg << "yes" << std::endl;
#else
  msg << "no" << std::endl;
#endif

  msg << "\nFakeGPU Runtime Configuration:" << std::endl;
  FakeGPU::print_configuration(msg, detail);
}

}  // namespace Impl
}  // namespace Experimental
}  // namespace Kokkos

#else
void KOKKOS_CORE_SRC_IMPL_FAKEGPU_PREVENT_LINK_ERROR() {}
#endif  // defined( KOKKOS_ENABLE_FAKEGPU )
