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

/// \file Kokkos_FakeGPU.hpp
/// \brief Declaration and definition of Kokkos::FakeGPU device.

#ifndef KOKKOS_FAKEGPU_HPP
#define KOKKOS_FAKEGPU_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_FAKEGPU)

#include <cstddef>
#include <iosfwd>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Parallel.hpp>
//#include <Kokkos_TaskScheduler.hpp>
#include <Kokkos_LogicalSpaces.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <impl/Kokkos_Tags.hpp>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_Tools.hpp>
#include <impl/Kokkos_ExecSpaceInitializer.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <Kokkos_UniqueToken.hpp>

namespace Kokkos {
namespace Experimental {

class FakeGPU;

struct FakeGPUMemSpaceNamer {
	static const char* get_name() { return "FakeGPUSpace"; }
};
struct FakeHostMemSpaceNamer {
  static const char* get_name() { return "FakeHostSpace"; }
};

using FakeGPUSpace = Kokkos::Experimental::LogicalMemorySpace<Kokkos::HostSpace, FakeGPU, FakeGPUMemSpaceNamer, false>;
using FakeHostSpace =
    Kokkos::Experimental::LogicalMemorySpace<Kokkos::HostSpace, Kokkos::Serial,
                                             FakeHostMemSpaceNamer, false>;

/// \class FakeGPU
/// \brief Kokkos device for non-parallel execution
///
/// A "device" represents a parallel execution model.  It tells Kokkos
/// how to parallelize the execution of kernels in a parallel_for or
/// parallel_reduce.  For example, the Threads device uses Pthreads or
/// C++11 threads on a CPU, the OpenMP device uses the OpenMP language
/// extensions, and the Cuda device uses NVIDIA's CUDA programming
/// model.  The FakeGPU device executes "parallel" kernels
/// sequentially.  This is useful if you really do not want to use
/// threads, or if you want to explore different combinations of MPI
/// and shared-memory parallel programming models.
class FakeGPU {
 public:
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as an execution space:
  using execution_space = FakeGPU;
  //! This device's preferred memory space.
  using memory_space = FakeGPUSpace;
  //! The size_type alias best suited for this device.
  using size_type = memory_space::size_type;
  //! This execution space preferred device_type
  using device_type = Kokkos::Device<execution_space, memory_space>;

  //! This device's preferred array layout.
  // TODO DZP: LayoutLeft to match CUDA. Good idea?
  using array_layout = LayoutLeft;

  /// \brief  Scratch memory space
  using scratch_memory_space = ScratchMemorySpace<Kokkos::Experimental::FakeGPU>;

  //@}

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  ///
  /// For the FakeGPU device, this method <i>always</i> returns false,
  /// because parallel_for or parallel_reduce with the FakeGPU device
  /// always execute sequentially.
  inline static int in_parallel() { return false; }

  /// \brief Wait until all dispatched functors complete.
  ///
  /// The parallel_for or parallel_reduce dispatch of a functor may
  /// return asynchronously, before the functor completes.  This
  /// method does not return until all dispatched functors on this
  /// device have completed.
  static void impl_static_fence() {}

  void fence() const {}

  /** \brief  Return the maximum amount of concurrency.  */
  static int concurrency() { return 1; }

  //! Print configuration information to the given output stream.
  static void print_configuration(std::ostream&,
                                  const bool /* detail */ = false) {}

  static void impl_initialize();

  static bool impl_is_initialized();

  //! Free any resources being consumed by the device.
  static void impl_finalize();

  //--------------------------------------------------------------------------

  inline static int impl_thread_pool_size(int = 0) { return 1; }
  KOKKOS_INLINE_FUNCTION static int impl_thread_pool_rank() { return 0; }

  //--------------------------------------------------------------------------

  KOKKOS_INLINE_FUNCTION static unsigned impl_hardware_thread_id() {
    return impl_thread_pool_rank();
  }
  inline static unsigned impl_max_hardware_threads() {
    return impl_thread_pool_size(0);
  }

  uint32_t impl_instance_id() const noexcept { return 0; }
  static bool s_on_gpu;
  static bool is_on_gpu() { return s_on_gpu; }
  static void set_on_gpu(bool on_gpu) { s_on_gpu = on_gpu; }
  static const char* name();
  //--------------------------------------------------------------------------
};
}  // namespace Experimental

template <bool WantGPU>
struct CheckOnGPU {
  static void check() {
    bool error_state = Kokkos::Experimental::FakeGPU::is_on_gpu() != WantGPU;
    if (error_state) {
      std::string error_string =
          "Invalid " +
          std::string(Kokkos::Experimental::FakeGPU::is_on_gpu() ? "GPU"
                                                                 : "CPU") +
          " access\n";
      Kokkos::abort(error_string.c_str());
    }
  }
};
namespace Impl {
template <typename ValueType, class Verifier>
struct CheckedFetch {
  ValueType* m_ptr;
  template <typename iType>
  KOKKOS_INLINE_FUNCTION ValueType& operator[](const iType& i) {
    Verifier::check();
    return m_ptr[i];
  }

  KOKKOS_INLINE_FUNCTION
  operator const ValueType*() const { return m_ptr; }

  KOKKOS_INLINE_FUNCTION
  CheckedFetch() : m_ptr() {}

  KOKKOS_DEFAULTED_FUNCTION
  ~CheckedFetch() = default;

  KOKKOS_INLINE_FUNCTION
  CheckedFetch(const CheckedFetch& rhs) : m_ptr(rhs.m_ptr) {}

  KOKKOS_INLINE_FUNCTION
  CheckedFetch(CheckedFetch&& rhs) : m_ptr(rhs.m_ptr) {}

  KOKKOS_INLINE_FUNCTION
  CheckedFetch& operator=(const CheckedFetch& rhs) {
    m_ptr = rhs.m_ptr;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION
  CheckedFetch& operator=(CheckedFetch&& rhs) {
    m_ptr = rhs.m_ptr;
    return *this;
  }
  // TODO DZP this seems wrong
  inline explicit CheckedFetch(ValueType* const arg_ptr) : m_ptr(arg_ptr) {}
  template <class MemorySpace>
  inline explicit CheckedFetch(
      const ValueType* const arg_ptr,
      Kokkos::Impl::SharedAllocationRecord<MemorySpace, void>*)
      : m_ptr(arg_ptr) {}

  KOKKOS_INLINE_FUNCTION
  CheckedFetch(CheckedFetch const rhs, size_t offset)
      : m_ptr(rhs.m_ptr + offset) {}
};

struct FakeSpaceTag {};
template <class Traits>
class ViewDataHandle<
    Traits, typename std::enable_if<std::is_same<typename Traits::specialize,
                                                 FakeSpaceTag>::value>::type> {
 public:
  using track_type = Kokkos::Impl::SharedAllocationTracker;

  using value_type  = typename Traits::value_type;
  using return_type = typename Traits::value_type&;
  using memspace    = typename Traits::memory_space;
  using handle_type = Kokkos::Impl::CheckedFetch<
      value_type, CheckOnGPU<std::is_same<
                      memspace, Kokkos::Experimental::FakeGPUSpace>::value>>;

  KOKKOS_INLINE_FUNCTION
  static handle_type const& assign(handle_type const& arg_handle,
                                   track_type const& /* arg_tracker */) {
    return arg_handle;
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type const assign(handle_type const& arg_handle,
                                  size_t offset) {
    return handle_type(arg_handle, offset);
  }

  KOKKOS_INLINE_FUNCTION
  static handle_type assign(value_type* arg_data_ptr,
                            track_type const& arg_tracker) {
    if (arg_data_ptr == nullptr) return handle_type();

    // Assignment of texture = non-texture requires creation of a texture object
    // which can only occur on the host.  In addition, 'get_record' is only
    // valid if called in a host execution space

    using memory_space = typename Traits::memory_space;
    using record = typename Impl::SharedAllocationRecord<memory_space, void>;

    record* const r = arg_tracker.template get_record<memory_space>();

    return handle_type(arg_data_ptr, r);
  }
};

}  // namespace Impl

// this stuff is in raw namespace Kokkos
template <class... Prop>
struct ViewTraits<void, Kokkos::Experimental::FakeGPUSpace, Prop...> {
  using specialize      = Impl::FakeSpaceTag;
  using memory_space    = Kokkos::Experimental::FakeGPUSpace;
  using execution_space = memory_space::execution_space;
  //using array_layout    = execution_space::array_layout;
  using array_layout    = Kokkos::LayoutLeft;
  using HostMirror      = Kokkos::Experimental::FakeHostSpace;
  using HostMirrorSpace = Kokkos::Experimental::FakeHostSpace;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  // using reference_type = typename
  // Kokkos::Impl::ViewDataHandle<memory_traits>::return_type;
};
template <class... Prop>
struct ViewTraits<void, Kokkos::Experimental::FakeHostSpace, Prop...> {
  using specialize      = Impl::FakeSpaceTag;
  using memory_space    = Kokkos::Experimental::FakeHostSpace;
  using execution_space = memory_space::execution_space;
  //using array_layout    = execution_space::array_layout;
  using array_layout    = Kokkos::LayoutRight;
  using HostMirror      = Kokkos::Experimental::FakeHostSpace;
  using HostMirrorSpace = Kokkos::Experimental::FakeHostSpace;
  using memory_traits   = typename ViewTraits<void, Prop...>::memory_traits;
  // using reference_type = typename
  // Kokkos::Impl::ViewDataHandle<memory_traits>::return_type;
};

namespace Impl {
template <class Traits>
class ViewMapping<Traits, FakeSpaceTag> {
 private:
  template <class, class...>
  friend class ViewMapping;
  template <class, class...>
  friend class Kokkos::View;

  typedef ViewOffset<typename Traits::dimension, typename Traits::array_layout,
                     void>
      offset_type;

  typedef typename ViewDataHandle<Traits>::handle_type handle_type;
  mutable handle_type
      m_handle;  // TODO DZP: what on earth is Kokkos even doing in this design.
  offset_type m_offset;

  KOKKOS_INLINE_FUNCTION
  ViewMapping(const handle_type& arg_handle, const offset_type& arg_offset)
      : m_handle(arg_handle), m_offset(arg_offset) {}

 public:
  typedef void printable_label_typedef;
  enum { is_managed = Traits::is_managed };

  //----------------------------------------
  // Domain dimensions

  enum { Rank = Traits::dimension::rank };

  template <typename iType>
  KOKKOS_INLINE_FUNCTION constexpr size_t extent(const iType& r) const {
    return m_offset.m_dim.extent(r);
  }

  KOKKOS_INLINE_FUNCTION constexpr typename Traits::array_layout layout()
      const {
    return m_offset.layout();
  }

  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_0() const {
    return m_offset.dimension_0();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_1() const {
    return m_offset.dimension_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_2() const {
    return m_offset.dimension_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_3() const {
    return m_offset.dimension_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_4() const {
    return m_offset.dimension_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_5() const {
    return m_offset.dimension_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_6() const {
    return m_offset.dimension_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t dimension_7() const {
    return m_offset.dimension_7();
  }

  // Is a regular layout with uniform striding for each index.
  using is_regular = typename offset_type::is_regular;

  KOKKOS_INLINE_FUNCTION constexpr size_t stride_0() const {
    return m_offset.stride_0();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_1() const {
    return m_offset.stride_1();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_2() const {
    return m_offset.stride_2();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_3() const {
    return m_offset.stride_3();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_4() const {
    return m_offset.stride_4();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_5() const {
    return m_offset.stride_5();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_6() const {
    return m_offset.stride_6();
  }
  KOKKOS_INLINE_FUNCTION constexpr size_t stride_7() const {
    return m_offset.stride_7();
  }

  template <typename iType>
  KOKKOS_INLINE_FUNCTION void stride(iType* const s) const {
    m_offset.stride(s);
  }

  //----------------------------------------
  // Range span

  /** \brief  Span of the mapped range */
  KOKKOS_INLINE_FUNCTION constexpr size_t span() const {
    return m_offset.span();
  }

  /** \brief  Is the mapped range span contiguous */
  KOKKOS_INLINE_FUNCTION constexpr bool span_is_contiguous() const {
    return m_offset.span_is_contiguous();
  }

  typedef typename ViewDataHandle<Traits>::return_type reference_type;
  typedef typename Traits::value_type* pointer_type;

  /** \brief  Query raw pointer to memory */
  KOKKOS_INLINE_FUNCTION constexpr pointer_type data() const {
    return m_handle.m_ptr;
  }

  //----------------------------------------
  // The View class performs all rank and bounds checking before
  // calling these element reference methods.

  KOKKOS_FORCEINLINE_FUNCTION
  reference_type reference() const { return m_handle[0]; }

  template <typename I0>
  KOKKOS_FORCEINLINE_FUNCTION
      typename std::enable_if<(std::is_integral<I0>::value &&
                               // if layout is neither stride nor irregular,
                               // then just use the handle directly
                               !(std::is_same<typename Traits::array_layout,
                                              Kokkos::LayoutStride>::value ||
                                 !is_regular::value)),
                              reference_type>::type
      reference(const I0& i0) const {
    return m_handle[i0];
  }

  template <typename I0>
  KOKKOS_FORCEINLINE_FUNCTION
      typename std::enable_if<(std::is_integral<I0>::value &&
                               // if the layout is strided or irregular, then
                               // we have to use the offset
                               (std::is_same<typename Traits::array_layout,
                                             Kokkos::LayoutStride>::value ||
                                !is_regular::value)),
                              reference_type>::type
      reference(const I0& i0) const {
    return m_handle[m_impl_offset(i0)];
  }

  template <typename I0, typename I1>
  KOKKOS_FORCEINLINE_FUNCTION reference_type reference(const I0& i0,
                                                       const I1& i1) const {
    return m_handle[m_impl_offset(i0, i1)];
  }

  template <typename I0, typename I1, typename I2>
  KOKKOS_FORCEINLINE_FUNCTION reference_type reference(const I0& i0,
                                                       const I1& i1,
                                                       const I2& i2) const {
    return m_handle[m_impl_offset(i0, i1, i2)];
  }

  template <typename I0, typename I1, typename I2, typename I3>
  KOKKOS_FORCEINLINE_FUNCTION reference_type
  reference(const I0& i0, const I1& i1, const I2& i2, const I3& i3) const {
    return m_handle[m_impl_offset(i0, i1, i2, i3)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4>
  KOKKOS_FORCEINLINE_FUNCTION reference_type reference(const I0& i0,
                                                       const I1& i1,
                                                       const I2& i2,
                                                       const I3& i3,
                                                       const I4& i4) const {
    return m_handle[m_impl_offset(i0, i1, i2, i3, i4)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5>
  KOKKOS_FORCEINLINE_FUNCTION reference_type
  reference(const I0& i0, const I1& i1, const I2& i2, const I3& i3,
            const I4& i4, const I5& i5) const {
    return m_handle[m_impl_offset(i0, i1, i2, i3, i4, i5)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6>
  KOKKOS_FORCEINLINE_FUNCTION reference_type
  reference(const I0& i0, const I1& i1, const I2& i2, const I3& i3,
            const I4& i4, const I5& i5, const I6& i6) const {
    return m_handle[m_impl_offset(i0, i1, i2, i3, i4, i5, i6)];
  }

  template <typename I0, typename I1, typename I2, typename I3, typename I4,
            typename I5, typename I6, typename I7>
  KOKKOS_FORCEINLINE_FUNCTION reference_type
  reference(const I0& i0, const I1& i1, const I2& i2, const I3& i3,
            const I4& i4, const I5& i5, const I6& i6, const I7& i7) const {
    return m_handle[m_impl_offset(i0, i1, i2, i3, i4, i5, i6, i7)];
  }

  //----------------------------------------

 private:
  enum { MemorySpanMask = 8 - 1 /* Force alignment on 8 byte boundary */ };
  enum { MemorySpanSize = sizeof(typename Traits::value_type) };

 public:
  /** \brief  Span, in bytes, of the referenced memory */
  KOKKOS_INLINE_FUNCTION constexpr size_t memory_span() const {
    return (m_offset.span() * sizeof(typename Traits::value_type) +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  //----------------------------------------

  KOKKOS_INLINE_FUNCTION ~ViewMapping() {}
  KOKKOS_INLINE_FUNCTION ViewMapping() : m_handle(), m_offset() {}
  KOKKOS_INLINE_FUNCTION ViewMapping(const ViewMapping& rhs)
      : m_handle(rhs.m_handle), m_offset(rhs.m_offset) {}
  KOKKOS_INLINE_FUNCTION ViewMapping& operator=(const ViewMapping& rhs) {
    m_handle = rhs.m_handle;
    m_offset = rhs.m_offset;
    return *this;
  }

  KOKKOS_INLINE_FUNCTION ViewMapping(ViewMapping&& rhs)
      : m_handle(rhs.m_handle), m_offset(rhs.m_offset) {}
  KOKKOS_INLINE_FUNCTION ViewMapping& operator=(ViewMapping&& rhs) {
    m_handle = rhs.m_handle;
    m_offset = rhs.m_offset;
    return *this;
  }

  //----------------------------------------

  /**\brief  Span, in bytes, of the required memory */
  KOKKOS_INLINE_FUNCTION
  static constexpr size_t memory_span(
      typename Traits::array_layout const& arg_layout) {
    typedef std::integral_constant<unsigned, 0> padding;
    return (offset_type(padding(), arg_layout).span() * MemorySpanSize +
            MemorySpanMask) &
           ~size_t(MemorySpanMask);
  }

  /**\brief  Wrap a span of memory */
  template <class... P>
  KOKKOS_INLINE_FUNCTION ViewMapping(
      Kokkos::Impl::ViewCtorProp<P...> const& arg_prop,
      typename Traits::array_layout const& arg_layout)
      : m_handle(
            ((Kokkos::Impl::ViewCtorProp<void, pointer_type> const&)arg_prop)
                .value

        ) {
    typedef typename Traits::value_type value_type;
    typedef std::integral_constant<
        unsigned, Kokkos::Impl::ViewCtorProp<P...>::allow_padding
                      ? sizeof(value_type)
                      : 0>
        padding;

    typename Traits::array_layout layout;
    for (int i = 0; i < Traits::rank; i++)
      layout.dimension[i] = arg_layout.dimension[i];
    m_offset = offset_type(padding(), layout);
  }

  /**\brief  Assign data */
  KOKKOS_INLINE_FUNCTION
  void assign_data(pointer_type arg_ptr) { m_handle = handle_type(arg_ptr); }

  //----------------------------------------
  /*  Allocate and construct mapped array.
   *  Allocate via shared allocation record and
   *  return that record for allocation tracking.
   */
  template <class... P>
  Kokkos::Impl::SharedAllocationRecord<>* allocate_shared(
      Kokkos::Impl::ViewCtorProp<P...> const& arg_prop,
      typename Traits::array_layout const& arg_layout) {
    using alloc_prop = Kokkos::Impl::ViewCtorProp<P...>;

    using execution_space = typename alloc_prop::execution_space;
    using memory_space    = typename Traits::memory_space;
    using value_type      = typename Traits::value_type;
    using functor_type    = ViewValueFunctor<execution_space, value_type>;
    using record_type =
        Kokkos::Impl::SharedAllocationRecord<memory_space, functor_type>;

    // Query the mapping for byte-size of allocation.
    // If padding is allowed then pass in sizeof value type
    // for padding computation.
    using padding = std::integral_constant<
        unsigned int, alloc_prop::allow_padding ? sizeof(value_type) : 0>;

    m_offset = offset_type(padding(), arg_layout);

    const size_t alloc_size =
        (m_offset.span() * MemorySpanSize + MemorySpanMask) &
        ~size_t(MemorySpanMask);
    const std::string& alloc_name =
        static_cast<Kokkos::Impl::ViewCtorProp<void, std::string> const&>(
            arg_prop)
            .value;
    // Create shared memory tracking record with allocate memory from the memory
    // space
    record_type* const record = record_type::allocate(
        static_cast<Kokkos::Impl::ViewCtorProp<void, memory_space> const&>(
            arg_prop)
            .value,
        alloc_name, alloc_size);

    m_handle = handle_type(reinterpret_cast<pointer_type>(record->data()));

    //  Only initialize if the allocation is non-zero.
    //  May be zero if one of the dimensions is zero.
    if (alloc_size && alloc_prop::initialize) {
      // Assume destruction is only required when construction is requested.
      // The ViewValueFunctor has both value construction and destruction
      // operators.
      record->m_destroy = functor_type(
          static_cast<Kokkos::Impl::ViewCtorProp<void, execution_space> const&>(
              arg_prop)
              .value,
          (value_type*)m_handle.m_ptr, m_offset.span(), alloc_name);

      // Construct values
      record->m_destroy.construct_shared_allocation();
    }

    return record;
  }
  //  template <class... P>
  //  Kokkos::Impl::SharedAllocationRecord<> *
  //  allocate_shared(Kokkos::Impl::ViewCtorProp<P...> const &arg_prop,
  //                  typename Traits::array_layout const &arg_layout) {
  //    typedef Kokkos::Impl::ViewCtorProp<P...> alloc_prop;
  //
  //    typedef typename alloc_prop::execution_space execution_space;
  //    typedef typename Traits::memory_space memory_space;
  //    typedef typename Traits::value_type value_type;
  //    typedef ViewValueFunctor<execution_space, value_type> functor_type;
  //    typedef Kokkos::Impl::SharedAllocationRecord<memory_space, functor_type>
  //        record_type;
  //
  //    // Query the mapping for byte-size of allocation.
  //    // If padding is allowed then pass in sizeof value type
  //    // for padding computation.
  //    typedef std::integral_constant<
  //        unsigned, alloc_prop::allow_padding ? sizeof(value_type) : 0>
  //        padding;
  //
  //    typename Traits::array_layout layout;
  //    for (int i = 0; i < Traits::rank; i++)
  //      layout.dimension[i] = arg_layout.dimension[i];
  //    m_offset = offset_type(padding(), layout);
  //
  //    const size_t alloc_size = memory_span();
  //
  //    // Create shared memory tracking record with allocate memory from the
  //    memory
  //    // space
  //    auto memspace = ((Kokkos::Impl::ViewCtorProp<void, memory_space> const
  //    &)arg_prop)
  //            .value;
  //    record_type *const record = record_type::allocate(
  //        ((Kokkos::Impl::ViewCtorProp<void, memory_space> const &)arg_prop)
  //            .value,
  //        ((Kokkos::Impl::ViewCtorProp<void, std::string> const
  //        &)arg_prop).value, alloc_size);
  //
  //#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  //    if (alloc_size) {
  //#endif
  //      //m_handle =
  //      handle_type(reinterpret_cast<pointer_type>(record->data()) , record //
  //      TODO DZP: what should this be? m_handle =
  //      handle_type(reinterpret_cast<pointer_type>(record->data(), record) //
  //      TODO DZP: what should this be?
  //                             );
  //#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  //    }
  //#endif
  //
  //    //  Only initialize if the allocation is non-zero.
  //    //  May be zero if one of the dimensions is zero.
  //    if (alloc_size && alloc_prop::initialize) {
  //      // Construct values
  //      record->m_destroy.construct_shared_allocation();
  //    }
  //
  //    return record;
  //  }
};

}  // namespace Impl
namespace Experimental {
namespace Impl {

class FakeGPUSpaceInitializer : public Kokkos::Impl::ExecSpaceInitializerBase {
 public:
  FakeGPUSpaceInitializer()  = default;
  ~FakeGPUSpaceInitializer() = default;
  void initialize(const InitArguments& args) final;
  void finalize(const bool) final;
  void fence() final;
  void print_configuration(std::ostream& msg, const bool detail) final;
};

}  // namespace Impl
}  // namespace Experimental
namespace Tools {
namespace Experimental {
template <>
struct DeviceTypeTraits<Kokkos::Experimental::FakeGPU> {
  static constexpr DeviceType id = DeviceType::Unknown;
};
}  // namespace Experimental
}  // namespace Tools
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::Experimental::FakeGPU::memory_space,
                         Kokkos::Experimental::FakeGPU::scratch_memory_space> {
  enum : bool { assignable = false };
  enum : bool { accessible = true };
  enum : bool { deepcopy = false };
};


template <>
struct VerifyExecutionCanAccessMemorySpace<
    Kokkos::Experimental::FakeGPU::memory_space, Kokkos::Experimental::FakeGPU::scratch_memory_space> {
  enum : bool { value = true };
  inline static void verify(void) {}
  inline static void verify(const void*) {}
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
	namespace Experimental {
namespace Impl {

// Resize thread team data scratch memory
void fake_gpu_resize_thread_team_data(size_t pool_reduce_bytes,
                                    size_t team_reduce_bytes,
                                    size_t team_shared_bytes,
                                    size_t thread_local_bytes);

Kokkos::Impl::HostThreadTeamData* fake_gpu_get_thread_team_data();

} /* namespace Impl */
} /* namespace Experimental */
} /* namespace Kokkos */

namespace Kokkos {
namespace Impl {

/*
 * < Kokkos::FakeGPU , WorkArgTag >
 * < WorkArgTag , Impl::enable_if< std::is_same< Kokkos::FakeGPU ,
 * Kokkos::DefaultExecutionSpace >::value >::type >
 *
 */
template <class... Properties>
class TeamPolicyInternal<Kokkos::Experimental::FakeGPU, Properties...>
    : public PolicyTraits<Properties...> {
 private:
  size_t m_team_scratch_size[2];
  size_t m_thread_scratch_size[2];
  int m_league_size;
  int m_chunk_size;

 public:
  //! Tag this class as a kokkos execution policy
  using execution_policy = TeamPolicyInternal;

  using traits = PolicyTraits<Properties...>;

  //! Execution space of this execution policy:
  using execution_space = Kokkos::Experimental::FakeGPU;

  const typename traits::execution_space& space() const {
    static typename traits::execution_space m_space;
    return m_space;
  }

  template <class ExecSpace, class... OtherProperties>
  friend class TeamPolicyInternal;

  template <class... OtherProperties>
  TeamPolicyInternal(
      const TeamPolicyInternal<Kokkos::Experimental::FakeGPU, OtherProperties...>& p) {
    m_league_size            = p.m_league_size;
    m_team_scratch_size[0]   = p.m_team_scratch_size[0];
    m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
    m_team_scratch_size[1]   = p.m_team_scratch_size[1];
    m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
    m_chunk_size             = p.m_chunk_size;
  }

  //----------------------------------------

  template <class FunctorType>
  int team_size_max(const FunctorType&, const ParallelForTag&) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_max(const FunctorType&, const ParallelReduceTag&) const {
    return 1;
  }
  template <class FunctorType, class ReducerType>
  int team_size_max(const FunctorType&, const ReducerType&,
                    const ParallelReduceTag&) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_recommended(const FunctorType&, const ParallelForTag&) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_recommended(const FunctorType&,
                            const ParallelReduceTag&) const {
    return 1;
  }
  template <class FunctorType, class ReducerType>
  int team_size_recommended(const FunctorType&, const ReducerType&,
                            const ParallelReduceTag&) const {
    return 1;
  }

  //----------------------------------------

  inline int team_size() const { return 1; }
  inline bool impl_auto_team_size() const { return false; }
  inline bool impl_auto_vector_length() const { return false; }
  inline void impl_set_team_size(size_t) {}
  inline void impl_set_vector_length(size_t) {}
  inline int league_size() const { return m_league_size; }
  inline size_t scratch_size(const int& level, int = 0) const {
    return m_team_scratch_size[level] + m_thread_scratch_size[level];
  }

  inline int impl_vector_length() const { return 1; }
  inline static int vector_length_max() {
    return 1024;
  }  // Use arbitrary large number, is meant as a vectorizable length

  inline static int scratch_size_max(int level) {
    return (level == 0 ? 1024 * 32 : 20 * 1024 * 1024);
  }
  /** \brief  Specify league size, request team size */
  TeamPolicyInternal(const execution_space&, int league_size_request,
                     int team_size_request, int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0},
        m_thread_scratch_size{0, 0},
        m_league_size(league_size_request),
        m_chunk_size(32) {
    if (team_size_request > 1)
      Kokkos::abort("Kokkos::abort: Requested Team Size is too large!");
  }

  TeamPolicyInternal(const execution_space& space, int league_size_request,
                     const Kokkos::AUTO_t& /**team_size_request*/,
                     int vector_length_request = 1)
      : TeamPolicyInternal(space, league_size_request, -1,
                           vector_length_request) {}

  TeamPolicyInternal(const execution_space& space, int league_size_request,
                     const Kokkos::AUTO_t& /* team_size_request */
                     ,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space, league_size_request, -1, -1) {}

  TeamPolicyInternal(const execution_space& space, int league_size_request,
                     int team_size_request,
                     const Kokkos::AUTO_t& /* vector_length_request */
                     )
      : TeamPolicyInternal(space, league_size_request, team_size_request, -1) {}

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t& team_size_request,
                     const Kokkos::AUTO_t& vector_length_request)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}
  TeamPolicyInternal(int league_size_request, int team_size_request,
                     const Kokkos::AUTO_t& vector_length_request)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int vector_length_request = 1)
      : TeamPolicyInternal(typename traits::execution_space(),
                           league_size_request, team_size_request,
                           vector_length_request) {}

  inline int chunk_size() const { return m_chunk_size; }

  /** \brief set chunk_size to a discrete value*/
  inline TeamPolicyInternal& set_chunk_size(
      typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch
   * hierarchy */
  inline TeamPolicyInternal& set_scratch_size(const int& level,
                                              const PerTeamValue& per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch
   * hierarchy */
  inline TeamPolicyInternal& set_scratch_size(
      const int& level, const PerThreadValue& per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of
   * the scratch hierarchy */
  inline TeamPolicyInternal& set_scratch_size(
      const int& level, const PerTeamValue& per_team,
      const PerThreadValue& per_thread) {
    m_team_scratch_size[level]   = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  using member_type = Impl::HostThreadTeamMember<Kokkos::Experimental::FakeGPU>;
};
} /* namespace Impl */
} /* namespace Kokkos */

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::FakeGPU with RangePolicy */

namespace Kokkos {
namespace Impl {

struct RAIIGPUContext {
  RAIIGPUContext() { Kokkos::Experimental::FakeGPU::set_on_gpu(true); }
  ~RAIIGPUContext() { Kokkos::Experimental::FakeGPU::set_on_gpu(false); }
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>, Kokkos::Experimental::FakeGPU> {
 private:
  using Policy = Kokkos::RangePolicy<Traits...>;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  typename std::enable_if<std::is_same<TagType, void>::value>::type exec()
      const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i);
    }
  }

  template <class TagType>
  typename std::enable_if<!std::is_same<TagType, void>::value>::type exec()
      const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i);
    }
  }

 public:
  inline void execute() const {
    RAIIGPUContext ctx;
    this->template exec<typename Policy::work_tag>();
  }

  inline ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

/*--------------------------------------------------------------------------*/

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::FakeGPU> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;

  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i, update);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(reference_type update) const {
    const TagType t{};

    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i, update);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    Kokkos::Experimental::Impl::fake_gpu_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *Kokkos::Experimental::Impl::fake_gpu_get_thread_team_data();

    pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

    reference_type update =
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer), ptr);

    this->template exec<WorkTag>(update);

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);
  }

  template <class HostViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const HostViewType& arg_result_view,
      typename std::enable_if<Kokkos::is_view<HostViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    // TODO DZP: check this
    static_assert(Kokkos::is_view<HostViewType>::value,
                  "Kokkos::FakeGPU reduce result must be a View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename HostViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::FakeGPU reduce result must be a View in HostSpace");
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

/*--------------------------------------------------------------------------*/

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::Experimental::FakeGPU> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i, update, true);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(reference_type update) const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i, update, true);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size  = Analysis::value_size(m_functor);
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    Kokkos::Experimental::Impl::fake_gpu_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *Kokkos::Experimental::Impl::fake_gpu_get_thread_team_data();

    reference_type update =
        ValueInit::init(m_functor, pointer_type(data.pool_reduce_local()));

    this->template exec<WorkTag>(update);
  }

  inline ParallelScan(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

/*--------------------------------------------------------------------------*/
template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::Experimental::FakeGPU> {
 private:
  using Policy  = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  ReturnType& m_returnvalue;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(i, update, true);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(reference_type update) const {
    const TagType t{};
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      m_functor(t, i, update, true);
    }
  }

 public:
  inline void execute() {
    const size_t pool_reduce_size  = Analysis::value_size(m_functor);
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    Kokkos::Experimental::Impl::fake_gpu_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *Kokkos::Experimental::Impl::fake_gpu_get_thread_team_data();

    reference_type update =
        ValueInit::init(m_functor, pointer_type(data.pool_reduce_local()));

    this->template exec<WorkTag>(update);

    m_returnvalue = update;
  }

  inline ParallelScanWithTotal(const FunctorType& arg_functor,
                               const Policy& arg_policy,
                               ReturnType& arg_returnvalue)
      : m_functor(arg_functor),
        m_policy(arg_policy),
        m_returnvalue(arg_returnvalue) {}
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::FakeGPU with MDRangePolicy */

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::FakeGPU> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using iterate_type = typename Kokkos::Impl::HostIterateTile<
      MDRangePolicy, FunctorType, typename MDRangePolicy::work_tag, void>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;

  void exec() const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      iterate_type(m_mdr_policy, m_functor)(i);
    }
  }

 public:
  inline void execute() const { this->exec(); }

  inline ParallelFor(const FunctorType& arg_functor,
                     const MDRangePolicy& arg_policy)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::FakeGPU> {
 private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy        = typename MDRangePolicy::impl_range_policy;

  using WorkTag = typename MDRangePolicy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                   MDRangePolicy, FunctorType>;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using value_type     = typename Analysis::value_type;
  using reference_type = typename Analysis::reference_type;

  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, reference_type>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  inline void exec(reference_type update) const {
    const typename Policy::member_type e = m_policy.end();
    for (typename Policy::member_type i = m_policy.begin(); i < e; ++i) {
      iterate_type(m_mdr_policy, m_functor, update)(i);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));
    const size_t team_reduce_size  = 0;  // Never shrinks
    const size_t team_shared_size  = 0;  // Never shrinks
    const size_t thread_local_size = 0;  // Never shrinks

    Kokkos::Experimental::Impl::fake_gpu_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *Kokkos::Experimental::Impl::fake_gpu_get_thread_team_data();

    pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

    reference_type update =
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer), ptr);

    this->exec(update);

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);
  }

  template <class HostViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const MDRangePolicy& arg_policy,
      const HostViewType& arg_result_view,
      typename std::enable_if<Kokkos::is_view<HostViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result_view.data()) {
    static_assert(Kokkos::is_view<HostViewType>::value,
                  "Kokkos::FakeGPU reduce result must be a View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename HostViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Kokkos::FakeGPU reduce result must be a View in HostSpace");
  }

  inline ParallelReduce(const FunctorType& arg_functor,
                        MDRangePolicy arg_policy, const ReducerType& reducer)
      : m_functor(arg_functor),
        m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                                    , Kokkos::HostSpace >::value
      , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
      );*/
  }
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/
/* Parallel patterns for Kokkos::FakeGPU with TeamPolicy */

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::FakeGPU> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy = TeamPolicyInternal<Kokkos::Experimental::FakeGPU, Properties...>;
  using Member = typename Policy::member_type;

  const FunctorType m_functor;
  const int m_league;
  const int m_shared;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      HostThreadTeamData& data) const {
    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(Member(data, ileague, m_league));
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(HostThreadTeamData& data) const {
    const TagType t{};
    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(t, Member(data, ileague, m_league));
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size  = 0;  // Never shrinks
    const size_t team_reduce_size  = TEAM_REDUCE_SIZE;
    const size_t team_shared_size  = m_shared;
    const size_t thread_local_size = 0;  // Never shrinks

    Kokkos::Experimental::Impl::fake_gpu_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *Kokkos::Experimental::Impl::fake_gpu_get_thread_team_data();

    this->template exec<typename Policy::work_tag>(data);
  }

  ParallelFor(const FunctorType& arg_functor, const Policy& arg_policy)
      : m_functor(arg_functor),
        m_league(arg_policy.league_size()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(arg_functor, 1)) {}
};

/*--------------------------------------------------------------------------*/

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::FakeGPU> {
 private:
  enum { TEAM_REDUCE_SIZE = 512 };

  using Policy = TeamPolicyInternal<Kokkos::Experimental::FakeGPU, Properties...>;

  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;

  using Member  = typename Policy::member_type;
  using WorkTag = typename Policy::work_tag;

  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;

  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;

  using pointer_type   = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const int m_league;
  const ReducerType m_reducer;
  pointer_type m_result_ptr;
  const int m_shared;

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type exec(
      HostThreadTeamData& data, reference_type update) const {
    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(Member(data, ileague, m_league), update);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  exec(HostThreadTeamData& data, reference_type update) const {
    const TagType t{};

    for (int ileague = 0; ileague < m_league; ++ileague) {
      m_functor(t, Member(data, ileague, m_league), update);
    }
  }

 public:
  inline void execute() const {
    const size_t pool_reduce_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

    const size_t team_reduce_size  = TEAM_REDUCE_SIZE;
    const size_t team_shared_size  = m_shared;
    const size_t thread_local_size = 0;  // Never shrinks

    Kokkos::Experimental::Impl::fake_gpu_resize_thread_team_data(pool_reduce_size, team_reduce_size,
                                   team_shared_size, thread_local_size);

    HostThreadTeamData& data = *Kokkos::Experimental::Impl::fake_gpu_get_thread_team_data();

    pointer_type ptr =
        m_result_ptr ? m_result_ptr : pointer_type(data.pool_reduce_local());

    reference_type update =
        ValueInit::init(ReducerConditional::select(m_functor, m_reducer), ptr);

    this->template exec<WorkTag>(data, update);

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), ptr);
  }

  template <class ViewType>
  ParallelReduce(
      const FunctorType& arg_functor, const Policy& arg_policy,
      const ViewType& arg_result,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void*>::type = nullptr)
      : m_functor(arg_functor),
        m_league(arg_policy.league_size()),
        m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(m_functor, 1)) {
    static_assert(Kokkos::is_view<ViewType>::value,
                  "Reduction result on Kokkos::FakeGPU must be a Kokkos::View");

    static_assert(
        Kokkos::Impl::MemorySpaceAccess<typename ViewType::memory_space,
                                        Kokkos::HostSpace>::accessible,
        "Reduction result on Kokkos::FakeGPU must be a Kokkos::View in "
        "HostSpace");
  }

  inline ParallelReduce(const FunctorType& arg_functor, Policy arg_policy,
                        const ReducerType& reducer)
      : m_functor(arg_functor),
        m_league(arg_policy.league_size()),
        m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(arg_functor, 1)) {
    /*static_assert( std::is_same< typename ViewType::memory_space
                            , Kokkos::HostSpace >::value
    , "Reduction result on Kokkos::OpenMP must be a Kokkos::View in HostSpace"
    );*/
  }
};

}  // namespace Impl
}  // namespace Kokkos

/*--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------*/

namespace Kokkos {
namespace Experimental {

template <>
class UniqueToken<FakeGPU, UniqueTokenScope::Instance> {
 public:
  using execution_space = FakeGPU;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const& = execution_space()) noexcept {}

  /// \brief create object size for requested size on given instance
  ///
  /// It is the users responsibility to only acquire size tokens concurrently
  UniqueToken(size_type, execution_space const& = execution_space()) {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept { return 1; }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const noexcept { return 0; }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release(int) const noexcept {}
};

template <>
class UniqueToken<FakeGPU, UniqueTokenScope::Global> {
 public:
  using execution_space = FakeGPU;
  using size_type       = int;

  /// \brief create object size for concurrency on the given instance
  ///
  /// This object should not be shared between instances
  UniqueToken(execution_space const& = execution_space()) noexcept {}

  /// \brief upper bound for acquired values, i.e. 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int size() const noexcept { return 1; }

  /// \brief acquire value such that 0 <= value < size()
  KOKKOS_INLINE_FUNCTION
  int acquire() const noexcept { return 0; }

  /// \brief release a value acquired by generate
  KOKKOS_INLINE_FUNCTION
  void release(int) const noexcept {}
};

}  // namespace Experimental
namespace Impl {
namespace detail {
template <class...>
struct Pack {};

template <class, class...>
struct converter;

template <class... ProcessedProps>
struct converter<Pack<ProcessedProps...>> {
  using type = Kokkos::View<ProcessedProps...>;
};

template <class... ProcessedProps, class BaseSpace, class Exec, class Namer,
          bool Shares, class... Props>
struct converter<
    Pack<ProcessedProps...>,
    Kokkos::Experimental::LogicalMemorySpace<BaseSpace, Exec, Namer, Shares>,
    Props...> {
  using type =
      typename converter<Pack<ProcessedProps..., BaseSpace>, Props...>::type;
};

template <class... ProcessedProps, class Prop, class... Props>
struct converter<Pack<ProcessedProps...>, Prop, Props...> {
  using type =
      typename converter<Pack<ProcessedProps..., Prop>, Props...>::type;
};

}  // namespace detail

template <class View>
struct LogicalToBase;

template <class... Props>
struct LogicalToBase<Kokkos::View<Props...>> {
  using type = typename detail::converter<detail::Pack<>, Props...>::type;
};

template <class Value>
struct LogicalToBase {
  using type = Value;
};

template <class View>
using not_logical_view = std::is_same<View, typename LogicalToBase<View>::type>;
}  // namespace Impl
template <typename Src>
inline auto create_mirror(
    const Src& src,
    typename std::enable_if<!Impl::not_logical_view<Src>::value>::type* =
        nullptr) {
  using mirror = typename Src::HostMirror;
  mirror mirr(
      std::string(src.label()).append("_mirror"),
      src.rank_dynamic > 0 ? src.extent(0) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 1 ? src.extent(1) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 2 ? src.extent(2) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 3 ? src.extent(3) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 4 ? src.extent(4) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 5 ? src.extent(5) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 6 ? src.extent(6) : KOKKOS_IMPL_CTOR_DEFAULT_ARG,
      src.rank_dynamic > 7 ? src.extent(7) : KOKKOS_IMPL_CTOR_DEFAULT_ARG);
  return mirr;
}
}  // namespace Kokkos

//#include <impl/Kokkos_FakeGPU_Task.hpp>

#endif  // defined( KOKKOS_ENABLE_FAKEGPU )
#endif  /* #define KOKKOS_FAKEGPU_HPP */
