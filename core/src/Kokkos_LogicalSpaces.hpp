#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <iostream>
namespace Kokkos {

class DefaultNamer {
  static constexpr const char* get_name() { return "DEFAULT"; }
};

struct EmptyModifier {
  static void exec(){};
};

template<class BaseSpace, class PreferredMemorySpace = void, class Namer = DefaultNamer, class Prologue = EmptyModifier, class Epilogue = EmptyModifier>
class LogicalExecutionSpace {
 BaseSpace space;
 public:
   using pre = Prologue;
   using post = Epilogue;
   template<typename... Args>
   LogicalExecutionSpace(Args... args) : space(args...) {};
  //! \name Type declarations that all Kokkos devices must provide.
  //@{

  //! Tag this class as an execution space:
  using execution_space = LogicalExecutionSpace<BaseSpace, PreferredMemorySpace, Namer, Prologue, Epilogue>;
  //! The size_type alias best suited for this device.
  using size_type = typename BaseSpace::size_type;
  //! This device's preferred memory space.
  using memory_space = std::conditional<std::is_same<PreferredMemorySpace,void>::value,typename BaseSpace::memory_space, PreferredMemorySpace>;
  //! This execution space preferred device_type
  using device_type = Kokkos::Device<execution_space, memory_space>;

  //! This device's preferred array layout.
  using array_layout = typename BaseSpace::array_layout; // TODO: better

  /// \brief  Scratch memory space
  using scratch_memory_space = Kokkos::ScratchMemorySpace<BaseSpace>;

  //@}

  /// \brief True if and only if this method is being called in a
  ///   thread-parallel function.
  ///
  /// For the Serial device, this method <i>always</i> returns false,
  /// because parallel_for or parallel_reduce with the Serial device
  /// always execute sequentially.
  inline static int in_parallel() { return BaseSpace::in_parallel(); }

  /// \brief Wait until all dispatched functors complete.
  ///
  /// The parallel_for or parallel_reduce dispatch of a functor may
  /// return asynchronously, before the functor completes.  This
  /// method does not return until all dispatched functors on this
  /// device have completed.
  static void impl_static_fence() { BaseSpace::impl_static_fence(); }

  void fence() const { space.fence(); }

  /** \brief  Return the maximum amount of concurrency.  */
  static int concurrency() { return BaseSpace::concurrency(); }

  //! Print configuration information to the given output stream.
  static void print_configuration(std::ostream& out,
                                  const bool detail = false) {
    BaseSpace::print_configuration(out, detail);
  }

  static void impl_initialize() { BaseSpace::impl_initialize(); };

  static bool impl_is_initialized() { return BaseSpace::impl_is_initialized(); };

  //! Free any resources being consumed by the device.
  static void impl_finalize() { BaseSpace::finalize(); };

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
  static const char* name() { return Namer::get_name(); }
};

namespace Impl {

template<class BaseSpace, class FunctorType, template<class...> class SpecifiedPolicy,class... PolicyTraits, class... SpaceTraits>
class ParallelFor<FunctorType, SpecifiedPolicy<PolicyTraits...>, LogicalExecutionSpace<BaseSpace, SpaceTraits...>> {
  using LogicalSpace = LogicalExecutionSpace<BaseSpace, SpaceTraits...>;
  using Policy = SpecifiedPolicy<PolicyTraits...>;
  using real_runner = ParallelFor<FunctorType, Policy, BaseSpace>;
  real_runner runner;
  public:
  inline void execute() const {
    LogicalSpace::pre::exec();
    runner.execute();
    LogicalSpace::post::exec();
  }
  template<typename... Args>
  inline ParallelFor(Args... args) : runner(args...) {}
};

template<class BaseSpace, class FunctorType, template<class...> class SpecifiedPolicy, class ReducerType, class... PolicyTraits, class... SpaceTraits>
class ParallelReduce<FunctorType, SpecifiedPolicy<PolicyTraits...>, ReducerType, LogicalExecutionSpace<BaseSpace,SpaceTraits...>> {
  using LogicalSpace = LogicalExecutionSpace<BaseSpace, SpaceTraits...>;
  using real_runner = ParallelReduce<FunctorType, SpecifiedPolicy<PolicyTraits...>, ReducerType, BaseSpace>;
  real_runner runner;
  public:
  inline void execute() const {
    LogicalSpace::pre::exec();
    runner.execute();
    LogicalSpace::post::exec();
  }
  template<typename... Args>
  inline ParallelReduce(Args... args) : runner(args...) {}
};

template<class BaseSpace, class FunctorType, template<class...> class SpecifiedPolicy,class... PolicyTraits, class... SpaceTraits>
class ParallelScan<FunctorType, SpecifiedPolicy<PolicyTraits...>, LogicalExecutionSpace<BaseSpace, SpaceTraits...>> {
  using LogicalSpace = LogicalExecutionSpace<BaseSpace, SpaceTraits...>;
  using Policy = SpecifiedPolicy<PolicyTraits...>;
  using real_runner = ParallelScan<FunctorType, Policy, BaseSpace>;
  real_runner runner;
  public:
  inline void execute() {
    LogicalSpace::pre::exec();
    runner.execute();
    LogicalSpace::post::exec();
  }
  template<typename... Args>
  inline ParallelScan(Args... args) : runner(args...) {}
};
template<class BaseSpace, class FunctorType, template<class...> class SpecifiedPolicy, class ReturnType, class... PolicyTraits, class... SpaceTraits>
class ParallelScanWithTotal<FunctorType, SpecifiedPolicy<PolicyTraits...>, ReturnType, LogicalExecutionSpace<BaseSpace, SpaceTraits...>> {
  using LogicalSpace = LogicalExecutionSpace<BaseSpace, SpaceTraits...>;
  using Policy = SpecifiedPolicy<PolicyTraits...>;
  using real_runner = ParallelScanWithTotal<FunctorType, Policy, ReturnType, BaseSpace>;
  real_runner runner;
  public:
  inline void execute() {
    LogicalSpace::pre::exec();
    runner.execute();
    LogicalSpace::post::exec();
  }
  template<typename... Args>
  inline ParallelScanWithTotal(Args... args) : runner(args...) {}
};

}// Impl
namespace Tools {
	namespace Experimental {
template <class BaseSpace, class... SpaceTraits>
struct DeviceTypeTraits<Kokkos::LogicalExecutionSpace<BaseSpace,SpaceTraits...>> {
  static constexpr DeviceType id = DeviceType::Logical;
};
  
	}}
}// Kokkos
