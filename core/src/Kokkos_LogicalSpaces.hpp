#pragma once

#include <Kokkos_Macros.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <iostream>
namespace Kokkos {

struct DefaultExecutionSpaceNamer {
  static constexpr const char* get_name() { return "DefaultLogicalExecutionSpaceName"; }
};
struct DefaultMemorySpaceNamer {
  static constexpr const char* get_name() {
    return "DefaultLogicalMemorySpaceName";
  }
};

struct EmptyModifier {
  static void exec(){};
};

template <class BaseSpace, class PreferredMemorySpace = void,
          class Namer    = DefaultExecutionSpaceNamer,
          class Prologue = EmptyModifier, class Epilogue = EmptyModifier>
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
  using memory_space =
      typename std::conditional<std::is_same<PreferredMemorySpace, void>::value,
                                typename BaseSpace::memory_space,
                                PreferredMemorySpace>::type;
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

        /// \class LogicalMemorySpace
        /// \brief
        ///
        /// LogicalMemorySpace is a space that is identical to another space,
        /// but differentiable by name and template argument

        template <class BaseSpace, class DefaultExecutionSpace = void,
                  class Namer               = DefaultMemorySpaceNamer,
                  bool SharesAccessWithBase = true>
        class LogicalMemorySpace {
         public:
          //! Tag this class as a kokkos memory space
          using memory_space =
              LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                 SharesAccessWithBase>;
          using size_type = size_t;

          /// \typedef execution_space
          /// \brief Default execution space for this memory space.
          ///
          /// Every memory space has a default execution space.  This is
          /// useful for things like initializing a View (which happens in
          /// parallel using the View's default execution space).

          using execution_space = typename std::conditional<
              std::is_void<DefaultExecutionSpace>::value,
              typename BaseSpace::execution_space, DefaultExecutionSpace>::type;

          using device_type = Kokkos::Device<execution_space, memory_space>;

          /**\brief  Default memory space instance */
          LogicalMemorySpace() : underlying_allocator(){};
          LogicalMemorySpace(LogicalMemorySpace&& rhs)      = default;
          LogicalMemorySpace(const LogicalMemorySpace& rhs) = default;
          LogicalMemorySpace& operator=(LogicalMemorySpace&&) = default;
          LogicalMemorySpace& operator=(const LogicalMemorySpace&) = default;
          ~LogicalMemorySpace()                                    = default;

          BaseSpace underlying_allocator;

          template <typename... Args>
          LogicalMemorySpace(Args&&... args) : underlying_allocator(args...) {}

          /**\brief  Allocate untracked memory in the space */
          void* allocate(const size_t arg_alloc_size) const {
            return allocate("[unlabeled]",arg_alloc_size);
          }
          void* allocate(const char* arg_label, const size_t arg_alloc_size, const size_t arg_logical_size= 0) const {
            return underlying_allocator.allocate(arg_label, arg_alloc_size, arg_logical_size);
	  }
          /**\brief  Deallocate untracked memory in the space */
          void deallocate(void* const arg_alloc_ptr,
                          const size_t arg_alloc_size) const {
            return underlying_allocator.deallocate(arg_alloc_ptr,
                                                   arg_alloc_size);
          }

          /**\brief Return Name of the MemorySpace */
          constexpr static const char* name() { return Namer::get_name(); }

         private:
          friend class Kokkos::Impl::SharedAllocationRecord<memory_space, void>;
        };

        }  // namespace Kokkos

        //----------------------------------------------------------------------------

        namespace Kokkos {

        namespace Impl {

        template <typename BaseSpace, typename DefaultExecutionSpace,
                  class Namer, typename OtherSpace>
        struct MemorySpaceAccess<
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       true>,
            OtherSpace> {
          enum {
            assignable = MemorySpaceAccess<BaseSpace, OtherSpace>::assignable
          };
          enum {
            accessible = MemorySpaceAccess<BaseSpace, OtherSpace>::accessible
          };
          enum {
            deepcopy = MemorySpaceAccess<BaseSpace, OtherSpace>::deepcopy
          };
        };

        template <typename BaseSpace, typename DefaultExecutionSpace,
                  class Namer, typename OtherSpace>
        struct MemorySpaceAccess<
            OtherSpace, Kokkos::LogicalMemorySpace<
                            BaseSpace, DefaultExecutionSpace, Namer, true>> {
          enum {
            assignable = MemorySpaceAccess<OtherSpace, BaseSpace>::assignable
          };
          enum {
            accessible = MemorySpaceAccess<OtherSpace, BaseSpace>::accessible
          };
          enum {
            deepcopy = MemorySpaceAccess<OtherSpace, BaseSpace>::deepcopy
          };
        };

        template <typename BaseSpace, typename DefaultExecutionSpace,
                  class Namer>
        struct MemorySpaceAccess<
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       true>,
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       true>> {
          enum { assignable = true };
          enum { accessible = true };
          enum { deepcopy = true };
        };

        template <typename BaseSpace, typename DefaultExecutionSpace,
                  class Namer>
        struct MemorySpaceAccess<
            Kokkos::AnonymousSpace,
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       true>> {
          enum { assignable = true };
          enum { accessible = true };
          enum { deepcopy = true };
        };

        }  // namespace Impl

        }  // namespace Kokkos

        //----------------------------------------------------------------------------

        namespace Kokkos {

        namespace Impl {

        template <class BaseSpace, class DefaultExecutionSpace, class Namer,
                  bool SharesAccessSemanticsWithBase>
        class SharedAllocationRecord<
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       SharesAccessSemanticsWithBase>,
            void> : public SharedAllocationRecord<void, void> {
         private:
          using SpaceType =
              Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace,
                                         Namer, SharesAccessSemanticsWithBase>;
          using RecordBase = SharedAllocationRecord<void, void>;

          friend SpaceType;
          SharedAllocationRecord(const SharedAllocationRecord&) = delete;
          SharedAllocationRecord& operator=(const SharedAllocationRecord&) =
              delete;

          static void deallocate(RecordBase* arg_rec) {
            delete static_cast<SharedAllocationRecord*>(arg_rec);
          }

#ifdef KOKKOS_DEBUG
          /**\brief  Root record for tracked allocations from this
           * LogicalMemorySpace instance */
          static RecordBase s_root_record;
#endif

          const SpaceType m_space;

         protected:
          ~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
            if (Kokkos::Profiling::profileLibraryLoaded()) {
              Kokkos::Profiling::deallocateData(
                  Kokkos::Profiling::make_space_handle(m_space.name()),
                  RecordBase::m_alloc_ptr->m_label, data(), size());
            }
#endif

            m_space.deallocate(
                SharedAllocationRecord<void, void>::m_alloc_ptr,
                SharedAllocationRecord<void, void>::m_alloc_size);
          }
          SharedAllocationRecord() = default;

          SharedAllocationRecord(
              const SpaceType& arg_space, const std::string& arg_label,
              const size_t arg_alloc_size,
              const RecordBase::function_type arg_dealloc = &deallocate)
              : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
                    &SharedAllocationRecord<SpaceType, void>::s_root_record,
#endif
                    reinterpret_cast<SharedAllocationHeader*>(
                        arg_space.allocate(arg_label.c_str(),sizeof(SharedAllocationHeader) +
                                           arg_alloc_size, arg_alloc_size)),
                    sizeof(SharedAllocationHeader) + arg_alloc_size,
                    arg_dealloc),
                m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
            if (Kokkos::Profiling::profileLibraryLoaded()) {
              Kokkos::Profiling::allocateData(
                  Kokkos::Profiling::make_space_handle(arg_space.name()),
                  arg_label, data(), arg_alloc_size);
            }
#endif
            // Fill in the Header information
            RecordBase::m_alloc_ptr->m_record =
                static_cast<SharedAllocationRecord<void, void>*>(this);

            strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
                    SharedAllocationHeader::maximum_label_length);
            // Set last element zero, in case c_str is too long
            RecordBase::m_alloc_ptr
                ->m_label[SharedAllocationHeader::maximum_label_length - 1] =
                (char)0;
          }

         public:
          inline std::string get_label() const {
            return std::string(RecordBase::head()->m_label);
          }
          KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
              const SpaceType& arg_space, const std::string& arg_label,
              const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
            return new SharedAllocationRecord(arg_space, arg_label,
                                              arg_alloc_size);
#else
            (void)arg_space;
            (void)arg_label;
            (void)arg_alloc_size;
            return (SharedAllocationRecord*)0;
#endif
          }

          /**\brief  Allocate tracked memory in the space */
          static void* allocate_tracked(const SpaceType& arg_space,
                                        const std::string& arg_label,
                                        const size_t arg_alloc_size) {
            if (!arg_alloc_size) return (void*)0;

            SharedAllocationRecord* const r =
                allocate(arg_space, arg_label, arg_alloc_size);

            RecordBase::increment(r);

            return r->data();
          }

          /**\brief  Reallocate tracked memory in the space */
          static void* reallocate_tracked(void* const arg_alloc_ptr,
                                          const size_t arg_alloc_size) {
            SharedAllocationRecord* const r_old = get_record(arg_alloc_ptr);
            SharedAllocationRecord* const r_new =
                allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

            Kokkos::Impl::DeepCopy<SpaceType, SpaceType>(
                r_new->data(), r_old->data(),
                std::min(r_old->size(), r_new->size()));

            RecordBase::increment(r_new);
            RecordBase::decrement(r_old);

            return r_new->data();
          }
          /**\brief  Deallocate tracked memory in the space */
          static void deallocate_tracked(void* const arg_alloc_ptr) {
            if (arg_alloc_ptr != nullptr) {
              SharedAllocationRecord* const r = get_record(arg_alloc_ptr);

              RecordBase::decrement(r);
            }
          }

          static SharedAllocationRecord* get_record(void* alloc_ptr) {
            using Header     = SharedAllocationHeader;
            using RecordHost = SharedAllocationRecord<SpaceType, void>;

            SharedAllocationHeader const* const head =
                alloc_ptr ? Header::get_header(alloc_ptr)
                          : (SharedAllocationHeader*)nullptr;
            RecordHost* const record =
                head ? static_cast<RecordHost*>(head->m_record)
                     : (RecordHost*)0;

            if (!alloc_ptr || record->m_alloc_ptr != head) {
              Kokkos::Impl::throw_runtime_exception(std::string(
                  "Kokkos::Impl::SharedAllocationRecord< SpaceType , "
                  "void >::get_record ERROR"));
            }

            return record;
          }
#ifdef KOKKOS_DEBUG
          static void print_records(std::ostream& s, const SpaceType&,
                                    bool detail = false) {
            SharedAllocationRecord<void, void>::print_host_accessible_records(
                s, "HostSpace", &s_root_record, detail);
          }
#else
          static void print_records(std::ostream&, const SpaceType&,
                                    bool detail = false) {
            (void)detail;
            throw_runtime_exception(
                "SharedAllocationRecord<HostSpace>::print_records only works "
                "with "
                "KOKKOS_DEBUG enabled");
          }
#endif
        };
#ifdef KOKKOS_DEBUG
        /**\brief  Root record for tracked allocations from this HostSpace
         * instance */
        template <const char* Name, class BaseSpace,
                  class DefaultExecutionSpace,
                  bool SharesAccessSemanticsWithBase>
        RecordBase<BaseSpace, DefaultExecutionSpace,
                   SharesAccessSemanticsWithBase>::s_root_record;
#endif

        }  // namespace Impl

        }  // namespace Kokkos

        //----------------------------------------------------------------------------

        namespace Kokkos {

        namespace Impl {

        template <class Namer, class BaseSpace, class DefaultExecutionSpace,
                  bool SharesAccessSemanticsWithBase, class ExecutionSpace>
        struct DeepCopy<
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       SharesAccessSemanticsWithBase>,
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       SharesAccessSemanticsWithBase>,
            ExecutionSpace> {
          DeepCopy(void* dst, void* src, size_t n) {
            DeepCopy<BaseSpace, BaseSpace, ExecutionSpace>(dst, src, n);
          }
          DeepCopy(const ExecutionSpace& exec, void* dst, void* src, size_t n) {
            DeepCopy<BaseSpace, BaseSpace, ExecutionSpace>(exec, dst, src, n);
          }
        };

        template <class Namer, class BaseSpace, class DefaultExecutionSpace,
                  bool SharesAccessSemanticsWithBase, class ExecutionSpace,
                  class SourceSpace>
        struct DeepCopy<
            SourceSpace,
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       SharesAccessSemanticsWithBase>,
            ExecutionSpace> {
          DeepCopy(void* dst, void* src, size_t n) {
            DeepCopy<SourceSpace, BaseSpace, ExecutionSpace>(dst, src, n);
          }
          DeepCopy(const ExecutionSpace& exec, void* dst, void* src, size_t n) {
            DeepCopy<SourceSpace, BaseSpace, ExecutionSpace>(exec, dst, src, n);
          }
        };

        template <class Namer, class BaseSpace, class DefaultExecutionSpace,
                  bool SharesAccessSemanticsWithBase, class ExecutionSpace,
                  class DestinationSpace>
        struct DeepCopy<
            Kokkos::LogicalMemorySpace<BaseSpace, DefaultExecutionSpace, Namer,
                                       SharesAccessSemanticsWithBase>,
            DestinationSpace, ExecutionSpace> {
          DeepCopy(void* dst, void* src, size_t n) {
            DeepCopy<BaseSpace, DestinationSpace, ExecutionSpace>(dst, src, n);
          }
          DeepCopy(const ExecutionSpace& exec, void* dst, void* src, size_t n) {
            DeepCopy<BaseSpace, DestinationSpace, ExecutionSpace>(exec, dst,
                                                                  src, n);
          }
        };
        }  // namespace Impl

        }  // namespace Kokkos
