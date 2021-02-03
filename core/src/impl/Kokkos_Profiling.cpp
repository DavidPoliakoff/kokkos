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

#include <Kokkos_Macros.hpp>
#include <Kokkos_Tuners.hpp>
#include <impl/Kokkos_Profiling.hpp>
#if defined(KOKKOS_ENABLE_LIBDL)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iostream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace Kokkos {

namespace Tools {

namespace Experimental {
#ifdef KOKKOS_ENABLE_TUNING
static size_t kernel_name_context_variable_id;
static size_t kernel_type_context_variable_id;
static std::unordered_map<size_t, std::unordered_set<size_t>>
    features_per_context;
static std::unordered_set<size_t> active_features;
static std::unordered_map<size_t, VariableValue> feature_values;
static std::unordered_map<size_t, VariableInfo> variable_metadata;
#endif
static EventSet current_callbacks;
static EventSet backup_callbacks;
static EventSet no_profiling;

bool eventSetsEqual(const EventSet& l, const EventSet& r) {
  return l.init == r.init && l.finalize == r.finalize &&
         l.parse_args == r.parse_args && l.print_help == r.print_help &&
         l.begin_parallel_for == r.begin_parallel_for &&
         l.end_parallel_for == r.end_parallel_for &&
         l.begin_parallel_reduce == r.begin_parallel_reduce &&
         l.end_parallel_reduce == r.end_parallel_reduce &&
         l.begin_parallel_scan == r.begin_parallel_scan &&
         l.end_parallel_scan == r.end_parallel_scan &&
         l.push_region == r.push_region && l.pop_region == r.pop_region &&
         l.allocate_data == r.allocate_data &&
         l.deallocate_data == r.deallocate_data &&
         l.create_profile_section == r.create_profile_section &&
         l.start_profile_section == r.start_profile_section &&
         l.stop_profile_section == r.stop_profile_section &&
         l.destroy_profile_section == r.destroy_profile_section &&
         l.profile_event == r.profile_event &&
         l.begin_deep_copy == r.begin_deep_copy &&
         l.end_deep_copy == r.end_deep_copy && l.begin_fence == r.begin_fence &&
         l.end_fence == r.end_fence && l.sync_dual_view == r.sync_dual_view &&
         l.modify_dual_view == r.modify_dual_view &&
         l.declare_metadata == r.declare_metadata &&
         l.declare_input_type == r.declare_input_type &&
         l.declare_output_type == r.declare_output_type &&
         l.end_tuning_context == r.end_tuning_context &&
         l.begin_tuning_context == r.begin_tuning_context &&
         l.request_output_values == r.request_output_values &&
         l.declare_optimization_goal == r.declare_optimization_goal;
}
}  // namespace Experimental
bool profileLibraryLoaded() {
  return !Experimental::eventSetsEqual(Experimental::current_callbacks,
                                       Experimental::no_profiling);
}

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID) {
}

void endParallelFor(const uint64_t kernelID) {
}

void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID) {
}

void endParallelScan(const uint64_t kernelID) {
}

void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID) {
}

void endParallelReduce(const uint64_t kernelID) {
}

void pushRegion(const std::string& kName) {
}

void popRegion() {
}

void allocateData(const SpaceHandle space, const std::string label,
                  const void* ptr, const uint64_t size) {
}

void deallocateData(const SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size) {
}

void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size) {
}

void endDeepCopy() {
}

void beginFence(const std::string name, const uint32_t deviceId,
                uint64_t* handle) {
}

void endFence(const uint64_t handle) {
}

void createProfileSection(const std::string& sectionName, uint32_t* secID) {
}

void startSection(const uint32_t secID) {
}

void stopSection(const uint32_t secID) {
}

void destroyProfileSection(const uint32_t secID) {
}

void markEvent(const std::string& eventName) {
}

bool printHelp(const std::string& args) {
  return false;
}

void parseArgs(int _argc, char** _argv) {
  if (Experimental::current_callbacks.parse_args != nullptr && _argc > 0) {
    (*Experimental::current_callbacks.parse_args)(_argc, _argv);
  }
}

void parseArgs(const std::string& args) {
  if (Experimental::current_callbacks.parse_args == nullptr) {
    return;
  }
  using strvec_t = std::vector<std::string>;
  auto tokenize  = [](const std::string& line, const std::string& delimiters) {
    strvec_t _result{};
    std::size_t _bidx = 0;  // position that is the beginning of the new string
    std::size_t _didx = 0;  // position of the delimiter in the string
    while (_bidx < line.length() && _didx < line.length()) {
      // find the first character (starting at _didx) that is not a delimiter
      _bidx = line.find_first_not_of(delimiters, _didx);
      // if no more non-delimiter chars, done
      if (_bidx == std::string::npos) break;
      // starting at the position of the new string, find the next delimiter
      _didx = line.find_first_of(delimiters, _bidx);
      // starting at the position of the new string, get the characters
      // between this position and the next delimiter
      std::string _tmp = line.substr(_bidx, _didx - _bidx);
      // don't add empty strings
      if (!_tmp.empty()) _result.emplace_back(_tmp);
    }
    return _result;
  };
  auto vargs = tokenize(args, " \t");
  if (vargs.size() == 0) return;
  auto _argc          = static_cast<int>(vargs.size());
  char** _argv        = new char*[_argc + 1];
  _argv[vargs.size()] = nullptr;
  for (int i = 0; i < _argc; ++i) {
    auto& _str = vargs.at(i);
    _argv[i]   = new char[_str.length() + 1];
    std::memcpy(_argv[i], _str.c_str(), _str.length() * sizeof(char));
    _argv[i][_str.length()] = '\0';
  }
  parseArgs(_argc, _argv);
  for (int i = 0; i < _argc; ++i) {
    delete[] _argv[i];
  }
  delete[] _argv;
}

SpaceHandle make_space_handle(const char* space_name) {
  SpaceHandle handle;
  strncpy(handle.name, space_name, 63);
  return handle;
}

void initialize(const std::string& profileLibrary) {
  // Make sure initialize calls happens only once
  static int is_initialized = 0;
  if (is_initialized) return;
  is_initialized = 1;

#ifdef KOKKOS_ENABLE_LIBDL
  void* firstProfileLibrary = nullptr;

  if (profileLibrary.empty()) return;

  char* envProfileLibrary = const_cast<char*>(profileLibrary.c_str());

  char* envProfileCopy =
      (char*)malloc(sizeof(char) * (strlen(envProfileLibrary) + 1));
  sprintf(envProfileCopy, "%s", envProfileLibrary);

  char* profileLibraryName = strtok(envProfileCopy, ";");

  if ((profileLibraryName != nullptr) &&
      (strcmp(profileLibraryName, "") != 0)) {
    firstProfileLibrary = dlopen(profileLibraryName, RTLD_NOW | RTLD_GLOBAL);

    if (firstProfileLibrary == nullptr) {
      std::cerr << "Error: Unable to load KokkosP library: "
                << profileLibraryName << std::endl;
      std::cerr << "dlopen(" << profileLibraryName
                << ", RTLD_NOW | RTLD_GLOBAL) failed with " << dlerror()
                << '\n';
    } else {
#ifdef KOKKOS_ENABLE_PROFILING_LOAD_PRINT
      std::cout << "KokkosP: Library Loaded: " << profileLibraryName
                << std::endl;
#endif
      // dlsym returns a pointer to an object, while we want to assign to
      // pointer to function A direct cast will give warnings hence, we have to
      // workaround the issue by casting pointer to pointers.
      auto p1 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_for");
      Experimental::set_begin_parallel_for_callback(
          *reinterpret_cast<beginFunction*>(&p1));
      auto p2 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_scan");
      Experimental::set_begin_parallel_scan_callback(
          *reinterpret_cast<beginFunction*>(&p2));
      auto p3 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_reduce");
      Experimental::set_begin_parallel_reduce_callback(
          *reinterpret_cast<beginFunction*>(&p3));

      auto p4 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_scan");
      Experimental::set_end_parallel_scan_callback(
          *reinterpret_cast<endFunction*>(&p4));
      auto p5 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_for");
      Experimental::set_end_parallel_for_callback(
          *reinterpret_cast<endFunction*>(&p5));
      auto p6 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_reduce");
      Experimental::set_end_parallel_reduce_callback(
          *reinterpret_cast<endFunction*>(&p6));

      auto p7 = dlsym(firstProfileLibrary, "kokkosp_init_library");
      Experimental::set_init_callback(*reinterpret_cast<initFunction*>(&p7));
      auto p8 = dlsym(firstProfileLibrary, "kokkosp_finalize_library");
      Experimental::set_finalize_callback(
          *reinterpret_cast<finalizeFunction*>(&p8));

      auto p9 = dlsym(firstProfileLibrary, "kokkosp_push_profile_region");
      Experimental::set_push_region_callback(
          *reinterpret_cast<pushFunction*>(&p9));
      auto p10 = dlsym(firstProfileLibrary, "kokkosp_pop_profile_region");
      Experimental::set_pop_region_callback(
          *reinterpret_cast<popFunction*>(&p10));

      auto p11 = dlsym(firstProfileLibrary, "kokkosp_allocate_data");
      Experimental::set_allocate_data_callback(
          *reinterpret_cast<allocateDataFunction*>(&p11));
      auto p12 = dlsym(firstProfileLibrary, "kokkosp_deallocate_data");
      Experimental::set_deallocate_data_callback(
          *reinterpret_cast<deallocateDataFunction*>(&p12));

      auto p13 = dlsym(firstProfileLibrary, "kokkosp_begin_deep_copy");
      Experimental::set_begin_deep_copy_callback(
          *reinterpret_cast<beginDeepCopyFunction*>(&p13));
      auto p14 = dlsym(firstProfileLibrary, "kokkosp_end_deep_copy");
      Experimental::set_end_deep_copy_callback(
          *reinterpret_cast<endDeepCopyFunction*>(&p14));

      auto p15 = dlsym(firstProfileLibrary, "kokkosp_begin_fence");
      Experimental::set_begin_fence_callback(
          *reinterpret_cast<beginFenceFunction*>(&p15));
      auto p16 = dlsym(firstProfileLibrary, "kokkosp_end_fence");
      Experimental::set_end_fence_callback(
          *reinterpret_cast<endFenceFunction*>(&p16));

      auto p17 = dlsym(firstProfileLibrary, "kokkosp_dual_view_sync");
      Experimental::set_dual_view_sync_callback(
          *reinterpret_cast<dualViewSyncFunction*>(&p17));
      auto p18 = dlsym(firstProfileLibrary, "kokkosp_dual_view_modify");
      Experimental::set_dual_view_modify_callback(
          *reinterpret_cast<dualViewModifyFunction*>(&p18));

      auto p19 = dlsym(firstProfileLibrary, "kokkosp_declare_metadata");
      Experimental::set_declare_metadata_callback(
          *(reinterpret_cast<declareMetadataFunction*>(&p19)));

      auto p20 = dlsym(firstProfileLibrary, "kokkosp_create_profile_section");
      Experimental::set_create_profile_section_callback(
          *(reinterpret_cast<createProfileSectionFunction*>(&p20)));
      auto p21 = dlsym(firstProfileLibrary, "kokkosp_start_profile_section");
      Experimental::set_start_profile_section_callback(
          *reinterpret_cast<startProfileSectionFunction*>(&p21));
      auto p22 = dlsym(firstProfileLibrary, "kokkosp_stop_profile_section");
      Experimental::set_stop_profile_section_callback(
          *reinterpret_cast<stopProfileSectionFunction*>(&p22));
      auto p23 = dlsym(firstProfileLibrary, "kokkosp_destroy_profile_section");
      Experimental::set_destroy_profile_section_callback(
          *(reinterpret_cast<destroyProfileSectionFunction*>(&p23)));

      auto p24 = dlsym(firstProfileLibrary, "kokkosp_profile_event");
      Experimental::set_profile_event_callback(
          *reinterpret_cast<profileEventFunction*>(&p24));

#ifdef KOKKOS_ENABLE_TUNING
      auto p25 = dlsym(firstProfileLibrary, "kokkosp_declare_output_type");
      Experimental::set_declare_output_type_callback(
          *reinterpret_cast<Experimental::outputTypeDeclarationFunction*>(
              &p25));

      auto p26 = dlsym(firstProfileLibrary, "kokkosp_declare_input_type");
      Experimental::set_declare_input_type_callback(
          *reinterpret_cast<Experimental::inputTypeDeclarationFunction*>(&p26));
      auto p27 = dlsym(firstProfileLibrary, "kokkosp_request_values");
      Experimental::set_request_output_values_callback(
          *reinterpret_cast<Experimental::requestValueFunction*>(&p27));
      auto p28 = dlsym(firstProfileLibrary, "kokkosp_end_context");
      Experimental::set_end_context_callback(
          *reinterpret_cast<Experimental::contextEndFunction*>(&p28));
      auto p29 = dlsym(firstProfileLibrary, "kokkosp_begin_context");
      Experimental::set_begin_context_callback(
          *reinterpret_cast<Experimental::contextBeginFunction*>(&p29));
      auto p30 =
          dlsym(firstProfileLibrary, "kokkosp_declare_optimization_goal");
      Experimental::set_declare_optimization_goal_callback(
          *reinterpret_cast<Experimental::optimizationGoalDeclarationFunction*>(
              &p30));
#endif  // KOKKOS_ENABLE_TUNING

      auto p31 = dlsym(firstProfileLibrary, "kokkosp_print_help");
      Experimental::set_print_help_callback(
          *reinterpret_cast<printHelpFunction*>(&p31));
      auto p32 = dlsym(firstProfileLibrary, "kokkosp_parse_args");
      Experimental::set_parse_args_callback(
          *reinterpret_cast<parseArgsFunction*>(&p32));
    }
  }
#else
  (void)profileLibrary;
#endif  // KOKKOS_ENABLE_LIBDL
  if (Experimental::current_callbacks.init != nullptr) {
    (*Experimental::current_callbacks.init)(
        0, (uint64_t)KOKKOSP_INTERFACE_VERSION, (uint32_t)0, nullptr);
  }

#ifdef KOKKOS_ENABLE_TUNING
  Experimental::VariableInfo kernel_name;
  kernel_name.type = Experimental::ValueType::kokkos_value_string;
  kernel_name.category =
      Experimental::StatisticalCategory::kokkos_value_categorical;
  kernel_name.valueQuantity =
      Experimental::CandidateValueType::kokkos_value_unbounded;

  std::array<std::string, 4> candidate_values = {
      "parallel_for",
      "parallel_reduce",
      "parallel_scan",
      "parallel_copy",
  };

  Experimental::SetOrRange kernel_type_variable_candidates =
      Experimental::make_candidate_set(4, candidate_values.data());

  Experimental::kernel_name_context_variable_id =
      Experimental::declare_input_type("kokkos.kernel_name", kernel_name);

  Experimental::VariableInfo kernel_type;
  kernel_type.type = Experimental::ValueType::kokkos_value_string;
  kernel_type.category =
      Experimental::StatisticalCategory::kokkos_value_categorical;
  kernel_type.valueQuantity =
      Experimental::CandidateValueType::kokkos_value_set;
  kernel_type.candidates = kernel_type_variable_candidates;
  Experimental::kernel_type_context_variable_id =
      Experimental::declare_input_type("kokkos.kernel_type", kernel_type);

#endif

  Experimental::no_profiling.init     = nullptr;
  Experimental::no_profiling.finalize = nullptr;

  Experimental::no_profiling.begin_parallel_for    = nullptr;
  Experimental::no_profiling.begin_parallel_scan   = nullptr;
  Experimental::no_profiling.begin_parallel_reduce = nullptr;
  Experimental::no_profiling.end_parallel_scan     = nullptr;
  Experimental::no_profiling.end_parallel_for      = nullptr;
  Experimental::no_profiling.end_parallel_reduce   = nullptr;

  Experimental::no_profiling.push_region     = nullptr;
  Experimental::no_profiling.pop_region      = nullptr;
  Experimental::no_profiling.allocate_data   = nullptr;
  Experimental::no_profiling.deallocate_data = nullptr;

  Experimental::no_profiling.begin_deep_copy = nullptr;
  Experimental::no_profiling.end_deep_copy   = nullptr;

  Experimental::no_profiling.create_profile_section  = nullptr;
  Experimental::no_profiling.start_profile_section   = nullptr;
  Experimental::no_profiling.stop_profile_section    = nullptr;
  Experimental::no_profiling.destroy_profile_section = nullptr;

  Experimental::no_profiling.profile_event = nullptr;

  Experimental::no_profiling.declare_input_type    = nullptr;
  Experimental::no_profiling.declare_output_type   = nullptr;
  Experimental::no_profiling.request_output_values = nullptr;
  Experimental::no_profiling.end_tuning_context    = nullptr;
#ifdef KOKKOS_ENABLE_LIBDL
  free(envProfileCopy);
#endif
}

void finalize() {
  // Make sure finalize calls happens only once
  static int is_finalized = 0;
  if (is_finalized) return;
  is_finalized = 1;

  if (Experimental::current_callbacks.finalize != nullptr) {
    (*Experimental::current_callbacks.finalize)();

    Experimental::pause_tools();
  }
#ifdef KOKKOS_ENABLE_TUNING
  // clean up string candidate set
  for (auto& metadata_pair : Experimental::variable_metadata) {
    auto metadata = metadata_pair.second;
    if ((metadata.type == Experimental::ValueType::kokkos_value_string) &&
        (metadata.valueQuantity ==
         Experimental::CandidateValueType::kokkos_value_set)) {
      auto candidate_set = metadata.candidates.set;
      delete[] candidate_set.values.string_value;
    }
  }
#endif
}

void syncDualView(const std::string& label, const void* const ptr,
                  bool to_device) {
}
void modifyDualView(const std::string& label, const void* const ptr,
                    bool on_device) {
}

void declareMetadata(const std::string& key, const std::string& value) {
}

}  // namespace Tools

namespace Tools {
namespace Experimental {
void set_init_callback(initFunction callback) {
}
void set_finalize_callback(finalizeFunction callback) {
}
void set_parse_args_callback(parseArgsFunction callback) {
}
void set_print_help_callback(printHelpFunction callback) {
}
void set_begin_parallel_for_callback(beginFunction callback) {
}
void set_end_parallel_for_callback(endFunction callback) {
}
void set_begin_parallel_reduce_callback(beginFunction callback) {
}
void set_end_parallel_reduce_callback(endFunction callback) {
}
void set_begin_parallel_scan_callback(beginFunction callback) {
}
void set_end_parallel_scan_callback(endFunction callback) {
}
void set_push_region_callback(pushFunction callback) {
}
void set_pop_region_callback(popFunction callback) {
}
void set_allocate_data_callback(allocateDataFunction callback) {
}
void set_deallocate_data_callback(deallocateDataFunction callback) {
}
void set_create_profile_section_callback(
    createProfileSectionFunction callback) {
}
void set_start_profile_section_callback(startProfileSectionFunction callback) {
}
void set_stop_profile_section_callback(stopProfileSectionFunction callback) {
}
void set_destroy_profile_section_callback(
    destroyProfileSectionFunction callback) {
}
void set_profile_event_callback(profileEventFunction callback) {
}
void set_begin_deep_copy_callback(beginDeepCopyFunction callback) {
}
void set_end_deep_copy_callback(endDeepCopyFunction callback) {
}
void set_begin_fence_callback(beginFenceFunction callback) {
}
void set_end_fence_callback(endFenceFunction callback) {
}

void set_dual_view_sync_callback(dualViewSyncFunction callback) {
}
void set_dual_view_modify_callback(dualViewModifyFunction callback) {
}
void set_declare_metadata_callback(declareMetadataFunction callback) {
}

void set_declare_output_type_callback(outputTypeDeclarationFunction callback) {
}
void set_declare_input_type_callback(inputTypeDeclarationFunction callback) {
}
void set_request_output_values_callback(requestValueFunction callback) {
}
void set_end_context_callback(contextEndFunction callback) {
}
void set_begin_context_callback(contextBeginFunction callback) {
}
void set_declare_optimization_goal_callback(
    optimizationGoalDeclarationFunction callback) {
}

void pause_tools() {
}

void resume_tools() {}

EventSet get_callbacks() { return current_callbacks; }
void set_callbacks(EventSet new_events) {}
}  // namespace Experimental
}  // namespace Tools

namespace Profiling {
bool profileLibraryLoaded() { return false; }

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID) {
}
void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID) {
}
void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID) {
}
void endParallelFor(const uint64_t kernelID) {
}
void endParallelReduce(const uint64_t kernelID) {
}
void endParallelScan(const uint64_t kernelID) {
}

void pushRegion(const std::string& kName) {}
void popRegion() {}

void createProfileSection(const std::string& sectionName, uint32_t* secID) {
}
void destroyProfileSection(const uint32_t secID) {
}

void startSection(const uint32_t secID) {}

void stopSection(const uint32_t secID) {}

void markEvent(const std::string& eventName) {
}
void allocateData(const SpaceHandle handle, const std::string name,
                  const void* data, const uint64_t size) {
}
void deallocateData(const SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size) {
}

void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size) {
}
void endDeepCopy() {}

void finalize() {}
void initialize(const std::string& profileLibrary) {
}

bool printHelp(const std::string& args) {
	return false;
}
void parseArgs(const std::string& args) {}
void parseArgs(int _argc, char** _argv) {
}

SpaceHandle make_space_handle(const char* space_name) {
  return Kokkos::Tools::make_space_handle(space_name);
}
}  // namespace Profiling

}  // namespace Kokkos

// Tuning

namespace Kokkos {
namespace Tools {
namespace Experimental {
static size_t& get_context_counter() {
  static size_t x;
  return x;
}
static size_t& get_variable_counter() {
  static size_t x;
  return ++x;
}

size_t get_new_context_id() { return ++get_context_counter(); }
size_t get_current_context_id() { return get_context_counter(); }
void decrement_current_context_id() { --get_context_counter(); }
size_t get_new_variable_id() { return get_variable_counter(); }

size_t declare_output_type(const std::string& variableName, VariableInfo info) {
  size_t variableId = get_new_variable_id();
  return variableId;
}

size_t declare_input_type(const std::string& variableName, VariableInfo info) {
  size_t variableId = get_new_variable_id();
  return variableId;
}

void set_input_values(size_t contextId, size_t count, VariableValue* values) {
}
#include <iostream>
void request_output_values(size_t contextId, size_t count,
                           VariableValue* values) {
}

#ifdef KOKKOS_ENABLE_TUNING
static std::unordered_map<size_t, size_t> optimization_goals;
#endif

void begin_context(size_t contextId) {
}
void end_context(size_t contextId) {
}

bool have_tuning_tool() {
  return false;
}

VariableValue make_variable_value(size_t id, int64_t val) {
  VariableValue variable_value;
  variable_value.type_id         = id;
  variable_value.value.int_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, double val) {
  VariableValue variable_value;
  variable_value.type_id            = id;
  variable_value.value.double_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, const std::string& val) {
  VariableValue variable_value;
  variable_value.type_id = id;
  strncpy(variable_value.value.string_value, val.c_str(),
          KOKKOS_TOOLS_TUNING_STRING_LENGTH - 1);
  return variable_value;
}
SetOrRange make_candidate_set(size_t size, std::string* data) {
  SetOrRange value_set;
  value_set.set.values.string_value = new TuningString[size];
  for (size_t x = 0; x < size; ++x) {
    strncpy(value_set.set.values.string_value[x], data[x].c_str(),
            KOKKOS_TOOLS_TUNING_STRING_LENGTH - 1);
  }
  value_set.set.size = size;
  return value_set;
}
SetOrRange make_candidate_set(size_t size, int64_t* data) {
  SetOrRange value_set;
  value_set.set.size             = size;
  value_set.set.values.int_value = data;
  return value_set;
}
SetOrRange make_candidate_set(size_t size, double* data) {
  SetOrRange value_set;
  value_set.set.size                = size;
  value_set.set.values.double_value = data;
  return value_set;
}
SetOrRange make_candidate_range(double lower, double upper, double step,
                                bool openLower = false,
                                bool openUpper = false) {
  SetOrRange value_range;
  value_range.range.lower.double_value = lower;
  value_range.range.upper.double_value = upper;
  value_range.range.step.double_value  = step;
  value_range.range.openLower          = openLower;
  value_range.range.openUpper          = openUpper;
  return value_range;
}

SetOrRange make_candidate_range(int64_t lower, int64_t upper, int64_t step,
                                bool openLower = false,
                                bool openUpper = false) {
  SetOrRange value_range;
  value_range.range.lower.int_value = lower;
  value_range.range.upper.int_value = upper;
  value_range.range.step.int_value  = step;
  value_range.range.openLower       = openLower;
  value_range.range.openUpper       = openUpper;
  return value_range;
}

size_t get_new_context_id();
size_t get_current_context_id();
void decrement_current_context_id();
size_t get_new_variable_id();
void declare_optimization_goal(const size_t context,
                               const OptimizationGoal& goal) {
}
}  // end namespace Experimental
}  // end namespace Tools

}  // end namespace Kokkos
