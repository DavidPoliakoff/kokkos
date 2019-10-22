/*
 //@HEADER
 // ************************************************************************
 //
 //                        Kokkos v. 2.0
 //              Copyright (2014) Sandia Corporation
 //
 // Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
 // THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
 // EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 // PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
#include <dlfcn.h>
#include <vector>
#if defined(KOKKOS_ENABLE_PROFILING)
#define KOKKOS_ENABLE_TUNING  // TODO DZP: make this a build system option
#include <impl/Kokkos_Profiling.hpp>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
namespace Kokkos {
namespace Profiling {

static initFunction initProfileLibrary         = nullptr;
static finalizeFunction finalizeProfileLibrary = nullptr;

static beginFunction beginForCallback    = nullptr;
static beginFunction beginScanCallback   = nullptr;
static beginFunction beginReduceCallback = nullptr;
static endFunction endForCallback        = nullptr;
static endFunction endScanCallback       = nullptr;
static endFunction endReduceCallback     = nullptr;

static pushFunction pushRegionCallback = nullptr;
static popFunction popRegionCallback   = nullptr;

static allocateDataFunction allocateDataCallback     = nullptr;
static deallocateDataFunction deallocateDataCallback = nullptr;

static beginDeepCopyFunction beginDeepCopyCallback = nullptr;
static endDeepCopyFunction endDeepCopyCallback     = nullptr;

static createProfileSectionFunction createSectionCallback   = nullptr;
static startProfileSectionFunction startSectionCallback     = nullptr;
static stopProfileSectionFunction stopSectionCallback       = nullptr;
static destroyProfileSectionFunction destroySectionCallback = nullptr;

static profileEventFunction profileEventCallback = nullptr;
}  // end namespace Profiling

namespace Tuning {
size_t getNewContextId();
size_t getCurrentContextId();
void decrementCurrentContextId();
size_t getNewVariableId();

static size_t kernel_name_context_variable_id;
static size_t kernel_type_context_variable_id;
static tuningVariableDeclarationFunction tuningVariableDeclarationCallback =
    nullptr;
static tuningVariableValueFunction tuningVariableValueCallback = nullptr;
static contextVariableDeclarationFunction contextVariableDeclarationCallback =
    nullptr;
static contextEndFunction contextEndCallback = nullptr;

}  // end namespace Tuning

namespace Profiling {

bool profileLibraryLoaded() { return (initProfileLibrary != nullptr); }

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID) {
  if (beginForCallback != nullptr) {
    Kokkos::fence();
    (*beginForCallback)(kernelPrefix.c_str(), devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    size_t uniqIds[] = {Kokkos::Tuning::kernel_name_context_variable_id,
                        Kokkos::Tuning::kernel_type_context_variable_id};
    Kokkos::Tuning::VariableValue contextValues[] = {
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_name_context_variable_id,
            kernelPrefix.c_str()),
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_type_context_variable_id, "parallel_for")};
    Kokkos::Tuning::declareContextVariableValues(Tuning::getNewContextId(), 2,
                                                 uniqIds, contextValues);
#endif
  }
}

void endParallelFor(const uint64_t kernelID) {
  if (endForCallback != nullptr) {
    Kokkos::fence();
    (*endForCallback)(kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Kokkos::Tuning::endContext(Kokkos::Tuning::getCurrentContextId());
#endif
  }
}

void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID) {
  if (beginScanCallback != nullptr) {
    Kokkos::fence();
    (*beginScanCallback)(kernelPrefix.c_str(), devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    size_t uniqIds[] = {Kokkos::Tuning::kernel_name_context_variable_id,
                        Kokkos::Tuning::kernel_type_context_variable_id};
    Kokkos::Tuning::VariableValue contextValues[] = {
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_name_context_variable_id,
            kernelPrefix.c_str()),
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_type_context_variable_id, "parallel_for")};
    Kokkos::Tuning::declareContextVariableValues(
        Kokkos::Tuning::getNewContextId(), 2, uniqIds, contextValues);
#endif
  }
}

void endParallelScan(const uint64_t kernelID) {
  if (endScanCallback != nullptr) {
    Kokkos::fence();
    (*endScanCallback)(kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    Kokkos::Tuning::endContext(Kokkos::Tuning::getCurrentContextId());
#endif
  }
}

void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID) {
  if (beginReduceCallback != nullptr) {
    Kokkos::fence();
    (*beginReduceCallback)(kernelPrefix.c_str(), devID, kernelID);
#ifdef KOKKOS_ENABLE_TUNING
    size_t uniqIds[] = {Kokkos::Tuning::kernel_name_context_variable_id,
                        Kokkos::Tuning::kernel_type_context_variable_id};
    Kokkos::Tuning::VariableValue contextValues[] = {
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_name_context_variable_id,
            kernelPrefix.c_str()),
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_type_context_variable_id, "parallel_for")};
    Kokkos::Tuning::declareContextVariableValues(
        Kokkos::Tuning::getNewContextId(), 2, uniqIds, contextValues);
#endif
  }
}

void endParallelReduce(const uint64_t kernelID) {
  if (endReduceCallback != nullptr) {
    Kokkos::fence();
    (*endReduceCallback)(kernelID);
  }
#ifdef KOKKOS_ENABLE_TUNING
  Kokkos::Tuning::endContext(Kokkos::Tuning::getCurrentContextId());
#endif
}

void pushRegion(const std::string& kName) {
  if (pushRegionCallback != nullptr) {
    Kokkos::fence();
    (*pushRegionCallback)(kName.c_str());
  }
}

void popRegion() {
  if (popRegionCallback != nullptr) {
    Kokkos::fence();
    (*popRegionCallback)();
  }
}

void allocateData(const SpaceHandle space, const std::string label,
                  const void* ptr, const uint64_t size) {
  if (allocateDataCallback != nullptr) {
    (*allocateDataCallback)(space, label.c_str(), ptr, size);
  }
}

void deallocateData(const SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size) {
  if (deallocateDataCallback != nullptr) {
    (*deallocateDataCallback)(space, label.c_str(), ptr, size);
  }
}

void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size) {
  if (beginDeepCopyCallback != nullptr) {
    (*beginDeepCopyCallback)(dst_space, dst_label.c_str(), dst_ptr, src_space,
                             src_label.c_str(), src_ptr, size);
#ifdef KOKKOS_ENABLE_TUNING
    size_t uniqIds[] = {Kokkos::Tuning::kernel_name_context_variable_id,
                        Kokkos::Tuning::kernel_type_context_variable_id};
    Kokkos::Tuning::VariableValue contextValues[] = {
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_name_context_variable_id,
            "deep_copy_kernel"),
        Kokkos::Tuning::make_variable_value(
            Kokkos::Tuning::kernel_type_context_variable_id,
            "deep_copy")};  // TODO DZP: should deep copy have context variables
                            // for source and destination features?
    Kokkos::Tuning::declareContextVariableValues(
        Kokkos::Tuning::getNewContextId(), 2, uniqIds, contextValues);
#endif
  }
}

void endDeepCopy() {
  if (endDeepCopyCallback != nullptr) {
    (*endDeepCopyCallback)();
#ifdef KOKKOS_ENABLE_TUNING
    Kokkos::Tuning::endContext(Kokkos::Tuning::getCurrentContextId());
#endif
  }
}

void createProfileSection(const std::string& sectionName, uint32_t* secID) {
  if (createSectionCallback != nullptr) {
    (*createSectionCallback)(sectionName.c_str(), secID);
  }
}

void startSection(const uint32_t secID) {
  if (startSectionCallback != nullptr) {
    (*startSectionCallback)(secID);
  }
}

void stopSection(const uint32_t secID) {
  if (stopSectionCallback != nullptr) {
    (*stopSectionCallback)(secID);
  }
}

void destroyProfileSection(const uint32_t secID) {
  if (destroySectionCallback != nullptr) {
    (*destroySectionCallback)(secID);
  }
}

void markEvent(const std::string& eventName) {
  if (profileEventCallback != nullptr) {
    (*profileEventCallback)(eventName.c_str());
  }
}

SpaceHandle make_space_handle(const char* space_name) {
  SpaceHandle handle;
  handle.name = space_name;
  return handle;
}

}  // end namespace Profiling

namespace Tuning {

static std::unordered_map<size_t, std::unordered_set<size_t>>
    features_per_context;
static std::unordered_set<size_t> active_features;
static std::unordered_map<size_t, VariableValue> feature_values;

void declareTuningVariable(const std::string& variableName, size_t uniqID,
                           VariableInfo info) {
  if (tuningVariableDeclarationCallback != nullptr) {
    (*tuningVariableDeclarationCallback)(variableName.c_str(), uniqID, info);
  }
}

void declareContextVariable(const std::string& variableName, size_t uniqID,
                            VariableInfo info,
                            Kokkos::Tuning::SetOrRange candidate_values) {
  if (contextVariableDeclarationCallback != nullptr) {
    (*contextVariableDeclarationCallback)(variableName.c_str(), uniqID, info,
                                          candidate_values);
  }
}

void declareContextVariableValues(size_t contextId, size_t count,
                                  size_t* uniqIds, VariableValue* values) {
  if (features_per_context.find(contextId) == features_per_context.end()) {
    features_per_context[contextId] =
        std::unordered_set<size_t>(uniqIds, uniqIds + count);
  } else {
    features_per_context[contextId].insert(uniqIds, uniqIds + count);
  }
  active_features.insert(uniqIds, uniqIds + count);
  for (size_t x = 0; x < count; ++x) {
    feature_values[uniqIds[x]] = values[x];
  }
}

void requestTuningVariableValues(size_t contextId, size_t count,
                                 size_t* uniqIds, VariableValue* values,
                                 Kokkos::Tuning::SetOrRange* candidate_values) {
  std::vector<size_t> context_ids;
  std::vector<VariableValue> context_values;
  for (auto id : active_features) {
    context_ids.push_back(id);
    context_values.push_back(feature_values[id]);
  }
  if (tuningVariableValueCallback != nullptr) {
    (*tuningVariableValueCallback)(contextId, context_ids.size(),
                                   context_ids.data(), context_values.data(),
                                   count, uniqIds, values, candidate_values);
  }
}

void endContext(size_t contextId) {
  for (auto id : features_per_context[contextId]) {
    active_features.erase(id);
  }
  if (Kokkos::Tuning::contextEndCallback != nullptr) {
    (*contextEndCallback)(contextId);
  }
  decrementCurrentContextId();
}

bool haveTuningTool() { return (tuningVariableValueCallback != nullptr); }

VariableValue make_variable_value(size_t id, bool val) {
  VariableValue variable_value;
  variable_value.id               = id;
  variable_value.value.bool_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, int val) {
  VariableValue variable_value;
  variable_value.id              = id;
  variable_value.value.int_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, double val) {
  VariableValue variable_value;
  variable_value.id                 = id;
  variable_value.value.double_value = val;
  return variable_value;
}
VariableValue make_variable_value(size_t id, const char* val) {
  VariableValue variable_value;
  variable_value.id                 = id;
  variable_value.value.string_value = val;
  return variable_value;
}

}  // end namespace Tuning

namespace Profiling {

void initialize() {
  // Make sure initialize calls happens only once
  static int is_initialized = 0;
  if (is_initialized) return;
  is_initialized = 1;

  void* firstProfileLibrary;

  char* envProfileLibrary = getenv("KOKKOS_PROFILE_LIBRARY");

  // If we do not find a profiling library in the environment then exit
  // early.
  if (envProfileLibrary == nullptr) {
    return;
  }

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
      beginForCallback = reinterpret_cast<beginFunction>(p1);
      auto p2 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_scan");
      beginScanCallback = reinterpret_cast<beginFunction>(p2);
      auto p3 = dlsym(firstProfileLibrary, "kokkosp_begin_parallel_reduce");
      beginReduceCallback = reinterpret_cast<beginFunction>(p3);

      auto p4         = dlsym(firstProfileLibrary, "kokkosp_end_parallel_scan");
      endScanCallback = reinterpret_cast<endFunction>(p4);
      auto p5         = dlsym(firstProfileLibrary, "kokkosp_end_parallel_for");
      endForCallback  = reinterpret_cast<endFunction>(p5);
      auto p6 = dlsym(firstProfileLibrary, "kokkosp_end_parallel_reduce");
      endReduceCallback = reinterpret_cast<endFunction>(p6);

      auto p7            = dlsym(firstProfileLibrary, "kokkosp_init_library");
      initProfileLibrary = reinterpret_cast<initFunction>(p7);
      auto p8 = dlsym(firstProfileLibrary, "kokkosp_finalize_library");
      finalizeProfileLibrary = reinterpret_cast<finalizeFunction>(p8);

      auto p9 = dlsym(firstProfileLibrary, "kokkosp_push_profile_region");
      pushRegionCallback = reinterpret_cast<pushFunction>(p9);
      auto p10 = dlsym(firstProfileLibrary, "kokkosp_pop_profile_region");
      popRegionCallback = reinterpret_cast<popFunction>(p10);

      auto p11 = dlsym(firstProfileLibrary, "kokkosp_allocate_data");
      allocateDataCallback = reinterpret_cast<allocateDataFunction>(p11);
      auto p12 = dlsym(firstProfileLibrary, "kokkosp_deallocate_data");
      deallocateDataCallback = reinterpret_cast<deallocateDataFunction>(p12);

      auto p13 = dlsym(firstProfileLibrary, "kokkosp_begin_deep_copy");
      beginDeepCopyCallback = reinterpret_cast<beginDeepCopyFunction>(p13);
      auto p14            = dlsym(firstProfileLibrary, "kokkosp_end_deep_copy");
      endDeepCopyCallback = reinterpret_cast<endDeepCopyFunction>(p14);

      auto p15 = dlsym(firstProfileLibrary, "kokkosp_create_profile_section");
      createSectionCallback =
          reinterpret_cast<createProfileSectionFunction>(p15);
      auto p16 = dlsym(firstProfileLibrary, "kokkosp_start_profile_section");
      startSectionCallback = reinterpret_cast<startProfileSectionFunction>(p16);
      auto p17 = dlsym(firstProfileLibrary, "kokkosp_stop_profile_section");
      stopSectionCallback = reinterpret_cast<stopProfileSectionFunction>(p17);
      auto p18 = dlsym(firstProfileLibrary, "kokkosp_destroy_profile_section");
      destroySectionCallback =
          reinterpret_cast<destroyProfileSectionFunction>(p18);

      auto p19 = dlsym(firstProfileLibrary, "kokkosp_profile_event");
      profileEventCallback = reinterpret_cast<profileEventFunction>(p19);

      // TODO DZP: move to its own section
      auto p20 = dlsym(firstProfileLibrary, "kokkosp_declare_tuning_variable");
      Kokkos::Tuning::tuningVariableDeclarationCallback =
          reinterpret_cast<Kokkos::Tuning::tuningVariableDeclarationFunction>(
              p20);
      auto p21 = dlsym(firstProfileLibrary, "kokkosp_declare_context_variable");
      Kokkos::Tuning::contextVariableDeclarationCallback =
          reinterpret_cast<Kokkos::Tuning::contextVariableDeclarationFunction>(
              p21);
      auto p22 =
          dlsym(firstProfileLibrary, "kokkosp_request_tuning_variable_values");
      Kokkos::Tuning::tuningVariableValueCallback =
          reinterpret_cast<Kokkos::Tuning::tuningVariableValueFunction>(p22);
      auto p23 = dlsym(firstProfileLibrary, "kokkosp_end_context");
      Kokkos::Tuning::contextEndCallback =
          reinterpret_cast<Kokkos::Tuning::contextEndFunction>(p23);

      Kokkos::Tuning::VariableInfo kernel_name;
      kernel_name.type = Kokkos::Tuning::ValueType::kokkos_value_text;
      kernel_name.category =
          Kokkos::Tuning::StatisticalCategory::kokkos_value_categorical;
      kernel_name.valueQuantity =
          Kokkos::Tuning::CandidateValueType::kokkos_value_unbounded;
      Kokkos::Tuning::kernel_name_context_variable_id =
          Kokkos::Tuning::getNewVariableId();
      Kokkos::Tuning::kernel_type_context_variable_id =
          Kokkos::Tuning::getNewVariableId();

      Kokkos::Tuning::SetOrRange kernel_type_variable_candidates;
      kernel_type_variable_candidates.set.size = 4;
      kernel_type_variable_candidates.set.id =
          Kokkos::Tuning::kernel_type_context_variable_id;

      std::array<Kokkos::Tuning::VariableValue, 4> candidate_values = {
          Kokkos::Tuning::make_variable_value(
              Kokkos::Tuning::kernel_type_context_variable_id, "parallel_for"),
          Kokkos::Tuning::make_variable_value(
              Kokkos::Tuning::kernel_type_context_variable_id,
              "parallel_reduce"),
          Kokkos::Tuning::make_variable_value(
              Kokkos::Tuning::kernel_type_context_variable_id, "parallel_scan"),
          Kokkos::Tuning::make_variable_value(
              Kokkos::Tuning::kernel_type_context_variable_id, "parallel_copy"),
      };

      kernel_type_variable_candidates.set.values = candidate_values.data();

      Kokkos::Tuning::SetOrRange
          kernel_name_candidates;  // TODO DZP: an empty set in SetOrRange if
                                   // things are unbounded? Maybe an empty
                                   // struct in the union just for
                                   // clarification? Or unify the tag and the
                                   // data
      kernel_name_candidates.set.size = 0;
      kernel_name_candidates.set.id =
          Kokkos::Tuning::kernel_name_context_variable_id;

      Kokkos::Tuning::declareContextVariable(
          "kokkos.kernel_name", Kokkos::Tuning::kernel_name_context_variable_id,
          kernel_name, kernel_name_candidates);

      Kokkos::Tuning::VariableInfo kernel_type;
      kernel_type.type = Kokkos::Tuning::ValueType::kokkos_value_text;
      kernel_type.category =
          Kokkos::Tuning::StatisticalCategory::kokkos_value_categorical;
      kernel_type.valueQuantity =
          Kokkos::Tuning::CandidateValueType::kokkos_value_set;

      Kokkos::Tuning::declareContextVariable(
          "kokkos.kernel_type", Kokkos::Tuning::kernel_type_context_variable_id,
          kernel_type, kernel_type_variable_candidates);
    }
  }

  if (initProfileLibrary != nullptr) {
    (*initProfileLibrary)(0, (uint64_t)KOKKOSP_INTERFACE_VERSION, (uint32_t)0,
                          nullptr);
  }

  free(envProfileCopy);
}

void finalize() {
  // Make sure finalize calls happens only once
  static int is_finalized = 0;
  if (is_finalized) return;
  is_finalized = 1;

  if (finalizeProfileLibrary != nullptr) {
    (*finalizeProfileLibrary)();

    // Set all profile hooks to nullptr to prevent
    // any additional calls. Once we are told to
    // finalize, we mean it
    initProfileLibrary     = nullptr;
    finalizeProfileLibrary = nullptr;

    beginForCallback    = nullptr;
    beginScanCallback   = nullptr;
    beginReduceCallback = nullptr;
    endScanCallback     = nullptr;
    endForCallback      = nullptr;
    endReduceCallback   = nullptr;

    pushRegionCallback = nullptr;
    popRegionCallback  = nullptr;

    allocateDataCallback   = nullptr;
    deallocateDataCallback = nullptr;

    beginDeepCopyCallback = nullptr;
    endDeepCopyCallback   = nullptr;

    createSectionCallback  = nullptr;
    startSectionCallback   = nullptr;
    stopSectionCallback    = nullptr;
    destroySectionCallback = nullptr;

    profileEventCallback = nullptr;
    // TODO DZP: move to its own section
    Kokkos::Tuning::tuningVariableDeclarationCallback  = nullptr;
    Kokkos::Tuning::contextVariableDeclarationCallback = nullptr;
    Kokkos::Tuning::tuningVariableValueCallback        = nullptr;
    Kokkos::Tuning::contextEndCallback                 = nullptr;
  }
}
}  // namespace Profiling

namespace Tuning {

static size_t& getContextCounter() {
  static size_t x;
  return x;
}
static size_t& getVariableCounter() {
  static size_t x;
  return ++x;
}

size_t getNewContextId() { return ++getContextCounter(); }
size_t getCurrentContextId() { return getContextCounter(); }
void decrementCurrentContextId() { --getContextCounter(); }
size_t getNewVariableId() { return getVariableCounter(); }

}  // end namespace Tuning

}  // namespace Kokkos

#else

// TODO DZP, handle the off case

#include <impl/Kokkos_Profiling_Interface.hpp>
#include <cstring>

namespace Kokkos {
namespace Profiling {

bool profileLibraryLoaded() { return false; }

void beginParallelFor(const std::string&, const uint32_t, uint64_t*) {}
void endParallelFor(const uint64_t) {}
void beginParallelScan(const std::string&, const uint32_t, uint64_t*) {}
void endParallelScan(const uint64_t) {}
void beginParallelReduce(const std::string&, const uint32_t, uint64_t*) {}
void endParallelReduce(const uint64_t) {}

void pushRegion(const std::string&) {}
void popRegion() {}
void createProfileSection(const std::string&, uint32_t*) {}
void startSection(const uint32_t) {}
void stopSection(const uint32_t) {}
void destroyProfileSection(const uint32_t) {}

void markEvent(const std::string&) {}

void allocateData(const SpaceHandle, const std::string, const void*,
                  const uint64_t) {}
void deallocateData(const SpaceHandle, const std::string, const void*,
                    const uint64_t) {}

void beginDeepCopy(const SpaceHandle, const std::string, const void*,
                   const SpaceHandle, const std::string, const void*,
                   const uint64_t) {}
void endDeepCopy() {}

void initialize() {}
void finalize() {}

}  // namespace Profiling

namespace Tuning {
void declareTuningVariable(const std::string& variableName, int uniqID,
                           VariableInfo info) {}

void declareContextVariable(const std::string& variableName, int uniqID,
                            VariableInfo info) {}

void declareContextVariableValues(int contextId, int count, int* uniqIds,
                                  VariableValue* values) {}

void endContext(int contextId) {}

void requestTuningVariableValues(int count, int* uniqIds,
                                 VariableValue* values);
size_t getNewContextId() { return 0; }
size_t getCurrentContextId() { return 0; }
size_t getNewVariableId() { return 0; }

}  // end namespace Tuning

}  // namespace Kokkos

#endif
