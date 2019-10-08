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

#ifndef KOKKOSP_INTERFACE_HPP
#define KOKKOSP_INTERFACE_HPP

#include <Kokkos_Macros.hpp>
#include <cinttypes>
#include <cstddef>
#include <Kokkos_Core_fwd.hpp>
#include <string>

#include <iostream>
#include <cstdlib>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <dlfcn.h>

#include <impl/Kokkos_Profiling_DeviceInfo.hpp>

#define KOKKOSP_INTERFACE_VERSION 20191080
#define KOKKOS_ENABLE_TUNING // TODO DZP: this needs to be a proper build system option
namespace Kokkos {
namespace Profiling {

struct SpaceHandle {
  SpaceHandle(const char* space_name);
  char name[64];
};

} // end namespace Profiling

namespace Tuning {
struct ValueSet {
  int size;
  void* values;
};

struct ValueRange {
  void* lower;
  void* upper;
  bool openLower;
  bool openUpper;
};

struct VariableInfo {
  enum valueType {
    floating_point, // TODO DZP: single and double? One or the other?
    integer,
    text,
    boolean
  };
  enum statisticalCategory {
    categorical, // unordered distinct objects
    ordinal,     // ordered distinct objects
    interval,    // ordered distinct objects for which distance matters
    ratio        // ordered distinct objects for which distance matters, division matters, and the concept of zero exists
  };
  enum candidateValueType {
    set,        // I am one of [2,3,4,5]
    range,      // I am somewhere in [2,12)
    unbounded   // I am text, but we don't know at declaration time what values are appropriate. Only valid for Context Variables
    // TODO DZP: not handled: 1 + 3x, sets of ranges, range with hole (zero). Do these matter?
  };
  /** 
  enum candidateValueQuantity {
    point,
    list,
  };
  */
  union setOrRange {
    ValueSet set;
    ValueRange range; 
  };
  valueType type;
  statisticalCategory category;
  candidateValueType valueQuantity;
  setOrRange  value;
};

// TODO DZP: VariableInfo subclasses to automate some of this

struct VariableValue  {
  union ValueUnion {
     int int_value;
     double double_value;
     char* string_value;
  };
  ValueUnion value;
  VariableValue(int val) { value.int_value = val; }
  VariableValue(double val) { value.double_value = val; }
  VariableValue(char* val) { value.string_value = val; }
  VariableValue(const char* val) { value.string_value = const_cast<char*>(val); }
};

template<typename T>
VariableValue make_variable_value(T&& in){
  return VariableValue(std::forward<T>(in));        
}

} // end namespace Tuning

namespace Profiling {

typedef void (*initFunction)(const int, const uint64_t, const uint32_t,
                             KokkosPDeviceInfo*);
typedef void (*finalizeFunction)();
typedef void (*beginFunction)(const char*, const uint32_t, uint64_t*);
typedef void (*endFunction)(uint64_t);

typedef void (*pushFunction)(const char*);
typedef void (*popFunction)();

typedef void (*allocateDataFunction)(const SpaceHandle, const char*,
                                     const void*, const uint64_t);
typedef void (*deallocateDataFunction)(const SpaceHandle, const char*,
                                       const void*, const uint64_t);

typedef void (*createProfileSectionFunction)(const char*, uint32_t*);
typedef void (*startProfileSectionFunction)(const uint32_t);
typedef void (*stopProfileSectionFunction)(const uint32_t);
typedef void (*destroyProfileSectionFunction)(const uint32_t);

typedef void (*profileEventFunction)(const char*);

typedef void (*beginDeepCopyFunction)(SpaceHandle, const char*, const void*,
                                      SpaceHandle, const char*, const void*,
                                      uint64_t);
typedef void (*endDeepCopyFunction)();
} //end namespace Profiling

namespace Tuning {
typedef void (*tuningVariableDeclarationFunction)(const char*, const size_t, VariableInfo info); 
typedef void (*contextVariableDeclarationFunction)(const char*, const size_t, VariableInfo info); 
typedef void(*tuningVariableValueFunction)(const size_t count, const size_t* uniqIds, VariableValue*);
typedef void (*contextVariableValueFunction)(const size_t contextId, const size_t count, const size_t* uniqIds, VariableValue* values);
typedef void (*contextEndFunction)(const size_t);

} // end namespace Tuning

namespace Profiling {
bool profileLibraryLoaded();

void beginParallelFor(const std::string& kernelPrefix, const uint32_t devID,
                      uint64_t* kernelID);
void endParallelFor(const uint64_t kernelID);
void beginParallelScan(const std::string& kernelPrefix, const uint32_t devID,
                       uint64_t* kernelID);
void endParallelScan(const uint64_t kernelID);
void beginParallelReduce(const std::string& kernelPrefix, const uint32_t devID,
                         uint64_t* kernelID);
void endParallelReduce(const uint64_t kernelID);

void pushRegion(const std::string& kName);
void popRegion();

void createProfileSection(const std::string& sectionName, uint32_t* secID);
void startSection(const uint32_t secID);
void stopSection(const uint32_t secID);
void destroyProfileSection(const uint32_t secID);

void markEvent(const std::string* evName);

void allocateData(const SpaceHandle space, const std::string label,
                  const void* ptr, const uint64_t size);
void deallocateData(const SpaceHandle space, const std::string label,
                    const void* ptr, const uint64_t size);

void beginDeepCopy(const SpaceHandle dst_space, const std::string dst_label,
                   const void* dst_ptr, const SpaceHandle src_space,
                   const std::string src_label, const void* src_ptr,
                   const uint64_t size);
void endDeepCopy();

void initialize();
void finalize();

}  // namespace Profiling
namespace Tuning  {
void declareTuningVariable(const std::string& variableName, size_t uniqID, VariableInfo info); 

void declareContextVariable(const std::string& variableName, size_t uniqID, VariableInfo info); 

void declareContextVariableValues(size_t contextId, size_t count, size_t* uniqIds, VariableValue* values);

void endContext(size_t contextId);

void requestTuningVariableValues(size_t count, size_t* uniqIds, VariableValue* values);

bool haveTuningTool();

} // end namespace Tuning
}  // namespace Kokkos

#else
namespace Kokkos {
namespace Profiling {

struct SpaceHandle {
  SpaceHandle(const char* space_name);
  char name[64];
};

bool profileLibraryLoaded();

void beginParallelFor(const std::string&, const uint32_t, uint64_t*);
void endParallelFor(const uint64_t);
void beginParallelScan(const std::string&, const uint32_t, uint64_t*);
void endParallelScan(const uint64_t);
void beginParallelReduce(const std::string&, const uint32_t, uint64_t*);
void endParallelReduce(const uint64_t);

void pushRegion(const std::string&);
void popRegion();
void createProfileSection(const std::string&, uint32_t*);
void startSection(const uint32_t);
void stopSection(const uint32_t);
void destroyProfileSection(const uint32_t);

void markEvent(const std::string&);

void allocateData(const SpaceHandle, const std::string, const void*,
                  const uint64_t);
void deallocateData(const SpaceHandle, const std::string, const void*,
                    const uint64_t);

void beginDeepCopy(const SpaceHandle, const std::string, const void*,
                   const SpaceHandle, const std::string, const void*,
                   const uint64_t);
void endDeepCopy();

void initialize();
void finalize();

} // end namespace Profiling
namespace Tuning  {
void declareTuningVariable(const std::string& variableName, int uniqID, VariableInfo info); 

void declareContextVariable(const std::string& variableName, int uniqID, VariableInfo info); 

void declareContextVariableValues(int contextId, int count, int* uniqIds, VariableValue* values);

void endContext(int contextId);

void requestTuningVariableValues(int count, int* uniqIds, VariableValue* values);

} // end namespace Tuning
} // end namespace Kokkos
#endif
#endif
